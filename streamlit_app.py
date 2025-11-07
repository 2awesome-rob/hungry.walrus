import sqlite3
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st


# Page config
st.set_page_config(page_title="HockeyStat", page_icon="🏒")
st.title("🏒 Warrior Stats")

@st.cache_data
def load_dfs_from_database(season: int = 2025, db_path: str = "data/HockeyStat.db") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the relevant tables from the HockeyStat SQLite DB.
    - Filters `Teams` by season.
    - Loads Players, Games, PlayerGameStats and GoalieGameStats restricted to teams in that season.
    Returns five dataframes in the order: teams, rosters, games, player_stats, goalie_stats.
    If the DB is missing or tables are not present, returns empty dataframes with sensible columns.
    """
    # helper empty dataframes
    df_teams = pd.DataFrame(columns=["team_id", "club", "team", "season"])
    df_rosters = pd.DataFrame(columns=["player_id", "team_id", "jersey_num", "name", "position"])
    df_games = pd.DataFrame(columns=["game_id", "date", "home_team_id", "home_score", "away_team_id", "away_score"])
    df_players = pd.DataFrame(columns=["game_id", "player_id", "goals", "assists", "penalty_min", "active"])
    df_goalies = pd.DataFrame(columns=["game_id", "player_id", "shots_faced", "saves", "goals_allowed", "result", "active"])

    try:
        with sqlite3.connect(db_path) as conn:
            # Teams for season
            q = "SELECT team_id, club, team, season FROM Teams WHERE season = ?"
            df_teams = pd.read_sql_query(q, conn, params=(season,))

            if df_teams.empty:
                return df_teams, df_rosters, df_games, df_players, df_goalies

            team_ids = df_teams["team_id"].unique().tolist()

            # helper to build IN clause
            placeholders = ",".join(["?"] * len(team_ids))

            # Players / Rosters
            q = f"SELECT player_id, team_id, jersey_num, name, position FROM Players WHERE team_id IN ({placeholders})"
            df_rosters = pd.read_sql_query(q, conn, params=team_ids)

            # Games involving these teams
            q = f"SELECT game_id, date, home_team_id, home_score, away_team_id, away_score FROM Games WHERE home_team_id IN ({placeholders}) OR away_team_id IN ({placeholders})"
            df_games = pd.read_sql_query(q, conn, params=team_ids + team_ids)
            if not df_games.empty and "date" in df_games.columns:
                df_games["date"] = pd.to_datetime(df_games["date"], errors="coerce")
                df_games = df_games.sort_values("date")

            # Player game stats: restrict to player_ids and game_ids we loaded (if available)
            player_ids = df_rosters["player_id"].unique().tolist()
            game_ids = df_games["game_id"].unique().tolist()

            if player_ids and game_ids:
                p_placeholders = ",".join(["?"] * len(player_ids))
                g_placeholders = ",".join(["?"] * len(game_ids))
                q = f"SELECT game_id, player_id, goals, assists, penalty_min, active FROM PlayerGameStats WHERE player_id IN ({p_placeholders}) AND game_id IN ({g_placeholders})"
                df_players = pd.read_sql_query(q, conn, params=player_ids + game_ids)
            else:
                # try to at least load any PlayerGameStats if present
                try:
                    q = "SELECT game_id, player_id, goals, assists, penalty_min, active FROM PlayerGameStats"
                    df_players = pd.read_sql_query(q, conn)
                except Exception:
                    df_players = df_players

            # Compute points safely
            for col in ("goals", "assists"):
                if col in df_players.columns:
                    df_players[col] = pd.to_numeric(df_players[col], errors="coerce").fillna(0).astype(int)
            if "goals" in df_players.columns and "assists" in df_players.columns:
                df_players["points"] = df_players["goals"] + df_players["assists"]

            # Goalies
            if game_ids:
                g_placeholders = ",".join(["?"] * len(game_ids))
                q = f"SELECT game_id, player_id, shots_faced, saves, goals_allowed, result, active FROM GoalieGameStats WHERE game_id IN ({g_placeholders})"
                df_goalies = pd.read_sql_query(q, conn, params=game_ids)
            else:
                try:
                    q = "SELECT game_id, player_id, shots_faced, saves, goals_allowed, result, active FROM GoalieGameStats"
                    df_goalies = pd.read_sql_query(q, conn)
                except Exception:
                    df_goalies = df_goalies

            # Compute save percentage safely
            if not df_goalies.empty:
                df_goalies["shots_faced"] = pd.to_numeric(df_goalies["shots_faced"], errors="coerce").fillna(0).astype(int)
                df_goalies["saves"] = pd.to_numeric(df_goalies["saves"], errors="coerce").fillna(0).astype(int)
                df_goalies["save_pct"] = np.where(df_goalies["shots_faced"] > 0, df_goalies["saves"] / df_goalies["shots_faced"], np.nan)

    except Exception as exc:
        # If DB is missing or other DB error, return empty frames but log to the app
        st.warning(f"Could not load DB '{db_path}': {exc}")

    return df_teams, df_rosters, df_games, df_players, df_goalies


# Load data (safe: function returns empty dataframes if DB or tables are missing)
df_teams, df_rosters, df_games, df_players, df_goalies = load_dfs_from_database()

### make this a session state variable?
selected_team_name = "10UBlack Warriors"

### --- Streamlit UI scaffold ---
tab1, tab2, tab3 = st.tabs(["Team", "Players", "Admin"])

with tab1:
    st.subheader(f"{selected_team_name} Overview", divider="red")

    if df_teams.empty:
        st.info("No teams found for the selected season (or DB not found). Use Admin to add teams.")
    else:
        team_map = df_teams.set_index("team_id")["team"].to_dict()
        team_options = df_teams["team"].tolist().sort()
        #selected_team_name = st.selectbox("Select team", options=team_options, index=7, disabled=True)
        selected_team_id = int(df_teams[df_teams["team"] == selected_team_name].iloc[0]["team_id"])

        # Games involving this team
        if df_games.empty:
            st.info("No games found for this team/season.")
        else:
            is_home = df_games["home_team_id"] == selected_team_id
            is_away = df_games["away_team_id"] == selected_team_id
            team_games = df_games[is_home | is_away].copy()

            if team_games.empty:
                st.info("No games for the selected team in this season.")
            else:
                # Compute PF, PA and W/L/T
                def result_for_row(row, team_id):
                    hs = row.get("home_score")
                    as_ = row.get("away_score")
                    try:
                        hs = int(hs)
                        as_ = int(as_)
                    except Exception:
                        return "?"
                    if hs == as_:
                        return "T"
                    if row.get("home_team_id") == team_id:
                        return "W" if hs > as_ else "L"
                    else:
                        return "W" if as_ > hs else "L"

                team_games["team_score"] = team_games.apply(lambda r: r["home_score"] if r["home_team_id"] == selected_team_id else r["away_score"], axis=1)
                team_games["opp_score"] = team_games.apply(lambda r: r["away_score"] if r["home_team_id"] == selected_team_id else r["home_score"], axis=1)
                team_games["result_for_team"] = team_games.apply(lambda r: result_for_row(r, selected_team_id), axis=1)

                PF = pd.to_numeric(team_games["team_score"], errors="coerce").fillna(0).astype(int).sum()
                PA = pd.to_numeric(team_games["opp_score"], errors="coerce").fillna(0).astype(int).sum()
                W = (team_games["result_for_team"] == "W").sum()
                L = (team_games["result_for_team"] == "L").sum()
                T = (team_games["result_for_team"] == "T").sum()

                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Wins", int(W))
                mcol2.metric("Losses", int(L))
                mcol3.metric("Ties", int(T))

                mcol4, mcol5, mcol6 = st.columns(3)
                mcol4.metric("PF", int(PF))
                mcol5.metric("PA", int(PA))
                sd_val = int(PF - PA)
                mcol6.metric("SD", sd_val)

                # Prepare display table: map team ids to names and format date as date-only
                display = team_games.copy()
                if "date" in display.columns:
                    try:
                        display["date"] = pd.to_datetime(display["date"], errors="coerce")
                        # show most recent first for the table
                        display = display.sort_values("date", ascending=False)
                        display["date"] = display["date"].dt.strftime("%Y-%m-%d")
                    except Exception:
                        pass

                st.subheader(f"Recent Games", divider="red")
                display["home_team"] = display["home_team_id"].map(team_map).fillna(display["home_team_id"])
                display["visiting_team"] = display["away_team_id"].map(team_map).fillna(display["away_team_id"])

                cols = ["date", "home_team", "home_score", "visiting_team", "away_score", "result_for_team"]
                st.dataframe(display[cols].rename(columns={"result_for_team": "Result"}))

with tab2:
    def _active_mask(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s.fillna(False)
        num = pd.to_numeric(s, errors="coerce")
        if not num.isna().all():
            return num.fillna(0).astype(int) != 0
        return s.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
                
    if df_teams.empty or df_rosters.empty:
        st.info("No roster data available.")
    else:
        team_names = df_teams["team"].tolist().sort()
#        selected_team_name = st.selectbox("Team (for player list)", options=team_names, index=7, disabled=True)
        selected_team_id = int(df_teams[df_teams["team"] == selected_team_name].iloc[0]["team_id"])

        team_players = df_rosters[df_rosters["team_id"] == selected_team_id]
        if team_players.empty:
            st.info("No players found for this team.")
        else:
            player_options = team_players["name"].tolist()
            selected_player_name = st.selectbox("Select player", options=player_options)
            selected_player_id = int(team_players[team_players["name"] == selected_player_name].iloc[0]["player_id"])

            st.subheader(f"{selected_player_name} - Season Stats", divider="red")

            # If the selected player appears in goalie stats, show aggregated goalie metrics first
            g_stats = df_goalies[df_goalies["player_id"] == selected_player_id]
            if not g_stats.empty:
                "Goalie Stats"
                if "active" in g_stats.columns:
                    g_active = g_stats[_active_mask(g_stats["active"])].copy()
                else:
                    g_active = g_stats.copy()

                if not g_active.empty:
                    gp = int(g_active["game_id"].nunique())
                    shots = int(g_active["shots_faced"].sum()) if "shots_faced" in g_active.columns else 0
                    saves = int(g_active["saves"].sum()) if "saves" in g_active.columns else 0
                    ga = int(g_active["goals_allowed"].sum()) if "goals_allowed" in g_active.columns else 0
                    save_pct = round(float(saves) / shots, 3) if shots > 0 else None
                    # compute W/L from result column (case-insensitive)
                    if "result" in g_active.columns:
                        res = g_active["result"].astype(str).str.upper()
                        wins = int((res == "W").sum())
                        losses = int((res == "L").sum())
                        shutout_count = int(g_active[(res == "W") & (g_active["goals_allowed"] == 0)].shape[0])
                        shutouts = shutout_count if shutout_count > 0 else None
                    else:
                        wins = 0
                        losses = 0
                        shutouts = None

                    gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)
                    gcol1.metric("W", wins, delta=shutouts)
                    gcol2.metric("L", losses)
                    gcol3.metric("Shots On", shots)
                    gcol4.metric("Saves", saves)
                    gcol5.metric("Save %", f"{save_pct*100:.1f}%" if save_pct is not None else "N/A")
                    st.markdown("---")

            # Player stats
            p_stats = df_players[df_players["player_id"] == selected_player_id]
            if p_stats.empty:
                st.info("No player game stats available for this player.")
            else:
                "Player Stats"
                if "active" in p_stats.columns:
                    active_mask = _active_mask(p_stats["active"])
                    p_stats_active = p_stats[active_mask].copy()
                else:
                    p_stats_active = p_stats.copy()

                if p_stats_active.empty:
                    st.info("Player has no active games this season.")
                else:
                    games_count = int(p_stats_active["game_id"].nunique())
                    goals_total = int(p_stats_active["goals"].sum())
                    assists_total = int(p_stats_active["assists"].sum())
                    points_total = int(p_stats_active.get("points", pd.Series(dtype=int)).sum()) if "points" in p_stats_active.columns else int(goals_total + assists_total)
                    penalty_total = int(pd.to_numeric(p_stats_active.get("penalty_min", pd.Series(0)), errors="coerce").fillna(0).sum())
                    hat_tricks = p_stats_active[p_stats_active["goals"] >= 3].shape[0]
                    hat_trick_label = f"{hat_tricks} HatTricks" if hat_tricks > 0 else None
                    play_makers = p_stats_active[p_stats_active["assists"] >= 3].shape[0]
                    play_maker_label = f"{play_makers} PlayMakers" if play_makers > 1 else None

                    pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
                    pcol1.metric("Games", games_count)
                    pcol2.metric("Goals", goals_total, delta=hat_trick_label)
                    pcol3.metric("Assists", assists_total, delta=play_maker_label)
                    pcol4.metric("Points", points_total)
                    pcol5.metric("PIM", penalty_total)

                    # per-game averages (safe formatting)
                    "Per Game Averages"
                    ga_goals = round(goals_total / games_count, 2) if games_count > 0 else 0.0
                    ga_assists = round(assists_total / games_count, 2) if games_count > 0 else 0.0
                    ga_points = round(points_total / games_count, 2) if games_count > 0 else 0.0
                    ga_pim = round(penalty_total / games_count, 2) if games_count > 0 else 0.0
                    pcol6, pcol7, pcol8, pcol9, pcol10 = st.columns(5)
                    pcol7.metric("GPG", ga_goals)
                    pcol8.metric("APG", ga_assists)
                    pcol9.metric("PPG", ga_points)
                    pcol10.metric("PIM/G", ga_pim)

with tab3:
    st.header("Admin Panel")
    tab31, tab32 = st.tabs(["Edit Games", "Edit Teams"])

    with tab31:
        st.subheader("Add Games")

        game_date = st.date_input("Game Date")
        home_team = st.selectbox("Home Team", df_teams["team"], index=7)
        away_team = st.selectbox("Away Team", df_teams["team"], index=1)

        with st.form("game_form"):
            st.write("Add / Edit Games here.")

            home_score = st.number_input("Home Score", min_value=0, step=1)
            away_score = st.number_input("Away Score", min_value=0, step=1)

            st.markdown("### Player Stats")
            selected_home_team_id = df_teams[df_teams["team"] == home_team]["team_id"].values[0]
            selected_away_team_id = df_teams[df_teams["team"] == away_team]["team_id"].values[0]

            home_roster = df_rosters[df_rosters["team_id"] == selected_home_team_id]
            away_roster = df_rosters[df_rosters["team_id"] == selected_away_team_id]
            combined_roster = pd.concat([home_roster, away_roster])

            player_stats = []
            for _, player in combined_roster.iterrows():
                st.markdown(f"**{player['name']} ({player['position']})**")
                active = st.checkbox(f"Active - {player['name']}", value=True, key=f"31active_{player['player_id']}")
                goals = st.number_input("Goals", min_value=0, step=1, key=f"31goals_{player['player_id']}")
                assists = st.number_input("Assists", min_value=0, step=1, key=f"31assists_{player['player_id']}")
                penalty_min = st.number_input("Penalty Minutes", min_value=0, step=1, key=f"31penalty_{player['player_id']}")
                player_stats.append({
                    "player_id": player["player_id"],
                    "active": active,
                    "goals": goals,
                    "assists": assists,
                    "penalty_min": penalty_min
                })

            st.markdown("### Goalie Stats")
            goalie_stats = []
            goalies = combined_roster[combined_roster["position"] == "G"]
            for _, goalie in goalies.iterrows():
                st.markdown(f"**{goalie['name']}**")
                active = st.checkbox(f"Active - {goalie['name']}", key=f"31g_active_{goalie['player_id']}")
                shots_faced = st.number_input("Shots Faced", min_value=0, step=1, key=f"31shots_{goalie['player_id']}")
                saves = st.number_input("Saves", min_value=0, step=1, key=f"saves_{goalie['player_id']}")
                goals_allowed = st.number_input("Goals Allowed", min_value=0, step=1, key=f"31ga_{goalie['player_id']}")
                result = st.radio("Result", ["W", "-", "L"], key=f"31result_{goalie['player_id']}", horizontal=True, index=1)
                goalie_stats.append({
                    "player_id": goalie["player_id"],
                    "active": active,
                    "shots_faced": shots_faced,
                    "saves": saves,
                    "goals_allowed": goals_allowed,
                    "result": result
                })

            submitted = st.form_submit_button("Submit")
        if submitted:

            with sqlite3.connect("data/HockeyStat.db") as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO Games (date, home_team_id, home_score, away_team_id, away_score)
                    VALUES (?, ?, ?, ?, ?)
                """, (game_date.isoformat(), selected_home_team_id, home_score, selected_away_team_id, away_score))
                game_id = cursor.lastrowid  

                for stat in player_stats:
                    cursor.execute("""
                        INSERT INTO PlayerGameStats (game_id, player_id, goals, assists, penalty_min, active)
                        VALUES (?, ?, ?, ?, ?, ?)
                """, (game_id, stat["player_id"], stat["goals"], stat["assists"], stat["penalty_min"], int(stat["active"])))

                for gstat in goalie_stats:
                    cursor.execute("""
                        INSERT INTO GoalieGameStats (game_id, player_id, shots_faced, saves, goals_allowed, result, active)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (game_id, gstat["player_id"], gstat["shots_faced"], gstat["saves"], gstat["goals_allowed"], gstat["result"], int(gstat["active"])))
                
                conn.commit()

            st.success("Game data submitted; Database updated.")

    with tab32:
        st.subheader("Manage Teams and Rosters")

        with st.form("team_form"):
            st.write("Add / Edit Teams and Rosters here.")

            team_mode = st.radio("Team Action", ["Add New Team", "Edit Existing Team"])
            if team_mode == "Add New Team":
                new_club = st.text_input("Club Name")
                new_team = st.text_input("Team Name")
                new_season = st.number_input("Season", min_value=2020, max_value=2030, value=2025)
            else:
                selected_team = st.selectbox("Select Team to Edit", df_teams["team"])
                team_row = df_teams[df_teams["team"] == selected_team].iloc[0]
                new_club = st.text_input("Club Name", value=team_row["club"])
                new_team = st.text_input("Team Name", value=team_row["team"])
                new_season = st.number_input("Season", min_value=2020, max_value=2030, value=team_row["season"])

            st.markdown("### Roster Management")
            roster_mode = st.radio("Roster Action", ["Add Player", "Edit Player"])
            if roster_mode == "Add Player":
                player_name = st.text_input("Player Name")
                jersey_num = st.number_input("Jersey Number", min_value=0, step=1)
                position = st.selectbox("Position", ["F", "D", "G"])
            else:
                team_id = df_teams[df_teams["team"] == new_team]["team_id"].values[0]
                team_roster = df_rosters[df_rosters["team_id"] == team_id]
                selected_player = st.selectbox("Select Player", team_roster["name"])
                player_row = team_roster[team_roster["name"] == selected_player].iloc[0]
                player_name = st.text_input("Player Name", value=player_row["name"])
                jersey_num = st.number_input("Jersey Number", min_value=0, step=1, value=player_row["jersey_num"])
                position = st.selectbox("Position", ["F", "D", "G"], index=["F", "D", "G"].index(player_row["position"]))

            submitted = st.form_submit_button("Submit")
            if submitted:
                with sqlite3.connect("data/HockeyStat.db") as conn:
                    cursor = conn.cursor()
                    
                    if team_mode == "Add New Team":
                        cursor.execute("""
                            INSERT INTO Teams (club, team, season)
                            VALUES (?, ?, ?)
                        """, (new_club, new_team, new_season))
                    else:
                        team_id = df_teams[df_teams["team"] == selected_team]["team_id"].values[0]
                        cursor.execute("""
                            UPDATE Teams SET club = ?, team = ?, season = ?
                            WHERE team_id = ?
                        """, (new_club, new_team, new_season, team_id))
                    
                    if roster_mode == "Add Player":
                        cursor.execute("""
                            INSERT INTO Players (team_id, jersey_num, name, position)
                            VALUES (?, ?, ?, ?)
                        """, (team_id, jersey_num, player_name, position))
                    else:
                        player_id = player_row["player_id"]
                        cursor.execute("""
                            UPDATE Players SET jersey_num = ?, name = ?, position = ?
                            WHERE player_id = ?
                        """, (jersey_num, player_name, position, player_id))

                    conn.commit()

                st.success("Team/Roster data submitted; Database updated.")

