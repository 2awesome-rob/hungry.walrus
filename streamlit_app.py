import sqlite3
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st


### Page config ###
st.set_page_config(page_title="HockeyStat", page_icon="🏒")

### Data loading function ###
@st.cache_data
def load_dfs_from_database(season: int = 2025, db_path: str = "data/HockeyStat.db") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the relevant tables from the HockeyStat SQLite DB.
    - Filters `Teams` by season.
    - Loads Players, Games, PlayerGameStats and GoalieGameStats restricted to teams in that season.
    Returns five dataframes in the order: teams, rosters, games, player_stats, goalie_stats.
    If the DB is missing or tables are not present, returns empty dataframes with sensible columns.
    """
    # helper empty dataframes
    df_teams = pd.DataFrame(columns=["team_id", "club", "team", "season", "location", "coach"])
    df_rosters = pd.DataFrame(columns=["player_id", "team_id", "jersey_num", "name", "position"])
    df_games = pd.DataFrame(columns=["game_id", "date", "home_team_id", "home_score", "away_team_id", "away_score", "overtime", "shootout"])
    df_players = pd.DataFrame(columns=["game_id", "player_id", "goals", "assists", "penalty_min", "active"])
    df_goalies = pd.DataFrame(columns=["game_id", "player_id", "shots_faced", "saves", "goals_allowed", "result", "active"])

    try:
        with sqlite3.connect(db_path) as conn:
            # Load Teams for the specified season
            q = "SELECT team_id, club, team, season FROM Teams WHERE season = ?"
            df_teams = pd.read_sql_query(q, conn, params=(season,))
            if df_teams.empty:
                return df_teams, df_rosters, df_games, df_players, df_goalies
            team_ids = df_teams["team_id"].unique().tolist()
            t_placeholders = ",".join(["?"] * len(team_ids))

            # Load Rosters for these teams
            q = f"SELECT player_id, team_id, jersey_num, name, position FROM Players WHERE team_id IN ({t_placeholders})"
            df_rosters = pd.read_sql_query(q, conn, params=team_ids)

            # Load Games involving these teams
            q = f"SELECT game_id, date, home_team_id, home_score, away_team_id, away_score FROM Games WHERE home_team_id IN ({t_placeholders}) OR away_team_id IN ({t_placeholders})"
            df_games = pd.read_sql_query(q, conn, params=team_ids + team_ids)
            if not df_games.empty and "date" in df_games.columns:
                df_games["date"] = pd.to_datetime(df_games["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                df_games = df_games.sort_values("date")

            # Load player and goalie stats: restrict to current rosters and season games only
            player_ids = df_rosters["player_id"].unique().tolist()
            game_ids = df_games["game_id"].unique().tolist()

            if player_ids and game_ids:
                p_placeholders = ",".join(["?"] * len(player_ids))
                g_placeholders = ",".join(["?"] * len(game_ids))
                q = f"SELECT game_id, player_id, goals, assists, penalty_min, active FROM PlayerGameStats WHERE player_id IN ({p_placeholders}) AND game_id IN ({g_placeholders})"
                df_players = pd.read_sql_query(q, conn, params=player_ids + game_ids)
                q = f"SELECT game_id, player_id, shots_faced, saves, goals_allowed, result, active FROM GoalieGameStats WHERE game_id IN ({g_placeholders})"
                df_goalies = pd.read_sql_query(q, conn, params=game_ids)
            else:
                df_players = df_players
                df_goalies = df_goalies

            # Add computed points column to player stats
            for col in ("goals", "assists"):
                if col in df_players.columns:
                    df_players[col] = pd.to_numeric(df_players[col], errors="coerce").fillna(0).astype(int)
            if "goals" in df_players.columns and "assists" in df_players.columns:
                df_players["points"] = df_players["goals"] + df_players["assists"]

            # Compute save percentage safely
            if not df_goalies.empty:
                df_goalies["shots_faced"] = pd.to_numeric(df_goalies["shots_faced"], errors="coerce").fillna(0).astype(int)
                df_goalies["saves"] = pd.to_numeric(df_goalies["saves"], errors="coerce").fillna(0).astype(int)
                df_goalies["save_pct"] = np.where(df_goalies["shots_faced"] > 0, df_goalies["saves"] / df_goalies["shots_faced"], np.nan)

    except Exception as exc:
        # If DB is missing or other DB error, return empty frames but log to the app
        st.warning(f"Could not load DB '{db_path}': {exc}")

    return df_teams, df_rosters, df_games, df_players, df_goalies

### helper functions
def _active_mask(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    num = pd.to_numeric(s, errors="coerce")
    if not num.isna().all():
        return num.fillna(0).astype(int) != 0
    return s.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])


### UI Header
st.title("🏒 HockeyStat Dashboard", help="Hockey Team and Player Statistics Tracker")
# Load data 
df_teams, df_rosters, df_games, df_players, df_goalies = load_dfs_from_database()

### make this a session state variable?
selected_team_name = "10UBlack Warriors"

if df_teams.empty:
    st.info("No teams found for the selected season (or DB not found). Use Admin to add teams.")
else:
    team_map = df_teams.set_index("team_id")["team"].to_dict()
    player_map = df_rosters.set_index("player_id")["name"].to_dict()
    team_options = sorted(df_teams["team"].tolist())
    #selected_team_name = st.selectbox("Select team", options=team_options, index=7, disabled=True)
    if selected_team_name not in team_options:
        selected_team_name = team_options[-1]
        st.info(f"Default Team not found; using '{selected_team_name}'")
    selected_team_id = int(df_teams[df_teams["team"] == selected_team_name].iloc[0]["team_id"])

    ### --- Streamlit UI scaffold ---
    tab1, tab2, tab3 = st.tabs(["Team", "Players", "Admin"])

    with tab1:
        st.subheader(f"{selected_team_name} Overview", divider="red")

        if df_games.empty:
            st.info("No games found for this season.")
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
                mcol6.metric("SD", int(PF - PA))

                # Prepare display table: map team ids to names and format date as date-only
                display = team_games.copy()
                if "date" in display.columns:
                    try:
                        display = display.sort_values("date", ascending=False)
                    except Exception:
                        pass

#                st.subheader(f"Recent Games", divider="red")
                st.badge("Recent Games", color="red")
                display["home_team"] = display["home_team_id"].map(team_map).fillna(display["home_team_id"])
                display["visiting_team"] = display["away_team_id"].map(team_map).fillna(display["away_team_id"])

                cols = ["date", "result_for_team", "away_score", "visiting_team", "home_team", "home_score"]
                st.dataframe(display[cols].rename(columns={"date":"Date",
                                                           "result_for_team": "Result",
                                                           "away_score": " Score",
                                                           "visiting_team": "Visitor",
                                                           "home_team": "at Home",
                                                           "home_score": "Score "
                                                           }), hide_index=True)
                st.markdown("---")
                st.badge("Goaie Stats", color="red")
                df_stats = df_goalies.merge(team_games[['game_id']], on="game_id", how="inner")
                df_stats = df_stats.merge(df_rosters[['player_id', 'name']], on='player_id', how='left')
                df_stats['wins'] = df_stats['result'].apply(lambda x: 1 if str(x).upper() == 'W' else 0)
                df_stats['losses'] = df_stats['result'].apply(lambda x: 1 if str(x).upper() == 'L' else 0)   
                df_stats = df_stats.groupby("name", as_index=False).agg({
                                            "wins": "sum", 
                                            "losses": "sum",
                                            "shots_faced": "sum",
                                            "saves": "sum",
                                            "goals_allowed": "sum"})
                df_stats = df_stats.sort_values(by="wins", ascending=False) 
                df_stats['save_pct'] = np.where(df_stats["shots_faced"] > 0, df_stats["saves"] / df_stats["shots_faced"], np.nan)   
                st.dataframe(df_stats.rename(columns={
                    "name": "Goalie",
                    "wins": "W",
                    "losses": "L",
                    "shots_faced": "SOG",
                    "saves": "Saves",
                    "save_pct": "Save %",
                    "goals_allowed": "Goals"
                }), hide_index=True)
                
                st.badge("Player Stats", color="red")
                df_stats = df_players.merge(team_games[['game_id']], on="game_id", how="inner")
                df_stats = df_stats.merge(df_rosters[['player_id', 'name']], on='player_id', how='left')
                df_stats = df_stats.groupby("name", as_index=False).agg({
                                            "active": "sum",   
                                            "goals": "sum",
                                            "assists": "sum",
                                            "points": "sum",
                                            "penalty_min": "sum"})
                df_stats = df_stats.sort_values(by="points", ascending=False) 
                st.dataframe(df_stats.rename(columns={
                    "name": "Player",
                    "active": "Games",
                    "goals": "Goals",
                    "assists": "Assists",
                    "points": "Points",
                    "penalty_min": "PIM"
                }), hide_index=True)


    with tab2:    
        if df_teams.empty or df_rosters.empty:
            st.info("No roster data available.")
        else:
            team_players = df_rosters[df_rosters["team_id"] == selected_team_id]
            if team_players.empty:
                st.info("No players found for this team.")
            else:
                player_options = sorted(team_players["name"].tolist())
                selected_player_name = st.selectbox("Select player", options=player_options)
                selected_player_id = int(team_players[team_players["name"] == selected_player_name].iloc[0]["player_id"])

                st.subheader(f"{selected_player_name} - Season Stats", divider="red")

                # If the selected player appears in goalie stats, show aggregated goalie metrics first
                g_stats = df_goalies[df_goalies["player_id"] == selected_player_id]
                if not g_stats.empty:
                    st.badge("Goalie Stats", color="red")
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
                    st.badge("Season Totals", color="red")
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

                        st.badge("Game Averages", color="red")
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

        st.text_input("admin password", type="password", key="admin_password_input")
        if st.session_state.get("admin_password_input", "") != "rip":
            st.warning("Input admin password.")
            st.stop()
        else:
            tab31, tab32 = st.tabs(["Edit Games", "Edit Teams"])

            with tab31:
                df_edit_games = df_games.copy(deep=True)
                selected_players_game_df = df_players.copy(deep=True)
                selected_goalies_game_df = df_goalies.copy(deep=True)
                df_edit_games["home_team"] = df_edit_games["home_team_id"].map(team_map).fillna(df_edit_games["home_team_id"])
                df_edit_games["away_team"] = df_edit_games["away_team_id"].map(team_map).fillna(df_edit_games["away_team_id"])
                cols = ["game_id", "date", "away_score", "away_team", "home_team", "home_score"]
                st.data_editor(df_edit_games[cols], height=200, use_container_width=True, num_rows="dynamic", key="game_data_editor")
                ### ensure changes are written back to the DB? -- TODO
                st.button("Save Game Changes", key="save_game_stats_button")
                if st.session_state.get("save_game_stats_button", False):
                    st.success("Game changes saved. (Functionality to write back to DB is not yet implemented.)")
                ### ensure changes are complient with data types and constraints? -- TODO

                selected_game = st.selectbox("Edit Game", df_edit_games['game_id'], index=None)
                selected_players_game_df = selected_players_game_df[selected_players_game_df["game_id"] == selected_game]
                selected_goalies_game_df = selected_goalies_game_df[selected_goalies_game_df["game_id"] == selected_game]
                selected_players_game_df = selected_players_game_df.merge(df_rosters[['player_id', 'name']], on='player_id', how='left')
                selected_goalies_game_df = selected_goalies_game_df.merge(df_rosters[['player_id', 'name']], on='player_id', how='left')
            
                cols = ["name", "goals", "assists", "penalty_min", "active"]
                st.data_editor(selected_players_game_df[cols], height=400, use_container_width=True, num_rows="fixed", key="player_stats_editor")
                cols = ["name", "shots_faced", "saves", "goals_allowed", "result", "active"]
                st.data_editor(selected_goalies_game_df[cols], height=100, use_container_width=True, num_rows="dynamic", key="goalie_stats_editor")
                ### ensure changes are written back to the DB? -- TODO
                ### ensure changes are complient with data types and constraints? -- TODO
                
            with tab32:
                df_edit_teams = df_teams.copy(deep=True)
                df_edit_players = df_rosters.copy(deep=True)
                st.data_editor(df_edit_teams[["club", "team", "season"]], height=300, use_container_width=True, num_rows="dynamic", key="team_data_editor")

                selected_team = st.selectbox("Edit Team Roster", df_edit_teams['team'], index=None)
            
                if selected_team:
                    team_id = int(df_edit_teams[df_edit_teams["team"] == selected_team].iloc[0]["team_id"])
                    df_edit_players = df_edit_players[df_edit_players["team_id"] == team_id]
                    st.data_editor(df_edit_players[["jersey_num", "name", "position"]], height=400, use_container_width=True, num_rows="dynamic", key="player_data_editor")
                    ### ensure changes are written back to the DB? -- TODO
                    ### ensure changes are complient with data types and constraints? -- TODO


