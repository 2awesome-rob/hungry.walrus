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
def check_admin_password(key="admin_password_input", expected="rip") -> bool:
    st.text_input("Admin password", type="password", key=key)
    if st.session_state.get(key, "") != expected:
        st.warning("Input admin password.")
        st.stop()
    return True

def active_mask(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    num = pd.to_numeric(s, errors="coerce")
    if not num.isna().all():
        return num.fillna(0).astype(int) != 0
    return s.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])

def compute_game_result(row: pd.Series, team_id: int) -> str:
    try:
        hs, as_ = int(row["home_score"]), int(row["away_score"])
    except Exception:
        return "?"
    if hs == as_:
        return "T"
    is_home = row["home_team_id"] == team_id
    return "W" if (hs > as_ if is_home else as_ > hs) else "L"

def summarize_team_games(df_games: pd.DataFrame, team_id: int) -> pd.DataFrame:
    is_home = df_games["home_team_id"] == team_id
    is_away = df_games["away_team_id"] == team_id
    df = df_games[is_home | is_away].copy()
    if df.empty:
        return df
    else:
        df["team_score"] = df.apply(lambda r: r["home_score"] if r["home_team_id"] == team_id else r["away_score"], axis=1)
        df["opp_score"] = df.apply(lambda r: r["away_score"] if r["home_team_id"] == team_id else r["home_score"], axis=1)
        df["result_for_team"] = df.apply(lambda r: compute_game_result(r, team_id), axis=1)
        return df.sort_values("date", ascending=False)

def display_game_table(df: pd.DataFrame, team_map: dict):
    df["home_team"] = df["home_team_id"].map(team_map).fillna(df["home_team_id"])
    df["visiting_team"] = df["away_team_id"].map(team_map).fillna(df["away_team_id"])
    cols = ["date", "result_for_team", "away_score", "visiting_team", "home_team", "home_score"]
    ### consider narrower columns in the dataframe display? -- TODO
    ### Consider conditional formatting for W/L/T? -- TODO
    ### consider centering the results and scores? -- TODO
    st.dataframe(df[cols].rename(columns={
        "date": "Date", "result_for_team": "Result", "away_score": " Score",
        "visiting_team": "Visitor", "home_team": "at Home", "home_score": "Score "
    }), hide_index=True)

def display_player_tables(df: pd.DataFrame):
    if "shots_faced" in df.columns.tolist():
        st.dataframe(df.rename(columns={
            "name": "Goalie", "wins": "W", "losses": "L", "shots_faced": "SOG",
            "saves": "Saves", "save_pct": "Save %", "goals_allowed": "Goals"
        }), hide_index=True)
    else:
        st.dataframe(df.rename(columns={
            "name": "Player", "active": "Games", "goals": "Goals",
            "assists": "Assists", "points": "Points", "penalty_min": "PIM"
        }), hide_index=True, height="stretch")

def aggregate_goalie_stats(df_goalies: pd.DataFrame, df_games: pd.DataFrame, df_rosters: pd.DataFrame) -> pd.DataFrame:
    df = df_goalies.merge(df_games[["game_id"]], on="game_id")
    df = df.merge(df_rosters[["player_id", "name"]], on="player_id")
    df["wins"] = df["result"].str.upper().eq("W").astype(int)
    df["losses"] = df["result"].str.upper().eq("L").astype(int)
    df = df.groupby("name", as_index=False).agg({
        "wins": "sum", "losses": "sum", "shots_faced": "sum", "saves": "sum", "goals_allowed": "sum"
    })
    df["save_pct"] = np.where(df["shots_faced"] > 0, df["saves"] / df["shots_faced"], np.nan)
    return df.sort_values("wins", ascending=False)

def summarize_goalie_stats(df_goalies: pd.DataFrame) -> dict:
    df = df_goalies.copy()
    df = df[active_mask(df["active"])] if "active" in df.columns else df
    if df.empty:
        return {}

    gp = df["game_id"].nunique()
    shots = int(df.get("shots_faced", 0).sum())
    saves = int(df.get("saves", 0).sum())
    ga = int(df.get("goals_allowed", 0).sum())
    save_pct = round(saves / shots, 3) if shots > 0 else None

    res = df.get("result", pd.Series(dtype=str)).astype(str).str.upper()
    wins = int((res == "W").sum())
    losses = int((res == "L").sum())
    shutouts = int(df[(res == "W") & (df["goals_allowed"] == 0)].shape[0]) or None

    return {
        "W": wins, "L": losses, "Shots": shots, "Saves": saves,
        "SavePct": save_pct, "Shutouts": shutouts
    }

def display_goalie_metrics(stats: dict):
    if not stats:
        st.info("No active goalie stats available.")
        return
    gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)
    gcol1.metric("W", stats["W"], delta=(f"{stats['Shutouts']} Shutouts" if stats["Shutouts"] else None))
    gcol2.metric("L", stats["L"])
    gcol3.metric("Shots On", stats["Shots"])
    gcol4.metric("Saves", stats["Saves"])
    gcol5.metric("Save %", f"{stats['SavePct']*100:.1f}%" if stats["SavePct"] is not None else "N/A")

def display_goalie_game_log(df_log: pd.DataFrame, df_games: pd.DataFrame):
    st.badge("Game Log", color="red")
    df = df_log[df_log["active"]==True]
    df = df.merge(df_games, on="game_id", how="left", suffixes=("", "_game")).sort_values("date", ascending=False)
    cols = ['date', 'result', 'shots_faced', 'saves', 'goals_allowed']
    st.dataframe(df[cols].rename(columns={
            "date": "Date", "result": "W/L", "shots_faced": "SOG", 
            "saves": "Saves", "goals_allowed": "Goals"
        }), hide_index = True)

def aggregate_player_stats(df_players: pd.DataFrame, df_games: pd.DataFrame, df_rosters: pd.DataFrame) -> pd.DataFrame:
    df = df_players.merge(df_games[["game_id"]], on="game_id")
    df = df.merge(df_rosters[["player_id", "name"]], on="player_id")
    df = df.groupby("name", as_index=False).agg({
        "active": "sum", "goals": "sum", "assists": "sum", "points": "sum", "penalty_min": "sum"
    })
    return df.sort_values("points", ascending=False)

def summarize_player_stats(df: pd.DataFrame) -> dict:
    df = df.copy()
    df = df[active_mask(df["active"])] if "active" in df.columns else df
    if df.empty:
        return {}

    games = df["game_id"].nunique()
    goals = int(df["goals"].sum())
    assists = int(df["assists"].sum())
    points = int(df.get("points", df["goals"] + df["assists"]).sum())
    pim = int(pd.to_numeric(df.get("penalty_min", 0), errors="coerce").fillna(0).sum())

    hat_tricks = df[df["goals"] >= 3].shape[0]
    play_makers = df[df["assists"] >= 3].shape[0]

    return {
        "Games": games, "Goals": goals, "Assists": assists, "Points": points, "PIM": pim,
        "HatTricks": hat_tricks, "PlayMakers": play_makers,
        "GPG": round(goals / games, 2) if games else 0.0,
        "APG": round(assists / games, 2) if games else 0.0,
        "PPG": round(points / games, 2) if games else 0.0,
        "PIMPG": round(pim / games, 2) if games else 0.0
    }

def display_player_metrics(stats: dict, position: str):
    if not stats:
        st.info("No active player stats available.")
        return
    
    st.badge("Offense Stats", color="red")
    pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
    pcol1.metric("Games", stats["Games"])
    pcol2.metric("Goals", stats["Goals"], delta=f"{stats['HatTricks']} HatTricks" if stats["HatTricks"] else None)
    pcol3.metric("Assists", stats["Assists"], delta=f"{stats['PlayMakers']} PlayMakers" if stats["PlayMakers"] > 1 else None)
    pcol4.metric("Points", stats["Points"])
    pcol5.metric("PIM", stats["PIM"])
    st.markdown("---")
    pcol6, pcol7, pcol8, pcol9, pcol10 = st.columns(5)
    #pcol6.metric("Position", position)
    pcol7.metric("GPG", stats["GPG"])
    pcol8.metric("APG", stats["APG"])
    pcol9.metric("PPG", stats["PPG"])
    pcol10.metric("PIM/G", stats["PIMPG"])

def display_player_game_log(df_log: pd.DataFrame, df_games: pd.DataFrame):
    st.badge("Game Log", color="red")
    df = df_log[df_log["active"]==True]
    df = df.merge(df_games, on="game_id", how="left", suffixes=("", "_game")).sort_values("date", ascending=False)
    cols = ['date', 'goals', 'assists', 'points', 'penalty_min']
    st.dataframe(df[cols].rename(columns={
            "date": "Date", "goals": "Goals", "assists": "Assists", 
            "points": "Points", "penalty_min": "PIM"
        }), hide_index = True)
    
def get_edit_games_df(df: pd.DataFrame, team_map: dict, next_game_id: int) -> pd.DataFrame:
    df_edit_games = df.copy(deep=True)
    df_edit_games["home_team"] = df_edit_games["home_team_id"].map(team_map).fillna(df_edit_games["home_team_id"])
    df_edit_games["away_team"] = df_edit_games["away_team_id"].map(team_map).fillna(df_edit_games["away_team_id"])
    df_next_game = pd.DataFrame([{
        "game_id": next_game_id,}])
    df_edit_games = pd.concat([df_edit_games, df_next_game], ignore_index=True)
    return df_edit_games
    
def get_edit_roster_df(df: pd.DataFrame, df_rosters: pd.DataFrame, selected_game: int) -> pd.DataFrame:
    selected_df = df.copy(deep=True)
    selected_df = selected_df[selected_df["game_id"] == selected_game]
    selected_df = selected_df.merge(df_rosters[['player_id', 'name']], on='player_id', how='left')
    return selected_df

def write_games_to_db(df: pd.DataFrame, teams: dict, conn: sqlite3.Connection):
    def _validate_game_row(row: pd.Series, teams: dict) -> bool:
        try:
            st.write(f"Validating game row: {row.to_dict()}")
            pd.to_datetime(row["date"])
            int(row["home_score"])
            int(row["away_score"])
            return row["away_team"] in teams.values() and row["home_team"] in teams.values()
        except Exception as e:
            st.warning(f"Validation failed: {e}")
            return False

    cursor = conn.cursor()
    for _, row in df.iterrows():
        if _validate_game_row(row, teams):
            try:
                row["home_team_id"] = [k for k, v in teams.items() if v == row["home_team"]][0]
                row["away_team_id"] = [k for k, v in teams.items() if v == row["away_team"]][0]
                game_id = row["game_id"]
                update_fields = ["date", "away_score", "away_team_id", "home_team_id", "home_score"]
                values = [row[col] for col in update_fields]
                set_clause = ", ".join([f"{col} = ?" for col in update_fields])
                values.append(game_id)
                cursor.execute(f"UPDATE Games SET {set_clause} WHERE game_id = ?", values)
            except Exception as e:
                st.warning(f"Update failed, trying insert: {e}")
                insert_clauseA = ", ".join(update_fields)
                insert_clauseB = ", ".join(["?" for _ in update_fields])
                cursor.execute(f"INSERT INTO Games ({insert_clauseA}) VALUES ({insert_clauseB})", values)
                # TODO: initialize player and goalie stats for new game
    conn.commit()
    conn.close()


def write_player_stats_to_db(df: pd.DataFrame, conn: sqlite3.Connection):
    def _validate_player_stats(df: pd.DataFrame) -> bool:
        required = ["goals", "assists", "penalty_min", "active"]
        return all(col in df.columns for col in required)

    cursor = conn.cursor()
    for _, row in df.iterrows():
        if _validate_player_stats(row):
            cursor.execute("""
                UPDATE PlayerGameStats SET goals=?, assists=?, penalty_min=?, active=?
                WHERE player_id=? AND game_id=?
            """, (row["goals"], row["assists"], row["penalty_min"], int(row["active"]), row["player_id"], row["game_id"]))
    conn.commit()
    conn.close()

def write_goalie_stats_to_db(df: pd.DataFrame, conn: sqlite3.Connection):
    def _validate_goalie_stats(df: pd.DataFrame) -> bool:
        required = ["shots_faced", "saves", "goals_allowed", "result", "active"]
        return all(col in df.columns for col in required)

    cursor = conn.cursor()
    for _, row in df.iterrows():
        if _validate_goalie_stats(row):
            cursor.execute("""
                UPDATE GoalieGameStats SET shots_faced=?, saves=?, goals_allowed=?, result=?, active=?
                WHERE player_id=? AND game_id=?
            """, (row["shots_faced"], row["saves"], row["goals_allowed"], row["result"], int(row["active"]), row["player_id"], row["game_id"]))
    conn.commit()
    conn.close()



### UI Header
st.title("🏒 HockeyStat Dashboard", help="Hockey Team and Player Statistics Tracker")
# Load data 
df_teams, df_rosters, df_games, df_players, df_goalies = load_dfs_from_database()

### make this a session state variable?
next_game_id = int(df_games["game_id"].max()) + 1 if not df_games.empty else 1
if "selected_team_name" not in st.session_state:
    st.session_state.selected_team_name = "10U Warriors, Black"

if df_teams.empty:
    st.info("No teams found for the selected season (or DB not found). Use Admin to add teams.")
else:
    team_map = df_teams.set_index("team_id")["team"].to_dict()
    player_map = df_rosters.set_index("player_id")["name"].to_dict()
    team_options = sorted(df_teams["team"].tolist())
#    selected_team_name = st.selectbox("Select team", options=team_options, index=1, disabled=True)
    selected_team_name = st.session_state.selected_team_name
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
            team_games = summarize_team_games(df_games, selected_team_id)

            if team_games.empty:
                st.info("No games for the selected team in this season.")
            else:
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

                st.markdown("---")
                st.badge("Recent Games", color="red")
                display_game_table(team_games.head(10), team_map)  
                st.markdown("---")
                st.badge("Goaie Stats", color="red")

                df_stats = aggregate_goalie_stats(df_goalies, team_games, df_rosters)
                display_player_tables(df_stats)
                
                st.badge("Player Stats", color="red")
                df_stats = aggregate_player_stats(df_players, team_games, df_rosters)
                display_player_tables(df_stats)

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

                selected_jersey = int(team_players[team_players["name"] == selected_player_name].iloc[0]["jersey_num"])
                selected_pos = str(team_players[team_players["name"] == selected_player_name].iloc[0]["position"])
                st.subheader(f"# {selected_jersey}   {selected_player_name}  ({selected_pos})", divider="red")

                #st.metric("Position", selected_pos)
                stats_df = df_goalies[df_goalies["player_id"] == selected_player_id]
                if not stats_df.empty:
                    st.badge("Goalie Stats", color="red")
                    stat_dict = summarize_goalie_stats(stats_df)
                    display_goalie_metrics(stat_dict)
                    st.markdown("---")
                    display_goalie_game_log(stats_df, df_games)

                stats_df = df_players[df_players["player_id"] == selected_player_id]
                if stats_df.empty:
                    st.info("No player game stats available for this player.")
                else:
                    stat_dict = summarize_player_stats(stats_df)
                    display_player_metrics(stat_dict, selected_pos)
                    st.markdown("---")
                    display_player_game_log(stats_df, df_games)

    with tab3:
        st.header("Admin Panel")

        if check_admin_password():
            tab31, tab32 = st.tabs(["Edit Games", "Edit Teams"])

            with tab31:
                df_edit_games = get_edit_games_df(df_games, team_map, next_game_id)
                cols = ["game_id", "date", "away_score", "away_team", "home_team", "home_score"]
                st.data_editor(df_edit_games[cols], height=200, use_container_width=True, hide_index=True, key="game_data_editor")

                st.button("Save Game Changes", key="save_game_stats_button")
                if st.session_state.get("save_game_stats_button", False):
                    edits = st.session_state.get("game_data_editor", {}).get("edited_rows", {})
                    # edits is a dictionary with key = df row # (not the index) and col name: new value for changes only!
                    # we need the entire row, or at least the game_id so we can update the db
                    # Get full edited rows by merging changes into original
                    df_edited = df_edit_games.loc[list(edits.keys())].copy()
                    for row_idx, changes in edits.items():
                         for col, new_val in changes.items():
                              df_edited.at[row_idx, col] = new_val
                    st.write("Merged Edited Rows:", df_edited)

                    #if isinstance(edits, dict):
                        #df_edited = pd.DataFrame.from_dict(edits, orient="index")
                        #st.write("Edited DataFrame:", df_edited)
#                    write_games_to_db(df_edited, team_map, sqlite3.connect("data/HockeyStat.db"))
                    ### helper function to reload dependent dataframes? -- TODO
#                    df_teams, df_rosters, df_games, df_players, df_goalies = load_dfs_from_database()
                    st.success("Game changes saved. (Functionality to write back to DB is not yet implemented.)")

                selected_game = st.selectbox("Edit Game", df_edit_games["game_id"].tolist(), index=None)
                if selected_game:
                    selected_players_game_df = get_edit_roster_df(df_players, df_rosters, selected_game)
                    cols = ["name", "goals", "assists", "penalty_min", "active"]
                    ### consider making active a checkbox column? -- TODO
                    ### cosider making player name read-only? -- TODO
                    st.data_editor(selected_players_game_df[cols], height="stretch", use_container_width=True, num_rows="fixed", key="player_stats_editor")

                    selected_goalies_game_df = get_edit_roster_df(df_goalies, df_rosters, selected_game)
                    cols = ["name", "shots_faced", "saves", "goals_allowed", "result", "active"]
                    ### consider making active a checkbox column? -- TODO
                    ### cosider making player name dropdown? -- TODO
                    st.data_editor(selected_goalies_game_df[cols], height="stretch", use_container_width=True, num_rows="dynamic", key="goalie_stats_editor")
                    ### ensure changes are written back to the DB? -- TODO
                    ### ensure changes are complient with data types and constraints? -- TODO
                    st.button("Save Game Changes", key="save_player_stats_button")
                    if st.session_state.get("save_player_stats_button", False):
                        ### helper function to ensure changes are complient with data types and constraints? -- TODO
                        ### should also ensure that a new player added to goalie stats is in the roster? -- TODO
                        ### helper function to write back changes to DB? -- TODO
                        ### helper function to reload dependent dataframes? -- TODO
                        st.success("Player and Goalie stats changes saved. (Functionality to write back to DB is not yet implemented.)")
                
            with tab32:
                df_edit_teams = df_teams.copy(deep=True)
                df_edit_players = df_rosters.copy(deep=True)
                st.data_editor(df_edit_teams[["club", "team", "season"]], height=300, use_container_width=True, num_rows="dynamic", key="team_data_editor")
                st.button("Save Team Changes", key="save_team_button")
                if st.session_state.get("save_team_button", False):
                    ### ensure changes are complient with data types and constraints? -- TODO
                    ### helper function to write back changes to DB? -- TODO
                    ### helper function to reload dependent dataframes? -- TODO
                    st.success("Team changes saved. (Functionality to write back to DB is not yet implemented.)")

                selected_team = st.selectbox("Edit Team Roster", df_edit_teams['team'], index=None)
            
                if selected_team:
                    team_id = int(df_edit_teams[df_edit_teams["team"] == selected_team].iloc[0]["team_id"])
                    df_edit_players = df_edit_players[df_edit_players["team_id"] == team_id]
                    ### consider making position a dropdown column? -- TODO
                    st.data_editor(df_edit_players[["jersey_num", "name", "position"]], height=400, use_container_width=True, num_rows="dynamic", key="player_data_editor")
                    st.button("Save Team Changes", key="save_roster_button")
                    if st.session_state.get("save_roster_button", False):
                        ### ensure changes are complient with data types and constraints? -- TODO
                        ### helper function to write back changes to DB? -- TODO
                        ### helper function to reload dependent dataframes? -- TODO
                        st.success("Team changes saved. (Functionality to write back to DB is not yet implemented.)")

st.write("")
st.write("")
st.markup("---")
st.link_button(label="MetroHockeyLeague", url="https://www.themhl.org/metropolitanhockeyleague/Standings")





