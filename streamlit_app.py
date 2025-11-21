import sqlite3
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

### Page config ###
st.set_page_config(page_title="HockeyStat", page_icon="🏒")

### Data loading function ###
@st.cache_data
def load_dfs_from_database(season: int, db_path: str = "data/HockeyStat.db") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the relevant tables from the HockeyStat SQLite DB.
    - Filters `Teams` by season.
    - Loads Players, Games, PlayerGameStats and GoalieGameStats restricted to teams in that season.
    Returns five dataframes in the order: teams, rosters, games, player_stats, goalie_stats.
    If the DB is missing or tables are not present, returns empty dataframes with sensible columns.
    """
    # helper empty dataframes
    df_teams = pd.DataFrame(columns=["team_id", "club", "team", "season", "location", "coach"])
    df_rosters = pd.DataFrame(columns=["player_id", "team_id", "jersey_num", "name", "position"])
    df_games = pd.DataFrame(columns=["game_id", "date", "home_team_id", "home_score", "away_team_id", "away_score", "overtime", "shootout", "league_play"])
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

            # Load Rosters for these teams
            t_placeholders = ",".join(["?"] * len(team_ids))
            q = f"SELECT player_id, team_id, jersey_num, name, position FROM Players WHERE team_id IN ({t_placeholders})"
            df_rosters = pd.read_sql_query(q, conn, params=team_ids)

            # Load Games involving these teams
            q = f"SELECT game_id, date, home_team_id, home_score, away_team_id, away_score, league_play FROM Games WHERE home_team_id IN ({t_placeholders}) OR away_team_id IN ({t_placeholders})"
            df_games = pd.read_sql_query(q, conn, params=team_ids + team_ids)
            if not df_games.empty and "date" in df_games.columns:
                df_games["date"] = pd.to_datetime(df_games["date"], errors="coerce")
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

    # indexes to useful values
    df_teams.set_index('team_id', drop=False, inplace=True)
    df_teams.index.rename('id', inplace=True)
    df_rosters.set_index('player_id', drop=False, inplace=True)
    df_rosters.index.rename('id', inplace=True)
    df_games.set_index('game_id', drop=False, inplace=True)
    df_games.index.rename('id', inplace=True)
    df_players.set_index(['game_id', 'player_id'], drop=False, inplace=True)
    df_players.index.rename(['g_id', 'p_id'], inplace=True)
    df_goalies.set_index(['game_id', 'player_id'], drop=False, inplace=True)
    df_goalies.index.rename(['g_id', 'p_id'], inplace=True)

    return df_teams, df_rosters, df_games, df_players, df_goalies

### Tab 0 helper functions
def summarize_team_games(df_games: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """converts from home_score and away_score team_score and opp_score 
    deterimines win-loss-tie result 
    returns df with added columns
    """
    def _compute_game_result(row: pd.Series, team_id: int) -> str:
        try:
            hs, as_ = int(row["home_score"]), int(row["away_score"])
        except Exception:
            return "?"
        if hs == as_:
            return "T"
        is_home = row["home_team_id"] == team_id
        return "W" if (hs > as_ if is_home else as_ > hs) else "L"

    is_home = df_games["home_team_id"] == team_id
    is_away = df_games["away_team_id"] == team_id
    df = df_games[is_home | is_away].copy()
    if df.empty:
        st.info("No games for the selected team in this season.")
        return df
    else:
        df["team_score"] = df.apply(lambda r: r["home_score"] if r["home_team_id"] == team_id else r["away_score"], axis=1)
        df["opp_score"] = df.apply(lambda r: r["away_score"] if r["home_team_id"] == team_id else r["home_score"], axis=1)
        df["result_for_team"] = df.apply(lambda r: _compute_game_result(r, team_id), axis=1)
        W = (df["result_for_team"] == "W").sum()
        L = (df["result_for_team"] == "L").sum()
        T = (df["result_for_team"] == "T").sum()
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Wins", int(W))
        mcol2.metric("Losses", int(L))
        mcol3.metric("Ties", int(T))
        PF = pd.to_numeric(df["team_score"], errors="coerce").fillna(0).astype(int).sum()
        PA = pd.to_numeric(df["opp_score"], errors="coerce").fillna(0).astype(int).sum()
        mcol4, mcol5, mcol6 = st.columns(3)
        mcol4.metric("PF", int(PF))
        mcol5.metric("PA", int(PA))
        mcol6.metric("SD", int(PF - PA))
        return df.sort_values("date", ascending=False)

def display_game_table(df_games: pd.DataFrame, team_map: dict):
    """ displays formatted dataframe with named columns and team names vs team id"""
    df = df_games.copy()
    df.loc[:, "home_team"] = df["home_team_id"].map(team_map).fillna(df["home_team_id"])
    df.loc[:, "visiting_team"] = df["away_team_id"].map(team_map).fillna(df["away_team_id"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    cols = ["date", "result_for_team", "away_score", "visiting_team", "home_team", "home_score"]
    ### Consider conditional formatting for W/L/T? -- TODO
    ### consider centering the results and scores? -- TODO
    st.dataframe(df[cols].rename(columns={
        "date": "Date", "result_for_team": "Result", "away_score": " Score",
        "visiting_team": "Visitor", "home_team": "at Home", "home_score": "Score "
    }), hide_index=True)

def display_season_total_stats(df_players: pd.DataFrame, df_games: pd.DataFrame, df_rosters: pd.DataFrame, team: int) -> pd.DataFrame:
    """ displays a df of season totals for each player on the team """
    df = df_players.merge(df_games[["game_id"]], on="game_id")
    df = df.merge(df_rosters[["player_id", "team_id", "name"]], on="player_id")
    df = df[df['team_id']==team]

    if "shots_faced" in df.columns:
        df["wins"] = df["result"].str.upper().eq("W").astype(int)
        df["losses"] = df["result"].str.upper().eq("L").astype(int)
        df = df.groupby("name", as_index=False).agg({
            "wins": "sum", "losses": "sum", "shots_faced": "sum", "saves": "sum", "goals_allowed": "sum"
            })
        df["save_pct"] = np.where(df["shots_faced"] > 0, df["saves"] / df["shots_faced"], np.nan)
        df.sort_values("wins", ascending=False) 
        st.dataframe(df.rename(columns={
            "name": "Goalie", "wins": "W", "losses": "L", "shots_faced": "SOG",
            "saves": "Saves", "save_pct": "Save %", "goals_allowed": "Goals"
            }).sort_values("W", ascending=False), hide_index=True)
    else:
        df = df.groupby("name", as_index=False).agg({
            "active": "sum", "goals": "sum", "assists": "sum", "points": "sum", "penalty_min": "sum"
        })
        st.dataframe(df.rename(columns={
            "name": "Player", "active": "Games", "goals": "Goals",
            "assists": "Assists", "points": "Points", "penalty_min": "PIM"
            }).sort_values("Points", ascending=False), hide_index=True, height="stretch")        

### Tab 1 helper functions
def display_summary_stats(stats_df: pd.DataFrame) -> dict:
    """summarizes performance summary stats for players and goalies"""
    def _active_mask(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s.fillna(False)
        num = pd.to_numeric(s, errors="coerce")
        if not num.isna().all():
            return num.fillna(0).astype(int) != 0
        return s.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
    def _display_goalie_metrics(stats: dict):
        if not stats:
            st.info("No active goalie stats available.")
            return
        st.badge("Goalie Stats", color="red")
        gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)
        gcol1.metric("W", stats["W"], delta=(f"{stats['Shutouts']} Shutouts" if stats["Shutouts"] else None))
        gcol2.metric("L", stats["L"])
        gcol3.metric("Shots On", stats["Shots"])
        gcol4.metric("Saves", stats["Saves"])
        gcol5.metric("Save %", f"{stats['SavePct']*100:.1f}%" if stats["SavePct"] is not None else "N/A")
    def _display_player_metrics(stats: dict):
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
        pcol7.metric("GPG", stats["GPG"])
        pcol8.metric("APG", stats["APG"])
        pcol9.metric("PPG", stats["PPG"])
        pcol10.metric("PIM/G", stats["PIMPG"])

    df = stats_df.copy()
    df = df[_active_mask(df["active"])] if "active" in df.columns else df
    if df.empty:
        return {}

    if "shots_faced" in df.columns:
        shots = int(df.get("shots_faced", 0).sum())
        saves = int(df.get("saves", 0).sum())
        save_pct = round(saves / shots, 3) if shots > 0 else None
        res = df.get("result", pd.Series(dtype=str)).astype(str).str.upper()
        wins = int((res == "W").sum())
        losses = int((res == "L").sum())
        shutouts = int(df[(res == "W") & (df["goals_allowed"] == 0)].shape[0]) or None
        stats = {"W": wins, "L": losses, "Shots": shots, "Saves": saves,
                 "SavePct": save_pct, "Shutouts": shutouts}
        _display_goalie_metrics(stats)

    else: 
        games = df["game_id"].nunique()
        goals = int(df["goals"].sum())
        assists = int(df["assists"].sum())
        points = int(df.get("points", df["goals"] + df["assists"]).sum())
        pim = int(pd.to_numeric(df.get("penalty_min", 0), errors="coerce").fillna(0).sum())

        hat_tricks = df[df["goals"] >= 3].shape[0]
        play_makers = df[df["assists"] >= 3].shape[0]
        stats = {"Games": games, "Goals": goals, "Assists": assists, "Points": points, "PIM": pim,
            "HatTricks": hat_tricks, "PlayMakers": play_makers, "GPG": round(goals / games, 2) if games else 0.0,
            "APG": round(assists / games, 2) if games else 0.0, "PPG": round(points / games, 2) if games else 0.0,
            "PIMPG": round(pim / games, 2) if games else 0.0}
        _display_player_metrics(stats)

def display_game_log(game_log_df: pd.DataFrame, df_games: pd.DataFrame):
    st.badge("Game Log", color="red")
    df = game_log_df[game_log_df["active"]==True].copy()
    df = df.merge(df_games, on="game_id", how="left", suffixes=("", "_game")).sort_values("date", ascending=False)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if 'shots_faced' in df.columns:
        cols = ['date', 'result', 'shots_faced', 'saves', 'goals_allowed']
        st.dataframe(df[cols].rename(columns={
                "date": "Date", "result": "W/L", "shots_faced": "SOG", 
                "saves": "Saves", "goals_allowed": "Goals"
            }), hide_index = True)
    else:
        cols = ['date', 'goals', 'assists', 'points', 'penalty_min']
        st.dataframe(df[cols].rename(columns={
                "date": "Date", "goals": "Goals", "assists": "Assists", 
                "points": "Points", "penalty_min": "PIM"
            }), hide_index = True)

### Tab 4 helper functions
def check_admin_password(key="admin_password_input", expected="rip") -> bool:
    st.text_input("Admin password", type="password", key=key)
    if st.session_state.get(key, "") != expected:
        st.warning("Input admin password.")
        st.stop()
    return True

def get_edit_games_df(df_games: pd.DataFrame, team_map: dict) -> pd.DataFrame:
    edit_games_df = df_games.copy(deep=True)
    edit_games_df["home_team"] = edit_games_df["home_team_id"].map(team_map).fillna(edit_games_df["home_team_id"])
    edit_games_df["away_team"] = edit_games_df["away_team_id"].map(team_map).fillna(edit_games_df["away_team_id"])
    return edit_games_df
    
def get_edit_game_roster_df(df: pd.DataFrame, df_rosters: pd.DataFrame, selected_game: int) -> pd.DataFrame:
    selected_df = df.copy(deep=True)
    selected_df = selected_df[selected_df["game_id"] == selected_game]
    selected_df = selected_df.merge(df_rosters[['player_id', 'name']], on='player_id', how='left')
    selected_df["edit_key"] = selected_df.apply(
            lambda row: f"{int(row['game_id'])}_{int(row['player_id'])}", axis=1)
    return selected_df

def write_games_to_db(edited_games_df: pd.DataFrame, 
                      df_rosters: pd.DataFrame, 
                      teams: dict, conn: sqlite3.Connection):
    cursor = conn.cursor()    
    def _validate_game_row(row: pd.Series, teams: dict) -> bool:
        try:
            pd.to_datetime(row["date"])
            vld = row["away_score"] in range(0, 100, 1) and row["home_score"] in range(0, 100, 1) and row["away_team"] in teams.values() and row["home_team"] in teams.values()
            return vld
        except Exception as e:
            st.warning(f"Validation failed: {e}")
            return False
    def _initialize_stats(team_id, game_id):
        players = df_rosters[df_rosters["team_id"] == team_id]
        if players.empty: 
            return

        for _, player in players.iterrows():
            cursor.execute("""
                SELECT 1 FROM PlayerGameStats 
                WHERE game_id = ? AND player_id = ?
                """, (game_id, player["player_id"]))
            if cursor.fetchone() is None:
                cursor.execute("""
                    INSERT INTO PlayerGameStats (game_id, player_id, goals, assists, penalty_min, active)
                    VALUES (?, ?, 0, 0, 0, TRUE)
                """, (game_id, player["player_id"]))

        goalies = players[players["position"] == "G"]
        for _, goalie in goalies.iterrows():
            cursor.execute("""
                SELECT 1 FROM GoalieGameStats 
                WHERE game_id = ? AND player_id = ?
                """, (game_id, goalie["player_id"]))
            if cursor.fetchone() is None:
                cursor.execute("""
                    INSERT INTO GoalieGameStats (game_id, player_id, saves, goals_allowed, shots_faced, active)
                    VALUES (?, ?, 0, 0, 0, FALSE)
                """, (game_id, goalie["player_id"]))

    edits = st.session_state.get("Games_data_editor", {}).get("edited_rows", {})
    df = edited_games_df.iloc[list(edits.keys())].copy()
    if df.empty:
        st.toast("🟩 No Changes to Commit")
        conn.close()
        return
    else:
        for game_id, row in df.iterrows():
            if _validate_game_row(row, teams):
                row["date"] = row["date"].strftime("%Y-%m-%d")
                row["home_team_id"] = [k for k, v in teams.items() if v == row["home_team"]][0]
                row["away_team_id"] = [k for k, v in teams.items() if v == row["away_team"]][0]
                update_fields = ["date", "away_score", "away_team_id", "home_team_id", "home_score", "league_play"]
                values = [row[col] for col in update_fields]

                cursor.execute("SELECT COUNT(*) FROM Games WHERE game_id = ?", (game_id,))
                exists = cursor.fetchone()[0] > 0
                if exists:
                    set_clause = ", ".join([f"{col} = ?" for col in update_fields])
                    values.append(game_id)
                    cursor.execute(f"UPDATE Games SET {set_clause} WHERE game_id = ?", values)
                    st.toast(f"Updated Game {game_id}")
                else:
                    insert_clauseA = ", ".join(update_fields)
                    insert_clauseB = ", ".join(["?" for _ in update_fields])
                    cursor.execute(f"INSERT INTO Games ({insert_clauseA}) VALUES ({insert_clauseB})", values)
                    game_id = cursor.lastrowid
                    for team in [row["home_team_id"], row["away_team_id"]]:
                        _initialize_stats(team, game_id)
                    st.toast(f"Added Game {game_id}")
            else: st.warning(f"Unable to Update Game {game_id}")
    conn.commit()
    conn.close()
    st.success("✅ Database Updated")

def write_stats_to_db(player_edited_df: pd.DataFrame, goalie_edited_df: pd.DataFrame,
                      df_rosters: pd.DataFrame, game_id: int,
                      conn: sqlite3.Connection):
    
    def _validate_player_stats(ds: pd.Series) -> bool:
        required = ["goals", "assists", "penalty_min", "active", "player_id"]
        return all(i in ds.index for i in required)

    def _validate_goalie_stats(ds: pd.Series) -> bool:
        required = ["shots_faced", "saves", "goals_allowed", "result", "active", "player_id"]
        return all(i in ds.index for i in required)

    def _get_df(edits_df, roster, edits):
        df = edits_df.merge(roster, on = 'name')
        return df.iloc[list(edits.keys())]

    edits = st.session_state.get("PlayerGameStats_editor", {}).get("edited_rows", {})
    update_players_df = _get_df(player_edited_df, df_rosters, edits)

    edits = st.session_state.get("GoalieGameStats_editor", {}).get("edited_rows", {})
    update_goalies_df = _get_df(goalie_edited_df, df_rosters, edits)

    if update_players_df.empty and update_goalies_df.empty:
        st.toast("🟩 No Changes to Commit")
        return

    cursor = conn.cursor()

    if len(update_players_df) > 0:
        for _, row in update_players_df.iterrows():
            if _validate_player_stats(row):
                cursor.execute("""
                    UPDATE PlayerGameStats
                    SET goals=?, assists=?, penalty_min=?, active=?
                    WHERE player_id=? AND game_id=?
                """, (row["goals"], row["assists"], row["penalty_min"], int(row["active"]), row["player_id"], game_id))
                st.toast(f"Stats updated for {row['name']}")

    if len(update_goalies_df) > 0:
        for _, row in update_goalies_df.iterrows():
            if _validate_goalie_stats(row):
                cursor.execute("SELECT COUNT(*) FROM GoalieGameStats WHERE player_id = ? game_id = ?", (row["player_id"], game_id))
                exists = cursor.fetchone()[0] > 0
                if exists:
                    cursor.execute("""
                        UPDATE GoalieGameStats
                        SET shots_faced=?, saves=?, goals_allowed=?, result=?, active=?
                        WHERE player_id=? AND game_id=?
                    """, (row["shots_faced"], row["saves"], row["goals_allowed"], row["result"], int(row["active"]), row["player_id"], game_id))
                    st.toast(f"Stats updated for goalie {row['name']}")
                else:
                    cursor.execute("""
                        INSERT INTO GoalieGameStats
                        shots_faced, saves, goals_allowed, result, active, player_id, game_id        
                        VALUES ?, ?, ?, ?, ?, ?, ?
                    """, (row["shots_faced"], row["saves"], row["goals_allowed"], row["result"], int(row["active"]), row["player_id"], game_id))
                    st.toast(f"Stats added for goalie {row['name']}")

    conn.commit()
    conn.close()
    st.toast("✅ Stat changes saved")

def write_to_db(edited_df: pd.DataFrame, tbl: str, conn: sqlite3.Connection):
    def _validated(df: pd.DataFrame, tbl: str) -> tuple[list[str], str]:
        if tbl == 'Teams':
            update_fields = ["club", "team", "season"]
            key_col = 'team_id'
        elif tbl == 'Players':
            update_fields = ["team_id", "jersey_num", "name", "position"]
            key_col = 'player_id'
        else: 
            st.toast("Unable to Make Changes")
            return [], ""
        if all(c in df.columns for c in update_fields):
            return update_fields, key_col
        else:
            st.warning("Table Type Mismatch")
            return [], ""

    edits = st.session_state.get(f"{tbl}_data_editor", {}).get("edited_rows", {})
    df = edited_df.iloc[list(edits.keys())].copy()
    update_fields, key_col = _validated(df, tbl)
    if not update_fields or df.empty:
        st.toast("🟩 No Changes to Commit")
        return

    cursor = conn.cursor()

    for id, row in df.iterrows():
            key = id
            values = [row[col] for col in update_fields]
            cursor.execute(f"SELECT COUNT(*) FROM {tbl} WHERE {key_col} = ?", (key,))
            exists = cursor.fetchone()[0] > 0
            if exists:
                set_clause = ", ".join([f"{col} = ?" for col in update_fields])
                values.append(key)
                cursor.execute(f"UPDATE {tbl} SET {set_clause} WHERE {key_col} = ?", values)
                st.success(f"Updated {tbl} {key}")
            else:
                insert_clauseA = ", ".join(update_fields)
                insert_clauseB = ", ".join(["?" for _ in update_fields])
                cursor.execute(f"INSERT INTO {tbl} ({insert_clauseA}) VALUES ({insert_clauseB})", values)
                key = cursor.lastrowid
                st.success(f"Added {tbl} {key}")

    conn.commit()
    conn.close()
    st.toast(f"✅ {tbl} changes saved")

### UI Header
st.title("🏒 HockeyStat Dashboard", help="Hockey Team and Player Statistics Tracker")
col00, col01, col02 = st.columns(3)

with col00:
    selected_season = st.selectbox("Selected Season", options=[2025, 2026, 2027], index=0, disabled=True)

# Load data - TODO: Add season selection option 
df_teams, df_rosters, df_games, df_players, df_goalies = load_dfs_from_database(int(selected_season))

if df_teams.empty or df_rosters.empty:
    st.info("No teams found for the selected season. Use Admin Panel to add teams and rosters.")
    df_select_games = df_games

else:
    team_map = df_teams.set_index("team_id")["team"].to_dict()
    player_map = df_rosters.set_index("player_id")["name"].to_dict()
    team_options = sorted(df_teams["team"].tolist())
    with col01:
        selected_team_name = st.selectbox("Selected Team", options=team_options, index=7, disabled=True)
        if selected_team_name not in team_options:
            selected_team_name = team_options[0]
            st.toast(f"Default Team not found; using '{selected_team_name}'")
    selected_team_id = int(df_teams[df_teams["team"] == selected_team_name].iloc[0]["team_id"])
    st.session_state.team_name = selected_team_name
    st.session_state.team_id = selected_team_id

    with col02:
        l_p = st.pills("League Play", ["League", "Non-League"], selection_mode="multi", default=["League"])
    if l_p == []:
        df_select_games = pd.DataFrame(columns=["game_id", "date", "home_team_id", "home_score", "away_team_id", "away_score", "overtime", "shootout", "league_play"])
    elif l_p == ["League"]:
        df_select_games = df_games[df_games['league_play']==1]
    elif l_p == ["Non-League"]:
        df_select_games = df_games[df_games['league_play']==0]
    else:
        df_select_games = df_games
    selected_game_ids = df_select_games["game_id"].unique()

### --- Streamlit UI scaffold ---
tabs = st.tabs(["Team", "Players", "About", "Admin"])
#if "active_tab" not in st.session_state:
#    st.session_state.active_tab = 2

with tabs[0]:
    if "team_name" in st.session_state:
        st.subheader(f"{st.session_state.team_name} Overview", divider="red")

        if df_select_games.empty:
            st.info("No games found for this season.")
        else:
            team_games = summarize_team_games(df_select_games, st.session_state.team_id)
            st.markdown("---")
            st.badge("Recent Games", color="red")
            display_game_table(team_games.head(10), team_map)  
            
            st.markdown("---")
            st.badge("Goaie Stats", color="red")
            display_season_total_stats(df_goalies, team_games, df_rosters, st.session_state.team_id)
                
            st.badge("Player Stats", color="red")
            display_season_total_stats(df_players, team_games, df_rosters, st.session_state.team_id)

with tabs[1]:
    team_players = df_rosters[df_rosters["team_id"] == st.session_state.team_id]
    if team_players.empty or df_rosters.empty:
        st.info("No roster data available.")
    else:
        player_options = sorted(team_players["name"].tolist())
        selected_player_name = st.selectbox("Select player", options=player_options)

        selected_player_id = int(team_players[team_players["name"] == selected_player_name].iloc[0]["player_id"])
        selected_jersey = int(team_players[team_players["name"] == selected_player_name].iloc[0]["jersey_num"])
        selected_pos = str(team_players[team_players["name"] == selected_player_name].iloc[0]["position"])
        
        st.subheader(f"# {selected_jersey}   {selected_player_name}  ({selected_pos})", divider="red")

        stats_df = df_goalies[
            (df_goalies["player_id"] == selected_player_id) & 
            (df_goalies["game_id"].isin(selected_game_ids))]
        if not stats_df.empty:
            display_summary_stats(stats_df)
            st.markdown("---")
            display_game_log(stats_df, df_select_games)
            st.markdown("---")

        stats_df = df_players[
            (df_players["player_id"] == selected_player_id) &
            (df_players["game_id"].isin(selected_game_ids))]
        if stats_df.empty:
            st.info("No player game stats available for this player.")
        else:
            display_summary_stats(stats_df)
            st.markdown("---")
            display_game_log(stats_df, df_select_games)

with tabs[2]:
    st.write("League Standing and Schedule:")
    st.link_button(label="MetroHockeyLeague", url="https://www.themhl.org/metropolitanhockeyleague/Standings")
    
    st.markdown("---")
    st.write("Other Hockey Links:")
    col21, col22, col23 = st.columns(3)
    with col21:
        st.link_button(label="Strategy", url="https://blueseatblogs.com/hockey-systems-strategy/")
    with col22:
        st.link_button(label="Drills and Practice", url="https://www.icehockeysystems.com/hockey-drills/drill-category/small-area-games")
    with col23:
        st.link_button(label="Playbook", url="https://www.jes-hockey.com/")

    st.markdown("---")        
    st.write("A simple streamlit app used to query and display" \
            " team and player statistics from a SQLite database." \
            " Admin privileges support updating and correcting the database. " \
            " Originally designed by Rob for tracking the 2025" \
            " 10U West Sound Warriors Stats.")
    st.write("")
    st.write("")
    st.write("Apache License Version 2.0, January 2004")    

with tabs[3]:
    st.header("Admin Panel")
    if check_admin_password():
        tab31, tab32 = st.tabs(["Update Games", "Manage Teams"])

        with tab31:
            edit_games_df = get_edit_games_df(df_games, team_map)
            cols = ["date", "away_score", "away_team", "home_team", "home_score", "league_play"]
            edited_games_df = st.data_editor(edit_games_df[cols],
                           num_rows = "dynamic",
                           height=200,
                           hide_index=True,
                           key="Games_data_editor",
                           column_config={
                               "date" : st.column_config.DateColumn("Date"),
                               "home_team": st.column_config.SelectboxColumn("Home Team", options=list(team_map.values())),
                               "away_team": st.column_config.SelectboxColumn("Away Team", options=list(team_map.values())),
                               "league_play": st.column_config.CheckboxColumn("League")
                           })
            st.button("Save Game Changes", key="save_game_stats_button")
            if st.session_state.get("save_game_stats_button", False):
                write_games_to_db(edited_games_df, df_rosters, team_map,
                                  sqlite3.connect("data/HockeyStat.db"))
                st.cache_data.clear()

            st.write("")
            st.markdown("---")
            st.write("")
            game_labels = {
                f"{row['date'].strftime('%Y-%m-%d')} — {row['away_team']} at {row['home_team']}": row["game_id"]
                    for _, row in edit_games_df.iterrows()}
            selected_label = st.selectbox("Edit Game", options=list(game_labels.keys()), index=None)
            selected_game = game_labels.get(selected_label)
            if selected_game:
                select_players_game_df = get_edit_game_roster_df(df_players, df_rosters, selected_game)
                cols = ["name", "goals", "assists", "penalty_min", "active"]
                select_players_game_df.sort_values("name", inplace=True)
                player_stat_edits_df = st.data_editor(select_players_game_df[cols],
                               height="stretch",
                               hide_index=True, 
                               num_rows="fixed",
                               key="PlayerGameStats_editor",
                               column_config={
                                    "active": st.column_config.CheckboxColumn("Active"),
                                    "name": st.column_config.TextColumn("Name", disabled=True)
                                    })
                select_goalies_game_df = get_edit_game_roster_df(df_goalies, df_rosters, selected_game)
                cols = ["name", "shots_faced", "saves", "goals_allowed", "result", "active"]
                select_goalies_game_df.sort_values("name", inplace=True)
                goalie_stat_edits_df = st.data_editor(select_goalies_game_df[cols],
                               height="stretch",
                               hide_index=True,  
                               num_rows="dynamic",
                               key="GoalieGameStats_editor",
                               column_config={
                                    "active": st.column_config.CheckboxColumn("Active"),
                                    "name": st.column_config.TextColumn("Goalie", disabled=True),
                                    "result": st.column_config.SelectboxColumn("Result", options=[None, "W", "L", "-"]) 
                                    })
                st.button("Save Stat Changes", key="save_player_stats_button")
                if st.session_state.get("save_player_stats_button", False):
                    write_stats_to_db(player_stat_edits_df,
                                      goalie_stat_edits_df,
                                      df_rosters,
                                      selected_game,
                                      sqlite3.connect("data/HockeyStat.db"))
                    st.cache_data.clear()

        with tab32:
            edit_roster_df = df_rosters.copy(deep=True)
            cols = ["club", "team", "season"]
            edit_teams_df = st.data_editor(df_teams[cols], height=300, num_rows="dynamic", key="Teams_data_editor")
            st.button("Save Team Changes", key="save_team_button")
            if st.session_state.get("save_team_button", False):
                write_to_db(edit_teams_df, "Teams", sqlite3.connect("data/HockeyStat.db"))
                st.cache_data.clear()
                
            selected_team = st.selectbox("Edit Team Roster", df_teams['team'], index=None)
            if selected_team:
                team_id = int(df_teams[df_teams["team"] == selected_team].iloc[0]["team_id"])
                cols = ["jersey_num", "name", "position"]
                edit_roster_df = st.data_editor(df_rosters[df_rosters["team_id"] == team_id][cols],
                                    height=400, 
                                    num_rows="dynamic", 
                                    key="Players_data_editor",
                                    column_config={
                                        "position": st.column_config.SelectboxColumn("Position", options=["G", "F", "D"])
                                        })
                st.button("Save Team Changes", key="save_roster_button")
                if st.session_state.get("save_roster_button", False):
                    edit_roster_df['team_id'] = team_id                       
                    write_to_db(edit_roster_df, sqlite3.connect("data/HockeyStat.db"))
                    st.cache_data.clear()