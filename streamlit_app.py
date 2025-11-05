import sqlite3
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st


# Page config
st.set_page_config(page_title="HockeyStat", page_icon="🏒")
st.title("🏒 Warriors")


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


### --- Streamlit UI scaffold ---
tab1, tab2 = st.tabs(["Games", "Player Stats"])

with tab1:
    st.header("Games")

    if df_teams.empty:
        st.info("No teams found for the selected season (or DB not found). Use Admin to add teams.")
    else:
        team_map = df_teams.set_index("team_id")["team"].to_dict()
        team_options = df_teams["team"].tolist()
        selected_team_name = st.selectbox("Select team", options=team_options, index=0)
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

                st.markdown(f"**Summary — W/L/T:** {W}/{L}/{T} — **PF/PA/SD:** {PF}/{PA}/{PF-PA}")

                # Show games table
                st.dataframe(team_games[["date", "home_team_id", "home_score", "away_team_id", "away_score", "result_for_team"]].rename(columns={"result_for_team": "Result"}))

with tab2:
    st.header("Player Stats")

    if df_teams.empty or df_rosters.empty:
        st.info("No roster data available.")
    else:
        # default team is first from tab1 selection if present
        team_names = df_teams["team"].tolist()
        selected_team_name = st.selectbox("Team (for player list)", options=team_names, index=0)
        selected_team_id = int(df_teams[df_teams["team"] == selected_team_name].iloc[0]["team_id"])

        team_players = df_rosters[df_rosters["team_id"] == selected_team_id]
        if team_players.empty:
            st.info("No players found for this team.")
        else:
            player_options = team_players["name"].tolist()
            selected_player_name = st.selectbox("Select player", options=player_options)
            selected_player_id = int(team_players[team_players["name"] == selected_player_name].iloc[0]["player_id"])

            # Player season totals from df_players
            p_stats = df_players[df_players["player_id"] == selected_player_id]
            if p_stats.empty:
                st.info("No player game stats available for this player.")
            else:
                totals = {
                    "Games": p_stats["game_id"].nunique(),
                    "Goals": int(p_stats["goals"].sum()),
                    "Assists": int(p_stats["assists"].sum()),
                    "Points": int(p_stats.get("points", pd.Series(dtype=int)).sum()) if "points" in p_stats.columns else int(p_stats["goals"].sum() + p_stats["assists"].sum()),
                }
                st.write(totals)

                # Last 5 games
                recent = p_stats.merge(df_games, on="game_id", how="left").sort_values("date", ascending=False).head(5)
                if not recent.empty:
                    st.subheader("Last 5 games")
                    st.dataframe(recent[["date", "goals", "assists", "points"]])
