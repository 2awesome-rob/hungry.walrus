import sqlite3

#### stub for manually updating the database ####

db_path = "data/HockeyStat.db"

def describe_db_tables():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    t = [table[0] for table in tables]
    print("Tables in the database:" , t)
    for table_name in t:
        print(f"\nSchema for table '{table_name}':")
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        for column in columns:
            print(column)
    conn.close()
    return

#describe_db_tables()

def add_team(team_id, club, season, team, location, coach):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
                   INSERT INTO Teams (team_id, club, season, team, location, coach)
                   VALUES (?, ?, ?, ?, ?, ?)
                   """,(team_id, club, season, team, location, coach))

    print(f"Added team {team} with ID {team_id}")
    conn.commit()
    conn.close()

#add_team(10,'Tri-City Jr Americans','2025','Tolar Jr Americans','Hapo Center',None)
#add_team(11,'Tri-City Jr Americans','2025','Geris Jr Americans','Hapo Center',None)
#add_team(12,'Tri-City Jr Americans','2025','Knighten Jr Americans','Hapo Center',None)

def print_table_contents(table_name, game_id=None):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if game_id is None:
        cursor.execute(f"SELECT * FROM {table_name}")
    else:
        cursor.execute(f"SELECT * FROM {table_name} WHERE game_id = {game_id}")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    conn.close()

#print_table_contents("Games")
#print_table_contents("Teams")
#print_table_contents("Players")
#print_table_contents("GoalieGameStats", 14)
#print_table_contents("PlayerGameStats", 14)



def add_game(game_id, date, home_team_id, home_score, away_team_id, away_score, league_play):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
                   INSERT INTO Games (game_id, date, home_team_id, home_score, away_team_id, away_score, league_play)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   """,(game_id, date, home_team_id, home_score, away_team_id, away_score, league_play))

    print(f"Added game with ID {game_id} on {date}")
    
    for player_id in range(1, 14):
        cursor.execute("""
            INSERT INTO PlayerGameStats (game_id, player_id, goals, assists, penalty_min, active)
            VALUES (?, ?, 0, 0, 0, 1)
        """, (game_id, player_id))
    
    for player_id in [4,9]:  # assuming player IDs 4 and 9 are goalies  
        cursor.execute("""
            INSERT INTO GoalieGameStats (game_id, player_id, shots_faced, saves, goals_allowed, active)
            VALUES (?, ?, 0, 0, 0, 1)
        """, (game_id, player_id))
    
    conn.commit()
    conn.close()

#add_game(14, '2026-01-23', 10, 4, 1, 5, 2)


def update_goalie_stats(game_id, player_id, shots_faced, saves, goals_allowed, active, result):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE GoalieGameStats
        SET shots_faced = ?, saves = ?, goals_allowed = ?, active=?, result=?
        WHERE game_id = ? AND player_id = ?
    """, (shots_faced, saves, goals_allowed, active, result, game_id, player_id))

    if cursor.rowcount == 1:
        print("✅ Update successful")
    elif cursor.rowcount == 0:
        print("🚫 No Record Found")
    else:
        print(f"Updated {cursor.rowcount} records, expected 1.")
    
    conn.commit()
    conn.close()


def update_player_stats(game_id, player_id, goals, assists, penalty_min, active):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
                   UPDATE PlayerGameStats
                   SET goals=?, assists=?, penalty_min=?, active=?
                   WHERE game_id=? AND player_id=?
                   """,(goals,assists,penalty_min,active,game_id,player_id))

    if cursor.rowcount == 1:
        print("✅ Update successful")
    elif cursor.rowcount == 0:
        print("🚫 No Record Found")
    else:
        print(f"Updated {cursor.rowcount} records, expected 1.")
    
    conn.commit()
    conn.close()



update_goalie_stats(14, 9, 33, 29, 4, 1, "W")
update_goalie_stats(14, 4, None, None, None, 0, None)

update_player_stats(14, 5, 3, 0, 0, 1)
update_player_stats(14, 13, 2, 0, 0, 1)
update_player_stats(14, 2, 0, 1, 0, 1)

update_player_stats(14, 1, 0, 0, 1, 1)
update_player_stats(14, 8, 0, 0, 1, 1)




















def update_player_active(game_id, player_id, active):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
                   UPDATE PlayerGameStats
                   SET active=?
                   WHERE game_id=? AND player_id=?
                   """,(active,game_id,player_id))

    if cursor.rowcount == 1:
        print("✅ Update successful")
    elif cursor.rowcount == 0:
        print("🚫 No Record Found")
    else:
        print(f"Updated {cursor.rowcount} records, expected 1.")
    
    conn.commit()
    conn.close()

#update_player_active(9, 9, False)
#update_player_active(7, 3, True)


def add_column(tbl, col):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} INTEGER DEFAULT 1")
    
    print(f"Added column {col} to table {tbl}")
    conn.commit()
    conn.close()

#add_column("Games", "game_type")

def drop_column(tbl, col):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"ALTER TABLE {tbl} DROP COLUMN {col}")
    
    conn.commit()
    conn.close()

#drop_column("Games", "league_play")

def update_game_type(game_id, game_type):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
                   UPDATE Games
                   SET game_type=?
                   WHERE game_id=?
                   """,(game_type,game_id))

    if cursor.rowcount == 1:
        print("✅ Update successful")
    elif cursor.rowcount == 0:
        print("🚫 No Record Found")
    else:
        print(f"Updated {cursor.rowcount} records, expected 1.")
    
    conn.commit()
    conn.close()
#update_game_type(9,0)