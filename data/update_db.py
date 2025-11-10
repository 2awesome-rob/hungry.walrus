import sqlite3

#### stub for manually updating the database ####

db_path = "data/HockeyStat.db"


def update_goalie_stats(game_id, player_id, shots_faced, saves, goals_allowed):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE GoalieGameStats
        SET shots_faced = ?, saves = ?, goals_allowed = ?
        WHERE game_id = ? AND player_id = ?
    """, (shots_faced, saves, goals_allowed, game_id, player_id))

    if cursor.rowcount == 1:
        print("✅ Update successful")
    elif cursor.rowcount == 0:
        print("🚫 No Record Found")
    else:
        print(f"Updated {cursor.rowcount} records, expected 1.")
    
    conn.commit()
    conn.close()

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

def delete_game(game_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"DELETE FROM Games WHERE game_id = {game_id}")

    if cursor.rowcount == 1:
        print("✅ Update successful")
    elif cursor.rowcount == 0:
        print("🚫 No Record Found")
    else:
        print(f"Updated {cursor.rowcount} records, expected 1.")
    
    conn.commit()
    conn.close()




def update_player_stats(game_id, player_id, goals, assists, penalty_min):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
                   UPDATE PlayerGameStats
                   SET goals=?, assists=?, penalty_min=?
                   WHERE game_id=? AND player_id=?
                   """,(goals,assists,penalty_min,game_id,player_id))

    if cursor.rowcount == 1:
        print("✅ Update successful")
    elif cursor.rowcount == 0:
        print("🚫 No Record Found")
    else:
        print(f"Updated {cursor.rowcount} records, expected 1.")
    
    conn.commit()
    conn.close()

#update_player_stats(6, 9, 3, 1, 1)
#update_goalie_stats(3, 9, 11, 10, 1)
#update_goalie_stats(5, 9, 11, 9, 2)

#print_table_contents("GoalieGameStats", 7)
#print_table_contents("PlayerGameStats", 7)
#print_table_contents("Teams")
#delete_game(7)


print_table_contents("Games")
#print_table_contents("Players")



