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

def print_table_contents(table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    for row in rows:
        print(row)

    conn.close()



#update_goalie_stats(3, 9, 11, 10, 1)
#update_goalie_stats(5, 9, 11, 9, 2)

#print_table_contents("GoalieGameStats")
#print_table_contents("PlayerGameStats")
print_table_contents("Teams")

