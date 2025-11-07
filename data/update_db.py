import sqlite3

#### stub for manually updating the database ####

db_path = "HockeyStat.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
    UPDATE GoalieGameStats
    SET shots_faced = ?, saves = ?, goals_allowed = ?,
    WHERE game_id = ? AND player_id = ?
""", (11, 10, 1, 3, 9))

if cursor.rowcount == 1:
    print("✅ Update successful")
elif cursor.rowcount == 0:
    print("🚫 No Record Found")
else:
    print(f"Updated {cursor.rowcount} records, expected 1.")

cursor.execute("""
    UPDATE GoalieGameStats
    SET shots_faced = ?, saves = ?, goals_allowed = ?,
    WHERE game_id = ? AND player_id = ?
""", (11, 10, 1, 5, 9))

if cursor.rowcount == 1:
    print("✅ Update successful")
elif cursor.rowcount == 0:
    print("🚫 No Record Found")
else:
    print(f"Updated {cursor.rowcount} records, expected 1.")


conn.commit()
conn.close()