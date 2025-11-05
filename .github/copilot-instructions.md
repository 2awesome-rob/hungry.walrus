# AI Agent Instructions for VB_scorekeeper

  ## Project Overview
  This is a web application for viewing hockey player statistics.
  To support this, it tracks game by game stats for players on a team. In the initial usecase we only care about the stats for players on our team, so opposing team rosters and stats may not be input. However, the app should be extensible to tracking statistics across a small rec league. Anticipate the app will be run on phones and tablets - so screen layout efficiency is a must! Rec leagues record simple statistics, shift level stats like +/- are usually not recorded.
  A separate admin mode is used to support data entry and update - this may be run on a fullsize computer 

  ## Core Architecture
  -- **Persistent Data Records**
    The HockeyStat.db has following existing tables Teams, Players, Games, PlayerGameStats, and GoalieGameStats with SCHEMA:

    TABLE Teams (
                team_id INTEGER PRIMARY KEY AUTOINCREMENT,
                club TEXT NOT NULL,
                season YEAR,
                team TEXT NOT NULL,
                location TEXT,
                coach TEXT
            )

    TABLE Players (
                player_id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER NOT NULL,
                jersey_num INTEGER,
                name TEXT NOT NULL,
                position TEXT CHECK (position IN ('F', 'D', 'G', 'C', 'W')),
                FOREIGN KEY (team_id) REFERENCES Team(team_id)
            )

    TABLE Games (
                game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                home_team_id INTEGER,
                away_team_id INTEGER,
                home_score INTEGER,
                away_score INTEGER,
                result TEXT CHECK (result IN ('W', 'L', 'T')),
                point_diff INTEGER,
                overtime BOOLEAN DEFAULT FALSE,
                shootout BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (home_team_id) REFERENCES Team(team_id),
                FOREIGN KEY (away_team_id) REFERENCES Team(team_id)
            )

    TABLE PlayerGameStats (
                game_id INTEGER NOT NULL,
                player_id INTEGER NOT NULL,
                goals INTEGER DEFAULT 0,
                assists INTEGER DEFAULT 0,
                penalty_min INTEGER DEFAULT 0,
		        active BOOL DEFAULT TRUE,
                PRIMARY KEY (game_id, player_id),
                FOREIGN KEY (game_id) REFERENCES Game(game_id),
                FOREIGN KEY (player_id) REFERENCES Player(player_id)
            )


    TABLE GoalieGameStats (
                game_id INTEGER NOT NULL,
                player_id INTEGER NOT NULL,
                shots_faced INTEGER DEFAULT 0,
                saves INTEGER DEFAULT 0,
                goals_allowed INTEGER DEFAULT 0,
                result TEXT CHECK (result IN ('W', 'L', '-')),
		        active BOOL DEFAULT TRUE,
                PRIMARY KEY (game_id, player_id),
                FOREIGN KEY (game_id) REFERENCES Game(game_id),
                FOREIGN KEY (player_id) REFERENCES Player(player_id)
            )


  -- **User Interface**: Use Streamlit and organize the user app into two primary tabs (mobile/tablet-first layout). 
     1. Games
        - View past game results, including active / inactive roster for the game (note inactive players should be crossed out, backup goalie may skate as a player)
	- Query and Sort by 
     2. Player Stats
        - Select a player and View /Explore stats (eg, season totals, per game stats, last 5 games, plots, etc)

  -- **Admin Interface**: Use Streamlit and organize the admin app mode to support data entry and updating. this may be the same app loaded with additional privileges or a second app specially designed for interacting with the same database
     1. Teams and Players
        - Add/Edit Teams and Rosters
     2. Games
        - Add/Edit Game Results
        - Include active/inactive roster, goalie stats, and player stats
