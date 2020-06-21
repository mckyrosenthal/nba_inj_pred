# Using Machine Learning to Classify NBA Injuries

All data used in the project [can be downloaded here](https://people.ucsc.edu/~mmrosent/nba_inj_pred.html).

This project aims to apply machine learning (ML) techniques to the problem of resting players in NBA games strategically. Specifically, we seek to understand what factors contribute to games in which players get injured, and also to see if we can train an ML algorithm to decide whether players should be rested for a given game. This second task is a difficult one, particularly because beforehand metrics must be established to determine whether a given algorithm actually aids in this task. Simply identifying a large fraction of games where players get injured is insufficent, as injured games are rare, and algorithms with even a modest false positive (FP) rate may result in players actually playing less games overall.

### Data scraping and cleaning

The data for this project was complied from three distinct sources: first, NBA injuries from the 2010-2020 were downloaded from the [Pro Sports Transactions](https://www.prosportstransactions.com/basketball/Search/Search.php) website, using both the "Missed games due to injury" and the "Movement to/from injured/inactive list (IL)" filters. The second filter is necessary as these two catergories are mutually exclusive, and the first catergory misses many injuries. See [this notebook](download_inj_data.ipynb) for details.

This list of injuries is then correlated with the game the player was injured in, using game logs from [Basketball reference](https://www.basketball-reference.com/) [(see this notebook).](make_inj_df.ipynb) We use custom scrapers to pull not just the games the player was injured in, but all stats from the 2010-2020 seasons for each player that appears in our list of injuries. We also pull metadata for each player, i.e. their height, weight, which hand they shoot with, and their listed position on Basketball reference. We also classify players as a "Guard," "Wing," or "Big" using their listed position on Basketball reference. Finally, we filter out all players who didn't average at least 15 minutes per game over their career.

This list of games will constitute the entries in our training data, but we obviously can't use the the statistics from the games themselves as features . One thing we can do immediately with this data is [determine the distribution of the number of games players miss due to injuries.](missed_games.ipynb).

![alt text](missed_games_fig.png "Number of games missed due to injury")

The data have mean 5.889, and standard deviation 10.470.

To turn the in game statistics into features, statistics are windowed over different number of games before the current game, taking care not to include statistics from the game in question. We use 7, 14 and 21 day windows, and also a 1 day window corresponding to back-to-back games. [See this notebook.](windowing.ipynb)

We also include [speed-distance tracking data](https://stats.nba.com/players/speed-distance/). This data is only available starting with the 2013 season, so we include further limit our data to only include games from 2013-2020. We use selenium to build another custom scraper to pull this data, using 7, 14, and 21 day windows [(see this notebook).](speed_dist_scrape.ipynb). Finally, the speed-distance data is combined with our previous data [to form our final dataset.](make_final_dataset.ipynb) We use one-hot encoding on all catergorical data, resulting in 141 total features and 133072 examples.

