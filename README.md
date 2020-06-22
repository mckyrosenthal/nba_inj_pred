# Using Machine Learning to Classify NBA Injuries

All data used in the project [can be downloaded here](https://people.ucsc.edu/~mmrosent/nba_inj_pred.html).

This project aims to apply machine learning (ML) techniques to the problem of resting players in NBA games strategically. Specifically, we seek to understand what factors contribute to games in which players get injured, and also to see if we can train an ML algorithm to decide whether players should be rested for a given game. This second task is a difficult one, particularly because beforehand metrics must be established to determine whether a given algorithm actually aids in this task. Simply identifying a large fraction of games where players get injured is insufficent, as injured games are rare, and algorithms with even a modest false positive (FP) rate may result in players actually playing less games overall.

### Data scraping and cleaning

The data for this project was complied from three distinct sources: first, NBA injuries from the 2010-2020 were downloaded from the [Pro Sports Transactions](https://www.prosportstransactions.com/basketball/Search/Search.php) website, using both the "Missed games due to injury" and the "Movement to/from injured/inactive list (IL)" filters. The second filter is necessary as these two catergories are mutually exclusive, and the first catergory misses many injuries. See [this notebook](download_inj_data.ipynb) for details.

This list of injuries is then correlated with the game the player was injured in, using game logs from [Basketball reference](https://www.basketball-reference.com/) [(see this notebook).](make_inj_df.ipynb) We use custom scrapers to pull not just the games the player was injured in, but all stats from the 2010-2020 seasons for each player that appears in our list of injuries. We also pull metadata for each player, i.e. their height, weight, which hand they shoot with, and their listed position on Basketball reference. We also classify players as a "Guard," "Wing," or "Big" using their listed position on Basketball reference. Finally, we filter out all players who didn't average at least 15 minutes per game over their career.

This list of games will constitute the entries in our training data, but we obviously can't use the the statistics from the games themselves as features . One thing we can do immediately with this data is [determine the distribution of the number of games players miss due to injuries.](missed_games.ipynb)

![alt text](missed_games_fig.png "Number of games missed due to injury")

The data have mean 5.889, and standard deviation 10.470.

To turn the in game statistics into features, statistics are windowed over different number of games before the current game, taking care not to include statistics from the game in question. We use 7, 14 and 21 day windows, and also a 1 day window corresponding to back-to-back games. [See this notebook.](windowing.ipynb)

We also include [speed-distance tracking data](https://stats.nba.com/players/speed-distance/). This data is only available starting with the 2013 season, so we include further limit our data to only include games from 2013-2020. We use selenium to build another custom scraper to pull this data, using 7, 14, and 21 day windows [(see this notebook).](speed_dist_scrape.ipynb). Finally, the speed-distance data is combined with our previous data [to form our final dataset.](make_final_dataset.ipynb) We use one-hot encoding on all catergorical data, resulting in 141 total features and 133072 examples.

### Data Analysis

We begin by attempt to model the data using a [simple logistic regression model](log_reg.ipynb). We begin by identifying the best features in a simple manner using the correlations between the features and whether the player was injured in the a given game. We indentify the 30 "best" features by keeping the 30 features with the highest correlation, but also dropping features that correspond to the same statistic but over a different time window. The best performing features found using this method are

![alt text](best_feats_corr.png "Best features from correlation with Was_Injured")

We then use 5-fold cross-validation to optimize the hyperparameters of the logistic regression before analyzing performance on the test set. The logistic regression classifier can correcrly identify around 50% of the injuries with a false positive rate of around 35%. This is a large absolute number of false positives given that the overall injury rate is around 3%

![alt text](con_mat_fig.png "Confusion matrix")

![alt text](roc_simple.png "Confusion matrix")

In order to investigate the best performing features in a more sophisticated manner, we also use recursive feature elimination (RFE) to identify the best performing subset of features. We again use 5-fold cross validation, and optimize the f1 score. The best scoring features are shown below.

![alt text](rfe_best_feat.png "RFE best features")

There is considerable overlap between these two sets of features. We see that the number of previous injuries and the player's age both appear, making it appear that long term wear and tear contributes substantially to the probability of an injury. The average number of free throw attempts and free throws the player averaged over the last 3 weeks appear as well, making it seem that going to the free throw line frequency is associated with injury. Both methods of selection favor some number of the speed-distance tracking stats, though RFE favors using more. While there are several differences between the two sets of feature, one of particular note is that features chosen through correlation include far more back-to-back statistics than RFE, which only chooses one b2b feature.

The model with features chosen using RFE was then compared to training data. The performance on the training set is quite similar to the performance above. 

#### Practical application of model

As it currently stands, while the model certainly useful for understanding what features are associated with injury, the model is not really practical for predicting NBA injuries. This stems from the model's FP rate, which misclassifies a large number of games. We can get a quantitative idea of how many games our classifier gains/loses by taking the number of injuries we correctly classify, multiplying by the number of games we gain by correctly classifying an injury, and subtracting the number of false positives, i.e.

\begin{align}
x = y
\end{align}
