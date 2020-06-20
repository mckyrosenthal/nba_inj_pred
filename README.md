# Using Machine Learning to Classify NBA Injuries

This project aims to apply machine learning (ML) techniques to the problem of resting players in NBA games strategically. Specifically, we seek to understand what factors contribute to games in which players get injured, and also to see if we can train an ML algorithm to decide whether players should be rested for a given game. This second task is a difficult one, particularly because beforehand metrics must be established to determine whether a given algorithm actually aids in this task. Simply identifying a large fraction of games where players get injured is insufficent, as injured games are rare, and algorithms with even a modest false positive (FP) rate may result in players actually playing less games overall.

The data for this project was complied from three distinct sources: first, NBA injuries from the 2010-2020 were downloaded from the [Pro Sports Transactions](https://www.prosportstransactions.com/basketball/Search/Search.php) website, using both the "Missed games due to injury" and the "Movement to/from injured/inactive list (IL)" filters. The second 

