# Pts-Machine-Learning

This repo contains a CSV with NBA player data and a function to train a KNN, Random Forrest, Gradient Boosting Regressor, and Linear machine learning models to predict player points in a game.  The training and testing results are written to seperate Text files to compare each models performance

## Required Libraries
* sklearn
* numpy
* pandas

## Data
The data in the CSV is either individual player data or team data which has been compared to league averages to standardize it.  For example if a team rebounds 85% of opponents misses and the league average is 82% the statistic will be standardized by .85/.82.  
All team data is based on the 5 starters in the starting lineup and then averaged.
