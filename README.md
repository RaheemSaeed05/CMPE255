# CMPE255


Article: https://medium.com/@raheem.saeed2005/from-tweets-to-sentiment-building-a-machine-learning-classifier-with-crisp-dm-7136cc181844
Youtube: 


# Assignment 1 — Twitter Sentiment Analysis

In Assignment 1 - using ChatGPT to perform an end-to-end data science analysis on a Kaggle Twitter Sentiment Analysis dataset.

For my scope the goal was to classify tweets as:

- Positive
- Neutral
- Negative

The project followed the CRISP-DM data science methodology and included exploratory data analysis, data cleaning, preprocessing, feature engineering, feature selection, clustering, machine learning model comparison, evaluation, and error analysis.

## Dataset

The project uses a Kaggle Twitter Sentiment Analysis dataset.

After cleaning, the dataset contained 3,534 valid tweets.

## Models Tested

- Multinomial Naive Bayes
- Complement Naive Bayes
- Logistic Regression
- Linear SVM
- SGDClassifier

## Final Model

The final model used:

**TF-IDF + Structural Tweet Features + Linear SVM**

Final holdout performance:

- Accuracy: 65.91%
- Macro-F1: 65.83%
- Majority Baseline Accuracy: 40.45%


## ChatGPT Transcript

The complete ChatGPT data science workflow is included in:

`255_HW_1.pdf`

The dataset used for the analysis is included as:

`test.csv`
