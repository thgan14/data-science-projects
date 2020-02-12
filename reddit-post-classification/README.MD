# Web API scrppuing & Reddit postClassification

## Overview:
This project involves scraping 2 Subreddits for their post then building a classification model that will predict which subreddit and unseen post is from

## Subreddits used:
World News - https://www.reddit.com/r/worldnews/
Today I learned - https://www.reddit.com/r/todayilearned/

I used the pushshift api to scrape around 3000 posts from each subreddit. 1,500 posts were used to build the classification model. The final model used was a voting classifier model which consisted of 5 models:
1. Logistic Regression
2. Multinomial Naive Bayes
3. Extra Trees classifier
4. Random Forest classifier
5. AdaBoost classifier (based on an Extra trees model)

The final accuracy achieved was 90.5% on unseen data 
