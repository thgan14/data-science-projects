# Question Answering System 

Question Answering system using Google's Natural Question dataset. This project is based on the kaggle competition from here: https://www.kaggle.com/c/tensorflow2-question-answering/overview

## Background
The dataset consists of 300,000 questions each based on a particular Wikipedia page. Each training example comes with candidate answers from the Wikipedia page. The no. of candidate answers range from 20 - 1000's. The goal was to predict the correct candidate answer from the choices. In addition to this, I also had to predict Short Answers for the questions. These short answers would either be made up of text from the Long answers or Yes/No answers.

## Model
I used Google's BERT model for this project. This model has achieved State of the Art results in many NLP tasks so seemed like the right way to go. It was a challenge to understand the model and learnt to implement it for this particular task.

I ended up creating 2 separate models for long and short answers. The short answer model would create candidate answers from the predicted long answer using text segmentation and named entity recognition and then make a prediction for the best fit short answer (if any).

## Metrics and evaluation
The Metric i used was the F1 score which takes into account both precision & recall. The Long answer model had an F1 score of 0.91 and worked fairly well in real world prediction tasks. However, the short answer model did not predict answers well despite having an F1 score of 0.89 in training validation. This could be due to the the flawed approach of creating candidates from the long answers. This is a cumbersome method and does not guarantee that the actual short answer will be in the list of candidates. The eventual Kaggle submission achieved a combined Micro f1 score of 0.31, this ranked about 800 of 1,300 entries.

## Improvements
I only trained the model on 10,000 of the 300,000 training data examples. The model performance would probably see improvements with longer training. I was also only ably to use a batch size of 16 as the model had 110M trainable parameters. Additional GPU's would have allowed me to use larger batch sizes which could have lead to improved predictions. Lastly, a new approach to the short answers should be implemented, the current one does not show much promise of accurately predicting answers.
