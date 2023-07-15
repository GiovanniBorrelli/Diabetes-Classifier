# Diabetes-Predictor
Artificial Intelligence University Project

Our project is about predicting if a patient has diabetes using a classifier.

## DATA ANALYSIS
The dataset used for training the model can be found here:
https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

In the Data Analysis folder, there is a matplotlib graphic for every attribute analysed.
There is a Correlation Matrix that shows the correlation between each attribute and the possibility to get diabetes.
There is a missingno graphic showing that there are no empty cells, and another graphic showing the duplicates.
The conclusion is that the dataset is imbalanced, with 91% negative cases for diabetes and 9% positive cases.

To balance the dataset, ADASYN and RandomUnderSampler were used to have a 50/50 ratio of positive/negative diabetes cases.
In order to do that, "gender" and "smoking_history" attributes were encoded.

## MODEL TRAINING
The model is trained using the diabetes dataset (Cross-validation is used).
The 3 best classifiers are: RandomForest, SVC, MLPClassifier:
All these classifiers have great scores of Accuracy, Precision, Recall and F1-Score.
The best one, RandomForest, was chosen.
The model trained with the unmodified dataset gave these scores:
- Precision: 0.93
- Accuracy: 0.97
- Recall: 0.69
- F1-Score: 0.79

Using the balanced dataset increased the overall performance of the model, having these scores:
- Precision: 0.97
- Accuracy: 0.94
- Recall: 0.90
- F1-Score: 0.93

### Iperparameters Optimization
Using GridSearchCV, the best Iperparameters settings were:
{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50},
which gave these scores:
- Precision: 0.97
- Accuracy: 0.94
- Recall: 0.91
- F1-Score: 0.93

## Conclusion
The performance of a Random Forest classifier was assessed in predicting diabetes using a dataset containing 94,000 records. 
The model underwent hyperparameter tuning to optimize its performance. 
The results revealed a precision rate of around 94%, indicating that the model accurately identified a significant portion of positive diabetes cases. 
Moreover, the model achieved a recall rate of 91%, demonstrating its ability to correctly identify a large proportion of actual diabetes cases. 
These findings suggest that the model is well-calibrated and exhibits a balanced performance in predicting diabetes based on various health indicators and lifestyle factors.
