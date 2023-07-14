import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

dataset = pd.read_excel("Diabete.xlsx")

attributes = ["gender", "age", "hypertension", "heart_disease", "smoking_history",
              "bmi", "HbA1c_level", "blood_glucose_level"]
X = dataset[attributes]
y = dataset["diabetes"]

X = pd.get_dummies(X)

# Griglia degli iperparametri
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

rf = RandomForestClassifier(random_state=42)

# Esegui la GridSearchCV per trovare la combinazione ottimale di iperparametri
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

results = grid_search.cv_results_

# Stampa le metriche di valutazione per ogni combinazione di iperparametri
for mean_score, params in zip(results['mean_test_score'], results['params']):
    classifier = RandomForestClassifier(**params, random_state=42)
    
    accuracy = cross_val_score(classifier, X, y, cv=5, scoring='accuracy').mean()
    precision = cross_val_score(classifier, X, y, cv=5, scoring='precision').mean()
    recall = cross_val_score(classifier, X, y, cv=5, scoring='recall').mean()
    f1 = cross_val_score(classifier, X, y, cv=5, scoring='f1').mean()

    print(f"Iperparametri: {params}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("--------------")
