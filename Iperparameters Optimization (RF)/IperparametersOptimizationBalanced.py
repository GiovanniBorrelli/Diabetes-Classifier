import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

# Leggi il dataset da un file Excel
dataset = pd.read_excel("DiabeteBalanced.xlsx")

# Seleziona gli attributi rilevanti per la classificazione
attributes = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", 
"blood_glucose_level", "gender_Female", "gender_Male", "gender_Other", "smoking_history_No Info", 
"smoking_history_current", "smoking_history_ever", "smoking_history_former", 
"smoking_history_never", "smoking_history_not current"]
X = dataset[attributes]
y = dataset["diabetes"]

# Trasforma le variabili categoriche in variabili dummy
X = pd.get_dummies(X)

# Definisci la griglia degli iperparametri
param_grid = {
    'n_estimators': [10, 50],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

# Crea il classificatore Random Forest
rf = RandomForestClassifier(random_state=42)

# Esegui la ricerca a griglia per trovare la combinazione ottimale di iperparametri
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Ottieni i risultati della ricerca a griglia
results = grid_search.cv_results_

# Stampa le metriche di valutazione per ogni combinazione di iperparametri
for mean_score, params in zip(results['mean_test_score'], results['params']):
    classifier = RandomForestClassifier(**params, random_state=42)
    
    # Esegui la cross-validation e calcola le metriche
    accuracy = cross_val_score(classifier, X, y, cv=5, scoring='accuracy').mean()
    precision = cross_val_score(classifier, X, y, cv=5, scoring='precision').mean()
    recall = cross_val_score(classifier, X, y, cv=5, scoring='recall').mean()
    f1 = cross_val_score(classifier, X, y, cv=5, scoring='f1').mean()

    # Stampa le metriche per ogni classificatore con parametri diversi
    print(f"Iperparametri: {params}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("--------------")
