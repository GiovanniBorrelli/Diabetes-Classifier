import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN

dataset = pd.read_excel("Diabete.xlsx")

dataset_encoded = pd.get_dummies(dataset, columns=['gender', 'smoking_history'])

X = dataset_encoded.drop("diabetes", axis=1)
y = dataset_encoded["diabetes"]

# Crea classi sintetiche
adasyn = ADASYN(sampling_strategy=0.5)
X_resampled_adasyn, y_resampled_adasyn = adasyn.fit_resample(X, y)

# Scegli randomicamente un sottogruppo di persone senza diabete
under_sampler = RandomUnderSampler(sampling_strategy=1)
X_resampled_under, y_resampled_under = under_sampler.fit_resample(X_resampled_adasyn, y_resampled_adasyn)

#Il dataset risultante avrà una ratio 50/50 di diabetici/non diabetici
balanced_dataset = pd.concat([X_resampled_under, y_resampled_under], axis=1)

balanced_dataset.to_excel("DiabeteBalanced.xlsx", index=False)
