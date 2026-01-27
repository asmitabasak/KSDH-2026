Kharagpur Data Science Hackathon (KDSH)

Starter Code Template

Author: Asmita Basak

import pandas as pd import numpy as np import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler from sklearn.metrics import accuracy_score, classification_report from sklearn.linear_model import LogisticRegression

-----------------------------

1. Load Dataset

-----------------------------

Update the path as per your dataset

data_path = "data/dataset.csv" data = pd.read_csv(data_path)

print("Dataset shape:", data.shape) print(data.head())

-----------------------------

2. Basic Data Exploration

-----------------------------

print(data.info()) print(data.describe())

-----------------------------

3. Preprocessing

-----------------------------

Example: handling missing values

data = data.dropna()

Separate features and target

X = data.drop(columns=["target"])  # change 'target' column name y = data["target"]

Train-test split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )

Feature scaling

scaler = StandardScaler() X_train = scaler.fit_transform(X_train) X_test = scaler.transform(X_test)

-----------------------------

4. Model Training

-----------------------------

model = LogisticRegression() model.fit(X_train, y_train)

-----------------------------

5. Evaluation