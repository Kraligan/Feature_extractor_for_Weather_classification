import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

with open("Data/features.pkl", "rb") as f:
    dataset = pickle.load(f)

X = dataset['features']
y = dataset['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify = y, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 30, 50],
    'min_samples_leaf': [1, 2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                    param_grid,
                    cv=5,
                    scoring='accuracy',
                    verbose=2)

grid.fit(X_train, y_train)
best_param = grid.best_params_
print("Best parameters:", best_param)

model = RandomForestClassifier(**best_param, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluation:
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Save model
os.makedirs("model", exist_ok=True)
with open("model/random_forest.pkl", "wb") as f:
    pickle.dump(model, f)
