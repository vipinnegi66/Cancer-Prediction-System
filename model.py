import pickle
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# select top 5 important features
selected_features = [
    "worst radius",
    "worst perimeter",
    "worst area",
    "worst concave points",
    "mean concave points"
]

X = X[selected_features]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

best_model = None
best_accuracy = 0
best_model_name = ""

# train and evaluate
for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    print(name, "Accuracy:", accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name


print("\nBest Model:", best_model_name)
print("Best Accuracy:", best_accuracy)

# save best model
pickle.dump(best_model, open("model.pkl", "wb"))

print("Best model saved successfully!")