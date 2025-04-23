import pandas as pd

# Load dataset from UCI (Iris data)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(url, header=None, names=columns)

# Display head
print("🔹 First 5 rows:")
print(df.head())

# Summary statistics
print("\n🔹 Summary statistics:")
print(df.describe())

# Class distribution
print("\n🔹 Class distribution:")
print(df['class'].value_counts())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier using Gini Index (default criterion)
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)

# 3. Plot the decision tree structure
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title("Decision Tree - Gini Criterion")
plt.show()

# 4. Evaluate accuracy on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 5. Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# 6. Predict class for a custom input (e.g., sepal=5.1, petal=1.5)
custom_input = np.array([[5.1, 3.5, 1.5, 0.2]])  # Example: sepal_length=5.1, sepal_width=3.5, petal_length=1.5, petal_width=0.2
prediction = model.predict(custom_input)
predicted_class = iris.target_names[prediction][0]
print(f"Predicted class for custom input {custom_input[0]}: {predicted_class}")

# 7. Apply pruning or depth control (max_depth) and compare accuracy

# Before pruning (no max_depth)
model_no_pruning = DecisionTreeClassifier(criterion='gini', random_state=42)
model_no_pruning.fit(X_train, y_train)
y_pred_no_pruning = model_no_pruning.predict(X_test)
accuracy_no_pruning = accuracy_score(y_test, y_pred_no_pruning)

# After pruning (max_depth=3)
model_pruned = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model_pruned.fit(X_train, y_train)
y_pred_pruned = model_pruned.predict(X_test)
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)

# Print accuracy before and after pruning
print(f"Accuracy before pruning (no max_depth): {accuracy_no_pruning:.4f}")
print(f"Accuracy after pruning (max_depth=3): {accuracy_pruned:.4f}")
