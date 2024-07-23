# Module 5: Model Evaluation and Selection

## 1. Cross-Validation
Cross-validation is a technique used to assess how well a model generalizes to new data. It involves splitting the dataset into multiple subsets (folds), training the model on some of the folds, and evaluating it on the remaining fold. This process is repeated multiple times, and the evaluation scores are averaged to obtain a more reliable estimate of the model's performance.

Example using k-fold cross-validation with logistic regression:
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Create an instance of the LogisticRegression model
classifier = LogisticRegression()

# Perform k-fold cross-validation (e.g., k=5)
cv_scores = cross_val_score(classifier, X, y, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())
```

## 2. Hyperparameter Tuning with Grid Search
Hyperparameter tuning involves selecting the best set of hyperparameters for a machine learning model to optimize its performance. Grid search is a common technique used for hyperparameter tuning, where you specify a grid of hyperparameter values, and the algorithm evaluates the model's performance for each combination of values.

Example using grid search for hyperparameter tuning with support vector machine (SVM):
```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Create an instance of the SVM model
svm = SVC()

# Define the hyperparameter grid
param_grid = {'C': [0.1, 1, 10],
              'gamma': [0.1, 0.01, 0.001],
              'kernel': ['rbf', 'linear']}

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X, y)

# Print the best hyperparameters and corresponding score
print("Best hyperparameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
```

## 3. Model Evaluation Metrics
Model evaluation metrics provide insights into a model's performance. Scikit-learn offers various evaluation metrics depending on the task, such as classification accuracy, precision, recall, F1-score, ROC-AUC for classification tasks, and mean squared error (MSE), R-squared for regression tasks.

Example using accuracy score for classification:
```python
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the LogisticRegression model
classifier = LogisticRegression()

# Train the model on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
