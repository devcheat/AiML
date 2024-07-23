# Module 3: Supervised Learning

## 1. Classification
Classification is a supervised learning task where the goal is to predict the class labels of new instances based on past observations. Scikit-learn provides various classification algorithms for this purpose, including k-Nearest Neighbors (kNN), Decision Trees, Support Vector Machines (SVM), and Logistic Regression.

Example using Logistic Regression for binary classification:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the LogisticRegression classifier
classifier = LogisticRegression()

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

In this example, we use Logistic Regression, a commonly used classification algorithm, to classify instances into two classes. We split the dataset into training and test sets, train the classifier on the training data, make predictions on the test data, and evaluate the accuracy of the classifier.

## 2. Regression
Regression is another supervised learning task where the goal is to predict continuous numerical values. Scikit-learn provides various regression algorithms, including Linear Regression, Ridge Regression, and Lasso Regression.

Example using Linear Regression for predicting house prices:
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the LinearRegression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the performance of the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```

In this example, we use Linear Regression to predict house prices based on features such as the number of rooms, the age of the house, etc. We split the dataset into training and test sets, train the model on the training data, make predictions on the test data, and evaluate the model's performance using Mean Squared Error (MSE).

These are just two examples of supervised learning tasks using scikit-learn. Depending on your specific problem and data, you may choose different algorithms and techniques to build and evaluate your models.

Certainly! Let's cover Module 3: Supervised Learning and Module 4: Unsupervised Learning, along with the two points inside each module.
