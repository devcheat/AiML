# Module 2: Data Handling with Scikit-learn

## 1. Loading Datasets
Scikit-learn provides several built-in datasets that you can use to practice and learn machine learning algorithms. These datasets are accessible through the `sklearn.datasets` module. Some commonly used datasets include Iris, Boston House Prices, and Breast Cancer datasets.

Example:
```python
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Access the features and target labels
X = iris.data  # Features
y = iris.target  # Target labels
```

In this example, we load the Iris dataset, which contains information about iris flowers, including sepal and petal dimensions, and the species of iris. We access the features using `iris.data` and the target labels using `iris.target`.

## 2. Data Preprocessing Techniques
Data preprocessing is an essential step in any machine learning pipeline. Scikit-learn provides various preprocessing techniques to clean, transform, and preprocess data before feeding it into machine learning models. Some common preprocessing techniques include feature scaling, handling missing values, and encoding categorical variables.

### Feature Scaling
Feature scaling ensures that all features have the same scale, which can be crucial for some machine learning algorithms.

Example using StandardScaler for feature scaling:
```python
from sklearn.preprocessing import StandardScaler

# Create an instance of the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)
```

### Handling Missing Values
Missing values in the dataset can cause issues during model training. Scikit-learn provides utilities to handle missing values, such as imputation.

Example using SimpleImputer for handling missing values:
```python
from sklearn.impute import SimpleImputer

# Create an instance of the SimpleImputer with a strategy (e.g., mean)
imputer = SimpleImputer(strategy='mean')

# Fit the imputer to the training data and transform it
X_train_imputed = imputer.fit_transform(X_train)

# Transform the test data using the same imputer
X_test_imputed = imputer.transform(X_test)
```

In these examples, we demonstrate how to use StandardScaler for feature scaling and SimpleImputer for handling missing values. These are just a couple of examples of data preprocessing techniques available in scikit-learn. Depending on your data and task, you may need to use other preprocessing techniques as well.
