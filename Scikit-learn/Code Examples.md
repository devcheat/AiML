# Code examples 
1. **Introduction to Scikit-learn**
   - Installing Scikit-learn
     ```python
     pip install -U scikit-learn
     ```
   - Importing Scikit-learn
     ```python
     import sklearn
     ```

2. **Data Preprocessing**
   - Handling Missing Data
     ```python
     from sklearn.impute import SimpleImputer
     imp = SimpleImputer(missing_values=np.nan, strategy='mean')
     imp.fit_transform(X)
     ```
   - Encoding Categorical Variables
     ```python
     from sklearn.preprocessing import LabelEncoder
     le = LabelEncoder()
     le.fit_transform(y)
     ```
   - Feature Scaling
     ```python
     from sklearn.preprocessing import StandardScaler
     sc = StandardScaler()
     sc.fit_transform(X)
     ```

3. **Supervised Learning**
   - Linear Regression
     ```python
     from sklearn.linear_model import LinearRegression
     model = LinearRegression()
     model.fit(X, y)
     ```
   - Logistic Regression
     ```python
     from sklearn.linear_model import LogisticRegression
     model = LogisticRegression()
     model.fit(X, y)
     ```
   - Decision Trees
     ```python
     from sklearn.tree import DecisionTreeClassifier
     model = DecisionTreeClassifier()
     model.fit(X, y)
     ```
   - Random Forests
     ```python
     from sklearn.ensemble import RandomForestClassifier
     model = RandomForestClassifier()
     model.fit(X, y)
     ```

4. **Unsupervised Learning**
   - K-Means Clustering
     ```python
     from sklearn.cluster import KMeans
     model = KMeans(n_clusters=3)
     model.fit(X)
     ```
   - Principal Component Analysis
     ```python
     from sklearn.decomposition import PCA
     pca = PCA(n_components=2)
     X_pca = pca.fit_transform(X)
     ```

5. **Model Evaluation**
   - Cross-Validation
     ```python
     from sklearn.model_selection import cross_val_score
     scores = cross_val_score(model, X, y, cv=5)
     ```
   - Grid Search
     ```python
     from sklearn.model_selection import GridSearchCV
     parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
     svc = svm.SVC()
     clf = GridSearchCV(svc, parameters)
     clf.fit(X, y)
     ```
   - Performance Metrics
     ```python
     from sklearn.metrics import confusion_matrix
     y_pred = model.predict(X_test)
     cm = confusion_matrix(y_test, y_pred)
     ```
