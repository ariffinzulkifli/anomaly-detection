import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, auc, accuracy_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = './dataset/loan_application.csv'  # Updated to the correct file path
loan_application_data = pd.read_csv(file_path)

# Missing Value Treatment
for i in loan_application_data.columns:
    clas = loan_application_data[i].dtypes
    if clas == 'object':
        loan_application_data[i].fillna(loan_application_data[i].mode()[0], inplace=True)
    else:
        loan_application_data[i].fillna(loan_application_data[i].mean(), inplace=True)

# Applying LabelEncoder for converting object data types to integer
le = LabelEncoder()
df4 = loan_application_data.copy()
for i in df4.columns:
    cls = df4[i].dtypes
    if cls == 'object':
        df4[i] = le.fit_transform(df4[i].astype(str))
    else:
        df4[i] = df4[i]

# Split the data into training and test sets
X = df4.drop('Anomaly', axis=1)  # Assuming 'Anomaly' is the target variable
y = df4['Anomaly']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Train the Logistic Regression model
classifier = LogisticRegression(max_iter=1000)  # Increased the number of iterations
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)
y_predd = ["Anomaly" if i == 1 else "Not Anomaly" for i in y_pred]

# Confusion Matrix and Accuracy
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Calculate probabilities, precision, recall, F1 score, and AUC
lr_probs = classifier.predict_proba(X_test)[:, 1]
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1 = f1_score(y_test, y_pred)
lr_auc = auc(lr_recall, lr_precision)

# Add predictions to X_test for review
X_test_with_predictions = X_test.copy()
X_test_with_predictions['Prediction'] = y_predd
