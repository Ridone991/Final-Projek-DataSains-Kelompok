
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

df = pd.read_csv("cardio_train.csv", sep=';')
df.drop(columns=['id'], inplace=True)
df['age'] = (df['age'] / 365).astype(int)


X = df.drop('cardio', axis=1)
y = df['cardio']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 6. Evaluasi
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Akurasi:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))


joblib.dump(model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
# Visualisasi confusion matrix
cm = confusion_matrix(y_test, y_pred)           
