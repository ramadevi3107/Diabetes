import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model_file = 'model.pkl'
scaler_file = 'scaler.pkl'
df = pd.read_csv('diabetes.csv')
print("Missing values:", df.isnull().sum().sum())
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
ms = MinMaxScaler()
df[features] = ms.fit_transform(df[features])

print("After scaling:")
print(df.head())
x = df.drop(columns='Outcome')
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
with open(scaler_file, 'wb') as f:
    pickle.dump(ms, f)
