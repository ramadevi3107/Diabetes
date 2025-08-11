import pandas as pd
import pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    ms = pickle.load(f)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
a = int(input("Enter Pregnancies: "))
b = int(input("Enter Glucose: "))
c = int(input("Enter BloodPressure: "))
d = int(input("Enter SkinThickness: "))
e = int(input("Enter Insulin: "))
f = float(input("Enter BMI: "))
g = float(input("Enter DiabetesPedigreeFunction: "))
h = int(input("Enter Age: "))
user_input = pd.DataFrame([[a, b, c, d, e, f, g, h]], columns=features)
scaled_input = ms.transform(user_input)
ans = model.predict(scaled_input)
print("Prediction:", "Diabetic" if ans[0] == 1 else "Not Diabetic")
