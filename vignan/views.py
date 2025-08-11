from django.shortcuts import render, redirect
import numpy as np
import pickle
import os

# Define paths to model and scaler
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'model.pkl')
scaler_path = os.path.join(base_dir, 'scaler.pkl')

# Load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

def home(request):
    if request.method == 'POST':
        try:
            # Get form inputs
            preg = float(request.POST.get('Pregnancies'))
            glucose = float(request.POST.get('Glucose'))
            bp = float(request.POST.get('BloodPressure'))
            skin = float(request.POST.get('SkinThickness'))
            insulin = float(request.POST.get('Insulin'))
            bmi = float(request.POST.get('BMI'))
            dpf = float(request.POST.get('DiabetesPedigreeFunction'))
            age = float(request.POST.get('Age'))

            # Create input array
            input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

            # Scale the data
            scaled_data = scaler.transform(input_data)

            # Predict
            prediction = model.predict(scaled_data)[0]
            result = 'Positive for Diabetes' if prediction == 1 else 'Negative for Diabetes'

            return redirect('hello', result=result)

        except Exception as e:
            return render(request, 'home.html', {'error': f'Invalid input: {str(e)}'})

    return render(request, 'home.html')

def hello(request, result):
    return render(request, 'result.html', {'result': result})
