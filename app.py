from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('ocean_acidification_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            # Extract form data with default value 0.0 if missing
            fields = [
                "pH_TS_insitu_measured", "Calcite", "Carbonate_insitu_calculated", "Aragonite", 
                "Carbonate_measured", "Carbonate_insitu_measured", "pH_TS_insitu_calculated", 
                "CTDTEMP_ITS90", "Oxygen", "recommended_Oxygen", "CTDOXY", "Niskin_ID", 
                "Longitude", "Accession", "Salinity_PSS78"
            ]

            # Convert input values to float
            features = [float(request.form.get(field, 0.0)) for field in fields]

            # Debugging: Print received features
            print("Received features:", features)
            print("Feature count:", len(features))  # Should print 15

            # Ensure the correct number of features
            if len(features) != 15:
                return f"Error: Expected 15 features, but got {len(features)}."

            # Convert to numpy array and reshape for prediction
            features = np.array(features).reshape(1, -1)

            # Make prediction using the trained model
            prediction = model.predict(features)[0]
            return render_template('result.html', prediction=round(prediction, 4))

        except ValueError as e:
            return f"Error: Invalid input. Please ensure all fields contain valid numbers. {e}"

        except Exception as e:
            return f"Error: {e}"

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
