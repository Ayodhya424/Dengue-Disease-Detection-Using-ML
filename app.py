import os
import io
from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import numpy as np
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)
model = joblib.load('rf_dengue_model.pkl')

# MongoDB Local connection
# Connect to local MongoDB instance running on default port 27017
try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    # Test connection
    client.admin.command('ping')
    db = client['dengue_detection_db']
    collection = db['user_predictions']
    print("✓ Connected to local MongoDB")
except Exception as e:
    print(f"⚠ Warning: Could not connect to MongoDB: {e}")
    print("Make sure MongoDB is running. Start with: mongod")
    client = None
    collection = None

# Field metadata with display names and ranges
FIELD_METADATA = {
    'Gender': {'type': 'select', 'options': ['Male', 'Female'], 'range': 'Male/Female'},
    'Age': {'type': 'number', 'range': '1-120 years', 'placeholder': 'e.g., 25'},
    'Hemoglobin(g/dl)': {'type': 'number', 'range': '7-20 g/dl', 'placeholder': 'e.g., 12.5'},
    'Neutrophils(%)': {'type': 'number', 'range': '0-100%', 'placeholder': 'e.g., 60'},
    'Lymphocytes(%)': {'type': 'number', 'range': '0-100%', 'placeholder': 'e.g., 30'},
    'Monocytes(%)': {'type': 'number', 'range': '0-100%', 'placeholder': 'e.g., 5'},
    'Eosinophils(%)': {'type': 'number', 'range': '0-10%', 'placeholder': 'e.g., 2'},
    'RBC': {'type': 'number', 'range': '3.5-6.0 (Million/µL)', 'placeholder': 'e.g., 4.5'},
    'HCT(%)': {'type': 'number', 'range': '30-55%', 'placeholder': 'e.g., 42'},
    'MCV(fl)': {'type': 'number', 'range': '80-100 fL', 'placeholder': 'e.g., 90'},
    'MCH(pg)': {'type': 'number', 'range': '27-33 pg', 'placeholder': 'e.g., 30'},
    'MCHC(g/dl)': {'type': 'number', 'range': '32-36 g/dl', 'placeholder': 'e.g., 34'},
    'RDW-CV(%)': {'type': 'number', 'range': '11-15%', 'placeholder': 'e.g., 13'},
    'Total Platelet Count(/cumm)': {'type': 'number', 'range': '150k-450k', 'placeholder': 'e.g., 250000'},
    'MPV(fl)': {'type': 'number', 'range': '7-11 fL', 'placeholder': 'e.g., 9'},
    'PDW(%)': {'type': 'number', 'range': '10-18%', 'placeholder': 'e.g., 15'},
    'PCT(%)': {'type': 'number', 'range': '0.15-0.40%', 'placeholder': 'e.g., 0.25'},
    'Total WBC count(/cumm)': {'type': 'number', 'range': '4k-11k', 'placeholder': 'e.g., 7000'},
    'Fever': {'type': 'select', 'options': ['Yes', 'No'], 'range': 'Yes/No'},
    'Severe_Body_Pain': {'type': 'select', 'options': ['Yes', 'No'], 'range': 'Yes/No'},
    'Headache': {'type': 'select', 'options': ['Yes', 'No'], 'range': 'Yes/No'},
    'Rash': {'type': 'select', 'options': ['Yes', 'No'], 'range': 'Yes/No'},
    'Bleeding_Signs': {'type': 'select', 'options': ['Yes', 'No'], 'range': 'Yes/No'},
    'Vomiting': {'type': 'select', 'options': ['Yes', 'No'], 'range': 'Yes/No'}
}

FEATURES = list(FIELD_METADATA.keys())

# Fields that are optional. Missing values are treated as 'No' (0)
OPTIONAL_FEATURES = {
    'Fever',
    'Severe_Body_Pain',
    'Headache',
    'Rash',
    'Bleeding_Signs',
    'Vomiting'
}

# NOTE: PDF/image text extraction was removed due to reliability and dependency issues.
# Uploading is limited to CSV files only.
def convert_form_data(form_data):
    """Convert form data to model input format."""
    data = {}
    for feature in FEATURES:
        value = form_data.get(feature, '')

        # Optional symptom fields are treated as No when left blank
        if not value and feature in OPTIONAL_FEATURES:
            data[feature] = 0
            continue

        if not value:
            return None, f"Missing value for {feature}"
        
        # Convert Yes/No to 1/0, Male/Female to 1/0
        if str(value).lower() in ['yes', 'male']:
            data[feature] = 1
        elif str(value).lower() in ['no', 'female']:
            data[feature] = 0
        else:
            try:
                data[feature] = float(value)
            except ValueError:
                return None, f"Invalid value for {feature}: {value}"
    
    return data, None

def make_prediction(data):
    """Make dengue prediction using the model."""
    if data is None:
        return None, None
    
    feature_array = np.array([data[feature] for feature in FEATURES]).reshape(1, -1)
    
    try:
        prediction = model.predict(feature_array)[0]
        probability = model.predict_proba(feature_array)[0]
        
        result = "🔴 DENGUE POSITIVE" if prediction == 1 else "🟢 DENGUE NEGATIVE"
        confidence = max(probability) * 100
        
        return result, round(confidence, 2)
    except Exception as e:
        return None, None

def store_prediction_data(data, result, confidence):
    """Store user prediction data in MongoDB."""
    if collection is None:
        print("MongoDB is not connected. Skipping data storage.")
        return
    
    try:
        document = {
            'timestamp': datetime.utcnow(),
            'input_data': data,
            'prediction': result,
            'confidence': confidence
        }
        collection.insert_one(document)
        print(f"✓ Prediction saved to MongoDB")
    except Exception as e:
        print(f"Error storing data in MongoDB: {e}")

@app.route('/')
def home():
    return render_template(
        'index.html',
        features=FEATURES,
        field_metadata=FIELD_METADATA,
        optional_features=OPTIONAL_FEATURES,
    )

@app.route('/download-template')
def download_template():
    """Download CSV template for easy data entry."""
    df = pd.DataFrame(columns=FEATURES)
    # Add example row
    df.loc[0] = ['Male', 30, 13.5, 65, 30, 3, 2, 4.5, 42, 90, 30, 34, 13, 250000, 9, 15, 0.25, 7000, 'Yes', 'Yes', 'No', 'No', 'No', 'No']
    
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='dengue_report_template.csv'
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('file')
        
        # If file is uploaded, extract data and show form for verification
        if file and file.filename:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                if len(df) == 0:
                    return render_template(
                        'index.html',
                        features=FEATURES,
                        field_metadata=FIELD_METADATA,
                        optional_features=OPTIONAL_FEATURES,
                        error="CSV file is empty. Please add patient data."
                    )
                
                extracted_list = df.to_dict('records')
                # Use first record for direct prediction
                csv_data = extracted_list[0]
                data, error = convert_form_data(csv_data)
                if error:
                    return render_template(
                        'index.html',
                        features=FEATURES,
                        field_metadata=FIELD_METADATA,
                        optional_features=OPTIONAL_FEATURES,
                        error=error
                    )
                
                result, confidence = make_prediction(data)
                
                # Store prediction data in MongoDB
                store_prediction_data(data, result, confidence)
                
                if result is None:
                    return render_template(
                        'index.html',
                        features=FEATURES,
                        field_metadata=FIELD_METADATA,
                        optional_features=OPTIONAL_FEATURES,
                        error="Error making prediction. Please check your data."
                    )
                
                return render_template(
                    'index.html',
                    features=FEATURES,
                    field_metadata=FIELD_METADATA,
                    optional_features=OPTIONAL_FEATURES,
                    result=result,
                    confidence=confidence,
                    input_data=data,
                    msg=f"Prediction made from uploaded CSV data."
                )
            else:
                return render_template(
                    'index.html',
                    features=FEATURES,
                    field_metadata=FIELD_METADATA,
                    optional_features=OPTIONAL_FEATURES,
                    error="Only CSV uploads are supported. Please use the CSV template or fill the form manually."
                )
        
        # If no file, process form data
        data, error = convert_form_data(request.form)
        if error:
            return render_template(
                'index.html',
                features=FEATURES,
                field_metadata=FIELD_METADATA,
                optional_features=OPTIONAL_FEATURES,
                error=error
            )
        
        result, confidence = make_prediction(data)
        
        # Store prediction data in MongoDB
        store_prediction_data(data, result, confidence)
        
        if result is None:
            return render_template(
                'index.html',
                features=FEATURES,
                field_metadata=FIELD_METADATA,
                optional_features=OPTIONAL_FEATURES,
                error="Error making prediction. Please check your data."
            )
        
        return render_template(
            'index.html',
            features=FEATURES,
            field_metadata=FIELD_METADATA,
            optional_features=OPTIONAL_FEATURES,
            result=result,
            confidence=confidence,
            input_data=data
        )

    except Exception as e:
        return render_template(
            'index.html',
            features=FEATURES,
            field_metadata=FIELD_METADATA,
            optional_features=OPTIONAL_FEATURES,
            error=f"Processing Error: {str(e)}"
        )

# if __name__ == '__main__':
#     app.run(debug=True, port=5500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)