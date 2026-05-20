## Dengue Disease Prediction Using ML System
A machine learning project that predicts the likelihood of dengue infection based on 
patient medical data and blood test values. This system uses a Random Forest Classifier to analyze input parameters and provide diagnostic predictions.


# Features
- Upload medical reports or enter patient data manually.

- Predict dengue infection using blood test values and symptoms.

- User‑friendly web interface with clear result visualization.

- Built with Python, Scikit‑learn, Flask/Streamlit (depending on your setup).

- Supports structured medical parameters such as hemoglobin, RBC, WBC counts, and differential percentages.

# Tech Stack
- Programming Language: Python
- Dataset: Medical blood test records (custom/preprocessed dataset)

Libraries:

- scikit-learn (Random Forest Classifier)

- pandas, numpy (data preprocessing)

- matplotlib, seaborn (visualization)

- flask (web interface)


# Installation & Setup
``` Bash
# Clone the repository
git clone https://github.com/Ayodhya424/Dengue-Disease-Detection-Using-ML.git
cd Dengue-Disease-Detection-Using-ML

# install dependencies
pip install -r requirements.txt

#Run the application
python app.py
```

# Input Parameters
The model accepts the following medical features:
- Gender
- Age
- Hemoglobin (g/dl)
- Neutrophils (%)
- Lymphocytes (%)
- Monocytes (%)
- Eosinophils (%)
- RBC count
- HCT (%)
- MCV (fl)
- MCH (pg)
- MCHC (g/dl)

# Model Details
- Algorithm: Random Forest Classifier
Reason for Choice:

- Handles high‑dimensional medical data
- Robust against overfitting
- Provides feature importance for medical insights

- Evaluation Metrics: Accuracy, Precision, Recall, F1‑Score

# sample Screenshot
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/f2c7680e-9d5e-49f7-b7cd-3e864b2c4fac" />

# Example Output
- No Dengue Detected → Green box with confirmation message
- Dengue Detected → Red box with alert message

## Project Structure
<img width="636" height="216" alt="image" src="https://github.com/user-attachments/assets/2f376394-90c3-4cf3-95b9-19535240a1c3" />

# Future Improvements
- Integration with hospital databases for real‑time data.
- Adding more clinical features (platelet count, fever duration, etc.).
- Deploying on cloud (AWS/GCP/Azure) for scalability.
- Mobile app integration for quick patient screening.




