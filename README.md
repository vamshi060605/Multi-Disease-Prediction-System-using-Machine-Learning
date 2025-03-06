# Multi-Disease Prediction System using Machine Learning

##  Project Overview
The **Multi-Disease Prediction System** is a machine learning-based solution designed to predict the likelihood of multiple diseases, including **Diabetes, Liver Disease, and Heart Disease**. Using **Random Forest Classifier**, this project leverages medical datasets to train models and provide accurate predictions based on patient health metrics.

##  Features
- **Predicts multiple diseases:** Diabetes, Liver Disease, and Heart Disease
- **Data Preprocessing:** Handling missing values, feature scaling, and normalization
- **Machine Learning Model:** Implements **Random Forest Classifier**
- **Model Evaluation:** Accuracy Score, Confusion Matrix, Classification Report, and ROC Curve
- **Visualization:** Data distribution, correlation heatmaps, and performance metrics
- **Interactive UI (Optional):** Can be integrated with Flask or Streamlit for a user-friendly interface

## 🛠 Tech Stack
- **Programming Language:** Python
- **Libraries:**
  - Data Processing: `pandas`, `numpy`
  - Machine Learning: `scikit-learn`
  - Visualization: `matplotlib`, `seaborn`
- **Development Environment:** Google Colab / Jupyter Notebook

##  Dataset Information
The datasets used in this project contain medical parameters for each disease. Each dataset includes features such as:
- **Diabetes:** Glucose Level, Blood Pressure, BMI, Insulin, etc.
- **Liver Disease:** Age, Total Bilirubin, Alkaline Phosphatase, Albumin, etc.
- **Heart Disease:** Age, Cholesterol, Resting Blood Pressure, Max Heart Rate, etc.

##  Installation & Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/multi-disease-prediction.git
   cd multi-disease-prediction
   ```
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook
   ```

##  Model Training & Evaluation
1. **Data Preprocessing:**
   - Handle missing values and clean the dataset
   - Normalize and scale features
   - Split dataset into **training (80%)** and **testing (20%)**
2. **Model Training:**
   - Uses `RandomForestClassifier` from `scikit-learn`
   - Trained on historical medical data
3. **Performance Evaluation:**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report
   - ROC Curve for model validation

##  Results
- The **Random Forest model** achieved high accuracy in disease prediction.
- The **ROC curve** indicates a strong ability to differentiate between diseased and non-diseased patients.
- Feature importance analysis shows which medical parameters contribute the most to predictions.

##  Future Enhancements
- Integrate a **Web Interface** using Flask or Streamlit.
- Expand the system to predict additional diseases.
- Use **Deep Learning models (Neural Networks)** for improved accuracy.


