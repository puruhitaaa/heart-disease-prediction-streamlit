import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_model_and_scaler():
    with open('heart_disease_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except:
    st.error("Error: Model files not found. Please ensure 'heart_disease_model.pkl' and 'scaler.pkl' are present.")
    st.stop()

def prediction(features):
    features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]
    return prediction[0], probability

def main():
    st.title("Heart Disease Prediction System")
    
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stAlert {
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox("Choose a page", ["Home", "Prediction", "Model Insights"])
    
    if page == "Home":
        st.markdown("""
            <div style="text-align: center">
                <h1>❤️ Welcome to the Heart Disease Prediction System</h1>
                <p>This application uses machine learning to predict the risk of heart disease based on patient health metrics.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                ### How to use:
                1. Navigate to the Prediction page
                2. Enter your health metrics
                3. Click 'Predict' to see your results
                4. View detailed model insights in the Model Insights page
            """)
        with col2:
            st.markdown("""
                ### Features used:
                - Age
                - Sex
                - Chest Pain Type
                - Blood Pressure
                - And more...
            """)
            
    elif page == "Prediction":
        st.header("Patient Health Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=20, max_value=100, value=40)
            sex = st.selectbox("Sex", options=["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", 
                            options=["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
            chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[False, True])
            restecg = st.selectbox("Resting ECG Results", 
                                 options=["normal", "st-t abnormality", "lv hypertrophy"])
            thalch = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise Induced Angina", options=[False, True])
            
        with col3:
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST", 
                               options=["upsloping", "flat", "downsloping"])
            ca = st.number_input("Number of Major Vessels", min_value=0, max_value=4, value=0)
            thal = st.selectbox("Thalassemia", 
                              options=["normal", "fixed defect", "reversable defect"])
        
        if st.button("Predict"):
            categorical_mappings = {
                'sex': {'Male': 1, 'Female': 0},
                'fbs': {True: 1, False: 0},
                'exang': {True: 1, False: 0},
                'cp': {'typical angina': 0, 'atypical angina': 1, 'non-anginal': 2, 'asymptomatic': 3},
                'restecg': {'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2},
                'slope': {'upsloping': 0, 'flat': 1, 'downsloping': 2},
                'thal': {'normal': 0, 'fixed defect': 1, 'reversable defect': 2}
            }
            
            features = [
                age,
                categorical_mappings['sex'][sex],
                categorical_mappings['cp'][cp],
                trestbps,
                chol,
                categorical_mappings['fbs'][fbs],
                categorical_mappings['restecg'][restecg],
                thalch,
                categorical_mappings['exang'][exang],
                oldpeak,
                categorical_mappings['slope'][slope],
                ca,
                categorical_mappings['thal'][thal]
            ]
            
            prediction_value, probability = prediction(features)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction Result")
                if prediction_value == 1:
                    st.error("❗ High Risk of Heart Disease")
                else:
                    st.success("✅ Low Risk of Heart Disease")
                    
            with col2:
                st.subheader("Confidence Score")
                st.write(f"Probability of heart disease: {probability:.2%}")
                
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.barh(['Risk'], [probability], color='red', alpha=0.3)
                ax.barh(['Risk'], [1-probability], left=[probability], color='green', alpha=0.3)
                ax.set_xlim(0, 1)
                plt.tight_layout()
                st.pyplot(fig)
                
    else:
        st.header("Model Insights and Analytics")
        
        try:
            df = pd.read_csv('heart.csv')
            
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image('correlation_heatmap.png', 
                        caption="Feature Correlation Heatmap",
                        use_container_width=True)
                
            with col2:
                st.image('age_distribution.png',
                        caption="Age Distribution by Heart Disease Status",
                        use_container_width=True)
            
            if isinstance(model, RandomForestClassifier):
                st.image('feature_importance.png',
                        caption="Feature Importance Plot",
                        use_container_width=True)
                
            st.subheader("Dataset Statistics")
            st.write("Basic statistics of numeric features:")
            st.write(df.describe())
            
        except Exception as e:
            st.error(f"Error loading model insights: {str(e)}")

if __name__ == '__main__':
    main()