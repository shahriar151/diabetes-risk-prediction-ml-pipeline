"""
Diabetes Risk Prediction - Streamlit Dashboard
Real-time prediction interface for diabetes risk assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .low-risk {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all trained models and preprocessing objects"""
    try:
        models = {
            'Random Forest': joblib.load('models/random_forest_model.pkl'),
            'Logistic Regression': joblib.load('models/logistic_regression_model.pkl'),
            'SVM': joblib.load('models/svm_model.pkl')
        }
        scaler = joblib.load('models/scaler.pkl')
        imputer = joblib.load('models/imputer.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        
        return models, scaler, imputer, feature_names
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run the training notebook first.")
        st.stop()


def predict_diabetes_risk(model, scaler, imputer, patient_data, feature_names):
    """Make prediction for a single patient"""
    # Convert to dataframe
    df = pd.DataFrame([patient_data], columns=feature_names)
    
    # Preprocess
    df_imputed = imputer.transform(df)
    df_scaled = scaler.transform(df_imputed)
    
    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]
    
    return prediction, probability


def create_gauge_chart(probability, threshold=0.5):
    """Create a gauge chart for risk probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Diabetes Risk Score", 'font': {'size': 24}},
        delta={'reference': threshold * 100, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#4caf50'},
                {'range': [30, 70], 'color': '#ff9800'},
                {'range': [70, 100], 'color': '#f44336'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'family': "Arial"}
    )
    
    return fig


def create_feature_importance_chart(patient_data, feature_names):
    """Create a bar chart showing patient's feature values"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(patient_data.values()),
            y=feature_names,
            orientation='h',
            marker=dict(
                color=list(patient_data.values()),
                colorscale='Viridis',
                showscale=True
            )
        )
    ])
    
    fig.update_layout(
        title="Patient Feature Values",
        xaxis_title="Value",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Health Risk Assessment</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading AI models..."):
        models, scaler, imputer, feature_names = load_models()
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    selected_model_name = st.sidebar.selectbox(
        "Select Prediction Model",
        options=list(models.keys()),
        help="Choose which machine learning model to use for prediction"
    )
    selected_model = models[selected_model_name]
    
    # Input method
    input_method = st.sidebar.radio(
        "Input Method",
        options=["Manual Input", "Upload CSV"],
        help="Choose how to provide patient data"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Information")
    st.sidebar.info(f"""
    **Selected Model:** {selected_model_name}
    
    **Features:** {len(feature_names)}
    
    **Model Type:** Supervised Classification
    
    **Dataset:** Synthetic diabetes risk data (educational purposes)
    """)
    
    # Main content
    if input_method == "Manual Input":
        st.subheader("📝 Enter Patient Information")
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=45,
                help="Patient's age in years"
            )
            
            bmi = st.number_input(
                "BMI (Body Mass Index)",
                min_value=10.0,
                max_value=60.0,
                value=25.0,
                step=0.1,
                help="Weight(kg) / Height(m)²"
            )
            
            glucose = st.number_input(
                "Glucose Level (mg/dL)",
                min_value=50,
                max_value=300,
                value=120,
                help="Blood glucose concentration"
            )
            
            blood_pressure = st.number_input(
                "Blood Pressure (mmHg)",
                min_value=40,
                max_value=200,
                value=80,
                help="Diastolic blood pressure"
            )
        
        with col2:
            insulin = st.number_input(
                "Insulin (µU/ml)",
                min_value=0,
                max_value=1000,
                value=100,
                help="Serum insulin level"
            )
            
            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0,
                max_value=100,
                value=20,
                help="Triceps skin fold thickness"
            )
            
            pregnancies = st.number_input(
                "Number of Pregnancies",
                min_value=0,
                max_value=20,
                value=1,
                help="Number of times pregnant"
            )
            
            diabetes_pedigree = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.01,
                help="Genetic predisposition score"
            )
        
        # Create patient data dictionary
        patient_data = {
            'Age': age,
            'BMI': bmi,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'Insulin': insulin,
            'SkinThickness': skin_thickness,
            'Pregnancies': pregnancies,
            'DiabetesPedigreeFunction': diabetes_pedigree
        }
        
        # Prediction button
        st.markdown("---")
        if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing patient data..."):
                prediction, probabilities = predict_diabetes_risk(
                    selected_model, scaler, imputer, patient_data, feature_names
                )
            
            # Display results
            st.markdown("## 📊 Prediction Results")
            
            # Create three columns for results
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Risk assessment
                risk_class = "High Risk" if prediction == 1 else "Low Risk"
                risk_style = "high-risk" if prediction == 1 else "low-risk"
                
                st.markdown(f"""
                <div class="prediction-box {risk_style}">
                    <h2>Prediction: {risk_class}</h2>
                    <h3>Confidence: {max(probabilities) * 100:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge chart
                gauge_fig = create_gauge_chart(probabilities[1])
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Probability breakdown
            st.markdown("### 📈 Probability Breakdown")
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                st.metric(
                    label="No Diabetes",
                    value=f"{probabilities[0] * 100:.1f}%",
                    delta=None
                )
            
            with prob_col2:
                st.metric(
                    label="Has Diabetes",
                    value=f"{probabilities[1] * 100:.1f}%",
                    delta=None
                )
            
            # Feature visualization
            st.markdown("### 🔬 Patient Feature Analysis")
            feature_chart = create_feature_importance_chart(patient_data, feature_names)
            st.plotly_chart(feature_chart, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            if prediction == 1:
                st.error("""
                **⚠️ High Risk Detected**
                
                Based on the analysis, this patient shows elevated risk for diabetes. Recommendations:
                - Consult with a healthcare professional immediately
                - Consider comprehensive diabetes screening
                - Monitor blood glucose levels regularly
                - Discuss lifestyle modifications (diet, exercise)
                - Review family history and genetic factors
                
                *Note: This is an AI prediction for educational purposes only. Not a medical diagnosis.*
                """)
            else:
                st.success("""
                **✅ Low Risk Detected**
                
                Based on the analysis, this patient shows lower risk for diabetes. Recommendations:
                - Maintain healthy lifestyle habits
                - Regular health check-ups
                - Monitor weight and BMI
                - Stay physically active
                - Balanced diet with limited sugar intake
                
                *Note: This is an AI prediction for educational purposes only. Continue regular health monitoring.*
                """)
    
    else:  # Upload CSV
        st.subheader("📤 Upload Patient Data (CSV)")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with patient data. Must contain all required features."
        )
        
        # Show sample format
        with st.expander("📋 View Required CSV Format"):
            sample_df = pd.DataFrame({
                'Age': [45, 52],
                'BMI': [25.6, 30.2],
                'Glucose': [120, 145],
                'BloodPressure': [80, 88],
                'Insulin': [100, 150],
                'SkinThickness': [20, 25],
                'Pregnancies': [1, 3],
                'DiabetesPedigreeFunction': [0.5, 0.8]
            })
            st.dataframe(sample_df, use_container_width=True)
        
        if uploaded_file is not None:
            # Read CSV
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Found {len(df)} patient records.")
                
                # Display data
                st.markdown("### 📊 Uploaded Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Predict button
                if st.button("🔍 Predict for All Patients", type="primary", use_container_width=True):
                    with st.spinner("Processing batch predictions..."):
                        # Preprocess
                        df_imputed = imputer.transform(df)
                        df_scaled = scaler.transform(df_imputed)
                        
                        # Predict
                        predictions = selected_model.predict(df_scaled)
                        probabilities = selected_model.predict_proba(df_scaled)[:, 1]
                        
                        # Add results to dataframe
                        results_df = df.copy()
                        results_df['Prediction'] = predictions
                        results_df['Risk_Probability'] = probabilities
                        results_df['Risk_Level'] = results_df['Prediction'].map({
                            0: 'Low Risk',
                            1: 'High Risk'
                        })
                    
                    # Display results
                    st.markdown("### 📊 Batch Prediction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Total Patients",
                            len(results_df)
                        )
                    
                    with col2:
                        high_risk_count = (predictions == 1).sum()
                        st.metric(
                            "High Risk Cases",
                            high_risk_count,
                            delta=f"{(high_risk_count/len(results_df)*100):.1f}%"
                        )
                    
                    with col3:
                        avg_risk = probabilities.mean()
                        st.metric(
                            "Average Risk Score",
                            f"{avg_risk * 100:.1f}%"
                        )
                    
                    # Risk distribution chart
                    st.markdown("### 📈 Risk Distribution")
                    fig = px.histogram(
                        results_df,
                        x='Risk_Probability',
                        nbins=20,
                        color='Risk_Level',
                        title="Distribution of Risk Probabilities",
                        labels={'Risk_Probability': 'Risk Probability', 'count': 'Number of Patients'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="diabetes_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure your CSV has the correct format and column names.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>⚠️ Disclaimer:</strong> This tool uses a machine learning model trained on synthetic data 
        for educational and demonstration purposes only. It should NOT be used for actual medical diagnosis 
        or treatment decisions. Always consult qualified healthcare professionals for medical advice.</p>
        
        <p style='margin-top: 1rem;'>
            <strong>Developer:</strong> Shahriar | <strong>Model:</strong> {selected_model_name} | 
            <strong>Framework:</strong> Streamlit + scikit-learn
        </p>
    </div>
    """.format(selected_model_name=selected_model_name), unsafe_allow_html=True)


if __name__ == "__main__":
    main()