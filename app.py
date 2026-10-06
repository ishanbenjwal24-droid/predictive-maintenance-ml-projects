import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib
import os
from data_preprocessing import preprocess_stage1, preprocess_stage2
from evaluation import evaluate_model

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://www.streamlit.io",
        "Report a bug": "https://github.com",
        "About": "Advanced Predictive Maintenance System v1.0"
    }
)

# Professional CSS Styling
st.markdown("""
<style>
    :root {
        --primary-color: #1e3c72;
        --secondary-color: #2a5298;
        --accent-color: #00d4ff;
        --success-color: #00d084;
        --warning-color: #ff6b35;
        --danger-color: #ff1744;
        --light-bg: #0f1419;
        --card-bg: #1a1f2e;
        --border-color: #2d3748;
    }
    
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    body {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        color: #e8eef5;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 1rem;
        letter-spacing: 1px;
        animation: fadeInDown 0.8s ease;
    }
    
    .main-subtitle {
        text-align: center;
        color: #b0bcc4;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(42, 82, 152, 0.2) 0%, rgba(0, 212, 255, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        border: 1px solid rgba(0, 212, 255, 0.6);
        box-shadow: 0 12px 48px rgba(0, 212, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .metric-label {
        color: #b0bcc4;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .metric-value {
        color: #00d4ff;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.5rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    
    .status-active {
        background: rgba(0, 208, 132, 0.2);
        color: #00d084;
        border: 1px solid #00d084;
    }
    
    .status-warning {
        background: rgba(255, 107, 53, 0.2);
        color: #ff6b35;
        border: 1px solid #ff6b35;
    }
    
    .status-danger {
        background: rgba(255, 23, 68, 0.2);
        color: #ff1744;
        border: 1px solid #ff1744;
    }
    
    .section-header {
        color: #00d4ff;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3);
        letter-spacing: 0.5px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background: rgba(26, 31, 46, 0.5);
        padding: 10px;
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(42, 82, 152, 0.2) 0%, rgba(0, 212, 255, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        color: #b0bcc4;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.3) 0%, rgba(0, 212, 255, 0.1) 100%) !important;
        color: #00d4ff !important;
        border: 1px solid rgba(0, 212, 255, 0.6) !important;
    }
    
    .dataframe {
        background: rgba(26, 31, 46, 0.5) !important;
        color: #e8eef5 !important;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(0, 212, 255, 0.1) !important;
    }
    
    .prediction-success {
        background: linear-gradient(135deg, rgba(0, 208, 132, 0.2) 0%, rgba(0, 208, 132, 0.1) 100%);
        border: 1px solid rgba(0, 208, 132, 0.5);
        padding: 1.5rem;
        border-radius: 12px;
        color: #00d084;
    }
    
    .prediction-danger {
        background: linear-gradient(135deg, rgba(255, 23, 68, 0.2) 0%, rgba(255, 23, 68, 0.1) 100%);
        border: 1px solid rgba(255, 23, 68, 0.5);
        padding: 1.5rem;
        border-radius: 12px;
        color: #ff1744;
    }
    
    .feature-input {
        background: rgba(26, 31, 46, 0.5);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .feature-input:focus {
        border: 1px solid rgba(0, 212, 255, 0.6);
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2a5298 0%, #00d4ff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.5);
    }
    
    .progress-container {
        background: rgba(26, 31, 46, 0.5);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .sidebar-title {
        color: #00d4ff;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .info-box {
        background: rgba(42, 82, 152, 0.2);
        border: 1px solid rgba(42, 82, 152, 0.5);
        border-left: 4px solid rgba(0, 212, 255, 0.8);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #b0bcc4;
    }
    
    .success-box {
        background: rgba(0, 208, 132, 0.1);
        border: 1px solid rgba(0, 208, 132, 0.4);
        border-left: 4px solid #00d084;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #00d084;
    }
    
    .warning-box {
        background: rgba(255, 107, 53, 0.1);
        border: 1px solid rgba(255, 107, 53, 0.4);
        border-left: 4px solid #ff6b35;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #ff6b35;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class=\"main-header\">⚙️ Predictive Maintenance System</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="main-subtitle">
    Advanced Machine Learning Platform for Equipment Failure Prediction
    <br/>
    <span style="font-size: 0.9rem; color: #00d4ff;">Real-time monitoring • Predictive analytics • Maintenance optimization</span>
</div>
""", unsafe_allow_html=True)

# Add a divider
st.markdown("---")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("predictive_maintenance.csv")
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure predictive_maintenance.csv exists")
        return None

@st.cache_resource
def load_models():
    models = {}
    if os.path.exists("stage1_voting_classifier.pkl"):
        models["stage1"] = joblib.load("stage1_voting_classifier.pkl")
    if os.path.exists("stage2_voting_classifier.pkl"):
        models["stage2"] = joblib.load("stage2_voting_classifier.pkl")
    return models

# Define the preprocessor globally or load it with the models
@st.cache_resource
def load_preprocessor(df):
    categorical_features = ["Type"]
    numerical_features = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )
    # Fit the preprocessor on the full training data to ensure consistent feature space
    X_full = df.drop(columns=["Target", "Failure Type"])
    preprocessor.fit(X_full)
    return preprocessor

def create_data_overview(df):
    st.markdown("<h2 class=\"section-header\">📊 Data Overview & Statistics</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">📈 Total Samples</div>
            <div class="metric-value">{:,}</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        failure_count = df["Target"].sum()
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">⚠️ Failure Cases</div>
            <div class="metric-value">{:,}</div>
        </div>
        """.format(int(failure_count)), unsafe_allow_html=True)
    
    with col3:
        failure_rate = (failure_count / len(df)) * 100
        status_class = "status-danger" if failure_rate > 5 else "status-warning" if failure_rate > 2 else "status-active"
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">📊 Failure Rate</div>
            <div class="metric-value">{:.2f}%</div>
            <div class="status-badge {}">{}</div>
        </div>
        """.format(failure_rate, status_class, "High" if failure_rate > 5 else "Medium" if failure_rate > 2 else "Low"), unsafe_allow_html=True)
    
    with col4:
        unique_types = df["Type"].nunique()
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">🏭 Product Types</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(unique_types), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data Sample Section
    st.markdown("<h3 style='color: #00d4ff; margin-top: 2rem; margin-bottom: 1rem;'>📋 Data Sample</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, hide_index=False)
    
    # Statistics Section
    st.markdown("<h3 style='color: #00d4ff; margin-top: 2rem; margin-bottom: 1rem;'>📈 Descriptive Statistics</h3>", unsafe_allow_html=True)
    st.dataframe(df.describe().T, use_container_width=True)

def create_visualizations(df):
    st.markdown("<h2 class=\"section-header\">📈 Advanced Data Visualizations</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Failure Distribution", "Feature Analysis", "Correlations", "Failure Types"])
    
    with tab1:
        st.markdown("<h3 style='color: #00d4ff; margin-bottom: 1rem;'>Failure & Success Distribution</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(df, names="Target", 
                           title="Distribution of Failure vs Success",
                           labels={0: "No Failure", 1: "Failure"},
                           color_discrete_sequence=["#00d084", "#ff1744"],
                           hole=0.4)
            fig_pie.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12, color="#b0bcc4")
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            failure_by_type = df.groupby(["Type", "Target"]).size().unstack(fill_value=0)
            fig_bar = px.bar(failure_by_type, 
                           title="Failures by Product Type",
                           color_discrete_sequence=["#00d084", "#ff1744"],
                           barmode="group")
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12, color="#b0bcc4")
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.markdown("<h3 style='color: #00d4ff; margin-bottom: 1rem;'>Feature-wise Analysis</h3>", unsafe_allow_html=True)
        numerical_cols = ["Air temperature [K]", "Process temperature [K]", 
                         "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
        
        selected_feature = st.selectbox("Select Feature for Deep Analysis", numerical_cols, key="feature_select")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(df, x=selected_feature, color="Target",
                                  title=f"Distribution: {selected_feature}",
                                  color_discrete_map={0: "#00d084", 1: "#ff1744"},
                                  nbins=30)
            fig_hist.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12, color="#b0bcc4")
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_box = px.box(df, x="Target", y=selected_feature,
                           title=f"Comparison: {selected_feature}",
                           color="Target",
                           color_discrete_map={0: "#00d084", 1: "#ff1744"})
            fig_box.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12, color="#b0bcc4")
            )
            st.plotly_chart(fig_box, use_container_width=True)
    
    with tab3:
        st.markdown("<h3 style='color: #00d4ff; margin-bottom: 1rem;'>Feature Correlation Matrix</h3>", unsafe_allow_html=True)
        corr_matrix = df[numerical_cols].corr()
        fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title="Feature Interdependencies",
                              color_continuous_scale="RdBu_r")
        fig_heatmap.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, color="#b0bcc4")
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab4:
        st.markdown("<h3 style='color: #00d4ff; margin-bottom: 1rem;'>Failure Type Breakdown</h3>", unsafe_allow_html=True)
        failure_types = df[df["Target"] == 1]["Failure Type"].value_counts()
        fig_failure_types = px.bar(x=failure_types.index, y=failure_types.values,
                                 title="Distribution of Failure Types",
                                 color=failure_types.values,
                                 color_continuous_scale="Reds")
        fig_failure_types.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, color="#b0bcc4"),
            xaxis_title="Failure Type",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_failure_types, use_container_width=True)

# Model training section removed — models are pre-trained and saved externally.

def prediction_section():
    st.markdown("<h2 class=\"section-header\">🔮 Real-time Predictive Analysis</h2>", unsafe_allow_html=True)
    
    models = load_models()
    df = load_data()
    preprocessor = load_preprocessor(df)

    # Compute automatic threshold to maximize recall (based on validation split)
    auto_threshold_pct = None
    try:
        if "stage1" in models:
            x_tr, x_te, y_tr, y_te, _ = preprocess_stage1(df)
            if x_te is not None:
                from sklearn.metrics import recall_score
                proba_te = models["stage1"].predict_proba(x_te)
                classes_stage1 = getattr(models["stage1"], 'classes_', None)
                try:
                    idx_fail = list(classes_stage1).index(1)
                except Exception:
                    idx_fail = 1 if proba_te.shape[1] > 1 else 0

                failure_probs_te = proba_te[:, idx_fail]

                # find smallest threshold that gives recall >= target_recall
                target_recall = 0.90
                thresh_found = None
                for t in [i / 1000 for i in range(0, 1001)]:
                    preds = (failure_probs_te >= t).astype(int)
                    r = recall_score(y_te, preds)
                    if r >= target_recall:
                        thresh_found = t
                        break

                if thresh_found is None:
                    # fallback: choose threshold that maximizes recall (will be minimal t)
                    thresh_found = float(min(failure_probs_te))

                auto_threshold_pct = int(max(1, round(thresh_found * 100)))
    except Exception:
        auto_threshold_pct = None

    if not models:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ No trained models found</strong><br/>
            Please navigate to the Model Training section and train the models first.
        </div>
        """, unsafe_allow_html=True)
        return

    # Show loaded model info for debugging
    loaded = ", ".join(models.keys()) if models else "None"
    st.markdown(f"<div style='color: #b0bcc4; font-size:0.9rem; margin-bottom:0.5rem;'>Loaded models: {loaded}</div>", unsafe_allow_html=True)
    for name, mdl in models.items():
        cls_info = getattr(mdl, 'classes_', None)
        human_classes = None
        if name == "stage1":
            # Map numeric stage1 classes to human labels
            if cls_info is not None:
                mapping = {0: "No Failure", 1: "Failure"}
                try:
                    human_classes = [mapping[int(c)] for c in cls_info]
                except Exception:
                    human_classes = list(cls_info)
            else:
                human_classes = 'N/A'
        elif name == "stage2":
            # Derive human-readable failure-type classes from dataset if possible
            try:
                df_failure_only = df[df["Target"] == 1]
                if len(df_failure_only) > 0:
                    le_temp = LabelEncoder()
                    le_temp.fit(df_failure_only["Failure Type"]) 
                    human_classes = list(le_temp.classes_)
                else:
                    human_classes = list(cls_info) if cls_info is not None else 'N/A'
            except Exception:
                human_classes = list(cls_info) if cls_info is not None else 'N/A'
        else:
            human_classes = list(cls_info) if cls_info is not None else 'N/A'

        st.markdown(f"<div style='color: #b0bcc4; font-size:0.85rem; margin-bottom:0.25rem;'>{name} type: {type(mdl).__name__} | classes: {human_classes}</div>", unsafe_allow_html=True)

    # Allow user to adjust failure decision threshold (percentage)
    default_thresh = auto_threshold_pct if auto_threshold_pct is not None else 50
    failure_threshold = st.slider("Failure decision threshold (%)", min_value=1, max_value=99, value=default_thresh, help="Mark as failure when stage1 failure probability >= threshold")
    if auto_threshold_pct is not None:
        st.markdown(f"<div style='color: #b0bcc4; font-size:0.9rem; margin-top:0.25rem;'>Auto threshold for recall ≥ 90% on validation: {auto_threshold_pct}%</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>📌 Instructions:</strong> Enter the machine parameters below to get a real-time failure prediction.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #00d4ff; margin-top: 1.5rem; margin-bottom: 1rem;'>⚙️ Machine Parameters</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div style='color: #b0bcc4; font-weight: 600; margin-bottom: 0.5rem;'>Temperature Parameters</div>", unsafe_allow_html=True)
        air_temp = st.slider("Air Temperature (K)", min_value=290.0, max_value=310.0, value=298.0, step=0.1, 
                            help="Ambient temperature around the machine")
        process_temp = st.slider("Process Temperature (K)", min_value=300.0, max_value=320.0, value=308.0, step=0.1,
                               help="Internal operating temperature")
    
    with col2:
        st.markdown("<div style='color: #b0bcc4; font-weight: 600; margin-bottom: 0.5rem;'>Mechanical Parameters</div>", unsafe_allow_html=True)
        rotational_speed = st.slider("Rotational Speed (rpm)", min_value=1000, max_value=3000, value=1500, step=50,
                                    help="Machine rotation speed in revolutions per minute")
        torque = st.slider("Torque (Nm)", min_value=0.0, max_value=100.0, value=40.0, step=1.0,
                          help="Applied mechanical torque")
    
    with col3:
        st.markdown("<div style='color: #b0bcc4; font-weight: 600; margin-bottom: 0.5rem;'>Maintenance Parameters</div>", unsafe_allow_html=True)
        tool_wear = st.slider("Tool Wear (min)", min_value=0, max_value=300, value=50, step=5,
                             help="Cumulative tool wear time in minutes")
        product_type = st.selectbox("Product Type", ["L", "M", "H"],
                                   help="Machine product classification (Low/Medium/High)")
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("<div style='color: #b0bcc4;'>Click the prediction button to analyze the equipment</div>", unsafe_allow_html=True)
    
    with col2:
        predict_button = st.button("🚀 Analyze Equipment", type="primary", use_container_width=True)
    
    if predict_button:
        input_data = pd.DataFrame({
            "UDI": [1],
            "Product ID": ["TEST001"],
            "Type": [product_type],
            "Air temperature [K]": [air_temp],
            "Process temperature [K]": [process_temp],
            "Rotational speed [rpm]": [rotational_speed],
            "Torque [Nm]": [torque],
            "Tool wear [min]": [tool_wear],
            "Target": [0],
            "Failure Type": ["No Failure"]
        })
        
        try:
            x_processed = preprocessor.transform(input_data.drop(columns=["UDI", "Product ID", "Target", "Failure Type"]))
            
            if "stage1" in models:
                # Get model outputs and classes
                classes_stage1 = getattr(models["stage1"], 'classes_', None)
                proba = models["stage1"].predict_proba(x_processed)[0]
                # Determine which index corresponds to failure label (1)
                try:
                    idx_failure = list(classes_stage1).index(1)
                except Exception:
                    # fallback: assume class 1 is at index 1
                    idx_failure = 1 if len(proba) > 1 else 0

                failure_prob = proba[idx_failure]
                # Use the user-adjustable threshold to decide
                is_failure = (failure_prob * 100) >= failure_threshold

                # Debug info
                st.markdown(f"<div style='color: #b0bcc4; font-size:0.9rem; margin-top:0.5rem;'>Stage1 classes: {list(classes_stage1)} | raw proba: {proba} | failure_prob: {failure_prob:.4f}</div>", unsafe_allow_html=True)
                
                st.markdown("<h3 style='color: #00d4ff; margin-top: 2rem;'>📊 Analysis Results</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if is_failure:
                        st.markdown("""
                        <div class="prediction-danger">
                            <h3 style='color: #ff1744; margin-top: 0;'>⚠️ FAILURE DETECTED</h3>
                            <div style='font-size: 1.2rem; margin: 1rem 0;'>
                                <strong>Failure Probability: {:.1f}%</strong>
                            </div>
                            <div style='font-size: 0.9rem; opacity: 0.9;'>
                                Immediate maintenance recommended
                            </div>
                        </div>
                        """.format(failure_prob * 100), unsafe_allow_html=True)
                        
                        if "stage2" in models:
                            # Build a stage-2 preprocessor fitted on failure-only data to match training
                            try:
                                df_failure_only = df[df["Target"] == 1]
                                if len(df_failure_only) > 0:
                                    cat_feats = ["Type"]
                                    num_feats = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
                                    preproc2 = ColumnTransformer(
                                        transformers=[
                                            ("num", "passthrough", num_feats),
                                            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_feats)
                                        ],
                                        remainder='drop'
                                    )
                                    preproc2.fit(df_failure_only.drop(columns=["Target", "Failure Type"]))
                                    x2 = preproc2.transform(input_data.drop(columns=["UDI", "Product ID", "Target", "Failure Type"]))
                                else:
                                    # fallback to using the global preprocessor transform
                                    x2 = x_processed
                            except Exception:
                                x2 = x_processed

                            # Predict failure type using the stage2 model
                            failure_pred_idx = models["stage2"].predict(x2)[0]
                            failure_proba = models["stage2"].predict_proba(x2)[0]

                            # Map encoded indices back to human-readable failure type names
                            try:
                                df_failure_only = df[df["Target"] == 1]
                                le2 = LabelEncoder()
                                le2.fit(df_failure_only["Failure Type"]) if len(df_failure_only) > 0 else None
                                if len(df_failure_only) > 0:
                                    predicted_failure_type = le2.inverse_transform([int(failure_pred_idx)])[0]
                                    stage2_class_names = list(le2.classes_)
                                    # map probabilities to names (probabilities align with encoded order)
                                    proba_named = dict(zip(stage2_class_names, failure_proba))
                                else:
                                    # fallback to generic mapping
                                    predicted_failure_type = str(failure_pred_idx)
                                    proba_named = {str(i): p for i, p in enumerate(failure_proba)}
                            except Exception:
                                predicted_failure_type = str(failure_pred_idx)
                                proba_named = {str(i): p for i, p in enumerate(failure_proba)}

                            # Show named probabilities for stage2
                            st.markdown(f"<div style='color: #b0bcc4; font-size:0.9rem; margin-top:0.25rem;'>Stage2 probabilities: {proba_named}</div>", unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div style='background: rgba(255, 23, 68, 0.1); border-left: 4px solid #ff1744; 
                                        padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                                <div style='color: #ff1744; font-weight: 600; margin-bottom: 0.5rem;'>
                                    Predicted Failure Type
                                </div>
                                <div style='color: #e8eef5; font-size: 1.1rem;'>
                                    <strong>{predicted_failure_type}</strong>
                                </div>
                                <div style='color: #b0bcc4; font-size: 0.9rem; margin-top: 0.5rem;'>
                                    Confidence: {max(proba_named.values()) * 100:.1f}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="prediction-success">
                            <h3 style='color: #00d084; margin-top: 0;'>✅ SYSTEM HEALTHY</h3>
                            <div style='font-size: 1.2rem; margin: 1rem 0;'>
                                <strong>No Failure Probability: {:.1f}%</strong>
                            </div>
                            <div style='font-size: 0.9rem; opacity: 0.9;'>
                                Equipment operating normally - Continue monitoring
                            </div>
                        </div>
                        """.format((1 - failure_prob) * 100), unsafe_allow_html=True)
                
                # Build chart probabilities in order [No Failure, Failure]
                chart_probs = [1 - failure_prob, failure_prob]

                with col2:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=["✅ No Failure", "⚠️ Failure"],
                            y=chart_probs,
                            marker_color=["#00d084", "#ff1744"],
                            text=[f"{p*100:.1f}%" for p in chart_probs],
                            textposition='outside'
                        )
                    ])
                    fig.update_layout(
                        title="<b>Prediction Probabilities</b>",
                        yaxis_title="Probability",
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=12, color="#b0bcc4"),
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Parameter Summary
                st.markdown("<h3 style='color: #00d4ff; margin-top: 2rem;'>📋 Input Summary</h3>", unsafe_allow_html=True)
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Temperature Diff</div>
                        <div class='metric-value'>{process_temp - air_temp:.1f} K</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with summary_col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Mechanical Power</div>
                        <div class='metric-value'>{(torque * rotational_speed * 2 * 3.14159) / 60:.0f} W</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with summary_col3:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Tool Wear Status</div>
                        <div class='metric-value'>{tool_wear}m</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ Prediction Error:</strong><br/>
                {str(e)}
            </div>
            """, unsafe_allow_html=True)

def model_performance_section():
    st.markdown("<h2 class=\"section-header\">📊 Model Performance Metrics</h2>", unsafe_allow_html=True)
    
    models = load_models()
    df = load_data()
    
    if not models or df is None:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ No trained models or data found</strong><br/>
            Please train models first in the Model Training section.
        </div>
        """, unsafe_allow_html=True)
        return
    
    tab1, tab2 = st.tabs(["Stage 1: Failure Prediction", "Stage 2: Failure Classification"])
    
    with tab1:
        st.markdown("<h3 style='color: #00d4ff; margin-bottom: 1rem;'>Stage 1 - Equipment Failure Prediction</h3>", unsafe_allow_html=True)
        
        if "stage1" in models:
            x_train, x_test, y_train, y_test, le = preprocess_stage1(df)
            
            y_pred = models["stage1"].predict(x_test)
            y_proba = models["stage1"].predict_proba(x_test)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h4 style='color: #00d4ff; margin-bottom: 1rem;'>Confusion Matrix</h4>", unsafe_allow_html=True)
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                                 title="Confusion Matrix",
                                 labels=dict(x="Predicted", y="Actual", color="Count"),
                                 x=["No Failure", "Failure"],
                                 y=["No Failure", "Failure"],
                                 color_continuous_scale="Blues")
                fig_cm.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12, color="#b0bcc4")
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                st.markdown("<h4 style='color: #00d4ff; margin-bottom: 1rem;'>ROC Curve</h4>", unsafe_allow_html=True)
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, 
                    name=f"ROC Curve (AUC = {roc_auc:.3f})",
                    line=dict(color="#00d4ff", width=3)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name="Random Classifier",
                    line=dict(color="#b0bcc4", dash="dash")
                ))
                fig_roc.update_layout(
                    title="<b>ROC Curve - Stage 1</b>",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12, color="#b0bcc4"),
                    hovermode="closest"
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            
            # Classification Report
            st.markdown("<h4 style='color: #00d4ff; margin-top: 2rem; margin-bottom: 1rem;'>Classification Report</h4>", unsafe_allow_html=True)
            report = classification_report(y_test, y_pred, target_names=["No Failure", "Failure"], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            
            styled_df = report_df.style.format("{:.3f}").background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("Stage 1 model not found. Please train it first.")
    
    with tab2:
        st.markdown("<h3 style='color: #00d4ff; margin-bottom: 1rem;'>Stage 2 - Failure Type Classification</h3>", unsafe_allow_html=True)
        
        if "stage2" in models:
            df_failure_only = df[df["Target"] == 1]
            X2_train, X2_test, y2_train, y2_test, le2 = preprocess_stage2(df_failure_only)
            
            y2_pred = models["stage2"].predict(X2_test)
            
            st.markdown("<h4 style='color: #00d4ff; margin-bottom: 1rem;'>Confusion Matrix</h4>", unsafe_allow_html=True)
            cm2 = confusion_matrix(y2_test, y2_pred)
            
            fig_cm2 = px.imshow(cm2, text_auto=True, aspect="auto",
                              title="Confusion Matrix - Failure Type Classification",
                              labels=dict(x="Predicted", y="Actual", color="Count"),
                              color_continuous_scale="Reds")
            fig_cm2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12, color="#b0bcc4")
            )
            st.plotly_chart(fig_cm2, use_container_width=True)
            
            # Classification Report
            st.markdown("<h4 style='color: #00d4ff; margin-top: 2rem; margin-bottom: 1rem;'>Classification Report</h4>", unsafe_allow_html=True)
            report2 = classification_report(y2_test, y2_pred, output_dict=True)
            report_df2 = pd.DataFrame(report2).transpose()
            
            styled_df2 = report_df2.style.format("{:.3f}").background_gradient(cmap="Reds", subset=["precision", "recall", "f1-score"])
            st.dataframe(styled_df2, use_container_width=True)
        else:
            st.warning("Stage 2 model not found. Please train it first.")

def main():
    df = load_data()
    
    if df is None:
        st.markdown("""
        <div class="warning-box" style='margin-top: 2rem;'>
            <strong>❌ Data Loading Error</strong><br/>
            Unable to load the predictive maintenance dataset. Please ensure the file exists.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>⚙️</div>
            <div class="sidebar-title">Control Panel</div>
            <div style='color: #b0bcc4; font-size: 0.85rem;'>Predictive Maintenance System</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Custom navigation buttons (replaces default radio look)
        if 'page' not in st.session_state:
            st.session_state.page = "🏠 Overview"

        nav_options = ["🏠 Overview", "📊 Data Analysis", "🔮 Prediction", "📈 Performance"]
        for opt in nav_options:
            if st.button(opt, key=f"nav_{opt}"):
                st.session_state.page = opt
        page = st.session_state.page
        
        st.markdown("---")
        
        st.markdown("<div style='color: #00d4ff; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;'>📌 Quick Stats</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Total Data</div>
                <div class='metric-value' style='font-size: 1.5rem;'>{len(df):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            failure_rate = (df["Target"].sum() / len(df)) * 100
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Failure Rate</div>
                <div class='metric-value' style='font-size: 1.5rem;'>{failure_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style='color: #b0bcc4; font-size: 0.85rem; text-align: center; margin-top: 2rem;'>
            <div style='margin-bottom: 0.5rem;'>Version 1.0</div>
            <div style='opacity: 0.7;'>Powered by ML & Streamlit</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content
    if "🏠 Overview" in page:
        create_data_overview(df)
    
    elif "📊 Data Analysis" in page:
        create_visualizations(df)
    
    elif "🔮 Prediction" in page:
        prediction_section()
    
    elif "📈 Performance" in page:
        model_performance_section()

if __name__ == "__main__":
    main()

