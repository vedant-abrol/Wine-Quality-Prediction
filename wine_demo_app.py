#!/usr/bin/env python3
"""
Wine Quality Prediction - Interactive Demo Application
=======================================================

A beautiful wine-themed Streamlit GUI for demonstrating the Wine Quality
Prediction model. Works without AWS ECS using a mock/lightweight model.

Author: Vedant Abrol
Course: CS643 - Cloud Computing

Run with: streamlit run wine_demo_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="🍷 Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# WINE-THEMED CUSTOM CSS
# =============================================================================

def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Lato:wght@300;400;700&display=swap');
    
    /* Main background - elegant wine gradient */
    .stApp {
        background: linear-gradient(135deg, #1a0a0a 0%, #2d1515 50%, #1a0a0a 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #d4a574 !important;
    }
    
    /* Body text */
    p, span, label, .stMarkdown {
        font-family: 'Lato', sans-serif !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d1515 0%, #4a1c1c 50%, #2d1515 100%) !important;
        border-right: 2px solid #8b4557;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #f5e6d3 !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #e8d5c4 !important;
        font-weight: 500;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #722f37 !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        background: linear-gradient(90deg, #4a1c1c, #8b4557, #d4a574) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #722f37 0%, #8b4557 50%, #a05568 100%) !important;
        color: #f5e6d3 !important;
        border: 2px solid #d4a574 !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-family: 'Playfair Display', serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(114, 47, 55, 0.4) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #8b4557 0%, #a05568 50%, #c77d8e 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(139, 69, 87, 0.5) !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #d4a574 !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 2.5rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #e8d5c4 !important;
        font-family: 'Lato', sans-serif !important;
    }
    
    /* Card-like containers */
    .wine-card {
        background: linear-gradient(135deg, rgba(45, 21, 21, 0.9) 0%, rgba(74, 28, 28, 0.8) 100%);
        border: 1px solid #8b4557;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Quality badge styles */
    .quality-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .quality-low {
        background: linear-gradient(135deg, #8b0000, #a52a2a);
        color: #ffeedd;
        border: 2px solid #cd5c5c;
    }
    
    .quality-medium {
        background: linear-gradient(135deg, #722f37, #8b4557);
        color: #f5e6d3;
        border: 2px solid #c77d8e;
    }
    
    .quality-high {
        background: linear-gradient(135deg, #d4a574, #c4915e);
        color: #2d1515;
        border: 2px solid #e8d5c4;
    }
    
    .quality-excellent {
        background: linear-gradient(135deg, #ffd700, #daa520);
        color: #2d1515;
        border: 2px solid #fff8dc;
    }
    
    /* Wine glass animation */
    .wine-glass {
        font-size: 4rem;
        animation: swirl 3s ease-in-out infinite;
    }
    
    @keyframes swirl {
        0%, 100% { transform: rotate(-5deg); }
        50% { transform: rotate(5deg); }
    }
    
    /* Divider */
    .wine-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #8b4557, #d4a574, #8b4557, transparent);
        margin: 2rem 0;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: rgba(45, 21, 21, 0.8) !important;
        border: 1px solid #8b4557 !important;
        border-radius: 10px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(74, 28, 28, 0.6) !important;
        border-radius: 10px !important;
        color: #d4a574 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(45, 21, 21, 0.6);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #e8d5c4 !important;
        background-color: transparent !important;
        border-radius: 8px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #722f37 !important;
        color: #d4a574 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a0a0a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #722f37;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #8b4557;
    }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

@st.cache_resource
def load_or_create_model():
    """Load training data and create a lightweight sklearn model for demo."""
    
    # Try to load the training data
    data_path = Path("TrainingDataset.csv")
    
    # Define column names explicitly (the CSV header is malformed with extra quotes)
    column_names = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol", "quality"
    ]
    
    feature_cols = column_names[:-1]  # All except 'quality'
    
    if data_path.exists():
        # Load data, skip the malformed header, use our clean column names
        df = pd.read_csv(data_path, sep=';', skiprows=1, names=column_names)
        
        # Verify data loaded correctly
        if df.empty or len(df.columns) != 12:
            return None, None, None, None, None
        
        X = df[feature_cols]
        y = df["quality"]
        
        # Train a Random Forest model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        accuracy = model.score(X_test_scaled, y_test)
        
        return model, scaler, df, accuracy, feature_cols
    else:
        # Return mock model if no data available
        return None, None, None, None, None


def predict_quality(model, scaler, features):
    """Make prediction using the trained model."""
    if model is None:
        # Mock prediction based on heuristics
        return mock_predict(features)
    
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    return int(prediction), probabilities


def mock_predict(features):
    """Mock prediction when no model is available."""
    # Simple heuristic based on key features
    alcohol = features[10]  # alcohol content
    volatile_acidity = features[1]
    sulphates = features[9]
    
    # Higher alcohol, lower volatile acidity, and moderate sulphates = better quality
    base_score = 5
    
    if alcohol > 11:
        base_score += 1
    if alcohol > 12:
        base_score += 1
    if volatile_acidity < 0.4:
        base_score += 1
    if 0.5 < sulphates < 1.0:
        base_score += 0.5
    if volatile_acidity > 0.7:
        base_score -= 1
    
    prediction = max(3, min(9, int(round(base_score))))
    
    # Create mock probability distribution
    probs = np.zeros(7)  # Qualities 3-9
    probs[prediction - 3] = 0.7
    if prediction > 3:
        probs[prediction - 4] = 0.15
    if prediction < 9:
        probs[prediction - 2] = 0.15
    
    return prediction, probs


def get_quality_description(quality):
    """Get wine quality description and rating."""
    descriptions = {
        3: ("Poor", "🍷", "This wine has significant quality issues.", "quality-low"),
        4: ("Below Average", "🍷🍷", "This wine needs improvement in several areas.", "quality-low"),
        5: ("Average", "🍷🍷🍷", "A decent everyday wine with room for improvement.", "quality-medium"),
        6: ("Good", "🍷🍷🍷🍷", "A well-balanced wine suitable for most occasions.", "quality-medium"),
        7: ("Very Good", "🍷🍷🍷🍷🍷", "An excellent wine with refined characteristics.", "quality-high"),
        8: ("Excellent", "⭐🍷🍷🍷🍷🍷", "A premium quality wine with exceptional balance.", "quality-excellent"),
        9: ("Outstanding", "⭐⭐🍷🍷🍷🍷🍷", "A world-class wine of extraordinary quality!", "quality-excellent"),
    }
    return descriptions.get(quality, ("Unknown", "🍷", "Unable to classify.", "quality-medium"))


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header():
    """Render the main header with wine theme."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div class="wine-glass">🍷</div>
            <h1 style="font-size: 3rem; margin-bottom: 0; background: linear-gradient(135deg, #d4a574, #f5e6d3, #d4a574); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                       text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                Wine Quality Predictor
            </h1>
            <p style="color: #e8d5c4; font-size: 1.2rem; font-style: italic; margin-top: 0.5rem;">
                AI-Powered Wine Analysis • CS643 Cloud Computing Project
            </p>
        </div>
        <div class="wine-divider"></div>
        """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with wine parameter inputs."""
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #d4a574; font-size: 1.8rem;">🍇 Wine Parameters</h2>
            <p style="color: #c4a484; font-size: 0.9rem;">Adjust the chemical properties</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Acidity Group
        st.markdown("#### 🧪 Acidity Profile")
        fixed_acidity = st.slider(
            "Fixed Acidity (g/L)",
            min_value=4.0, max_value=16.0, value=7.4, step=0.1,
            help="Tartaric acid concentration - primary fixed acid in wine"
        )
        
        volatile_acidity = st.slider(
            "Volatile Acidity (g/L)",
            min_value=0.1, max_value=1.6, value=0.5, step=0.01,
            help="Acetic acid concentration - too high leads to vinegar taste"
        )
        
        citric_acid = st.slider(
            "Citric Acid (g/L)",
            min_value=0.0, max_value=1.0, value=0.25, step=0.01,
            help="Adds freshness and flavor - found in small quantities"
        )
        
        st.markdown("---")
        
        # Sugar & Density
        st.markdown("#### 🍬 Sugar & Density")
        residual_sugar = st.slider(
            "Residual Sugar (g/L)",
            min_value=0.5, max_value=16.0, value=2.5, step=0.1,
            help="Sugar remaining after fermentation"
        )
        
        density = st.slider(
            "Density (g/cm³)",
            min_value=0.990, max_value=1.004, value=0.996, step=0.0001,
            format="%.4f",
            help="Density of wine - depends on alcohol and sugar content"
        )
        
        st.markdown("---")
        
        # Sulfur & Chlorides
        st.markdown("#### ⚗️ Additives")
        chlorides = st.slider(
            "Chlorides (g/L)",
            min_value=0.01, max_value=0.6, value=0.08, step=0.01,
            help="Salt content in wine"
        )
        
        free_sulfur_dioxide = st.slider(
            "Free SO₂ (mg/L)",
            min_value=1.0, max_value=70.0, value=15.0, step=1.0,
            help="Prevents microbial growth and oxidation"
        )
        
        total_sulfur_dioxide = st.slider(
            "Total SO₂ (mg/L)",
            min_value=5.0, max_value=300.0, value=45.0, step=1.0,
            help="Total amount of SO₂ (free + bound)"
        )
        
        sulphates = st.slider(
            "Sulphates (g/L)",
            min_value=0.3, max_value=2.0, value=0.6, step=0.01,
            help="Wine additive contributing to SO₂ levels"
        )
        
        st.markdown("---")
        
        # pH & Alcohol
        st.markdown("#### 🔬 pH & Alcohol")
        pH = st.slider(
            "pH Level",
            min_value=2.7, max_value=4.0, value=3.3, step=0.01,
            help="Acidity level (lower = more acidic)"
        )
        
        alcohol = st.slider(
            "Alcohol (%)",
            min_value=8.0, max_value=15.0, value=10.5, step=0.1,
            help="Alcohol content by volume"
        )
        
        st.markdown("---")
        
        # Preset buttons
        st.markdown("#### 🎯 Quick Presets")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🍷 Everyday", use_container_width=True):
                st.session_state.preset = "everyday"
        with col2:
            if st.button("⭐ Premium", use_container_width=True):
                st.session_state.preset = "premium"
        
        return [
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
            pH, sulphates, alcohol
        ]


def render_prediction_result(quality, probabilities, features):
    """Render the prediction result with beautiful styling."""
    
    label, rating, description, css_class = get_quality_description(quality)
    
    # Main prediction card
    st.markdown(f"""
    <div class="wine-card" style="text-align: center;">
        <h2 style="color: #d4a574; margin-bottom: 1rem;">🎯 Prediction Result</h2>
        <div class="quality-badge {css_class}">
            Quality Score: {quality}/9
        </div>
        <h3 style="color: #f5e6d3; margin: 1rem 0;">{rating}</h3>
        <p style="color: #d4a574; font-size: 1.5rem; font-weight: 600;">{label}</p>
        <p style="color: #c4a484; font-size: 1.1rem; max-width: 500px; margin: 1rem auto;">
            {description}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability distribution chart
    if probabilities is not None and len(probabilities) > 0:
        st.markdown("### 📊 Quality Probability Distribution")
        
        # Create labels for available quality scores (3-9)
        quality_labels = list(range(3, 3 + len(probabilities)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f"Quality {q}" for q in quality_labels],
            y=probabilities * 100,
            marker=dict(
                color=['#722f37' if q != quality else '#d4a574' for q in quality_labels],
                line=dict(color='#8b4557', width=2)
            ),
            text=[f"{p*100:.1f}%" for p in probabilities],
            textposition='outside',
            textfont=dict(color='#e8d5c4', size=12)
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(45,21,21,0.3)',
            font=dict(color='#e8d5c4', family='Lato'),
            xaxis=dict(
                title="Quality Rating",
                gridcolor='rgba(139,69,87,0.3)',
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title="Probability (%)",
                gridcolor='rgba(139,69,87,0.3)',
                range=[0, 100]
            ),
            height=350,
            margin=dict(t=30, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_wine_analysis(features, feature_cols):
    """Render detailed wine analysis."""
    
    st.markdown("### 🔍 Wine Composition Analysis")
    
    # Create radar chart of wine properties
    # Normalize features for radar chart (0-1 scale based on typical ranges)
    ranges = {
        "fixed acidity": (4, 16),
        "volatile acidity": (0.1, 1.6),
        "citric acid": (0, 1),
        "residual sugar": (0.5, 16),
        "chlorides": (0.01, 0.6),
        "free sulfur dioxide": (1, 70),
        "total sulfur dioxide": (5, 300),
        "density": (0.99, 1.004),
        "pH": (2.7, 4),
        "sulphates": (0.3, 2),
        "alcohol": (8, 15)
    }
    
    normalized = []
    for i, col in enumerate(feature_cols):
        min_val, max_val = ranges[col]
        norm_val = (features[i] - min_val) / (max_val - min_val)
        normalized.append(max(0, min(1, norm_val)))
    
    # Radar chart
    categories = ['Fixed Acid', 'Volatile Acid', 'Citric Acid', 'Sugar', 
                  'Chlorides', 'Free SO₂', 'Total SO₂', 'Density', 'pH', 
                  'Sulphates', 'Alcohol']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized + [normalized[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(212, 165, 116, 0.3)',
        line=dict(color='#d4a574', width=2),
        marker=dict(color='#d4a574', size=8)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='rgba(139,69,87,0.4)',
                tickfont=dict(color='#c4a484')
            ),
            angularaxis=dict(
                gridcolor='rgba(139,69,87,0.4)',
                tickfont=dict(color='#e8d5c4', size=11)
            ),
            bgcolor='rgba(45,21,21,0.3)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8d5c4', family='Lato'),
        height=450,
        margin=dict(t=30, b=30, l=80, r=80)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="wine-card">
            <h4 style="color: #d4a574;">🍇 Key Quality Factors</h4>
            <ul style="color: #e8d5c4; line-height: 2;">
                <li><strong>Alcohol Content:</strong> Higher levels often indicate better quality</li>
                <li><strong>Volatile Acidity:</strong> Lower is better (avoids vinegar taste)</li>
                <li><strong>Sulphates:</strong> Moderate levels enhance quality</li>
                <li><strong>Citric Acid:</strong> Adds freshness to the wine</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="wine-card">
            <h4 style="color: #d4a574;">📋 Your Wine Profile</h4>
            <ul style="color: #e8d5c4; line-height: 2;">
                <li><strong>Alcohol:</strong> {features[10]:.1f}% {"✓ Good" if features[10] > 10 else "○ Moderate"}</li>
                <li><strong>Volatile Acidity:</strong> {features[1]:.2f} g/L {"✓ Low" if features[1] < 0.5 else "⚠ High"}</li>
                <li><strong>pH Level:</strong> {features[8]:.2f} {"✓ Balanced" if 3.0 < features[8] < 3.5 else "○ Check"}</li>
                <li><strong>Sulphates:</strong> {features[9]:.2f} g/L {"✓ Good" if 0.5 < features[9] < 1.0 else "○ Adjust"}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def render_dataset_explorer(df):
    """Render dataset exploration section."""
    
    if df is None:
        st.warning("📂 Dataset not found. Place TrainingDataset.csv in the project folder for full functionality.")
        return
    
    st.markdown("### 📈 Dataset Insights")
    
    tabs = st.tabs(["📊 Distribution", "🔗 Correlations", "📋 Sample Data"])
    
    with tabs[0]:
        # Quality distribution
        quality_counts = df['quality'].value_counts().sort_index()
        
        fig = px.bar(
            x=quality_counts.index,
            y=quality_counts.values,
            labels={'x': 'Quality Score', 'y': 'Number of Wines'},
            color=quality_counts.values,
            color_continuous_scale=[[0, '#722f37'], [0.5, '#8b4557'], [1, '#d4a574']]
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(45,21,21,0.3)',
            font=dict(color='#e8d5c4', family='Lato'),
            xaxis=dict(gridcolor='rgba(139,69,87,0.3)', dtick=1),
            yaxis=dict(gridcolor='rgba(139,69,87,0.3)'),
            coloraxis_showscale=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Wines", f"{len(df):,}")
        with col2:
            st.metric("Avg Quality", f"{df['quality'].mean():.2f}")
        with col3:
            st.metric("Min Quality", f"{df['quality'].min()}")
        with col4:
            st.metric("Max Quality", f"{df['quality'].max()}")
    
    with tabs[1]:
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale=[[0, '#1a0a0a'], [0.5, '#722f37'], [1, '#d4a574']],
            aspect='auto'
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8d5c4', family='Lato', size=10),
            height=500,
            margin=dict(l=100, r=20, t=30, b=100)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        # Sample data - with wine-colored highlighting
        try:
            styled_df = df.head(20).style.background_gradient(
                cmap='Reds', 
                subset=['quality']
            )
            st.dataframe(styled_df, use_container_width=True, height=400)
        except ImportError:
            # Fallback if matplotlib not installed
            st.dataframe(df.head(20), use_container_width=True, height=400)


def render_about_section():
    """Render about/info section."""
    
    st.markdown("""
    <div class="wine-card">
        <h3 style="color: #d4a574; text-align: center;">🎓 About This Project</h3>
        <div style="color: #e8d5c4; padding: 1rem;">
            <p style="font-size: 1.1rem; line-height: 1.8;">
                This <strong>Wine Quality Prediction</strong> application is part of the 
                <strong>CS643 Cloud Computing</strong> course project. It demonstrates the use of 
                machine learning for predicting wine quality based on physicochemical properties.
            </p>
            
            <h4 style="color: #d4a574; margin-top: 1.5rem;">🔬 Technical Stack</h4>
            <ul style="line-height: 2;">
                <li><strong>Original Pipeline:</strong> Apache Spark MLlib on AWS EMR/ECS</li>
                <li><strong>Demo Application:</strong> Streamlit + Scikit-learn</li>
                <li><strong>Algorithms:</strong> Logistic Regression, Random Forest</li>
                <li><strong>Dataset:</strong> UCI Wine Quality Dataset (Red Wine)</li>
            </ul>
            
            <h4 style="color: #d4a574; margin-top: 1.5rem;">📊 Model Features</h4>
            <p>The model uses 11 physicochemical properties to predict wine quality on a scale of 3-9:</p>
            <ul style="line-height: 1.8;">
                <li>Fixed & Volatile Acidity</li>
                <li>Citric Acid & Residual Sugar</li>
                <li>Chlorides & Sulfur Dioxide (Free/Total)</li>
                <li>Density, pH, Sulphates, Alcohol</li>
            </ul>
            
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; 
                        background: rgba(114, 47, 55, 0.3); border-radius: 10px;">
                <p style="color: #d4a574; font-size: 1.2rem; margin: 0;">
                    👨‍💻 Developed by <strong>Vedant Abrol</strong>
                </p>
                <p style="color: #c4a484; margin: 0.5rem 0 0 0;">
                    CS643 - Cloud Computing • NJIT
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    if 'preset' not in st.session_state:
        st.session_state.preset = None
    
    # Load model
    model, scaler, df, accuracy, feature_cols = load_or_create_model()
    
    # Render header
    render_header()
    
    # Sidebar inputs
    features = render_sidebar()
    
    # Handle presets
    if st.session_state.preset == "everyday":
        features = [7.4, 0.5, 0.25, 2.5, 0.08, 15.0, 45.0, 0.996, 3.3, 0.6, 10.5]
        st.session_state.preset = None
        st.rerun()
    elif st.session_state.preset == "premium":
        features = [8.5, 0.35, 0.45, 2.0, 0.06, 25.0, 60.0, 0.994, 3.2, 0.85, 12.5]
        st.session_state.preset = None
        st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Predict button
        st.markdown("<div style='text-align: center; padding: 1rem;'>", unsafe_allow_html=True)
        predict_clicked = st.button("🍷 Predict Wine Quality", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if predict_clicked or 'last_prediction' in st.session_state:
            if predict_clicked:
                quality, probabilities = predict_quality(model, scaler, features)
                st.session_state.last_prediction = (quality, probabilities, features.copy())
            else:
                quality, probabilities, _ = st.session_state.last_prediction
            
            render_prediction_result(quality, probabilities, features)
    
    with col2:
        # Model info card
        if model is not None and accuracy is not None:
            st.markdown(f"""
            <div class="wine-card">
                <h3 style="color: #d4a574; text-align: center;">🤖 Model Status</h3>
                <div style="text-align: center; padding: 1rem;">
                    <p style="color: #4ade80; font-size: 1.2rem;">✓ Model Loaded Successfully</p>
                    <p style="color: #e8d5c4;">
                        <strong>Algorithm:</strong> Random Forest<br>
                        <strong>Test Accuracy:</strong> {accuracy*100:.1f}%<br>
                        <strong>Training Samples:</strong> {len(df):,}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="wine-card">
                <h3 style="color: #d4a574; text-align: center;">🤖 Demo Mode</h3>
                <div style="text-align: center; padding: 1rem;">
                    <p style="color: #fbbf24; font-size: 1.2rem;">⚡ Using Heuristic Model</p>
                    <p style="color: #e8d5c4;">
                        This demo uses intelligent heuristics<br>
                        when the training data is unavailable.<br>
                        <em>Add TrainingDataset.csv for full ML model.</em>
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Wine analysis
        if 'last_prediction' in st.session_state:
            render_wine_analysis(features, feature_cols if feature_cols else [
                "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                "pH", "sulphates", "alcohol"
            ])
    
    st.markdown("<div class='wine-divider'></div>", unsafe_allow_html=True)
    
    # Tabs for additional content
    tab1, tab2 = st.tabs(["📊 Dataset Explorer", "ℹ️ About Project"])
    
    with tab1:
        render_dataset_explorer(df)
    
    with tab2:
        render_about_section()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #8b4557; margin-top: 2rem;">
        <p>🍷 Wine Quality Predictor • CS643 Cloud Computing Project</p>
        <p style="font-size: 0.9rem;">Made with ❤️ using Streamlit & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
