import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Page config
st.set_page_config(
    page_title="EA Eye Scout Predictor",
    page_icon="⚽",
    layout="wide"
)

# Custom styling
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #2E5A88;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">⚽ EA Eye Scout Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Data-driven recruitment insights for East African football talent</div>', unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("🔍 Player Assessment")
st.sidebar.markdown("Enter player metrics to predict potential and market value")

# Generate synthetic East African player dataset for training
@st.cache_data
def load_training_data():
    np.random.seed(42)
    n_players = 500
    
    data = {
        'age': np.random.randint(17, 28, n_players),
        'goals_per_90': np.random.uniform(0, 1.2, n_players),
        'assists_per_90': np.random.uniform(0, 0.8, n_players),
        'pass_accuracy': np.random.uniform(65, 92, n_players),
        'tackles_per_90': np.random.uniform(0.5, 4.5, n_players),
        'dribbles_per_90': np.random.uniform(0.5, 5.0, n_players),
        'aerial_duels_won': np.random.uniform(40, 85, n_players),
        'key_passes_per_90': np.random.uniform(0, 2.5, n_players),
        'minutes_played': np.random.randint(500, 2500, n_players),
    }
    
    df = pd.DataFrame(data)
    
    # Create synthetic target variables
    # Potential score (0-100) based on performance metrics
    df['potential_score'] = (
        df['goals_per_90'] * 18 +
        df['assists_per_90'] * 14 +
        df['dribbles_per_90'] * 6 +
        df['key_passes_per_90'] * 10 +
        df['pass_accuracy'] / 100 * 8 +
        df['tackles_per_90'] * 4 +
        df['aerial_duels_won'] / 100 * 5 -
        (df['age'] - 21) * 1.5
    ).clip(40, 98) + np.random.normal(0, 3, n_players)
    
    # Market value (EUR) based on potential, age, minutes
    df['market_value'] = (
        df['potential_score'] * 25000 +
        np.exp(-0.15 * (df['age'] - 22)) * 150000 +
        df['minutes_played'] * 50
    ).clip(10000, 2500000) + np.random.normal(0, 50000, n_players)
    
    df['market_value'] = df['market_value'].astype(int)
    df['potential_score'] = df['potential_score'].clip(40, 98).astype(int)
    
    # Add region labels
    regions = ['Kenya', 'Uganda', 'Tanzania', 'Ethiopia', 'Rwanda']
    df['region'] = np.random.choice(regions, n_players, p=[0.35, 0.25, 0.20, 0.12, 0.08])
    
    return df

# Load data and train model
df = load_training_data()

feature_cols = ['age', 'goals_per_90', 'assists_per_90', 'pass_accuracy', 
                'tackles_per_90', 'dribbles_per_90', 'aerial_duels_won', 
                'key_passes_per_90', 'minutes_played']

X = df[feature_cols]
y_potential = df['potential_score']
y_value = df['market_value']

# Train models
@st.cache_resource
def train_models():
    # Potential model
    rf_potential = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_potential.fit(X, y_potential)
    
    # Value model
    rf_value = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_value.fit(X, y_value)
    
    scaler = StandardScaler()
    scaler.fit(X)
    
    return rf_potential, rf_value, scaler

model_potential, model_value, scaler = train_models()

# Model accuracy metrics
y_pred_pot = model_potential.predict(X)
y_pred_val = model_value.predict(X)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.metric("Potential MAE", f"±{mean_absolute_error(y_potential, y_pred_pot):.1f} pts")
st.sidebar.metric("Value MAE", f"€{mean_absolute_error(y_value, y_pred_val):,.0f}")
st.sidebar.metric("R² Score", f"{r2_score(y_value, y_pred_val):.2f}")

st.sidebar.markdown("---")

# Input fields
st.sidebar.subheader("📝 Player Metrics")

age = st.sidebar.number_input("Age", min_value=16, max_value=35, value=21)
goals = st.sidebar.number_input("Goals per 90 min", min_value=0.0, max_value=3.0, value=0.45, step=0.05)
assists = st.sidebar.number_input("Assists per 90 min", min_value=0.0, max_value=2.0, value=0.30, step=0.05)
pass_acc = st.sidebar.slider("Pass Accuracy (%)", 50, 95, 78)
tackles = st.sidebar.number_input("Tackles per 90 min", min_value=0.0, max_value=8.0, value=2.1, step=0.1)
dribbles = st.sidebar.number_input("Dribbles per 90 min", min_value=0.0, max_value=8.0, value=2.5, step=0.1)
aerials = st.sidebar.slider("Aerial Duels Won (%)", 30, 90, 62)
key_passes = st.sidebar.number_input("Key Passes per 90", min_value=0.0, max_value=4.0, value=1.2, step=0.1)
minutes = st.sidebar.number_input("Minutes Played (season)", min_value=0, max_value=3000, value=1200, step=100)

# Prepare input for prediction
input_data = pd.DataFrame([[age, goals, assists, pass_acc, tackles, 
                            dribbles, aerials, key_passes, minutes]], 
                          columns=feature_cols)

# Make prediction
potential_pred = model_potential.predict(input_data)[0]
value_pred = model_value.predict(input_data)[0]

# Determine recommendation
if potential_pred >= 75 and value_pred < 500000:
    recommendation = "🔴 **UNDERVALUED GEM** - Immediate scout recommendation"
    rec_color = "#e74c3c"
    action = "Priority: Schedule trial within 2 weeks"
elif potential_pred >= 70:
    recommendation = "🟠 **STRONG PROSPECT** - Watchlist candidate"
    rec_color = "#e67e22"
    action = "Action: Track next 5 matches"
elif potential_pred >= 60:
    recommendation = "🟡 **DEVELOPMENTAL** - Monitor progression"
    rec_color = "#f1c40f"
    action = "Action: Reassess in 3 months"
else:
    recommendation = "⚪ **DEPTH PLAYER** - Low priority"
    rec_color = "#7f8c8d"
    action = "Action: Regional database only"

# Main area - two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🎯 Prediction Results")
    
    st.markdown(f"""
    <div class="prediction-card">
        <h3>Potential Score</h3>
        <h1 style="font-size: 4rem; margin: 0;">{potential_pred:.0f}<span style="font-size: 1.5rem;">/100</span></h1>
        <p style="color: #666;">Based on performance metrics and age trajectory</p>
        <hr>
        <h3>Estimated Market Value</h3>
        <h1 style="font-size: 3rem; margin: 0; color: #27ae60;">€{value_pred:,.0f}</h1>
        <p style="color: #666;">Projected transfer value</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🧠 Scout Recommendation")
    st.markdown(f"""
    <div class="prediction-card" style="border-left: 5px solid {rec_color};">
        <p style="font-size: 1.1rem;"><strong>{recommendation}</strong></p>
        <p style="margin-top: 1rem;">📋 {action}</p>
        <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #555;">
        🔍 <strong>Insight:</strong> {
            f"This {age}-year-old outperforms {int(np.mean(df[df['age'].between(age-1, age+1)]['goals_per_90'] < goals)*100)}% of peers in goals." 
            if goals > df['goals_per_90'].mean() 
            else f"Focus on improving final third output - currently {goals:.2f} goals/90 vs regional avg {df['goals_per_90'].mean():.2f}."
        }
        </p>
    </div>
    """, unsafe_allow_html=True)

# Comparison chart
st.markdown("---")
st.markdown("### 📈 Player Comparison vs Regional Averages")

# Create comparison data
regional_avg = df.groupby('region')[feature_cols].mean().reset_index()
player_data = pd.DataFrame({
    'Metric': ['Goals/90', 'Assists/90', 'Pass Acc %', 'Tackles/90', 'Dribbles/90', 'Key Passes/90'],
    'Player': [goals, assists, pass_acc, tackles, dribbles, key_passes]
})

# Simple radar chart data
categories = ['Goals/90', 'Assists/90', 'Pass Acc %', 'Tackles/90', 'Dribbles/90', 'Key Passes/90']
player_values = [goals, assists, pass_acc/20, tackles/2, dribbles/2, key_passes]  # Normalized

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=player_values,
    theta=categories,
    fill='toself',
    name=f'Player (Age {age})',
    line_color='#e74c3c'
))

# Add regional average reference
regional_norm = [
    df['goals_per_90'].mean(),
    df['assists_per_90'].mean(),
    df['pass_accuracy'].mean()/20,
    df['tackles_per_90'].mean()/2,
    df['dribbles_per_90'].mean()/2,
    df['key_passes_per_90'].mean()
]

fig.add_trace(go.Scatterpolar(
    r=regional_norm,
    theta=categories,
    fill='toself',
    name='East African Average',
    line_color='#3498db',
    opacity=0.5
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 5])
    ),
    showlegend=True,
    title="Player Profile vs Regional Benchmark",
    height=450
)

st.plotly_chart(fig, use_container_width=True)

# Undervalued players leaderboard
st.markdown("---")
st.markdown("### 💎 Hidden Gems: Most Undervalued Players in Database")

# Calculate value gap (actual value - predicted value based on performance)
df['predicted_from_perf'] = model_value.predict(df[feature_cols])
df['value_gap_pct'] = ((df['predicted_from_perf'] - df['market_value']) / df['predicted_from_perf']) * 100
undervalued = df[df['value_gap_pct'] > 20].sort_values('value_gap_pct', ascending=False).head(5)

if len(undervalued) > 0:
    for idx, row in undervalued.iterrows():
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid #ddd;">
            <span>🇰🇪 Player #{idx+1}</span>
            <span><strong>{row['age']} yrs</strong></span>
            <span>⚽ {row['goals_per_90']:.2f} g/90</span>
            <span style="color: #e74c3c;">🔻 {row['value_gap_pct']:.0f}% undervalued</span>
            <span>💰 €{row['market_value']:,.0f}</span>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No extreme undervaluations detected in current dataset.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    ⚽ EA Eye Scout Predictor | Data-driven recruitment intelligence for East African football<br>
    <em>Built with Streamlit, Random Forest, and East African player data</em>
</div>
""", unsafe_allow_html=True)