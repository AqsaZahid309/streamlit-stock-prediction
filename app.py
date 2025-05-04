import streamlit as st
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ====================== 1. PAGE SETUP - BEAUTIFUL DESIGN ====================== #
st.set_page_config(
    page_title="Stock Prediction App 📊", 
    page_icon="💹", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(45deg, #2c3e50, #34495e);
        color: #ecf0f1;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(45deg, #f8c291, #e67e22) !important;
        color: white !important;
        border-radius: 18px;
        padding: 15px 30px;
        font-size: 20px;
        border: none;
        box-shadow: 0 6px 12px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.07);
        box-shadow: 0 8px 16px rgba(0,0,0,0.6);
    }
    .stSelectbox, .stTextInput {
        background-color: rgba(255,255,255,0.15) !important;
        border-radius: 12px;
        padding: 10px;
        font-size: 16px;
    }
    .stMarkdown h1 {
        color: #f8c291 !important;
        font-size: 42px;
        font-weight: bold;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
    }
    .stSidebar {
        background: rgba(52, 152, 219, 0.8) !important;
        border-right: 4px solid #e67e22;
    }
    .stMetric>div {
        background-color: rgba(52, 152, 219, 0.3);
        border-radius: 15px;
        padding: 12px;
        font-size: 20px;
        font-weight: bold;
    }
    .stTextInput>input {
        background-color: rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'data_clean' not in st.session_state:
    st.session_state.data_clean = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = {}

# ====================== 2. HEADER SECTION ====================== #
col1, col2 = st.columns([1, 3])

# Load the uploaded image (your GIF)
uploaded_image = st.file_uploader("Upload SpongeBob GIF for the Header", type=["gif"])

if uploaded_image:
    st.image(uploaded_image, width=250)  # Display the uploaded GIF image

with col2:
    st.title("💰 Stock Prediction Magic! 💹")
    st.markdown("""<div style="border-left: 5px solid #e67e22; padding-left: 15px; margin: 10px 0;">
    <i>"Unlock wealth secrets, train AI models!<br> Predict stock prices, grow your money!"</i></div>""", unsafe_allow_html=True)

# ====================== 3. SIDEBAR - DATA UPLOAD ====================== #
with st.sidebar:
    st.header("📂 Upload Your Data")
    st.image("https://media.giphy.com/media/3o7TKMt1VV26k5wOqA/giphy.gif", width=200)
    
    uploaded_file = st.file_uploader("Select CSV File", type=["csv"], help="Upload a file similar to 'all_stocks_5yr.csv'")
    
    if uploaded_file:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("✅ Data loaded successfully!")
            st.balloons()
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# ====================== 4. MAIN WORKFLOW ====================== #
if 'data' in st.session_state and st.session_state.data is not None:
    # Stock selection
    selected_stock = st.selectbox(
        "📈 Select Your Stock", 
        st.session_state.data['Name'].unique(),
        help="Choose which company's stock to analyze"
    )
    
    stock_data = st.session_state.data[st.session_state.data['Name'] == selected_stock]

    # Data cleaning
    if st.button("1️⃣ Clean Data (Remove Missing Values)"):
        with st.spinner("🧹 Cleaning data..."):
            st.session_state.data_clean = stock_data.dropna()
            missing_count = stock_data.isnull().sum().sum()
            st.success(f"🎉 Removed {missing_count} missing values!")
            st.dataframe(st.session_state.data_clean.head().style.highlight_null(props="color: red;"))

    # Model training
    if st.button("2️⃣ Train Model (AI Learning)"):
        if st.session_state.data_clean is None:
            st.error("Please clean data first!")
        else:
            with st.spinner("🤖 Training AI model..."):
                try:
                    X = st.session_state.data_clean[['open', 'high', 'low', 'volume']]
                    y = st.session_state.data_clean['close']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    
                    st.session_state.model = LinearRegression().fit(X_train, y_train)
                    st.session_state.test_data = {'X_test': X_test, 'y_test': y_test}
                    
                    # Calculate training score
                    train_score = st.session_state.model.score(X_train, y_train)
                    st.success(f"✅ Model trained! Training R² score: {train_score:.2f}")
                    st.image("https://media.giphy.com/media/3o7qE1H1FqJQxYxjWM/giphy.gif", width=300)
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

    # Predictions
    if st.button("3️⃣ Predict Future Prices"):
        if 'model' not in st.session_state or st.session_state.model is None:
            st.error("Please train model first!")
        else:
            with st.spinner("🔮 Predicting prices..."):
                try:
                    predictions = st.session_state.model.predict(st.session_state.test_data['X_test'])
                    results = pd.DataFrame({
                        "Actual Price 💵": st.session_state.test_data['y_test'].values,
                        "Predicted Price 🔮": predictions
                    }).reset_index(drop=True)
                    
                    # Calculate R² score
                    r2 = r2_score(results["Actual Price 💵"], results["Predicted Price 🔮"])
                    st.metric("Model Performance", f"R² Score: {r2:.2f}")

                    # Create Altair visualization
                    chart = alt.Chart(results.reset_index()).transform_fold(
                        ['Actual Price 💵', 'Predicted Price 🔮'],
                        as_=['Price Type', 'Value']
                    ).mark_line().encode(
                        x='index:Q',
                        y='Value:Q',
                        color='Price Type:N',
                        strokeDash='Price Type:N'
                    ).properties(
                        height=400,
                        title=f"{selected_stock} Price Predictions"
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Download results
                    st.download_button(
                        "📥 Download Results",
                        data=results.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv",
                        help="Save your predictions for later use"
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

# ====================== 5. FOOTER SECTION ====================== #
st.markdown("---")
footer_col1, footer_col2 = st.columns([2, 1])
with footer_col1:
    st.markdown("""
    <div style="background: rgba(52, 152, 219, 0.2); padding: 20px; border-radius: 15px;">
    <h4 style="color:#3498db">⚠️ Important Notice:</h4>
    <p>These predictions are for educational purposes only. Consult a financial expert before making real investments.</p>
    </div>
    """, unsafe_allow_html=True)
with footer_col2:
    # Footer GIF or Image you want to display
    st.image("https://media.giphy.com/media/3o7abKhOpu0NwenH3O/giphy.gif", width=150)

st.markdown("---")
st.markdown("### 🚀 Created by Aqsa - A New Marvel in AI World! 🌟")
