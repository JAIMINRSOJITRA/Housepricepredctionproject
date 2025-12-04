import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🏡 House Price Estimator",
    layout="wide",
    page_icon="🏠",
)

# =========================
# CUSTOM CSS (Modern Dashboard Style)
# =========================
st.markdown(
    """
    <style>
    body { background-color: #f5f7fb; }
    .main { background-color: #f5f7fb; }
    .big-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #111827;
    }
    .subtitle {
        font-size: 1rem;
        color: #4b5563;
    }
    .card {
        background: #ffffff;
        border-radius: 18px;
        padding: 20px 24px;
        box-shadow: 0 12px 30px rgba(15,23,42,0.08);
        margin-bottom: 18px;
    }
    .glass-card {
        background: rgba(255,255,255,0.92);
        border-radius: 18px;
        padding: 22px 26px;
        box-shadow: 0 10px 28px rgba(15,23,42,0.12);
        backdrop-filter: blur(8px);
        margin-bottom: 18px;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 14px 18px;
        box-shadow: 0 6px 12px rgba(15,23,42,0.06);
    }
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
        padding: 18px 0px 4px 0px;
    }
    .stButton>button {
        border-radius: 999px;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
        border: none;
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        color: white;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8, #4338ca);
    }
    .sidebar-title {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
    }
    .big-number {
        font-size: 2.4rem;
        font-weight: 800;
        color: #065f46;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# LOAD PIPELINE
# =========================
@st.cache_resource
def load_pipeline(model_path: str = "xgboost_pipeline.pkl"):
    if not os.path.exists(model_path):
        return None, f"⚠️ Model file `{model_path}` not found in current folder."
    try:
        pipe = joblib.load(model_path)
        return pipe, None
    except Exception as e:
        return None, f"❌ Failed to load model: {e}"

pipeline, model_error = load_pipeline()

# =========================
# LOAD DATA FOR INSIGHTS
# =========================
@st.cache_data
def load_data(csv_path: str = "Housing.csv"):
    if not os.path.exists(csv_path):
        return None, f"Dataset `{csv_path}` not found."
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower().str.strip()
        return df, None
    except Exception as e:
        return None, f"Failed to load dataset: {e}"

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.markdown("<p class='sidebar-title'>🏠 House Price Dashboard</p>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["Home", "Predict Price", "Insights", "About"], index=0)

st.sidebar.info("Use this app to estimate **house prices** based on property details.")
st.sidebar.markdown("---")
st.sidebar.write("Made with ❤️ by **Jaimin Sojitra**")

# =========================
# HELPER: PREPROCESS SINGLE INPUT (MATCH TRAINING)
# =========================
def build_feature_vector(
    area: float,
    bedrooms: int,
    bathrooms: int,
    mainroad: bool,
    guestroom: bool,
    basement: bool,
    hotwater: bool,
    ac: bool,
    prefarea: bool,
    furnishing: str,
):
    f_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
    b = lambda x: 1 if x else 0

    furn_code = f_map[furnishing]

    total_rooms = bedrooms + bathrooms + b(guestroom)
    is_luxury = 1 if area > 3000 else 0
    amenity_score = (
        b(mainroad) + b(guestroom) + b(basement) +
        b(hotwater) + b(ac) + b(prefarea)
    )
    area_furn = area * furn_code
    area_bucket = 0 if area <= 1500 else (1 if area <= 3000 else 2)
    bedrooms_sq = bedrooms ** 2
    log_area = np.log1p(area)

    X = np.array([[
        area,
        bedrooms,
        bathrooms,
        b(mainroad),
        b(guestroom),
        b(basement),
        b(hotwater),
        b(ac),
        b(prefarea),
        furn_code,
        total_rooms,
        0,              # const_zero
        is_luxury,
        amenity_score,
        area_furn,
        area_bucket,
        bedrooms_sq,
        log_area,
    ]])

    data_dict = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "mainroad": b(mainroad),
        "guestroom": b(guestroom),
        "basement": b(basement),
        "hotwaterheating": b(hotwater),
        "airconditioning": b(ac),
        "prefarea": b(prefarea),
        "furnishing": furn_code,
        "total_rooms": total_rooms,
        "const_zero": 0,
        "is_luxury": is_luxury,
        "amenity_score": amenity_score,
        "area_furn": area_furn,
        "area_bucket": area_bucket,
        "bedrooms_sq": bedrooms_sq,
        "log_area": log_area,
    }

    return X, data_dict

# =========================
# HOME PAGE
# =========================
if page == "Home":
    st.markdown("<div class='hero-card'>", unsafe_allow_html=True)
    st.markdown("<div class='big-title'>🏡 House Price Estimator</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <p class='subtitle'>
        A modern ML-powered tool to estimate property prices based on area, rooms, amenities and locality comfort.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Model Type", "XGBoost Pipeline")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Target", "House Price (₹)")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Made By", "Jaimin Sojitra")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='footer'>Made with ❤️ by <b>Jaimin Sojitra</b></div>",
        unsafe_allow_html=True,
    )

# =========================
# PREDICT PRICE PAGE
# =========================
elif page == "Predict Price":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📥 Enter Property Details")

    left_col, right_col = st.columns(2)

    with left_col:
        locality = st.text_input("Locality (optional)", placeholder="Example: Andheri West, Mumbai")
        area = st.number_input("Area (sq.ft)", min_value=200, max_value=20000, value=1500, step=50)
        bedrooms = st.slider("Bedrooms", 1, 10, 3)
        bathrooms = st.slider("Bathrooms", 1, 8, 2)

        furnishing = st.selectbox(
            "Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"]
        )

    with right_col:
        st.markdown("🏘 Amenities")
        mainroad = st.toggle("Main Road Access")
        guestroom = st.toggle("Guest Room")
        basement = st.toggle("Basement")
        hotwater = st.toggle("Hot Water Heating")
        ac = st.toggle("Air Conditioning")
        prefarea = st.toggle("Preferred Residential Area")

    st.write("")
    pred_col, gauge_col = st.columns([1.4, 1])

    with pred_col:
        if model_error:
            st.error(model_error)
        else:
            if st.button("🔮 Estimate House Price"):
                X, info = build_feature_vector(
                    area=area,
                    bedrooms=bedrooms,
                    bathrooms=bathrooms,
                    mainroad=mainroad,
                    guestroom=guestroom,
                    basement=basement,
                    hotwater=hotwater,
                    ac=ac,
                    prefarea=prefarea,
                    furnishing=furnishing,
                )
                try:
                    pred_price = pipeline.predict(X)[0]
                    st.markdown("### 💰 Estimated Price")
                    st.markdown(f"<div class='big-number'>₹ {pred_price:,.0f}</div>", unsafe_allow_html=True)
                    st.caption("*(This is a model-based estimate, not an exact market quote.)*")

                    low = pred_price * 0.95
                    high = pred_price * 1.05
                    st.write(f"**Confidence Range (±5%)**: ₹ {low:,.0f} — ₹ {high:,.0f}")

                    st.write("---")
                    st.markdown("**🏠 Property Summary**")
                    st.markdown(
                        f"""
                        - Area: **{area} sq.ft**  
                        - Bedrooms / Bathrooms: **{bedrooms} / {bathrooms}**  
                        - Furnishing: **{furnishing}**  
                        - Main Road: **{"Yes" if mainroad else "No"}**, Basement: **{"Yes" if basement else "No"}**  
                        - Guest Room: **{"Yes" if guestroom else "No"}**, AC: **{"Yes" if ac else "No"}**  
                        - Preferred Area: **{"Yes" if prefarea else "No"}**  
                        """
                    )

                except Exception as e:
                    st.error(f"⚠️ Prediction failed: {e}")

    with gauge_col:
        st.subheader("📊 Price Scale (Normalized)")
        if not model_error:
            try:
                X, _ = build_feature_vector(
                    area=area,
                    bedrooms=bedrooms,
                    bathrooms=bathrooms,
                    mainroad=mainroad,
                    guestroom=guestroom,
                    basement=basement,
                    hotwater=hotwater,
                    ac=ac,
                    prefarea=prefarea,
                    furnishing=furnishing,
                )
                pred_price = pipeline.predict(X)[0]
                price_score = float(np.log1p(pred_price))  # just for visualization

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=price_score,
                        title={"text": "Relative Price Score"},
                        gauge={
                            "axis": {"range": [0, 15]},
                            "bar": {"color": "#2563eb"},
                            "steps": [
                                {"range": [0, 5], "color": "#dcfce7"},
                                {"range": [5, 10], "color": "#fef9c3"},
                                {"range": [10, 15], "color": "#fee2e2"},
                            ],
                        },
                    )
                )
                fig.update_layout(height=260, margin=dict(l=15, r=15, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Run a prediction to see price score.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='footer'>Made with ❤️ by <b>Jaimin Sojitra</b></div>",
        unsafe_allow_html=True,
    )

# =========================
# INSIGHTS PAGE
# =========================
elif page == "Insights":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Housing Dataset Insights")

    df, data_error = load_data()
    if data_error:
        st.error(data_error)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.write("First few rows of the dataset:")
        st.dataframe(df.head())

        if "price" in df.columns:
            st.write("")
            st.markdown("**Price Distribution**")
            fig1, ax1 = plt.subplots()
            ax1.hist(df["price"], bins=30)
            ax1.set_xlabel("Price (₹)")
            ax1.set_ylabel("Frequency")
            st.pyplot(fig1)

        if "area" in df.columns:
            st.write("")
            st.markdown("**Area vs Price (Sample)**")
            fig2, ax2 = plt.subplots()
            sample = df.sample(min(200, len(df)), random_state=42)
            ax2.scatter(sample["area"], sample["price"], alpha=0.6)
            ax2.set_xlabel("Area (sq.ft)")
            ax2.set_ylabel("Price (₹)")
            st.pyplot(fig2)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='footer'>Made with ❤️ by <b>Jaimin Sojitra</b></div>",
        unsafe_allow_html=True,
    )

# =========================
# ABOUT PAGE
# =========================
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("ℹ️ About this App")

    st.markdown(
        """
        This application uses a **Machine Learning (XGBoost) regression model**
        wrapped in a full **sklearn Pipeline** to estimate house prices.

        ### 🧠 How it works
        1. You provide property details such as area, rooms, amenities and furnishing.  
        2. The app converts them into engineered numerical features (rooms, luxury flag, amenity score, etc.).  
        3. These features are scaled and passed to an XGBoost model.  
        4. The model returns an estimated **price in ₹**.

        ### 🛠 Tech Stack
        - 🐍 Python  
        - 🧮 NumPy, Pandas  
        - 🧠 XGBoost, Scikit-learn Pipeline  
        - 🌐 Streamlit for interactive UI  
        - 📊 Matplotlib & Plotly for visualizations  
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='footer'>Made with ❤️ by <b>Jaimin Sojitra</b></div>",
        unsafe_allow_html=True,
    )
