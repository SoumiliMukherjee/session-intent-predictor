import requests
import streamlit as st

st.set_page_config(page_title='Purchase Intent Predictor' , layout='centered')
st.title('Online Shopper Purchase Intent Predictor')
st.markdown("Dual-Stream PyTorch **LSTM + ANN** served via **FastAPI**")

with st.form('session-form'):
    st.subheader('1. Browsing Funnel Activity (Sequence Stream)')
    col1 , col2 = st.columns(2)
    with col1:
        admin_pages = st.number_input("Administrative Pages", min_value=0, value=2)
        info_pages = st.number_input("Informational Pages", min_value=0, value=1)
        prod_pages = st.number_input("Product Related Pages", min_value=0, value=18)
    with col2:
        admin_dur = st.number_input("Administrative Duration (sec)", min_value=0.0, value=45.0)
        info_dur = st.number_input("Informational Duration (sec)", min_value=0.0, value=12.5)
        prod_dur = st.number_input("Product Related Duration (sec)", min_value=0.0, value=640.0)
    st.subheader("2. Session Metrics & Visitor Context (Context Stream)")
    col3 , col4 = st.columns(2)
    with col3:
        bounce_rate = st.slider("Bounce Rate", 0.0, 0.2, 0.01, step=0.005)
        exit_rate = st.slider("Exit Rate", 0.0, 0.2, 0.025, step=0.005)
        page_values = st.number_input("Page Values", min_value=0.0, value=14.2)
        special_day = st.slider("Special Day Closeness", 0.0, 1.0, 0.0, step=0.2)
    with col4:
        month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=8)
        visitor_type = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"])
        weekend = st.checkbox("Weekend Session", value=False)
        traffic_type = st.number_input("Traffic Type ID", min_value=1, max_value=20, value=2)
    submitted = st.form_submit_button('Predict Purchase Intent' , use_container_width=True)
if submitted:
    payload = {
        "Administrative": admin_pages,
        "Administrative_Duration": admin_dur,
        "Informational": info_pages,
        "Informational_Duration": info_dur,
        "ProductRelated": prod_pages,
        "ProductRelated_Duration": prod_dur,
        "BounceRates": bounce_rate,
        "ExitRates": exit_rate,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": traffic_type,
        "VisitorType": visitor_type,
        "Weekend": weekend
    }
    try:
        response = requests.post('http://127.0.0.1:8000/predict' , json=payload)
        response.raise_for_status()
        result = response.json()
        prob = result['purchase_probability']
        st.divider()
        if result['predicted_intent'] == 'Purchase':
            st.success(f"**High Purchase Intent** — Conversion Probability: **{prob * 100:.1f}%**")
        else:
            st.warning(f"**Likely Drop-off** — Conversion Probability: **{prob * 100:.1f}%**")
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach FastAPI. Keep `uvicorn app:app --reload` running in Terminal 1.")
        
