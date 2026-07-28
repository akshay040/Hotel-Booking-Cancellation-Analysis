"""Streamlit interface for transparent booking cancellation risk scoring."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("artifacts/model.joblib")

st.set_page_config(page_title="Hotel cancellation risk", page_icon="🏨", layout="wide")
st.title("🏨 Hotel booking cancellation risk")
st.caption("Decision support—not an automated cancellation or guest-treatment system.")


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error("Model artifact not found. Run `python -m hotel_cancellation.train` first.")
    st.stop()

model = load_model(MODEL_PATH)
with st.sidebar:
    st.header("Booking details")
    hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
    lead_time = st.number_input("Lead time (days)", 0, 800, 30)
    adr = st.number_input("Average daily rate", 0.0, 5_000.0, 100.0)
    weekend = st.number_input("Weekend nights", 0, 30, 1)
    weekday = st.number_input("Week nights", 0, 60, 2)
    adults = st.number_input("Adults", 1, 10, 2)
    children = st.number_input("Children", 0, 10, 0)
    month = st.selectbox(
        "Arrival month",
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
    )
    market = st.selectbox(
        "Market segment",
        [
            "Online TA",
            "Offline TA/TO",
            "Direct",
            "Corporate",
            "Groups",
            "Complementary",
            "Aviation",
        ],
    )
    deposit = st.selectbox("Deposit type", ["No Deposit", "Non Refund", "Refundable"])
    requests = st.number_input("Special requests", 0, 5, 0)
    repeated = st.checkbox("Returning guest")

row = pd.DataFrame(
    [
        {
            "lead_time": lead_time,
            "stays_in_weekend_nights": weekend,
            "stays_in_week_nights": weekday,
            "adults": adults,
            "children": children,
            "babies": 0,
            "is_repeated_guest": int(repeated),
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "booking_changes": 0,
            "days_in_waiting_list": 0,
            "adr": adr,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": requests,
            "hotel": hotel,
            "arrival_date_month": month,
            "meal": "BB",
            "country": "PRT",
            "market_segment": market,
            "distribution_channel": "TA/TO",
            "reserved_room_type": "A",
            "deposit_type": deposit,
            "customer_type": "Transient",
        }
    ]
)

probability = float(model.predict_proba(row)[0, 1])
left, right = st.columns([1, 2])
with left:
    st.metric("Estimated cancellation probability", f"{probability:.1%}")
    if probability >= 0.5:
        st.warning(
            "Higher model-estimated risk: consider a routine confirmation, not punitive action."
        )
    else:
        st.success("Lower model-estimated risk.")
with right:
    st.subheader("How to use this result")
    st.write(
        "The score comes from a logistic-regression pipeline trained on historical bookings. "
        "It is an estimate, not certainty. Monitor performance over time and review impacts "
        "across guest groups."
    )
    st.progress(probability)

with st.expander("Model limitations"):
    st.markdown(
        "- The public dataset covers two Portuguese hotels from 2015–2017 and may not generalise.\n"
        "- Unknown categories are handled safely, but data drift can degrade accuracy.\n"
        "- Country is used as an operational market signal; do not use the score for "
        "discriminatory treatment.\n"
        "- The 0.50 threshold is a demonstration default and should be selected using "
        "intervention costs."
    )
