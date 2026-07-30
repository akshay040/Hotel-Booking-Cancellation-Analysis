"""Streamlit interface for transparent hotel cancellation risk scoring."""

import json
from pathlib import Path

import joblib
import streamlit as st

from hotel_cancellation.inference import booking_frame, threshold_message

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

st.set_page_config(page_title="Hotel cancellation risk", page_icon="🏨", layout="wide")
st.title("🏨 Hotel booking cancellation risk")
st.caption("Decision support—not an automated cancellation or guest-treatment system.")


@st.cache_resource
def load_model(path: Path):
    """Load the fitted training pipeline once per Streamlit process."""
    return joblib.load(path)


@st.cache_data
def load_metrics(path: Path) -> dict:
    """Load optional evaluation context without making it a deployment requirement."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


if not MODEL_PATH.exists():
    st.error("Model artifact not found")
    st.code(
        "python -m hotel_cancellation.train "
        "--data hotel_bookings_cleaned_enhanced.csv --output artifacts"
    )
    st.info("Docker images train and package this artifact during the image build.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except (OSError, ValueError, TypeError, EOFError, ImportError, AttributeError) as error:
    st.error("The model artifact could not be loaded. Retrain it with the command above.")
    st.exception(error)
    st.stop()

metrics = load_metrics(METRICS_PATH)
score_tab, model_tab = st.tabs(["Score a booking", "Model information"])

with score_tab:
    st.write("Enter information known at booking time. Fields are grouped for easier review.")
    with st.form("booking", clear_on_submit=False):
        booking, stay, history = st.tabs(["Booking", "Stay & guests", "History & requests"])
        with booking:
            col1, col2, col3 = st.columns(3)
            hotel = col1.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
            month = col2.selectbox(
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
            lead_time = col3.number_input("Lead time (days)", 0, 737, 30)
            market = col1.selectbox(
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
            channel = col2.selectbox(
                "Distribution channel", ["TA/TO", "Direct", "Corporate", "GDS"]
            )
            deposit = col3.selectbox("Deposit type", ["No Deposit", "Non Refund", "Refundable"])
            customer = col1.selectbox(
                "Customer type", ["Transient", "Transient-Party", "Contract", "Group"]
            )
            meal = col2.selectbox("Meal plan", ["BB", "HB", "SC", "FB", "Undefined"])
            room = col3.selectbox("Reserved room type", list("ABCDEFGHI") + ["L", "P"])
            country = (
                col1.text_input("Country code", "PRT", max_chars=3).strip().upper() or "Unknown"
            )
        with stay:
            col1, col2, col3 = st.columns(3)
            weekend = col1.number_input("Weekend nights", 0, 19, 1)
            weekday = col2.number_input("Week nights", 0, 50, 2)
            adr = col3.number_input("Average daily rate (€)", 0.0, 5400.0, 100.0, step=5.0)
            adults = col1.number_input("Adults", 0, 55, 2)
            children = col2.number_input("Children", 0, 10, 0)
            babies = col3.number_input("Babies", 0, 10, 0)
        with history:
            col1, col2, col3 = st.columns(3)
            repeated = col1.checkbox("Repeated guest")
            previous_cancellations = col2.number_input("Previous cancellations", 0, 26, 0)
            previous_kept = col3.number_input("Previous bookings not cancelled", 0, 72, 0)
            changes = col1.number_input("Booking changes", 0, 21, 0)
            waiting = col2.number_input("Days on waiting list", 0, 391, 0)
            parking = col3.number_input("Required parking spaces", 0, 8, 0)
            requests = col1.number_input("Special requests", 0, 5, 0)
        submitted = st.form_submit_button("Estimate cancellation risk", type="primary")

    if not submitted:
        st.info("Complete the form and select **Estimate cancellation risk**.")
    else:
        values = {
            "lead_time": lead_time,
            "stays_in_weekend_nights": weekend,
            "stays_in_week_nights": weekday,
            "adults": adults,
            "children": children,
            "babies": babies,
            "is_repeated_guest": int(repeated),
            "previous_cancellations": previous_cancellations,
            "previous_bookings_not_canceled": previous_kept,
            "booking_changes": changes,
            "days_in_waiting_list": waiting,
            "adr": adr,
            "required_car_parking_spaces": parking,
            "total_of_special_requests": requests,
            "hotel": hotel,
            "arrival_date_month": month,
            "meal": meal,
            "country": country,
            "market_segment": market,
            "distribution_channel": channel,
            "reserved_room_type": room,
            "deposit_type": deposit,
            "customer_type": customer,
        }
        try:
            probability = float(model.predict_proba(booking_frame(values))[0, 1])
            label, action = threshold_message(probability)
        except ValueError as error:
            st.error(str(error))
        else:
            left, right = st.columns([1, 2])
            left.metric("Estimated cancellation probability", f"{probability:.1%}")
            left.progress(probability)
            right.subheader(label)
            right.write(action)
            right.caption("The 0.50 threshold is a demonstration default, not a business policy.")

with model_tab:
    st.subheader("Evaluation context")
    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Holdout ROC-AUC", f"{metrics['roc_auc']:.3f}")
        col2.metric("Holdout average precision", f"{metrics['average_precision']:.3f}")
        col3.metric("Holdout rows", f"{metrics['test_rows']:,}")
        st.caption(
            f"Chronological holdout starts {metrics['test_start']}; model: {metrics['model']}."
        )
    else:
        st.info("Evaluation metrics are unavailable. Retraining creates `artifacts/metrics.json`.")
    st.markdown(
        "**Limitations**\n\n"
        "- The public data covers two Portuguese hotels from 2015–2017 and may not generalise.\n"
        "- Unknown categories are handled safely, but data drift can degrade accuracy.\n"
        "- Country may act as a sensitive proxy; review impacts and never use this score for "
        "adverse treatment.\n"
        "- Feature importance describes model reliance, not causation."
    )
