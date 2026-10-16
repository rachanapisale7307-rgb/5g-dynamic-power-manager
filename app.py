import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from river.drift import ADWIN

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="5G Dynamic Power Manager",
    page_icon="📡",
    layout="wide"
)

st.title("📡 5G Dynamic Power Manager")
st.caption("Adaptive 5G Network Monitoring | Concept Drift | Energy Efficiency")

FILE_NAME = "/content/Phase4_Streaming_Data.csv"
INITIAL_TRAINING = 745

FEATURES = [
    "Grid_ID",
    "Download_Bytes",
    "Upload_Bytes",
    "Packet_Counts",
    "Connected_UEs",
    "Call_SMS_Frequency",
    "Rolling_Avg_UEs",
    "Rolling_Avg_DL",
    "Spatial_Traffic_Gradient",
    "Total_Traffic",
    "Hour_of_Day",
    "Day_of_Week",
    "BMU_X",
    "BMU_Y",
    "Operational_State_ID",
    "SOM_Distance",
    "Stream_Index"
]

TARGET = "Base_Station_Sleep_State"

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv(FILE_NAME)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"], errors="coerce"
        )
        df = df.sort_values(
            "Timestamp", na_position="last"
        ).reset_index(drop=True)

    X = df[FEATURES].copy()

    # Convert Grid_ID labels such as Grid_1 into numeric codes.
    X["Grid_ID"] = (
        X["Grid_ID"].astype(str)
        .str.extract(r"(\d+)", expand=False)
    )
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid = y.notna()

    df = df.loc[valid].reset_index(drop=True)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)

    return df, X, y


try:
    df, X, y = load_data()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

if len(df) <= INITIAL_TRAINING:
    st.error(
        f"At least {INITIAL_TRAINING + 1} valid rows are required."
    )
    st.stop()

# -------------------- MODEL --------------------
def create_model():
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    return AdaBoostClassifier(
        estimator=tree,
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    )


def train_model(features, labels):
    model = create_model()
    model.fit(features, labels)
    return model


# -------------------- SESSION STATE --------------------
if "stream_index" not in st.session_state:
    st.session_state.stream_index = INITIAL_TRAINING

if "adaptive_model" not in st.session_state:
    st.session_state.adaptive_model = train_model(
        X.iloc[:INITIAL_TRAINING],
        y.iloc[:INITIAL_TRAINING]
    )

if "adwin" not in st.session_state:
    st.session_state.adwin = ADWIN()

if "predictions" not in st.session_state:
    st.session_state.predictions = []

if "actuals" not in st.session_state:
    st.session_state.actuals = []

if "accuracy_history" not in st.session_state:
    st.session_state.accuracy_history = []

if "drift_events" not in st.session_state:
    st.session_state.drift_events = []

if "event_message" not in st.session_state:
    st.session_state.event_message = "Monitoring network traffic."

if "last_drift" not in st.session_state:
    st.session_state.last_drift = False

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Stream Controls")

batch_size = st.sidebar.slider(
    "Observations per batch",
    min_value=1,
    max_value=100,
    value=20
)

adaptation_buffer = st.sidebar.slider(
    "Recalibration window",
    min_value=20,
    max_value=200,
    value=50
)

if st.sidebar.button("▶️ Process Next Batch", use_container_width=True):
    start = st.session_state.stream_index
    end = min(start + batch_size, len(df))

    if start >= len(df):
        st.session_state.event_message = (
            "Stream completed. Reset to start again."
        )
    else:
        model = st.session_state.adaptive_model
        detector = st.session_state.adwin
        drift_happened = False

        for idx in range(start, end):
            row = X.iloc[[idx]]
            actual = int(y.iloc[idx])
            prediction = int(model.predict(row)[0])

            st.session_state.predictions.append(prediction)
            st.session_state.actuals.append(actual)

            error = int(prediction != actual)
            detector.update(error)

            if detector.drift_detected:
                drift_happened = True

                st.session_state.drift_events.append({
                    "Observation": idx,
                    "Timestamp": str(df.iloc[idx].get("Timestamp", "")),
                    "Actual": actual,
                    "Prediction": prediction
                })

                # Recalibrate on the recent labelled data available
                # in this simulation.
                buffer_start = max(
                    0, idx - adaptation_buffer + 1
                )

                recent_X = X.iloc[buffer_start:idx + 1]
                recent_y = y.iloc[buffer_start:idx + 1]

                if recent_y.nunique() >= 2:
                    model = train_model(recent_X, recent_y)

                # Start a fresh detector after handling drift.
                detector = ADWIN()

        st.session_state.adaptive_model = model
        st.session_state.adwin = detector
        st.session_state.stream_index = end

        if drift_happened:
            st.session_state.event_message = (
                "🚨 Concept Drift Detected! Recalibrating..."
            )
        else:
            st.session_state.event_message = (
                "Monitoring network traffic."
            )

        all_predictions = st.session_state.predictions
        all_actuals = st.session_state.actuals

        running_accuracy = accuracy_score(
            all_actuals, all_predictions
        )

        st.session_state.accuracy_history.append({
            "Observation": end,
            "Accuracy (%)": running_accuracy * 100
        })

        st.rerun()

if st.sidebar.button("🔄 Reset Stream", use_container_width=True):
    for key in [
        "stream_index", "adaptive_model", "adwin",
        "predictions", "actuals", "accuracy_history",
        "drift_events", "event_message", "last_drift"
    ]:
        st.session_state.pop(key, None)
    st.rerun()

# -------------------- METRICS --------------------
processed = len(st.session_state.predictions)
remaining = len(df) - st.session_state.stream_index
drift_count = len(st.session_state.drift_events)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Dataset Records", f"{len(df):,}")
c2.metric("Processed Records", f"{processed:,}")
c3.metric("Drift Events", drift_count)
c4.metric("Remaining Records", f"{max(0, remaining):,}")

st.info(st.session_state.event_message)

# -------------------- ACCURACY CHART --------------------
st.subheader("📈 Live Model Accuracy")

if st.session_state.accuracy_history:
    accuracy_df = pd.DataFrame(
        st.session_state.accuracy_history
    ).set_index("Observation")

    st.line_chart(accuracy_df)
    st.metric(
        "Current Accuracy",
        f"{st.session_state.accuracy_history[-1]['Accuracy (%)']:.2f}%"
    )
else:
    st.write(
        "Click **Process Next Batch** in the sidebar to begin streaming."
    )

# -------------------- DRIFT EVENTS --------------------
st.subheader("🚨 Concept Drift Events")

if st.session_state.drift_events:
    st.dataframe(
        pd.DataFrame(st.session_state.drift_events),
        use_container_width=True
    )
else:
    st.write("No drift events detected yet.")

# -------------------- ENERGY ESTIMATE --------------------
st.subheader("🌱 Estimated Energy Savings")

if processed > 0:
    predictions = np.array(st.session_state.predictions)

    # Assumption: class 1 means sleep; confirm this mapping
    # against the dataset's label definition.
    sleep_count = int(np.sum(predictions == 1))
    normal_count = processed - sleep_count

    NORMAL_POWER_W = 1000
    SLEEP_POWER_W = 100
    INTERVAL_SECONDS = 900  # assumed 15-minute interval

    baseline_joules = (
        processed * NORMAL_POWER_W * INTERVAL_SECONDS
    )

    managed_joules = (
        normal_count * NORMAL_POWER_W * INTERVAL_SECONDS
        + sleep_count * SLEEP_POWER_W * INTERVAL_SECONDS
    )

    saved_joules = baseline_joules - managed_joules
    saving_pct = (
        100 * saved_joules / baseline_joules
        if baseline_joules else 0
    )

    e1, e2, e3 = st.columns(3)
    e1.metric("Predicted Sleep Records", sleep_count)
    e2.metric("Estimated Energy Saved", f"{saved_joules:,.0f} J")
    e3.metric("Estimated Energy Reduction", f"{saving_pct:.2f}%")

    st.caption(
        "Illustrative estimate using assumed power levels and a "
        "15-minute interval, not measured hardware energy consumption."
    )
else:
    st.write("Energy estimates will appear after processing a batch.")

# -------------------- SHAP --------------------
st.subheader("🔎 SHAP Feature Importance")

if st.button("Generate SHAP Beeswarm Plot"):
    try:
        sample_size = min(20, processed)

        if sample_size < 1:
            st.warning("Process a batch before generating the SHAP plot.")
        else:
            explain_X = X.iloc[
                st.session_state.stream_index - sample_size:
                st.session_state.stream_index
            ].copy()

            # Explain predicted class probabilities.
            explainer = shap.Explainer(
                st.session_state.adaptive_model.predict_proba,
                X.iloc[:min(30, len(X))],
                algorithm="permutation"
            )

            shap_values = explainer(
                explain_X,
                max_evals=2 * len(FEATURES) + 1
            )

            # Select class 1 when available.
            if len(shap_values.shape) == 3:
                class_index = list(
                    st.session_state.adaptive_model.classes_
                ).index(1) if 1 in st.session_state.adaptive_model.classes_ else 0

                shap_values = shap_values[:, :, class_index]

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.beeswarm(
                shap_values,
                max_display=15,
                show=False
            )
            st.pyplot(fig)
            plt.close("all")

    except Exception as e:
        st.error(f"SHAP plot could not be generated: {e}")

# -------------------- FOOTER --------------------
st.divider()
st.caption(
    "5G Dynamic Power Manager | Streaming is simulated from the "
    "prepared CSV dataset. Drift alarms depend on ADWIN detection."
)
