import streamlit as st
import mlflow
import os
import pandas as pd
import tempfile
from ultralytics import YOLO
from PIL import Image
import shutil

# Page Config
st.set_page_config(page_title="Mask Detection Inference", layout="wide")

st.title("😷 Face Mask Detection - MLflow Inference")

# -------------------------------------------------------------------------
# Sidebar: MLflow Configuration & Model Selection
# -------------------------------------------------------------------------
st.sidebar.header("MLflow Configuration")

# Path to local mlruns relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
mlruns_dir = os.path.join(script_dir, "mlruns")
tracking_uri = f"file:///{mlruns_dir}"

try:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    st.sidebar.success(f"Connected to MLflow at: {mlruns_dir}")
except Exception as e:
    st.sidebar.error(f"Could not connect to MLflow: {e}")
    st.stop()

# 1. Select Experiment
experiments = client.search_experiments()
exp_names = [e.name for e in experiments if e.lifecycle_stage == "active"]

if not exp_names:
    st.warning("No active experiments found.")
    st.stop()

selected_exp_name = st.sidebar.selectbox("Select Experiment", exp_names)
selected_exp = next(e for e in experiments if e.name == selected_exp_name)

# 2. Select Run
runs = client.search_runs(
    experiment_ids=[selected_exp.experiment_id], order_by=["attribute.start_time DESC"]
)
if not runs:
    st.sidebar.warning("No runs found for this experiment.")
    st.stop()

# Create a readable list of runs (Name + ID + Start Time)
run_options = {
    f"{r.data.tags.get('mlflow.runName', 'Unnamed')} ({r.info.run_id[:7]})": r
    for r in runs
}
selected_run_label = st.sidebar.selectbox("Select Run", list(run_options.keys()))
selected_run = run_options[selected_run_label]
run_id = selected_run.info.run_id

# 3. Select Model Artifact
model_type = st.sidebar.radio(
    "Model Version", ["Best (weights/best.pt)", "Last (weights/last.pt)"]
)
artifact_path = "weights/best.pt" if "Best" in model_type else "weights/last.pt"


# Download Logic
@st.cache_resource(show_spinner=False)
def load_yolo_model(run_id, artifact_path, tracking_uri):
    # Ensure raw URI doesn't break downloading
    mlflow.set_tracking_uri(tracking_uri)

    # Create a temp directory for the artifact
    local_dir = os.path.join(tempfile.gettempdir(), "mlflow_models", run_id)
    os.makedirs(local_dir, exist_ok=True)

    target_path = os.path.join(local_dir, os.path.basename(artifact_path))

    # Check if cached
    if not os.path.exists(target_path):
        with st.spinner(f"Downloading {artifact_path} from MLflow..."):
            try:
                # download_artifacts returns the local path
                downloaded_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=artifact_path, dst_path=local_dir
                )
                # If artifact_path was a dir, it downloads content. If file, it downloads file.
                # Adjust if download_artifacts creates nested structure
                if not os.path.exists(target_path):
                    # Try to find it if structure differs
                    possible = os.path.join(local_dir, artifact_path)
                    if os.path.exists(possible):
                        target_path = possible
            except Exception as e:
                return None, f"Error downloading artifact: {e}"

    if not os.path.exists(target_path):
        return None, f"Artifact not found at {target_path}"

    try:
        model = YOLO(target_path)
        return model, None
    except Exception as e:
        return None, f"Error loading YOLO model: {e}"


# Load Button
if st.sidebar.button("Load Model"):
    model, err = load_yolo_model(run_id, artifact_path, tracking_uri)
    if err:
        st.error(err)
    else:
        st.session_state["model"] = model
        st.session_state["model_name"] = f"{selected_exp_name} - {selected_run_label}"
        st.success(f"Loaded {st.session_state['model_name']}")

# -------------------------------------------------------------------------
# Main Logic: Inference
# -------------------------------------------------------------------------

if "model" in st.session_state:
    st.markdown(f"### Current Model: **{st.session_state['model_name']}**")

    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05)

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        col1, col2 = st.columns(2)

        # Display Input
        params = {"conf": conf_threshold}

        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # Inference
        model = st.session_state["model"]
        results = model.predict(image, conf=conf_threshold)

        # Plot
        res_plotted = results[0].plot()[:, :, ::-1]  # BGR to RGB

        with col2:
            st.image(res_plotted, caption="Prediction Result", use_container_width=True)

        # Detailed Results
        with st.expander("Detection Details"):
            boxes = results[0].boxes
            if len(boxes) > 0:
                data = []
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = results[0].names[cls_id]
                    data.append(
                        {
                            "Class": label,
                            "Confidence": conf,
                            "Box": box.xywh.tolist()[0],
                        }
                    )
                st.write(pd.DataFrame(data))
            else:
                st.info("No detections found.")

else:
    st.info("👈 Please select and load a model from the sidebar to start.")
