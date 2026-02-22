import streamlit as st
from PIL import Image

from heatmap import analyze_image_detailed, load_detector_model


st.set_page_config(page_title="AI Image Detector", layout="wide")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 20% 0%, #10243f 0%, #081322 35%, #050b14 100%);
}

.hero {
    background: linear-gradient(135deg, rgba(14,39,70,0.95), rgba(18,58,46,0.88));
    border: 1px solid rgba(132, 212, 255, 0.25);
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 16px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

.hero h1 {
    font-size: 2.1rem;
    margin: 0;
    letter-spacing: 0.3px;
}

.hero p {
    margin-top: 6px;
    color: #bbd4ec;
}

.panel {
    background: rgba(11, 24, 43, 0.8);
    border: 1px solid rgba(133, 153, 176, 0.24);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 14px;
}

.label-pill {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
}

.label-real {
    background: rgba(32, 181, 122, 0.2);
    border: 1px solid rgba(32, 181, 122, 0.6);
    color: #74f8bf;
}

.label-fake {
    background: rgba(255, 92, 115, 0.22);
    border: 1px solid rgba(255, 107, 127, 0.7);
    color: #ffd4da;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>AI Image Detector + Heatmap</h1>
  <p>Upload an image to classify it as REAL or FAKE and inspect model attention maps.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Workflow")
    st.markdown("1. Upload a face image")
    st.markdown("2. Review prediction + confidence")
    st.markdown("3. Inspect heatmap and overlay")
    st.markdown("---")
    st.caption("Model file: `ai_detector_model.h5`")


@st.cache_resource
def get_model():
    return load_detector_model("ai_detector_model.h5")


try:
    model = get_model()
except Exception as exc:
    st.error(f"Could not load model file 'ai_detector_model.h5': {exc}")
    st.stop()


uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"Invalid image file: {exc}")
        st.stop()

    with st.spinner("Running prediction and heatmap generation..."):
        result = analyze_image_detailed(model, image)

    label = str(result["label"])
    confidence = float(result["confidence"])
    original_rgb = result["original_rgb"]
    heatmap_rgb = result["heatmap_rgb"]
    overlay_rgb = result["overlay_rgb"]
    raw_score = float(result["score"])
    class0 = str(result["class0"])
    class1 = str(result["class1"])
    threshold = float(result.get("threshold", 0.5))
    img_size = result.get("img_size", [128, 128])

    pct = confidence * 100
    class_name = "label-real" if label.upper() == "REAL" else "label-fake"

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            f'Prediction: <span class="label-pill {class_name}">{label}</span>',
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{pct:.2f}%")
        st.progress(min(max(confidence, 0.0), 1.0))
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Inference Details**")
        st.write(f"Sigmoid score: `{raw_score:.4f}`")
        st.write(f"Class 0: `{class0}`")
        st.write(f"Class 1: `{class1}`")
        st.write(f"Threshold: `{threshold:.2f}`")
        st.write(f"Input size: `{img_size[0]}x{img_size[1]}`")
        st.markdown("</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Original", "Heatmap", "Overlay"])
    with tab1:
        st.image(original_rgb, caption="Preprocessed image (128x128)", use_container_width=True)
    with tab2:
        st.image(heatmap_rgb, caption="Model attention heatmap", use_container_width=True)
    with tab3:
        st.image(overlay_rgb, caption="Heatmap blended with image", use_container_width=True)
else:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.info("Upload an image to start analysis.")
    st.markdown("</div>", unsafe_allow_html=True)
