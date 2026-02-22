import streamlit as st
from PIL import Image

from heatmap import (
    analyze_image_detailed,
    load_detector_model,
)


st.set_page_config(page_title="NexJam Vision Lab", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;700;800&family=Manrope:wght@400;600;700&display=swap');

:root {
  --bg0: #06101b;
  --bg1: #0d2237;
  --card: rgba(9, 19, 32, 0.72);
  --stroke: rgba(134, 176, 218, 0.26);
  --text: #e8f2ff;
  --muted: #acc7e5;
  --ok: #36d89a;
  --warn: #ff6f8e;
  --accent: #57b7ff;
}

.stApp {
  background:
    radial-gradient(1000px 500px at 0% -20%, #1f4c79 0%, transparent 60%),
    radial-gradient(900px 500px at 100% 0%, #1a5f4a 0%, transparent 62%),
    linear-gradient(180deg, var(--bg1) 0%, var(--bg0) 100%);
  color: var(--text);
}

[class*="css"] {
  font-family: "Manrope", sans-serif;
}

.hero {
  border: 1px solid var(--stroke);
  border-radius: 20px;
  padding: 24px 24px 18px 24px;
  background: linear-gradient(135deg, rgba(20,52,84,0.88), rgba(10,31,55,0.86));
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
  margin-bottom: 14px;
}

.hero h1 {
  margin: 0;
  font-family: "Syne", sans-serif;
  font-size: clamp(1.6rem, 3vw, 2.4rem);
  letter-spacing: 0.2px;
}

.hero p {
  margin: 6px 0 0 0;
  color: var(--muted);
}

.chip {
  display: inline-block;
  margin-top: 10px;
  padding: 6px 10px;
  border: 1px solid var(--stroke);
  border-radius: 999px;
  font-size: 0.83rem;
  color: #c8dcf2;
  background: rgba(8, 25, 46, 0.5);
}

.panel {
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 14px;
  background: var(--card);
}

.big-label {
  font-family: "Syne", sans-serif;
  font-size: 1.6rem;
  margin-bottom: 6px;
}

.pill {
  display: inline-block;
  border-radius: 999px;
  padding: 6px 12px;
  font-weight: 700;
  font-size: 0.9rem;
}

.pill-real {
  border: 1px solid rgba(72, 224, 157, 0.7);
  background: rgba(28, 155, 104, 0.2);
  color: #90ffd0;
}

.pill-fake {
  border: 1px solid rgba(255, 130, 150, 0.8);
  background: rgba(174, 41, 75, 0.25);
  color: #ffd6df;
}

.tiny {
  font-size: 0.85rem;
  color: var(--muted);
}

@media (max-width: 900px) {
  .hero {
    padding: 18px 16px 14px 16px;
  }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<section class="hero">
  <h1>NexJam Vision Lab</h1>
  <p>Face authenticity screening with explainable AI overlays.</p>
  <span class="chip">Binary classifier + visual attention maps</span>
</section>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Controls")
    st.caption("Upload one image and inspect decision focus.")
    st.markdown("### Model")
    st.code("ai_detector_model.h5")


@st.cache_resource
def get_model():
    return load_detector_model("ai_detector_model.h5")


try:
    model = get_model()
except Exception as exc:
    st.error(f"Model load failed: {exc}")
    st.stop()

uploaded = st.file_uploader(
    "Upload face image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Accepted formats: JPG, JPEG, PNG, WEBP",
)

if uploaded is None:
    st.markdown('<div class="panel tiny">No image uploaded yet.</div>', unsafe_allow_html=True)
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception as exc:
    st.error(f"Invalid image: {exc}")
    st.stop()

with st.spinner("Analyzing image..."):
    result = analyze_image_detailed(model, image)

label = str(result["label"])
confidence = float(result["confidence"])
score = float(result["score"])
class0 = str(result["class0"])
class1 = str(result["class1"])
threshold = float(result.get("threshold", 0.5))
img_size = result.get("img_size", [128, 128])
original_rgb = result["original_rgb"]
heatmap_rgb = result["heatmap_rgb"]
overlay_rgb = result["overlay_rgb"]
headline = str(result["explain_headline"])
bullets = result["explain_bullets"]
stats = result["attention_stats"]

pill_class = "pill-real" if label.upper() == "REAL" else "pill-fake"

top_left, top_right = st.columns([1.2, 1], gap="large")
with top_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="big-label">Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="pill {pill_class}">{label}</span>',
        unsafe_allow_html=True,
    )
    st.metric("Confidence", f"{confidence * 100:.2f}%")
    st.progress(min(max(confidence, 0.0), 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

with top_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Inference Telemetry**")
    st.write(f"Sigmoid score: `{score:.4f}`")
    st.write(f"Class 0: `{class0}`")
    st.write(f"Class 1: `{class1}`")
    st.write(f"Threshold: `{threshold:.2f}`")
    st.write(f"Input size: `{img_size[0]}x{img_size[1]}`")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("### Why this result")
st.write(headline)
for item in bullets:
    st.write(f"- {item}")
st.caption(
    "Interpretability note: this explanation is based on model attention heatmaps and is supportive evidence, not a proof of authenticity."
)
st.caption(
    f"Attention metrics: center-focus={stats['center_focus_ratio']:.2f}, "
    f"edge-focus={stats['edge_focus_ratio']:.2f}, hotspot-ratio={stats['hotspot_ratio']:.2f}"
)
st.markdown("</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Input", "Attention", "Composite"])
with tab1:
    st.image(original_rgb, caption="Model input (128x128)", use_container_width=True)
with tab2:
    st.image(heatmap_rgb, caption="Attention heatmap", use_container_width=True)
with tab3:
    st.image(overlay_rgb, caption="Heatmap overlay", use_container_width=True)
