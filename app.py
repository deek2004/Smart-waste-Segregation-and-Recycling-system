# ============================================================
#  Smart Waste Segregation & Recycling System
#  Phase 2 + 3: Streamlit App with Scores & Disposal Guides
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Waste Segregation",
    page_icon="♻️",
    layout="centered",
)

# ── Phase 3: Recyclability Data ───────────────────────────────
WASTE_INFO = {
    "organic": {
        "label":       "Organic Waste",
        "emoji":       "🌱",
        "score":       30,
        "score_label": "Low Recyclability",
        "color":       "#1D9E75",
        "bg":          "#E1F5EE",
        "facts": [
            "Breaks down naturally within weeks",
            "Cannot enter standard recycling streams",
            "Rich source of nutrients when composted",
        ],
        "disposal": [
            "🪣  Place in your green compost bin",
            "🌿  Home composting bin or pile",
            "🏙️  Drop at community composting facility",
            "❌  Never mix with dry recyclables",
        ],
        "tip": "Composting organic waste reduces methane emissions from landfills by up to 50%.",
    },
    "plastic": {
        "label":       "Plastic Waste",
        "emoji":       "🧴",
        "score":       65,
        "score_label": "Moderate Recyclability",
        "color":       "#378ADD",
        "bg":          "#E6F1FB",
        "facts": [
            "Takes 400–1000 years to decompose naturally",
            "Only plastics marked ♳ ♴ ♷ are widely recyclable",
            "Must be clean and dry before recycling",
        ],
        "disposal": [
            "🔢  Check the resin code (number inside triangle)",
            "🚿  Rinse containers before placing in blue bin",
            "🛍️  Soft plastics (bags) go to supermarket drop-offs",
            "❌  Never recycle greasy or food-soiled plastic",
        ],
        "tip": "Recycling one plastic bottle saves enough energy to power a 60W bulb for 6 hours.",
    },
    "paper": {
        "label":       "Paper Waste",
        "emoji":       "📄",
        "score":       85,
        "score_label": "High Recyclability",
        "color":       "#BA7517",
        "bg":          "#FAEEDA",
        "facts": [
            "Can be recycled 5–7 times before fibres degrade",
            "Recycling paper uses 70% less energy than virgin production",
            "Includes cardboard, newspapers, office paper, magazines",
        ],
        "disposal": [
            "📦  Flatten cardboard boxes before placing in bin",
            "📰  Keep paper dry — wet paper cannot be recycled",
            "🗂️  Remove plastic windows from envelopes first",
            "❌  Avoid recycling greasy pizza boxes or tissue paper",
        ],
        "tip": "Recycling one tonne of paper saves 17 trees and 26,000 litres of water.",
    },
    "metal": {
        "label":       "Metal Waste",
        "emoji":       "🥫",
        "score":       95,
        "score_label": "Excellent Recyclability",
        "color":       "#5F5E5A",
        "bg":          "#F1EFE8",
        "facts": [
            "Metals can be recycled infinitely without quality loss",
            "Aluminium recycling uses 95% less energy than new production",
            "Steel is the world's most recycled material",
        ],
        "disposal": [
            "🥤  Rinse cans and tins before recycling",
            "♻️  Place loose in your recycling bin — not bagged",
            "🔋  Take batteries to dedicated battery drop-off points",
            "🏗️  Large metal items go to a scrap metal dealer",
        ],
        "tip": "A recycled aluminium can is back on the shelf as a new can within 60 days.",
    },
}

# ── Load Model (cached) ───────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/waste_model.h5")

# ── Prediction Function ───────────────────────────────────────
CLASS_NAMES = ["organic", "plastic", "paper", "metal"]

def predict(image: Image.Image, model):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx], probs

# ── Score Bar ─────────────────────────────────────────────────
def score_bar(score, color):
    st.markdown(f"""
    <div style="margin: 8px 0 16px;">
      <div style="display:flex; justify-content:space-between;
                  font-size:13px; margin-bottom:4px;">
        <span style="color:#666;">Recyclability Score</span>
        <span style="font-weight:600; color:{color};">{score}/100</span>
      </div>
      <div style="background:#eee; border-radius:8px; height:12px;">
        <div style="width:{score}%; background:{color};
                    border-radius:8px; height:12px;
                    transition:width 0.5s ease;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  UI LAYOUT
# ══════════════════════════════════════════════════════════════

st.title("♻️ Smart Waste Segregation")
st.markdown(
    "Upload a photo of any waste item and the AI will classify it, "
    "score its recyclability, and give you a disposal guide."
)
st.divider()

# ── Load model ───────────────────────────────────────────────
with st.spinner("Loading AI model..."):
    try:
        model = load_model()
        st.success("Model loaded!", icon="✅")
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.info("Make sure `models/waste_model.h5` exists in your project folder.")
        st.stop()

# ── Upload ───────────────────────────────────────────────────
st.subheader("📸 Upload Waste Image")
uploaded = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported: JPG, PNG, WEBP",
)

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Analysing waste..."):
            label, probs = predict(image, model)
            info = WASTE_INFO[label]

        # ── Prediction Header ─────────────────────────────
        st.markdown(f"""
        <div style="background:{info['bg']}; border-radius:12px;
                    padding:16px 20px; margin-bottom:16px;
                    border-left: 4px solid {info['color']};">
          <div style="font-size:32px; margin-bottom:4px;">{info['emoji']}</div>
          <div style="font-size:22px; font-weight:700;
                      color:{info['color']};">{info['label']}</div>
          <div style="font-size:13px; color:#666; margin-top:2px;">
              {info['score_label']}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Score Bar ─────────────────────────────────────
        score_bar(info["score"], info["color"])

        # ── Confidence Scores ─────────────────────────────
        st.markdown("**Confidence scores:**")
        for i, cls in enumerate(CLASS_NAMES):
            pct = float(probs[i]) * 100
            w   = WASTE_INFO[cls]
            st.markdown(f"""
            <div style="display:flex; align-items:center;
                        gap:8px; margin-bottom:6px; font-size:13px;">
              <span style="width:60px;">{w['emoji']} {cls.capitalize()}</span>
              <div style="flex:1; background:#eee;
                          border-radius:6px; height:8px;">
                <div style="width:{pct:.1f}%; background:{w['color']};
                            border-radius:6px; height:8px;"></div>
              </div>
              <span style="width:40px; text-align:right;
                           color:#555;">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Details Section ───────────────────────────────────
    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.subheader("📋 Disposal Guide")
        for step in info["disposal"]:
            st.markdown(f"- {step}")

    with col4:
        st.subheader("💡 Did You Know?")
        for fact in info["facts"]:
            st.markdown(f"- {fact}")

    # ── Eco Tip ───────────────────────────────────────────
    st.info(f"🌍 **Eco Tip:** {info['tip']}")

    # ── All Categories Reference ──────────────────────────
    st.divider()
    st.subheader("♻️ All Waste Categories")
    c1, c2, c3, c4 = st.columns(4)
    for col, cls in zip([c1, c2, c3, c4], CLASS_NAMES):
        w = WASTE_INFO[cls]
        col.markdown(f"""
        <div style="background:{w['bg']}; border-radius:10px;
                    padding:12px; text-align:center;
                    border: 1px solid {w['color']}33;">
          <div style="font-size:24px;">{w['emoji']}</div>
          <div style="font-size:12px; font-weight:600;
                      color:{w['color']}; margin-top:4px;">
              {w['label']}
          </div>
          <div style="font-size:18px; font-weight:700;
                      color:{w['color']};">{w['score']}</div>
          <div style="font-size:10px; color:#888;">/ 100</div>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Empty State ───────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:48px 24px;
                background:#f8f9fa; border-radius:16px;
                border: 2px dashed #dee2e6;">
      <div style="font-size:48px;">📷</div>
      <div style="font-size:18px; font-weight:600;
                  margin:12px 0 8px;">No image uploaded yet</div>
      <div style="color:#666; font-size:14px;">
          Upload a JPG or PNG of any waste item to get started
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:12px;'>"
    "Smart Waste Segregation System · Built with TensorFlow & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)