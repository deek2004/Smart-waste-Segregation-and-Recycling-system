# ============================================================
#  Smart Waste Segregation & Recycling System
#  Full Version: Classifier + NLP Chatbot + Analytics + Impact
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
# NEW — replace with this
from google import genai

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Waste Segregation",
    page_icon="♻️",
    layout="centered",
)

# ── Waste Info ────────────────────────────────────────────────
WASTE_INFO = {
    "organic": {
        "label": "Organic Waste", "emoji": "🌱",
        "score": 30, "score_label": "Low Recyclability",
        "color": "#1D9E75", "bg": "#E1F5EE",
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
        "label": "Plastic Waste", "emoji": "🧴",
        "score": 65, "score_label": "Moderate Recyclability",
        "color": "#378ADD", "bg": "#E6F1FB",
        "facts": [
            "Takes 400–1000 years to decompose naturally",
            "Only plastics marked ♳ ♴ ♷ are widely recyclable",
            "Must be clean and dry before recycling",
        ],
        "disposal": [
            "🔢  Check the resin code (number inside triangle)",
            "🚿  Rinse containers before placing in blue bin",
            "🛍️  Soft plastics go to supermarket drop-offs",
            "❌  Never recycle greasy or food-soiled plastic",
        ],
        "tip": "Recycling one plastic bottle saves enough energy to power a 60W bulb for 6 hours.",
    },
    "paper": {
        "label": "Paper Waste", "emoji": "📄",
        "score": 85, "score_label": "High Recyclability",
        "color": "#BA7517", "bg": "#FAEEDA",
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
        "label": "Metal Waste", "emoji": "🥫",
        "score": 95, "score_label": "Excellent Recyclability",
        "color": "#5F5E5A", "bg": "#F1EFE8",
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

# ── Environmental Impact Data ─────────────────────────────────
IMPACT = {
    "organic":  {"co2": 0.5,  "water": 10,  "energy": 2.0},
    "plastic":  {"co2": 1.5,  "water": 100, "energy": 5.7},
    "paper":    {"co2": 1.0,  "water": 26,  "energy": 4.1},
    "metal":    {"co2": 4.0,  "water": 40,  "energy": 14.0},
}

# ── Session State ─────────────────────────────────────────────
if "scan_log" not in st.session_state:
    st.session_state.scan_log = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/waste_model.h5")

CLASS_NAMES = ["organic", "plastic", "paper", "metal"]

def predict(image, model):
    img = image.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    return CLASS_NAMES[int(np.argmax(probs))], probs

def score_bar(score, color):
    st.markdown(f"""
    <div style="margin:8px 0 16px;">
      <div style="display:flex;justify-content:space-between;
                  font-size:13px;margin-bottom:4px;">
        <span style="color:#666;">Recyclability Score</span>
        <span style="font-weight:600;color:{color};">{score}/100</span>
      </div>
      <div style="background:#eee;border-radius:8px;height:12px;">
        <div style="width:{score}%;background:{color};
                    border-radius:8px;height:12px;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
st.title("♻️ Smart Waste Segregation")
st.markdown(
    "AI-powered waste classification with recyclability scoring, "
    "disposal guides, NLP chatbot, and real-time analytics."
)
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Classify", "📊 Analytics", "🤖 Chatbot"])

# ══════════════════════════════════════════════════════════════
#  TAB 1 — CLASSIFIER
# ══════════════════════════════════════════════════════════════
with tab1:
    with st.spinner("Loading AI model..."):
        try:
            model = load_model()
            st.success("Model loaded!", icon="✅")
        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.stop()

    st.subheader("📸 Upload Waste Image")
    uploaded = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded:
        image = Image.open(uploaded)
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(image, caption="Uploaded image", width=300)

        with col2:
            with st.spinner("Analysing waste..."):
                label, probs = predict(image, model)
                st.session_state.scan_log.append(label)
                info = WASTE_INFO[label]

            st.markdown(f"""
            <div style="background:{info['bg']};border-radius:12px;
                        padding:16px 20px;margin-bottom:16px;
                        border-left:4px solid {info['color']};">
              <div style="font-size:32px;margin-bottom:4px;">{info['emoji']}</div>
              <div style="font-size:22px;font-weight:700;
                          color:{info['color']};">{info['label']}</div>
              <div style="font-size:13px;color:#666;margin-top:2px;">
                  {info['score_label']}
              </div>
            </div>
            """, unsafe_allow_html=True)

            score_bar(info["score"], info["color"])

            st.markdown("**Confidence scores:**")
            for i, cls in enumerate(CLASS_NAMES):
                pct = float(probs[i]) * 100
                w = WASTE_INFO[cls]
                st.markdown(f"""
                <div style="display:flex;align-items:center;
                            gap:8px;margin-bottom:6px;font-size:13px;">
                  <span style="width:60px;">{w['emoji']} {cls.capitalize()}</span>
                  <div style="flex:1;background:#eee;
                              border-radius:6px;height:8px;">
                    <div style="width:{pct:.1f}%;background:{w['color']};
                                border-radius:6px;height:8px;"></div>
                  </div>
                  <span style="width:40px;text-align:right;
                               color:#555;">{pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        col3, col4 = st.columns(2, gap="large")

        with col3:
            st.subheader("📋 Disposal Guide")
            for step in info["disposal"]:
                st.markdown(f"- {step}")

        with col4:
            st.subheader("💡 Did You Know?")
            for fact in info["facts"]:
                st.markdown(f"- {fact}")

        st.info(f"🌍 **Eco Tip:** {info['tip']}")

        # ── Impact Calculator ─────────────────────────────────
        st.divider()
        st.subheader("🌍 Environmental Impact")
        impact = IMPACT[label]
        i1, i2, i3 = st.columns(3)
        i1.metric("CO₂ Saved", f"{impact['co2']} kg")
        i2.metric("Water Saved", f"{impact['water']} L")
        i3.metric("Energy Saved", f"{impact['energy']} kWh")
        st.caption("Impact per item recycled correctly")

        st.markdown("**If 1,000 people recycled this item correctly:**")
        c1, c2, c3 = st.columns(3)
        c1.metric("CO₂ Reduction", f"{impact['co2']*1000:,.0f} kg")
        c2.metric("Water Saved",   f"{impact['water']*1000:,.0f} L")
        c3.metric("Energy Saved",  f"{impact['energy']*1000:,.0f} kWh")

        st.divider()
        st.subheader("♻️ All Waste Categories")
        c1, c2, c3, c4 = st.columns(4)
        for col, cls in zip([c1, c2, c3, c4], CLASS_NAMES):
            w = WASTE_INFO[cls]
            col.markdown(f"""
            <div style="background:{w['bg']};border-radius:10px;
                        padding:12px;text-align:center;
                        border:1px solid {w['color']}33;">
              <div style="font-size:24px;">{w['emoji']}</div>
              <div style="font-size:12px;font-weight:600;
                          color:{w['color']};margin-top:4px;">
                  {w['label']}
              </div>
              <div style="font-size:18px;font-weight:700;
                          color:{w['color']};">{w['score']}</div>
              <div style="font-size:10px;color:#888;">/ 100</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:48px 24px;
                    background:#f8f9fa;border-radius:16px;
                    border:2px dashed #dee2e6;">
          <div style="font-size:48px;">📷</div>
          <div style="font-size:18px;font-weight:600;
                      margin:12px 0 8px;">No image uploaded yet</div>
          <div style="color:#666;font-size:14px;">
              Upload a JPG or PNG of any waste item to get started
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TAB 2 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Session Analytics Dashboard")

    if not st.session_state.scan_log:
        st.info("Upload some waste images in the Classify tab to see analytics here.")
    else:
        log = st.session_state.scan_log
        total = len(log)

        avg_score = sum(WASTE_INFO[l]["score"] for l in log) // total
        most_common = max(set(log), key=log.count)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Scans", total)
        m2.metric("Most Common", WASTE_INFO[most_common]["emoji"] + " " + most_common.capitalize())
        m3.metric("Avg Recyclability", f"{avg_score}/100")

        st.divider()

        from collections import Counter
        counts = Counter(log)
        categories  = [WASTE_INFO[c]["label"] for c in CLASS_NAMES if c in counts]
        values      = [counts[c] for c in CLASS_NAMES if c in counts]
        colors      = [WASTE_INFO[c]["color"] for c in CLASS_NAMES if c in counts]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Waste Distribution**")
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.pie(values, labels=categories, autopct="%1.0f%%",
                    colors=colors, startangle=90)
            ax1.set_title("By Category", fontweight="bold")
            st.pyplot(fig1)
            plt.close()

        with col2:
            st.markdown("**Scan Count by Category**")
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.bar(categories, values, color=colors)
            ax2.set_ylabel("Number of Scans")
            ax2.set_title("Scan Frequency", fontweight="bold")
            plt.xticks(rotation=15, fontsize=9)
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close()

        st.divider()
        st.markdown("**Recyclability Score per Scan**")
        scores = [WASTE_INFO[l]["score"] for l in log]
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        ax3.plot(range(1, total+1), scores, marker="o",
                 color="#1D9E75", linewidth=2, markersize=6)
        ax3.axhline(y=avg_score, color="#D85A30", linestyle="--",
                    alpha=0.7, label=f"Average: {avg_score}")
        ax3.set_xlabel("Scan Number")
        ax3.set_ylabel("Recyclability Score")
        ax3.set_ylim([0, 100])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close()

        st.divider()
        st.markdown("**Cumulative Environmental Impact of Your Scans**")
        total_co2    = sum(IMPACT[l]["co2"]    for l in log)
        total_water  = sum(IMPACT[l]["water"]  for l in log)
        total_energy = sum(IMPACT[l]["energy"] for l in log)

        e1, e2, e3 = st.columns(3)
        e1.metric("Total CO₂ Saved",    f"{total_co2:.1f} kg")
        e2.metric("Total Water Saved",  f"{total_water:.0f} L")
        e3.metric("Total Energy Saved", f"{total_energy:.1f} kWh")

        if st.button("🔄 Clear Session Data"):
            st.session_state.scan_log = []
            st.rerun()

# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
#  TAB 3 — NLP CHATBOT (GEMINI)
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🤖 AI Recycling Assistant")
    st.markdown(
        "Ask me anything about waste disposal, recycling rules, "
        "or sustainability. Powered by **Google Gemini**."
    )

    if not GEMINI_API_KEY:
        st.warning(
            "⚠️ Gemini API key not configured. "
            "Add `GEMINI_API_KEY` to Streamlit secrets to enable this feature.",
            icon="🔑",
        )
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_question = st.chat_input("e.g. Can I recycle a greasy pizza box?")

        if user_question:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Gemini is thinking..."):
                    try:
                        prompt_context = (
                            "You are an expert waste management and recycling assistant. "
                            "Answer the following question clearly and concisely in under 100 words. "
                            "Use bullet points for clarity when listing steps or options. "
                            "Only answer questions related to waste, recycling, composting, "
                            "or environmental sustainability. "
                            "Question: "
                        )
                        response = gemini_client.models.generate_content(
                            model="gemini-1.5-flash",
                            contents=prompt_context + user_question,
                        )
                        answer = response.text
                        st.write(answer)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer}
                        )
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            st.error(
                                "⏳ The AI assistant is temporarily unavailable "
                                "due to rate limits. Please try again in a minute."
                            )
                        else:
                            st.error(f"Gemini error: {e}")

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:12px;'>"
    "Smart Waste Segregation System · TensorFlow + Streamlit · "
    "MobileNetV2 Transfer Learning"
    "</div>",
    unsafe_allow_html=True,
)