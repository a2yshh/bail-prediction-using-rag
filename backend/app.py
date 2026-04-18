import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "scripts"))
from rag.rag_pipeline import predict_bail

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Bail Prediction Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Bail Prediction Assistant")
st.divider()

# ─────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Case Details")
    case_description = st.text_area(
        "Describe the facts of the case",
        placeholder=(
            "e.g. The accused is charged under IPC Section 302 for murder. "
            "The incident occurred on 12th March 2023 in Agra district. "
            "The accused has no prior criminal record and has been residing "
            "in the same locality for 15 years. The prosecution argues the "
            "accused is a flight risk due to family connections in another state..."
        ),
        height=200
    )

with col2:
    st.subheader("🗂 Additional Info")
    district = st.selectbox(
        "District Court",
        ["agra", "lucknow", "allahabad", "kanpur", "varanasi",
         "meerut", "ghaziabad", "mathura", "aligarh", "other"]
    )
    offence_type = st.selectbox(
        "Nature of Offence",
        ["Murder (IPC 302)", "Robbery (IPC 392)", "Assault (IPC 323)",
         "Drug offence (NDPS)", "Fraud/Cheating (IPC 420)",
         "Sexual offence (IPC 376)", "Kidnapping (IPC 363)", "Other"]
    )
    prior_record = st.radio("Prior Criminal Record?", ["No", "Yes"])
    custody_days = st.number_input("Days in custody so far", min_value=0, value=30)

# ─────────────────────────────────────────
# COMPOSE FULL QUERY
# ─────────────────────────────────────────
def build_user_query(description, district, offence, prior, days):
    return (
        f"District: {district}. Offence: {offence}. "
        f"Prior criminal record: {prior}. Days in custody: {days}. "
        f"Case facts: {description}"
    )

# ─────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────
st.divider()
predict_btn = st.button("🔍 Predict Bail Outcome", type="primary", use_container_width=True)

if predict_btn:
    if not case_description.strip():
        st.warning("Please describe the case facts before predicting.")
    else:
        query = build_user_query(
            case_description, district, offence_type, prior_record, custody_days
        )

        with st.spinner("Retrieving similar cases and analyzing..."):
            result = predict_bail(query)

        st.divider()

        # ── PREDICTION BANNER ──
        pred  = result["llm_prediction"]
        conf  = result["llm_confidence"]
        color = "green" if pred == "GRANTED" else "red"
        emoji = "✅" if pred == "GRANTED" else "❌"

        st.markdown(
            f"<h2 style='color:{color}; text-align:center'>"
            f"{emoji} Bail likely to be: {pred} &nbsp;|&nbsp; Confidence: {conf}"
            f"</h2>",
            unsafe_allow_html=True
        )

        # ── RETRIEVAL SIGNAL ──
        rv_label = result["retrieval_prediction"]
        rv_conf  = result["retrieval_confidence"]
        st.info(
            f"📊 **Retrieval Signal** (majority vote of {len(result['retrieved_chunks'])} "
            f"similar cases): **{rv_label}** ({rv_conf}% of retrieved cases had this outcome)"
        )

        st.divider()
        col_a, col_b = st.columns(2)

        # ── SALIENT SENTENCES ──
        with col_a:
            st.subheader("🔑 Key Sentences That Drove This Decision")
            for i, sentence in enumerate(result["salient_sentences"], 1):
                st.markdown(
                    f"<div style='background:#000000;padding:10px;border-left:4px solid #4a6fa5;"
                    f"margin-bottom:8px;border-radius:4px'>"
                    f"<b>{i}.</b> {sentence}</div>",
                    unsafe_allow_html=True
                )

        # ── EXPLANATION ──
        with col_b:
            st.subheader("📖 Legal Reasoning")
            st.markdown(
                f"<div style='background:#000000;padding:15px;border-radius:8px;'>"
                f"{result['explanation']}</div>",
                unsafe_allow_html=True
            )

        # ── RETRIEVED CASES TABLE ──
        st.divider()
        with st.expander("📂 View Retrieved Similar Cases"):
            for i, chunk in enumerate(result["retrieved_chunks"], 1):
                badge_color = "green" if chunk["label_str"] == "GRANTED" else "red"
                st.markdown(
                    f"**Case {i}** | **Case ID:** {chunk['case_id']} | "
                    f"<span style='color:{badge_color}'><b>{chunk['label_str']}</b></span> | "
                    f"District: {chunk['district']} | "
                    f"Section: {chunk['section']} | "
                    f"Similarity: {round((1 - chunk['distance']) * 100, 1)}%",
                    unsafe_allow_html=True
                )
                st.caption(chunk["text"][:300] + "...")
                st.markdown("---")