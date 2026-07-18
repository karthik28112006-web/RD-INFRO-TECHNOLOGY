# app.py

import streamlit as st

from config import (
    PAGE_CONFIG,
    APP_TITLE,
    APP_SUBTITLE,
    INPUT_PLACEHOLDER,
    MAX_CHARS,
    SENTIMENT_LABELS,
    SENTIMENT_COLORS,
)
from utils import load_or_train_model, predict_sentiment

st.set_page_config(**PAGE_CONFIG)

TEXT_INPUT_KEY = "sentiment_text_input"
RESULT_KEY = "sentiment_result"


@st.cache_resource
def get_model_bundle(force_retrain: bool = False):
    return load_or_train_model(force_retrain=force_retrain)


def render_result(result: dict) -> None:
    label = result["label"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]

    display_label = SENTIMENT_LABELS.get(label, label.title())
    color = SENTIMENT_COLORS.get(label, "#FFFFFF")

    st.markdown(
        f"""
        <div style="padding:1rem;border-radius:0.5rem;border:1px solid {color};background-color:{color}22;">
            <h3 style="color:{color};margin:0;">{display_label}</h3>
            <p style="margin:0;">Confidence: {confidence:.2%}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Class Probabilities")
    for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        cls_display = SENTIMENT_LABELS.get(cls, cls.title())
        st.write(f"{cls_display}")
        st.progress(min(max(prob, 0.0), 1.0))


def render_data_source_info(meta: dict) -> None:
    with st.expander("ℹ️ Model & data source"):
        if meta.get("live_data"):
            st.success(f"Trained on live fetched data ({meta['source']}) — {meta['n_samples']} samples.")
        else:
            st.warning(f"Offline fallback data used — {meta['n_samples']} samples. "
                       "Check network access to fetch a larger live dataset.")
            if meta.get("error"):
                st.caption(f"Reason: {meta['error']}")

        if st.button("🔄 Retrain from live data"):
            get_model_bundle.clear()
            st.session_state["force_retrain"] = True
            st.rerun()


def _clear_input() -> None:
    """Callback: reset text area and any previous result before rerun."""
    st.session_state[TEXT_INPUT_KEY] = ""
    st.session_state[RESULT_KEY] = None


def main() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    force_retrain = st.session_state.pop("force_retrain", False)
    bundle = get_model_bundle(force_retrain=force_retrain)
    model = bundle["pipeline"]

    render_data_source_info(bundle["meta"])

    if TEXT_INPUT_KEY not in st.session_state:
        st.session_state[TEXT_INPUT_KEY] = ""
    if RESULT_KEY not in st.session_state:
        st.session_state[RESULT_KEY] = None

    st.text_area(
        label="Enter text to analyze",
        placeholder=INPUT_PLACEHOLDER,
        max_chars=MAX_CHARS,
        height=180,
        key=TEXT_INPUT_KEY,
    )

    col_analyze, col_clear = st.columns([3, 1])

    with col_analyze:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

    with col_clear:
        st.button("🗑️ Clear", use_container_width=True, on_click=_clear_input)

    if analyze_clicked:
        current_text = st.session_state[TEXT_INPUT_KEY]
        if not current_text or not current_text.strip():
            st.warning("Please enter some text before analyzing.")
            st.session_state[RESULT_KEY] = None
        else:
            with st.spinner("Analyzing sentiment..."):
                result = predict_sentiment(model, current_text)
            st.session_state[RESULT_KEY] = result

    if st.session_state[RESULT_KEY] is not None:
        render_result(st.session_state[RESULT_KEY])


if __name__ == "__main__":
    main()