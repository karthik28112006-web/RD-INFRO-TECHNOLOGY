# app.py
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from utils import load_trained_model, preprocess_canvas_image

st.set_page_config(page_title="Digit Recognizer", layout="centered")


@st.cache_resource
def get_model():
    return load_trained_model("mnist_model.h5")


def main():
    st.title("Handwritten Digit Recognition")
    model = get_model()

    col1, col2 = st.columns(2)

    with col1:
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=15,
            stroke_color="white",
            background_color="black",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )

    with col2:
        predict_clicked = st.button("Predict")

        if predict_clicked:
            processed = preprocess_canvas_image(
                canvas_result.image_data if canvas_result is not None else None
            )

            if processed is None:
                st.warning("Please draw a digit before predicting.")
            else:
                predictions = model.predict(processed)[0]
                pred_digit = int(np.argmax(predictions))
                confidence = float(np.max(predictions))

                st.subheader(f"Predicted Digit: {pred_digit}")
                st.write(f"Confidence: {confidence:.2%}")
                st.bar_chart({"Confidence": predictions})


if __name__ == "__main__":
    main()
