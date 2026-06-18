import streamlit as st
from model import predict_review 

st.markdown(
    """
    <style>
    div[data-testid="stTextAreaRootElement"] {
        border: 1px solid #444 !important;
        border-radius: 12px !important;
    }

    div[data-testid="stTextAreaRootElement"]:focus-within {
        border: 2px solid #22c55e !important;
        box-shadow: 0 0 0 2px rgba(34,197,94,0.35) !important;
    }

    div[data-testid="stTextAreaRootElement"] textarea:focus,
    div[data-testid="stTextAreaRootElement"] textarea:focus-visible {
        outline: none !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Fake Review Detector")
st.write(
    "Paste a review below and the AI model will classify it."
)

with st.form("review_form"):
    review = st.text_area(
        "Enter a review"
    )

    submitted = st.form_submit_button("Analyze")

if submitted:

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        prediction, confidence = predict_review(review)

        if prediction == 1:
            st.success("Prediction: Label 1")
        else:
             st.info("Prediction: Label 0")

        st.metric(
            "Confidence",
             f"{confidence * 100:.2f}%"
        )
