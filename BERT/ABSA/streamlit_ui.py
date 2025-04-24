import streamlit as st
import pandas as pd
from pyabsa import ATEPCCheckpointManager

# Load the ABSA model once and cache it
@st.cache_resource
def load_model():
    checkpoint = 'english'  # Or 'multilingual', etc.
    return ATEPCCheckpointManager.get_aspect_extractor(checkpoint=checkpoint)

model = load_model()

# ABSA Wrapper Class
class ABSA_Transformers_PyABSA:
    def __init__(self, model):
        self.model = model

    def process_batch(self, texts):
        results = []
        outputs = self.model.extract_aspect(inference_source=texts, pred_sentiment=True)

        for output in outputs:
            sentence = output.get('sentence', '')
            aspects = output.get('aspect', [])
            sentiments = output.get('sentiment', [])

            sentiment_dict = {
                aspect: sentiment
                for aspect, sentiment in zip(aspects, sentiments)
            }

            results.append({
                'Text': sentence,
                'Aspects': ', '.join(aspects),
                'Sentiments': ', '.join([f"{asp}: {sent}" for asp, sent in sentiment_dict.items()])
            })

        return pd.DataFrame(results)

# Streamlit UI
st.title("🔍 Aspect-Based Sentiment Analysis (ABSA)")
st.markdown("Enter one or more reviews (each on a new line):")

user_input = st.text_area("Input Reviews", height=200)

if st.button("Analyze"):
    reviews = [line.strip() for line in user_input.split("\n") if line.strip()]
    
    if reviews:
        absa_pipeline = ABSA_Transformers_PyABSA(model)
        df = absa_pipeline.process_batch(reviews)

        st.subheader("🧾 Analysis Result")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("⚠️ Please enter at least one review.")
