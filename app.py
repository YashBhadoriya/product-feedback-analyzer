import streamlit as st
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import os
from openai import OpenAI

# Setup
openai_api_key = os.getenv("API_KEY")
client = OpenAI(api_key=openai_api_key)

st.set_page_config(page_title="Product Feedback Analyzer", layout="wide")
st.title("🧠 AI-powered Product Feedback Analyzer")

# File upload or text input
upload_file = st.file_uploader("📄 Upload feedback (.csv or .txt)", type=["csv", "txt"])
text_input = st.text_area("📝 Or paste your feedback here (one per line)")

# Load feedback
def load_feedback(upload_file, text_input):
    if upload_file is not None:
        if upload_file.name.endswith('.csv'):
            df = pd.read_csv(upload_file)
            feedbacks = df.iloc[:, 0].dropna().astype(str).tolist()
        else:
            content = upload_file.read().decode("utf-8")
            feedbacks = [line.strip() for line in content.splitlines() if line.strip()]
    else:
        feedbacks = [line.strip() for line in text_input.splitlines() if line.strip()]
    return feedbacks

# Sentiment analysis
def analyze_sentiment(feedback):
    analysis = TextBlob(feedback)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# GPT summary
def generate_gpt_summary(feedback_list):
    prompt = (
        "You are a helpful product manager assistant. Summarize the key pain points and positive feedback from these user comments:\n\n"
        + "\n".join(feedback_list[:50])  # limit for token safety
        + "\n\nProvide a concise summary."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error generating summary: {e}"

# Main analysis logic
if st.button("▶️ Analyze Feedback"):
    feedbacks = load_feedback(upload_file, text_input)

    if not feedbacks:
        st.error("❗ Please upload or paste some feedback data!")
        st.stop()

    # Sentiment processing
    sentiments = [analyze_sentiment(f) for f in feedbacks]
    df = pd.DataFrame({"Feedback": feedbacks, "Sentiment": sentiments})

    # Pie chart
    st.subheader("📊 Sentiment Distribution")
    fig, ax = plt.subplots()
    df['Sentiment'].value_counts().plot.pie(
        autopct='%1.1f%%',
        ax=ax,
        colors=['green', 'red', 'grey']
    )
    ax.set_ylabel('')
    st.pyplot(fig)

    # Data Table
    st.subheader("📋 Feedback with Sentiment")
    st.dataframe(df)

    # GPT Summary
    st.subheader("🧠 GPT-generated Summary")
    with st.spinner("Generating summary using GPT..."):
        summary = generate_gpt_summary(feedbacks)
    st.markdown(summary)

    # Download analyzed feedback
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Analyzed Feedback CSV",
        data=csv,
        file_name="analyzed_feedback.csv",
        mime="text/csv"
    )
