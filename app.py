import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

sentiment_model = pipeline('sentiment-analysis')

def analyze_sentiment(text):
    result = sentiment_model(text)
    return result[0]['label'], result[0]['score']

def plot_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()

st.title('Sentiment Analysis Dashboard')

st.subheader('Enter your text:')
input_text = st.text_area("Text input", height=150)

if st.button('Analyze Sentiment'):
    if input_text:
        sentiment, confidence = analyze_sentiment(input_text)
        st.write(f"Sentiment: **{sentiment}**")
        st.write(f"Confidence Score: **{confidence:.2f}**")

        sentiment_data = {'Sentiment': [sentiment], 'Confidence': [confidence]}
        sentiment_df = pd.DataFrame(sentiment_data)

        fig = px.bar(sentiment_df, x='Sentiment', y='Confidence', title='Sentiment Analysis Confidence')
        st.plotly_chart(fig)

        sentiment_counts = {'Positive': 0, 'Negative': 0}
        sentiment_counts[sentiment] = 1

        fig_pie, ax = plt.subplots()
        ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig_pie)

        plot_wordcloud(input_text)
