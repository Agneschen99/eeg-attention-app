import streamlit as st
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import io

st.title("🧠 EEG Attention Score Analyzer")
st.write("Upload a Muse EEG CSV file to calculate attention scores based on beta wave power.")

uploaded_file = st.file_uploader("Choose an EEG CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    channels = ['TP9', 'AF7', 'AF8', 'TP10']
    fs = 256
    scores = {}

    fig, ax = plt.subplots()
    for ch in channels:
        if ch in df.columns:
            signal = df[ch].dropna().values
            n = len(signal)
            yf = rfft(signal)
            xf = rfftfreq(n, 1 / fs)
            power = np.abs(yf) ** 2

            total_power = np.sum(power[(xf >= 1) & (xf <= 50)])
            beta_power = np.sum(power[(xf >= 13) & (xf <= 30)])

            score = (beta_power / total_power) * 100 if total_power > 0 else 0
            scores[ch] = round(score, 2)

    # Plot
    ax.bar(scores.keys(), scores.values(), color=['#4f81bd', '#c0504d', '#9bbb59', '#8064a2'])
    ax.set_title("Attention Score by Channel")
    ax.set_ylabel("Attention Score (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis='y')
    st.pyplot(fig)

    # Display table
    st.subheader("Attention Scores:")
    st.table(pd.DataFrame(list(scores.items()), columns=['Channel', 'Attention Score (%)']))

    # Conclusion
    top_channel = max(scores, key=scores.get)
    st.success(f"Most active attention region: **{top_channel}** with {scores[top_channel]}% beta activity")
