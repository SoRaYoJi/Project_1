import pandas as pd
import streamlit as st

def render_dashboard(history):
    df = pd.DataFrame(history)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total OCR", len(df))
    col2.metric("Avg Confidence", f"{df.confidence.mean():.2f}%")
    col3.metric("Digits OCR", df.digits_count.sum())

    st.bar_chart(df["digits_count"])
    st.line_chart(df.groupby("time").size())
