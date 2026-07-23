import streamlit as st

st.set_page_config(
    page_title="AI Traffic Analysis System",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI Traffic Analysis System")

st.markdown("""
## Welcome

Select a module from the sidebar.

### Available Modules

- 🚗 Traffic Analysis
- 🕳️ Pothole Detection
- 📊 Reports
- 🤖 ML Prediction
""")