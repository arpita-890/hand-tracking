import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Hand Tracking App")
st.write("Camera is working!")

webrtc_streamer(key="test")
