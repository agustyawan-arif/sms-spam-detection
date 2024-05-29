import streamlit as st
from classifier import SMSClassifier

model_path = "bin/model.pkl"
vector_path = "bin/vector.pkl"

cls = SMSClassifier(model_path, vector_path)

text = st.text_input("Masukkan SMS: ", "I want to go to you house at 2 PM")
if text:
  if st.button("Process"):
    result = cls.predict(text)
    st.write(result)