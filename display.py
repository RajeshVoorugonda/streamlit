import streamlit as st
import torch

st.title("User Details")

name = st.text_input("Enter your name").strip()
age = st.text_input("Enter your age").strip()
gender = st.text_input("Enter your gender").strip()

if st.button("Submit"):
    try:
        age = int(age)
    except:
        st.error("Invalid age format, age must be an integer")
        st.stop()

    user_details = [name, age, gender]

    st.write("Thank you for providing your details, {}! We have recorded that you are a {}-year-old {}.".format(*user_details))
