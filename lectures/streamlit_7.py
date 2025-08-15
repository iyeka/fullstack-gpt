import streamlit as st
import time
from langchain.prompts import PromptTemplate
from datetime import datetime

today = datetime.today().strftime("%H:%M:%S")
st.title(today)

st.subheader("Subheader")
st.markdown(
    """
    #### Markdown
"""
)

st.write("It will try to display whatever you pass.")

model = st.selectbox(
    "Select",
    (
        "option1",
        "option2",
    ),
)
if model == "option1":
    st.write("cheap")
else:
    st.write(PromptTemplate)

name = st.text_input("What is you name")
st.write(name)

value = st.slider(
    "temperature",
    min_value=0.1,
    max_value=1.0,
)
st.write(value)

p = PromptTemplate.from_template("PromptTemplate")
st.write(p)
p  # == st.write(p). it's called Magic.

st.sidebar.title("sidebar title")

with st.sidebar:
    st.text_input("sidebar text input")

t1, t2, t3, t4 = st.tabs(["T", "A", "B", "S"])
with t1:
    st.write("T1")

with st.chat_message("human"):
    st.write("I am Ahmad Jamal!")

with st.chat_message("ai"):
    st.write("You are not.")

st.chat_input("Write sth here.")

with st.status("Embedding file...", expanded=True) as status:
    time.sleep(2)
    st.write("Getting the file...")
    time.sleep(2)
    st.write("Embedding the file...")
    time.sleep(2)
    st.write("Embedding is Done!")
    status.update(label="Error", state="error")
