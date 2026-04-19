import streamlit as st
import time
from langchain.prompts import PromptTemplate
from datetime import datetime
from typing import Literal

st.title("#7 DocumentGPT")
st.subheader("Streamlit Tutorial")

st.markdown("""현재시각: """)
today = datetime.today().strftime("%H:%M:%S")
st.write(today)

widget = st.selectbox(
    label="Choose What you wanna learn:",
    options=("write", "magic", "input", "sidebar", "tab", "Document GPT preview"),
)

if widget == "write":
    st.write("Display whatever you give.")
    st.write([1, 2, 3, 4])
    st.write(PromptTemplate)
elif widget == "magic":
    p = PromptTemplate.from_template("xx")
    p
elif widget == "input":
    st.text_input("What is your name?")
    value = st.slider("temperature", min_value=0.1, max_value=1.0)
elif widget == "sidebar":
    st.sidebar.title("sidebar title")
    with st.sidebar:
        st.text_input("sidebar title")
    with st.status("Embedding file...", expanded=True) as status:
        time.sleep(1)
        st.write("Getting the file...")
        time.sleep(1)
        st.write("Embedding the file...")
        time.sleep(1)
        st.write("Caching the file...")
        status.update(label="Error", state="error")

elif widget == "tab":
    tab_one, tab_two, tab_three = st.tabs(["A", "B", "C"])

    with tab_one:
        st.write("a")
    with tab_two:
        st.write("b")
    with tab_three:
        st.write("c")

elif widget == "Document GPT preview":
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    def send_message(message, role: Literal["human", "ai"], save=True):
        with st.chat_message(role):
            st.write(message)
        if save:
            st.session_state["messages"].append({"message": message, "role": role})

    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

    message = st.chat_input("Send a message to the AI")

    if message:
        send_message(message, "human")
        time.sleep(2)
        send_message(f"You said: {message}", "ai")

        with st.sidebar:
            st.write(st.session_state)
