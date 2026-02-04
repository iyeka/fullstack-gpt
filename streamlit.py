from streamlit import session_state
import streamlit as st
from langchain.prompts import PromptTemplate
from datetime import datetime
import time
from typing import Literal

st.title("Streamlit Tutorial")
st.subheader("#7.0~")

today = datetime.today().strftime("%H:%M:%S")
st.markdown(f"#### whole page refeshed when you choose a value: {today}")

tutorial, session_state = st.tabs(["tutorial", "session_state"])
with tutorial:
    type = st.selectbox(
        "Choose the feature you want to learn",
        ("st.write", "st magic", "user input", "sidebar", "chat messages"),
    )

    if (
        type == "st.write"
    ):  # function 선택할 필요 없이 어떤 것이든 화면에 표시할 수 있다.
        st.write([1, 2, 3, 4])
        st.write({"x": 1})
        st.write(PromptTemplate)
        p = PromptTemplate.from_template("xx")
        st.write(p)
    elif type == "st magic":  # displayed without st.write
        p = PromptTemplate.from_template("xx")
        p
    elif type == "user input":
        name = st.text_input("What is your name")
        st.write(name)

        value = st.slider("temperature", min_value=0.1, max_value=1.0)
        st.write(value)
    elif type == "sidebar":
        st.sidebar.title("title")
        st.sidebar.text_input("text input")
        with st.sidebar:
            st.text_input("with text input")
    elif type == "chat messages":
        with st.chat_message("human"):
            "HELLLO!"
        with st.chat_message("ai"):
            st.write("How are you?")

        st.chat_input("Send a message to the AI")

        with st.status("Embedding the file...", expanded=True) as status:
            time.sleep(3)
            st.write("Getting the file")
            time.sleep(3)
            st.write("Embedding the file")

            status.update(label="Error", state="error")
with session_state:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    def send_message(message, role: Literal["human", "ai"], save=True):
        with st.chat_message(role):
            st.write(message)
        if save:
            st.session_state["messages"].append({"message": message, "role": role})

    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

    message = st.chat_input("Send a message to the ai")
    if message:
        send_message(message, "human")
        time.sleep(3)
        send_message(f"You said: {message}", "ai")

    with st.sidebar:
        st.write(st.session_state)
