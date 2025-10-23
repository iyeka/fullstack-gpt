import streamlit as st
import time
from langchain.prompts import PromptTemplate
from datetime import datetime


today = datetime.today().strftime("%H:%M:%S")

st.title(today)
st.subheader("Subheader")
st.markdown(
    """
    Markdown
"""
)

command = st.selectbox(
    "Choose the command you want to know",
    ("write", "with", "chat", "else"),
)

if command == "write":
    st.write("st.write")
    st.write(["Swiss", "Army", "Knife", "of", "Streamlit"])
    st.write({"st.write": "Swiss Amry knife of Streamlit"})
    st.write(PromptTemplate)
    prompt = PromptTemplate.from_template("from_template")
    st.write(prompt)
    prompt
elif command == "with":
    st.sidebar.title("sidebar title")
    st.sidebar.text_input("st.sidebar.text_input")

    with st.sidebar:
        st.text_input("with st.sidebar")

    t1, t2, t3 = st.tabs(["A", "B", "C"])
    with t1:
        st.write("A")
    with t2:
        st.write("B")
    with t3:
        st.write("C")

    with st.chat_message("human"):
        st.write("I am human")
    with st.chat_message("ai"):
        st.write("I am AI")
    st.chat_input("Send a message to AI")

    with st.status("Embedding file...", expanded=True) as status:
        time.sleep(3)
        st.write("Getting the file...")
        time.sleep(3)
        st.write("Embedding the file...")
        time.sleep(3)
        st.write("Caching the file...")
        status.update(label="Error", state="error")
elif command == "chat":
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    def send_message(message, role, save=True):
        with st.chat_message(role):
            st.write(message)
        if save:
            st.session_state["messages"].append({"message": message, "role": role})

    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

    message = st.chat_input("Send message to AI")
    if message:
        send_message(message, "human")
        time.sleep(2)
        send_message(f"You said: {message}", "ai")
else:
    st.text_input("What is your name?")
    st.slider("temparature", min_value=0.1, max_value=1.0)
