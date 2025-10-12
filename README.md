# 2.0 Goal

- Making GPT-powered apps, ChatGPT plugin with data.
- Using LangChain, Streamlit, Pinecone, HuggingFace, FastAPI

# 2.2 Advantage of LangChain

- Memory Module
- Swap components
- LangSmith Debugger
- Agent model

# 3.4 Chaining Chains

## LCEL working logic

- [Components those chains could have](https://python.langchain.com/docs/concepts/runnables/):
  | Component | Input Type | Output Type |
  | --- | --- | --- |
  | Prompt | dictionary | PromptValue |
  | ChatModel | a string, list of chat messages or a PromptValue | ChatMessage |
  | LLM | a string, list of chat messages or a PromptValue | String |
  | OutputParser | the output of an LLM or ChatModel | Depends on the parser |
  | Retriever | a string | List of Documents |
  | Tool | a string or dictionary, depending on the tool | Depends on the tool |
