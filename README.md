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

# 4.0 Modules: pre-made pieces

- Model I/O (input and output): ex. PromptTemplates, LMS, OutputParser
- Retrieval: 외부 데이터에 접근하여 모델에 제공 ex. DocumentLoaders, Transformers, TextEmbeddings, VectorStores, Retrievers
- Chains
- Agents: chain이 필요한 도구들을 직접 선택하여 사용하여, AI를 자동화 한다.
- Memory: Add memory to chatbot.
- Callbacks: model이 하고 있는 일을 답변 전에 확인한다.

# 4.1 PromptTemplate의 장점

- validation
- save and load template
