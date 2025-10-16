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

# 5 Memory

- Conversation Buffer Memory
  - Saves whole conversations.
  - Text completion 시 유용
  - 대화 기록을 모두 모델에 전달해야 하기 때문에 길어질수록 비효율적
- Conversation Buffer Window Memory
  - 최근 n개의 메모리만 저장
  - 메모리 크기를 일정하게 유지할 수 있다.
  - 오래된 메시지는 버리고 최근 대화에만 집중한다는 것이 단점.
- Conversation Summary Memory
  - 초반에는 토큰을 더 많이 차지하지만, 메시지가 많아질수록 유용해진다.
  - 메모리를 사용하는 데 비용이 든다.
- Conversation Summary Buffer Memory
  - ConversationSummaryMemory + ConversationBufferWindowMemory
  - limit에 다다르면 오래된 메시지부터 요약한다.
  - 메모리를 사용하는 데 비용이 든다.
- Conversation KGMemory
  - Entity의 Knowledge Graph(요약본)를 만든다.
