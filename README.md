# 2.0 Goal

- Making GPT-powered apps, ChatGPT plugin with data.
- Using LangChain, Streamlit, Pinecone, HuggingFace, FastAPI

## 2.2 Advantage of LangChain

- Memory Module
- Swap components
- LangSmith Debugger
- Agent model

## 3.4 Chaining Chains

### LCEL working logic

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

## 4.1 PromptTemplate의 장점

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

# 6 RAG: Retrieval Augmented Generation

- 검색하는 질문과 관련 있는 문서들을 함께 prompt로 보내는 방식

## 방식

    1. Stuff
    2. Refine
    3. Map Reduce

## 단계

1. 첫번째 단계인 [Retrieval](https://python.langchain.com/v0.1/assets/images/data_connection-95ff2033a8faa5f3ba41376c0f6dd32a.jpg)

## 6.3 Vectors

- ![3D 벡터](../../Downloads/vector_plot.png)
- [벡터를 사용하여 비슷한 문서들을 검색](https://turbomaze.github.io/word2vecjson/)
- [참고영상](https://www.youtube.com/watch?v=2eWuYf-aZE4&t=16s)

## 6.6 Off-the-shelf Document Chains [Legacy]

- 종류

  1. ![Stuff](https://python.langchain.com.cn/assets/images/stuff-f51054532840dfb3cbdf86670b48ac7f.jpg): Retrieve한 Documents를 모두 Prompt에 넣는다.

  2. ![Refine](https://python.langchain.com.cn/assets/images/refine-42297d920f42e9988a3e53982f8e83d6.jpg): model이 각 document를 읽으며 답변을 생성한다. 답변을 쌓으면서 가다듬는다.

  3. ![Map Reduce](https://python.langchain.com.cn/assets/images/map_reduce-aa3ba13ab16536d9f9e046276bd83dd2.jpg): Retrieve한 Document별로 각각 체인을 돌린다. 발췌한 부분을 그대로 또는 요약하여 합친 뒤 LLM에 전달해 최종 응답한다.

  - 순서
    1. chain.invoke(질문)
    2. retriever returns 질문과 관련 있는 List of docs
    3. for doc in docs | prompt: '각각의 doc을 읽고 사용자의 질문에 답변하기에 중요한 정보를 추출해 주세요.' | llm
    4. for response in list of llm responses | put the all responses in one document.
    5. final doc | prompt | llm

  4. ![Map Re-rank](https://python.langchain.com.cn/assets/images/map_rerank-3aeb2ae5718693e009aef486ff0e4365.jpg): 각 document를 읽으면서 답변을 생성하고 그 답변에 대해 점수를 매긴다. 가장 높은 점수를 획득한 답변과 점수를 반환한다.

- 용례 (Stuff VS MapReduce)
  retriever가 반환하는 document 수가 많으면 prompt에 document를 다 넣을 수 없기 때문에 Stuff 보다 각 document를 요약하는 MapReduce를 사용한다.

## 7.8 Process of chain VS non-chain

- Non-Chain

```python
docs = retreiver.invoke(message)
docs = "\n\n".join(doc.page_content for doc in docs)
prompt = template.format_messages(context=docs, question=message)
llm.predict_messages(prompt)
```

- Chain

```python
chain = {
  "context": retriever | RunnableLambda(lambda docs: "\n\n".join(doc.page_content for doc in docs)),
  "question": RunnablePassthrough()
} | prompt | llm
```
