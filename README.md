# 2.0 Goal

- Making GPT-powered apps and ChatGPT plugin with existing data.
- Using LangChain, Streamlit, Pinecone, HuggingFace, FastAPI

## 2.2 Advantage of LangChain

- Memory Module
- Swap components
- LangSmith Debugger
- Agent model

## 3.2 Prompt Templates

- 사용하는 이유
  - Validation
  - Save and load template to disk

- PromptTemplate
  - Use when predicting text
  - Create a template from string

- ChatPromptTemplate
  - Create a template from messages

## 3.3 LangChain Expression Language (LCEL)

- 다양한 template, LLM 호출, 여러 responses를 함께 사용할 수 있도록 chain으로 연결한다.
- 다음의 일들을 코드 한 줄로 합친다.

1. make template
2. format messages
3. predict
4. make parser
5. parse

## 3.4 LCEL working logic

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

- Model I/O (input and output)
  - PromptTemplates
  - Language Models
  - OutputParser
- Retrieval: 외부 데이터에 접근하여 모델에 제공 ex. DocumentLoaders, Transformers, TextEmbeddings, VectorStores, Retrievers
- Chains
- Agents: chain이 필요한 도구들을 직접 선택하여 사용하여, AI를 자동화 한다.
- Memory: Add memory to chatbot.
- Callbacks: model이 하고 있는 일을 답변 전에 확인한다.

## 4.1 FewShotPromptTemplate

- 모델에게 대답하는 양식에 대한 예제들을 준다.
- (ex) customer service database를 fewshot으로 format 시켜 customer support bot을 만든다.
- 사용방법:
  1. examples를 준비한다.
  2. create a example_prompt to format the examples
  3. FewShotPromptTemplate에 전달하면, fewshot이 examples 하나하나를 가져와 example_prompt로 format한다.
  4. human question

## 4.3 ExampleSelector

- choose and limit examples that goes into prompt
- 유저의 로그인 여부, 사용 언어 등의 기준으로 custom 할 수 있다.

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
  - ConversationSummaryMemory + ConversationBufferMemory
  - limit에 다다르면 오래된 메시지부터 요약한다.
  - 메모리를 사용하는 데 비용이 든다.
- Conversation KGMemory
  - Entity를 추출해 Knowledge Graph(요약본)를 만든다.

## 5.8 Memory on LLMChain VS LCEL chain

- LLMChain (off-the-shelf)
  - 자동: LLM response를 memory에 update
  - 수동: memory를 prompt에 넣는 것
- LCEL (custom chain)
  - 모두 수동: load and save memory

# 6 RAG: Retrieval Augmented Generation

- 검색하는 질문과 관련 있는 문서들을 함께 prompt로 보내는 방식

## 방식

1. Stuff
2. Refine
3. Map Reduce

## 단계

1. ![Retrieval](https://image.samsungsds.com/kr/insights/20240308-image2.png?queryString=20250214030334)

- Embedding: 사람이 읽는 문자를 컴퓨터가 이해할 수 있는 숫자로 변환하는 작업

## 6.3 Vectors

- ![3D 벡터](../../Downloads/vector_plot.png)
- [벡터를 사용하여 비슷한 문서들을 검색](https://turbomaze.github.io/word2vecjson/)
- [참고영상](https://www.youtube.com/watch?v=2eWuYf-aZE4&t=16s)
- Vector Store: Vector 공간을 검색할 수 있게 해주는 database
- 순서: Create Vectors -> Cache -> Put in Vectorstore -> Search for relavant docs

## 6.6 Off-the-shelf Document Chains [Legacy]

- 종류
  1. ![Stuff](https://python.langchain.com.cn/assets/images/stuff-f51054532840dfb3cbdf86670b48ac7f.jpg): Retrieve한 Documents를 모두 Prompt에 넣는다.

  2. ![Refine](https://python.langchain.com.cn/assets/images/refine-42297d920f42e9988a3e53982f8e83d6.jpg): model이 각 document를 읽으며 답변을 생성하며 이전 답변을 가다듬는다.

  3. ![Map Reduce](https://python.langchain.com.cn/assets/images/map_reduce-aa3ba13ab16536d9f9e046276bd83dd2.jpg)
  - Retrieve한 Document별로 각각 요약한다. 발췌한 부분을 그대로 또는 요약하여 합친 뒤 LLM에 전달해 최종 응답한다.
  - 순서
    1. chain.invoke(질문)
    2. retriever returns 질문과 관련 있는 List of docs
    3. for doc in docs | prompt: '각각의 doc을 읽고 사용자의 질문에 답변하기에 중요한 정보를 추출해 주세요.' | llm
    4. for response in list of llm responses | put the all responses in one document.
    5. final doc | prompt: '질문과 관련된 정보들입니다. 이를 토대로 답변해주세요.' | llm
  4. ![Map Re-rank](https://python.langchain.com.cn/assets/images/map_rerank-3aeb2ae5718693e009aef486ff0e4365.jpg): Retrieve한 documents를 각각 읽으면서 답변을 생성하고 그 답변에 대해 점수를 매긴다. 가장 높은 점수를 획득한 답변과 점수를 반환한다.

- Stuff VS MapReduce
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

# 8 PrivateGPT

## 8.1 Ways of using HuggingFace

A. HuggingFace Inference API: 유료로 허깅페이스 서버 사용

1. [모델 선택](https://huggingface.co/models)(ex. Mistral-7B, falcon-7b-instruct)
2. Deploy Inference Endpoints

B. HuggingFaceHub: 무료 허깅페이스 서버 사용

1. Inference API가 활성화되어 있는 모델 선택
2. HuggingFace Access Token 생성 후 .env 파일에 HUGGINGFACEHUB_API_TOKEN variable에 저장
3. from langchain.llms import HuggingFaceHub
4. ```python
   llm = HuggingFaceHub(
   repo_id="mistralai/Mistral-7B-v0.1",
   model_kwargs={"max_new_tokens": 250},
   )
   ```

C. 컴퓨터에 다운로드 받아 사용

1. from langchain.llms.huggingface_pipeline import HuggingFacePipeline
2. ```python
   llm = HuggingFacePipeline.from_model_id(
     model_id="gpt2",
     task="text-generation",
     pipeline_kwargs={
         "max_new_tokens": 50
     },
   )
   ```

## 8.3 Download GPT4All

1. Search for model(ex.falcon-q4)
2. llm = GPT4All(model="./falcon.bin")

## 8.4 Ollama

1. Download Ollama from [page](ollama.ai/download)
2. terminal에 ollama run mistral
3. Ollama가 하는 일

- 컴퓨터에 서버를 만든다.
- localhost/API로 send requests
- model을 사용하여 response를 준다.

## 9.8 Function Calling

- Give llm functions to use
- Only works for OpenAI models

## 10.3 Parsing Function

- filter_urls REGEX:
  - r"^(._\/ai-gateway\/)._" : scrape all sites those include /ai-gateway/
  - r"^(?!._\/vectorize\/)._" : scrape all sites except /vectorize/ is included.
