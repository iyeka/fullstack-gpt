# 2.0 Goal

- Making GPT-powered apps and ChatGPT plugin with existing data.
- Using LangChain, Streamlit, Pinecone, HuggingFace, FastAPI

## 2.2 Advantage of LangChain

- Memory Module
- Swap components
- LangSmith Debugger
- Agent model

## 2.5 Virtual Environment

```zsh
python -m venv ./env

source env/bin/activate

pip install -r requirements.txt
```

- .gitignore env/

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
  - 대화 중 중요한 내용만 뽑아내어 저장한다.
  - Entity를 추출해 Knowledge Graph(요약본)를 만든다.
- Conversation Token Buffer Memory
  - ConversationBufferWindowMemory와 비슷한데, interaction 값 대신 토큰 양을 limit으로 둔다.

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

1) Load and split the source
2) Embedding: 사람이 읽는 문자를 컴퓨터가 이해할 수 있는 숫자로 변환하는 작업
3) Store the embedded numbers and search

## 6.3 Vectors

- ![3D 벡터](.cache/files/vector_plot.png)
- [벡터를 사용하여 비슷한 문서들을 검색](https://turbomaze.github.io/word2vecjson/)
- [참고영상](https://www.youtube.com/watch?v=2eWuYf-aZE4&t=16s)
- Vector Store: Vector 공간을 검색할 수 있게 해주는 database
- 순서: Create Vectors -> Cache -> Put in Vectorstore -> Search for relavant docs

## 6.6 Off-the-shelf Document Chains [Legacy]

- 종류
  1. ![Stuff](https://python.langchain.com.cn/assets/images/stuff-f51054532840dfb3cbdf86670b48ac7f.jpg): Retrieve한 Documents를 모두 Prompt에 넣는다.

  2. ![Refine](https://python.langchain.com.cn/assets/images/refine-42297d920f42e9988a3e53982f8e83d6.jpg): model이 각 document를 읽으며 답변을 생성하며 이전 답변을 가다듬는다.

  3. ![Map Reduce](https://python.langchain.com.cn/assets/images/map_reduce-aa3ba13ab16536d9f9e046276bd83dd2.jpg)
  - Retrieve한 각각의 Document에서 질문과 관련된 부분을 발췌한다. 발췌문을 합쳐 LLM에 전달해 최종 응답한다.
  - 순서
    1. chain.invoke(질문)
    2. retriever returns 질문과 관련 있는 List of docs
    3. for doc in docs | prompt: '각각의 doc을 읽고 사용자의 질문에 답변하기에 중요한 정보를 추출해 주세요.' | llm
    4. for response in list of llm responses | put the all responses in one document.
    5. final doc | prompt: '질문과 관련된 정보들입니다. 이를 토대로 답변해주세요.' | llm
  4. ![Map Re-rank](https://python.langchain.com.cn/assets/images/map_rerank-3aeb2ae5718693e009aef486ff0e4365.jpg)
  - Retrieve한 documents를 각각 읽으면서 답변을 생성하고 그 답변에 대해 점수를 매긴다. 가장 높은 점수를 획득한 답변과 점수를 반환한다.
  - 순서
    1. retriever로부터 documents를 받는다.
    2. 두 체인을 만든다.
    - 첫 번째 체인
      - 각각의 doc을 llm에 전달하여 "doc만 사용하여 user question에 답변해줘"
      - llm에 "답변의 유용한 정도를 0점~5점으로 평가해줘"
    - 두 번째 체인
      - 답변과 점수를 다른 prompt에 넣고 "주어진 답변을 보고, 점수가 가장 높고, 가장 최근에 작성된 답변을 선택해줘"

- Stuff VS MapReduce
  - retriever가 반환하는 document 수가 많으면 prompt에 document를 다 넣을 수 없기 때문에 Stuff 보다 각 document를 요약하는 MapReduce를 사용한다
  - 비용을 아끼고 싶다면 stuff

## 7.7 @st.cache_data

- 함수 위에 데코레이터를 붙여 사용
- 함수의 인자가 처음 실행했을 떄와 동일하면, 함수를 재실행하지 않지 않고 이전 결과를 반환한다.

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
5. model instruction에 따라 prompt를 작성한다.

C. 컴퓨터에 다운로드 받아 사용

1. from langchain.llms.huggingface_pipeline import HuggingFacePipeline
2. ```python
   llm = HuggingFacePipeline.from_model_id(
     model_id="gpt2",
     task="text-generation",
     device=0, #GPU
     divice=-1, #CPU
     pipeline_kwargs={
         "max_new_tokens": 50
     },
   )
   ```

## 8.3 Download GPT4All

1. Search for model(ex.falcon-q4)
2. llm = GPT4All(model="./falcon.bin") # Where to save

## 8.4 Ollama

1. Download Ollama from [page](ollama.ai/download)
2. Find model from [page](https://ollama.com/library)
3. terminal에 ollama run model-name (ex. ollama run falcon:180b-chat)
4. Ollama가 하는 일

- 컴퓨터에 서버를 만든다.
- localhost/API로 send requests
- model을 사용하여 response를 준다.

5. 모델 삭제 시 ollama rm model-name

## 9.8 Function Calling

- Force llm output structure
- Give llm functions to call
- Only works for OpenAI models

## 10.1 AsyncChromiumLoader

- 목표: Scrape all the html from the website and clean it to text.
- 방법: 2 types of data loaders
  1. Playwright + Chromium, and BeautifulSoup:
  - Playwright
    - Browser control package like Selenium
    - 가상환경에서 console에 playwright install -> AsyncChromiumLoader를 documenet loader로 사용
    - 브라우저를 사용하기 때문에 느리다.
  - 언제 사용:
    - 웹사이트에 sitemap이 없을 때 사용
    - Javascript 코드가 많은 사이트를 추출할 때 사용. 접속 직후에 data들이 바로 로딩되지 않기 때문.
  2. Sitemap Loader
  - 웹사이트의 sitemap을 가져온다.
  - 주소 뒤에 /sitemap.xml 을 붙인다.
  - all the directories from the url 을 볼 수 있다.
  - text가 많은 정적인 사이트를 긁어올 때 사용

## 10.3 Parsing Function

- filter_urls REGEX:
  - r"^(._\/blog\/)._": scrape blog를 포함하는 urls
  - r"^(?!._\/blog\/)._": scrape blog를 포함하지 않는 urls

## 11.1 Audio Extraction

- FFmpeg: 비디오 CLI tool. 오디오 추출에 사용.

```console
brew install ffmpeg
```

- ffmpeg -항상덮어쓰기 -input video_path -ignore_the_video audio_path

```console
ffmpeg -y -i files/podcast.mp4 -vn files/audio.mp3
```

- subprocess: python code에서 command 실행
- pydub: python으로 오디오 조작
- glob: 패턴으로 파일 디렉토리 검색

## 12.1 Your First Agent

- AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION:
  - STRUCTURED: tools have multiple inputs
  - CHAT: optimized for chat models
  - ZERO_SHOT_REACT: react라는 논문에 기반한 type의 agent

## 12.2 How Do Agents Work

```python
next_action = agent.get_action(...)
while next_action != AgentFinish:
    observation = run(next_action)
    next_action = agent.get_action(..., next_action, observation)
return next_action
```

## 12.3 Zero-shot ReAct Agent

- AgentType.ZERO_SHOT_REACT_DESCRIPTION
  - most general purpose
  - STRUCTURED agent와 다르게 하나의 input만 받을 수 있음
  - ReAct: Reasoning and Acting 논문에 기반

## 12.4 OpenAI Functions Agent

- AgentType.OPENAI_FUNCTIONS
  - Pydantic으로 tool의 형식을 검증해야 한다.
  - input prompt가 짧다.

# 13 ChefGPT

- GPT Plugin -> Custom GPT with Actions(GPTs)
- CustomGPT: ChatGPT + pre-installed plugins + pre-loaded docs for RAG
- GPT Action: API 구성에 대한 설명을 바탕으로, GPT가 API에 request를 보낼지 판단한다.
  1. [Create a GPT](https://chatgpt.com/gpts)
  2. CustomGPT가 Action으로 사용할 API 서버를 만든다.
  - Python file로 FastAPI Server 작성
  - 서버 실행

  ```zsh
  uvicorn file_name:app --reload
  ```

  - url/docs: documentation page
  - url/openapi.json: api schema
  3. cloudflared 설치: local host를 외부에서 접근할 수 있도록 https url을 부여하는 CLI

  ```zsh
  brew install cloudflare/cloudflare/cloudflared
  cloudflared tunnel --url url
  ```

  4. 해당 url을 끝에 / 떼고 api server python file에 적는다.
  5. customGPT - configure - create new action - import from url - url/openapi.json 주소 붙여넣기 - import

## 13.5 API Key Auth

- customGPT authentication type
  - api key: 호출을 보낸 게 CustomGPT 임을 확인하기 위해 필요
  - OAuth: 사용자가 누군지 알고 싶을 때 사용
  - API Key는 임의로 설정

## 13.6 OAuth

- 소셜 아이디로 웹사이트 로그인 할 때 카카오 역할임:

1. 웹사이트가 사용자를 카카오로 보냄
2. 카카오 로그인
3. 카카오는 사용자와 token을 함께 원래의 웹사이트로 redirect
4. 웹사이트는 카카오에 token의 사용자 정보를 요청한다.

- Authorization URL: url/authorize
- Toekn URL: url/token
- Scope: user에게 허락하는 동작

## 14.6 Assistants API

1. add a message to thread
2. run the thread and see if it has any required output
3. submit required outputs
4. re-run
5. add messages again

## 15.1 AWS Bedrock

- foundation model 제공: your own data로 fine tune 가능
- Claude
  - huge context window
- .env file에 AWS_ACCESS_KEY 와 AWS_SECRET_KEY 붙여넣기
- pip install boto3: python-AWS client
- [Claude prompting documentation](https://platform.claude.com/docs/ko/build-with-claude/prompt-engineering/claude-prompting-best-practices)

## 15.4 AzureChatOpenAI

- 판권을 사서 more reliable 회사인 Microsoft가 오픈에이아이 모델 복사본을 가짐
- Not private. 모델에 보내는 data를 마소가 검열함
- AzureOpenAIService - Apply for access
- [Create Azure OpenAI Account, Model Deployment](portal.azure.com)
- .env file에 AZURE_OPENAI_API_KEY 와 AZURE_OPENAI_ENDPOINT 붙여넣기

## 16.2 Crews, Agents and Tasks

- Crews: Team of Agents
  - Tasks = [순서대로 실행]
  - Agents = [순서 상관 없음]
- Agent: Member of a team
  - What they do:
    - Perform tasks
    - Make decisions
    - Communicate with other agents
  - It has to have:
    - specific skills
    - a particular job to do
    - 여러 가지 일을 주면 잘 못함. 한 가지 일만 구체적으로 하라고 해야 함.
  - Attributes
    - Role
    - Goal
    - Backstory: 역할극을 할 때 답변을 더 잘 함. You are an expert chef.
- Task: Specific assignment completed by Agent
  - Description
  - Agent
  - **Expected Output**
- Tools: Agent가 현실에서 작동할 수 있도록 부여하는 도구. 파일 읽기, API 호출, 웹사이트 검색...

## 16.8 Custom Tools

- ManagerLLM: Tasks와 Agents를 살펴보고 Task 순서와 어떤 Agent가 수행할지 직접 지시한다.
