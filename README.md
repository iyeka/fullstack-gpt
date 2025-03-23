# 풀스택 GPT: #2.0 ~ #3.5

## 환경설정(#2.5)

1.  git init .
2.  가상환경 설정

```
python3.11.6 -m venv ./env
source env/bin/activate <---> deactivate
```

3.  package dependency 목록에 따라 설치하되, env 폴더의 package들 자체는 .gitignore

```
pip install -r requirements.txt
```

4. .env 파일을 만들어 .gitignore에 포함시키고 API Key 등 보안이 필요한 변수들을 넣는다.
5. python main.py 대신 jupyter notebook 파일을 만든다. -> Select Kernel

- 주피터를 쓰는 이유는 계속 print()를 하지 않아도 된다. 셀을 실행하면 print()하지 않아도 셀 내부 마지막 문장의 값을 보여준다.
- 값을 메모리에 저장해서 다른 셀에서 꺼내 쓸 수 있기에 처음부터 끝까지 실행하지 않아도 된다.

## How does the LangChain Expression Language works

- LangChain의 역할

  > - call prompt with model
  > - parse response
  > - turn response into AI Message

- [Components those chains can have](https://python.langchain.com/v0.1/docs/expression_language/interface/)
  | Component | Input Type | Output Type |
  | --- | --- | --- |
  | Prompt | Dictionary | \*PromptValue |
  | ChatModel | Single string, list of chat messages or a PromptValue | ChatMessage |
  | LLM | Single string, list of chat messages or a PromptValue | String |
  | OutputParser | The output of an LLM or ChatModel | Depends on the parser |
  | Retriever | Single string | List of Documents |
  | Tool | Single string or dictionary, depending on the tool | Depends on the tool |

  > \*PromptValue = formatted prompt

# 풀스택 GPT: #4.0 ~ #4.6

## [LangChain Modules](https://js.langchain.com/v0.1/docs/modules/)

1. Model I/O

- input: prompt template
- Language Models
- output: parsers

2. Retrieval: 외부 데이터를 모델에 주어 작업

- document loaders
- document transformers
- Text embedding models
- Vector stores
- retrievers

3. Agents

- AI 자동화. chain에 목표와 도구를 주면 chain이 도구를 선택해 목표를 달성.

4. Memory

- 챗봇에 메모리를 주어 기억 저장

5. Callbacks

- 모델이 생각하는 바를 실시간으로 알게 해준다.

## How to load prompt template from the disk

- 두 가지 타입의 prompt 작성

1. Json
2. Yaml

## [LangChain의 Third Party 제공업체 보기](https://python.langchain.com/docs/integrations/providers/)

# 풀스택 GPT: #5.0 ~ #5.8

## Tasks:

- [ ] 앞서 배운 메모리 클래스 중 하나를 사용하는 메모리로 LCEL 체인을 구현합니다.
- [ ] 이 체인은 영화 제목을 가져와 영화를 나타내는 세 개의 이모티콘으로 응답해야 합니다. (예: "탑건" -> "🛩️👨‍✈️🔥". "대부" -> "👨‍👨‍👦🔫🍝").
- [ ] 항상 세 개의 이모티콘으로 답장하도록 FewShotPromptTemplate 또는 FewShotChatMessagePromptTemplate을 사용하여 체인에 예시를 제공하세요.
  > - [ ] 요구조건에 맞는 답변 형식을 생성하도록 적절한 예시를 만들고, FewShotPromptTemplate 또는 FewShotChatMessagePromptTemplate를 이용하여 LLM에게 예시를 제공하세요.
  > - 자세한 사용법은 다음 공식 문서를 참고해보세요
  >   > - [Few-shot prompt templates](https://python.langchain.com/v0.1/docs/modules/model_io/prompts/few_shot_examples/)
  >   > - [Few-shot examples for chat models](https://python.langchain.com/v0.1/docs/modules/model_io/prompts/few_shot_examples_chat/)
- [ ] 메모리가 작동하는지 확인하려면 체인에 두 개의 영화에 대해 질문한 다음 다른 셀에서 체인에 먼저 질문한 영화가 무엇인지 알려달라고 요청하세요.

  > - ConversationBufferMemory 등 강의에서 배운 메모리 중 하나를 사용하여 이전 대화 기록을 기억하고 기록을 이용한 답변을 제공할 수 있도록 합니다.
  > - 채팅 형식의 메모리 기록을 프롬프트에 추가하고 싶을 때는 MessagesPlaceholder를 이용하세요. (공식문서 예시)
  > - RunnablePassthrough를 활용하면 LCEL 체인을 구현할 때 메모리 적용을 쉽게 할 수 있습니다. RunnablePassthrough는 메모리를 포함한 데이터를 체인의 각 단계에 전달하는 역할을 합니다. (강의 #5.7 1:04~ 참고)

# 풀스택 GPT: #6.0 ~ #6.10

- [ ] Stuff Documents 체인을 사용하여 완전한 RAG 파이프라인을 구현하세요.
- [ ] 체인을 수동으로 구현해야 합니다.
- [ ] 체인에 ConversationBufferMemory를 부여합니다.
- [ ] 이 문서를 사용하여 RAG를 수행하세요: https://gist.github.com/serranoarevalo/5acf755c2b8d83f1707ef266b82ea223
- [ ] 체인에 다음 질문을 합니다:
- [ ] Aaronson은 유죄인가요?
- [ ] 그가 테이블에 어떤 메시지를 썼나요?
- [ ] Julia는 누구인가요?

> - 다음과 같은 절차대로 구현하면 챌린지를 해결할 수 있습니다.
> - (1) 문서 로드하기: `TextLoader` 등 을 사용해서 파일에서 텍스트를 읽어옵니다. ( [Document Loaders 관련 문서](https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/)
> - (2) 문서 쪼개기: `CharacterTextSplitter` 등 을 사용해서 문서를 작은 문서 조각들로 나눕니다. [Character Split 관련 문서](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/character_text_splitter/)
> - (3) 임베딩 생성 및 캐시: `OpenAIEmbeddings`, `CacheBackedEmbeddings` 등 을 사용해 문서 조각들을 임베딩하고 임베딩을 저장합니다. [Caching 관련 문서](https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/caching_embeddings/)
> - (4) 벡터 스토어 생성: `FAISS` 등 을 사용해서 임베딩된 문서들을 저장하고 검색할 수 있는 데이터베이스를 만듭니다. [FAISS 관련 문서](https://python.langchain.com/v0.1/docs/integrations/vectorstores/faiss/)
> - (5) 대화 메모리와 질문 처리: `ConversationBufferMemory` 를 사용해 대화 기록을 관리합니다.
> - (6) 체인 연결: 앞에서 구현한 컴포넌트들을 적절하게 체인으로 연결합니다.
