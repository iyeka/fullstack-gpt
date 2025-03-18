# 풀스택 GPT: #2.0 ~ #3.5

## Tasks:

- [x] Create a Github Repository
- [x] Create a Python environment.
- [x] Install dependencies.
- [x] Create a Jupyter Notebook.
- [x] Setup your OpenAI Keys.
- [ ] Make two chains and chain them together using LCEL.
  - [ ] 프로그래밍 언어에 대한 시를 쓰는 데 특화된 체인과 시를 설명하는 데 특화된 체인을 만드세요.
  - [ ] LCEL을 사용해 두 체인을 서로 연결합니다.
  - [ ] 최종 체인은 프로그래밍 언어의 이름을 받고 시와 그 설명으로 응답해야 합니다.
  - [ ] 모델로는 "gpt-3.5-turbo"를 사용하고 프롬프트에는 ChatPromptTemplate을 사용하세요.
- [x] Push the code to Github

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

# 풀스택 GPT: #4.0 ~ #4.6

## Tasks:

- [ ] 영화 이름을 가지고 감독, 주요 출연진, 예산, 흥행 수익, 영화의 장르, 간단한 시놉시스 등 영화에 대한 정보로 답장하는 체인을 만드세요.
  > - LLM이 답변 형식을 학습하도록 다양한 영화에 대한 예시를 만들어야 합니다.
  > - 예시는 과제의 요구조건을 만족시키려면 `감독`, `주요 출연진`, `예산`, `흥행 수익`, `장르`, `간략한 줄거리` 가 포함되어야 합니다. LLM이 답변 형식을 효과적으로 학습하려면 모든 예시는 `동일한 형식`을 유지해야 합니다.
- [ ] LLM은 항상 동일한 형식을 사용하여 응답해야 하며, 이를 위해서는 원하는 출력의 예시를 LLM에 제공해야 합니다.
- [ ] 예제를 제공하려면 FewShotPromptTemplate 또는 FewShotChatMessagePromptTemplate을 사용하세요.
  > - 자세한 사용법은 다음 공식 문서를 참고해보세요
  > - [Few-shot prompt templates](https://python.langchain.com/v0.1/docs/modules/model_io/prompts/few_shot_examples/)
  > - [Few-shot examples for chat models](https://python.langchain.com/v0.1/docs/modules/model_io/prompts/few_shot_examples_chat/)
