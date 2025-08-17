# 풀스택 GPT: #7.0~7.10

## Tasks:

- [x] 이전 과제에서 구현한 RAG 파이프라인을 Streamlit으로 마이그레이션합니다.
- [x] 파일 업로드 및 채팅 기록을 구현합니다.
  > - [x] 파일 업로드를 구현하기 위해 [st.file_uploader](https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader)를 사용합니다. type 매개변수를 통해 허용할 파일 확장자를 지정할 수 있습니다.
  > - [x] 채팅 기록을 저장하기 위해 [Session State](https://docs.streamlit.io/library/api-reference/session-state)를 사용합니다.
- [ ] 사용자가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
  > - [ ] 사용자의 OpenAI API 키를 사용하기 위해 st.text_input을 이용하여 API 키를 입력받습니다. 그런 다음, ChatOpenAI와 OpenAIEmbeddings 클래스를 사용할 때 해당 API 키를 openai_api_key 매개변수로 넘깁니다. [st.text_input 공식 문서](https://docs.streamlit.io/library/api-reference/widgets/st.text_input)
- [ ] st.sidebar를 사용하여 스트림릿 앱의 코드와 함께 깃허브 리포지토리에 링크를 넣습니다.
  > - [ ] [여기에서 계정을 개설하세요](https://share.streamlit.io/)
  > - [ ] [다음 단계를 따르세요](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app#deploy-your-app-1)
  > - [ ] [with st.sidebar](https://docs.streamlit.io/library/api-reference/layout/st.sidebar)를 사용하면 사이드바와 관련된 코드를 더욱 깔끔하게 정리할 수 있습니다.
- [ ] 앱의 구조가 아래와 같은지 확인하고 배포 양식의 Main file path 에 app.py를 작성하세요.
- [ ] 과제 제출 링크는 반드시 streamlit.app URL 이 되도록 하세요.

```
your-repo/
├── .../
├── app.py
└── requirements.txt
```
