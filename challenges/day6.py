import streamlit as st

# 과제
md = st.markdown("""
    <details>
    <summary>풀스택 GPT: #9.0 ~ #9.9</summary>
    - [ ] QuizGPT를 구현하되 다음 기능을 추가합니다:
        - [ ] 함수 호출을 사용합니다.
            - [ ] 함수 호출 (Function Calling)을 활용하여 모델의 응답을 원하는 형식으로 변환합니다. [Attaching OpenAI functions](https://python.langchain.com/v0.1/docs/expression_language/primitives/binding/#attaching-openai-functions)
        - [ ] 유저가 시험의 난이도를 커스터마이징 할 수 있도록 하고 LLM이 어려운 문제 또는 쉬운 문제를 생성하도록 합니다.
            - [ ] 유저가 시험의 난이도를 선택할 수 있도록 st.selectbox 를 사용합니다. [st.selectbox 공식 문서](https://docs.streamlit.io/develop/api-reference/widgets/st.selectbox)
        - [ ] 퀴즈를 화면에 표시하여 유저가 풀 수 있도록 st.radio 를 사용합니다. [st.radio 공식 문서](https://docs.streamlit.io/develop/api-reference/widgets/st.radio)
        - [ ] 만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용합니다.
            - [ ] 만점 여부를 확인하기 위해 문제의 총 개수와 정답 개수가 같은지 비교합니다. 만약 같으면 st.balloons를 사용합니다. [st.balloons 공식 문서](https://docs.streamlit.io/develop/api-reference/status/st.balloons)
        - [ ] 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 로드합니다.
            - [ ] 유저의 자체 OpenAI API 키를 사용하기 위해 st.text_input 등 을 이용하여 API 키를 입력받습니다. 그런 다음, ChatOpenAI 클래스를 사용할 때 해당 API 키를 openai_api_key 매개변수로 넘깁니다. [st.text_input 공식 문서](https://docs.streamlit.io/library/api-reference/widgets/st.text_input)
        - [ ] st.sidebar를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크를 넣습니다.
    - [ ]이전 과제와 동일한 방식으로 앱을 Streamlit cloud 에 배포합니다.
    - [ ]제출은 streamlit.app URL로 이루어져야 합니다.
    </details>
""",
    unsafe_allow_html=True
)
