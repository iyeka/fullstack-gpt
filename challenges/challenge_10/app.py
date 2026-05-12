import streamlit as st

st.set_page_config(page_title="Fullstack GPT: #14.0 ~ #14.09", page_icon="🧙‍♀️")

st.title("OpenAI Assistants (Graduation Project)")

with st.expander("challenges"):
    st.markdown(
        """
        - 이전 과제에서 만든 에이전트를 OpenAI 어시스턴트로 리팩터링합니다.
            - 지난번 과제에서 만들었던 리서치 AI 에이전트 를 이번에는 OpenAI Assistant를 이용하여 구현해 보는 과제입니다.
            - [ ] 지난 과제에서 만들었던 도구들을 OpenAI Assistant에서 사용할 수 있도록 바꿔줍니다. (공식 문서의 [Function calling](https://platform.openai.com/docs/assistants/tools/function-calling) 부분을 참고하세요.)
            - [ ] 사용자의 OpenAI API 키를 입력 받고, Assistant, Thread 를 만들고 Thread에 사용자의 Message를 입력한 다음 Run을 만들어 실행하세요. (공식 문서의 [QuickStart](https://platform.openai.com/docs/assistants/quickstart) 를 보면 전체적인 흐름을 파악할 수 있습니다.)
            - [ ] Assistant, Thread, Run 이 화면 리렌더링 이후에도 유지될 수 있도록 session_state 을 사용합니다. ([Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state) 공식 문서 참고)
            - [ ] TIP : 주피터 노트북에서 할 때는 간단하지만 Streamlit과 함께 인터페이스에 적용하는 것은 생각보다 복잡합니다. 강의의 내용을 충분히 숙지한 후, [OpenAI Assistant](https://platform.openai.com/docs/assistants/overview) 공식문서를 천천히 읽어보면서 진행하세요.
            - [ ] TIP 2 : OpenAI Dashboard의 Threads 탭을 보면 생성된 Thread의 목록과 Thread 내에 있는 Message를 볼 수 있어서 개발 중에 유용하게 사용할 수 있습니다.
                - 기본적으로 Threads 탭이 비활성화 되어 있어서 설정에서 활성화해주어야 합니다.
                - 먼저, [OpenAI Organization Settings](https://platform.openai.com/settings/organization/general) 로 접속합니다.
                - Features and capabilities > Threads 에서 옵션을 Visible to organization owners 혹은 Visible to everyone 로 설정하고 밑에 Save 까지 눌러주세요.
                - 이제 OpenAI Dashboard 왼쪽 탭에 활성화된 [Threads 탭](https://platform.openai.com/threads)을 선택하면 스레드 목록이 보입니다.
        - [ ] 대화 기록을 표시하는 Streamlit 을 사용하여 유저 인터페이스를 제공하세요.
        - [ ] 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
        - [ ] st.sidebar를 사용하여 Streamlit app 의 코드과 함께 깃허브 리포지토리에 링크를 넣습니다.
    """
    )
