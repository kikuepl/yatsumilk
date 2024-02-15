import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

load_dotenv()

st.sidebar.title('タスク変更')

task_option = st.sidebar.selectbox(
    'タスクの選択',
    ('返信生成', 'スコアリング', "スコアリング + 返信生成")
)

scoring_number = 0

if task_option == "スコアリング":
    scoring_options = {
        "0": "knowledge",
        "1": "love",
        "2": "recommend",
        "3": "sales",
        "4": "全部"
    }

    scoring_number = st.sidebar.selectbox(
        "スコアリングタイプを選択してください:",
        options=list(scoring_options.keys()),
        format_func=lambda x: scoring_options[x] 
    )

    if scoring_number != "4":  # "全部"以外の場合
        with open(f"prompt/scoring/{scoring_options[scoring_number]}.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    else:  # "全部"の場合
        with open("prompt/all_scoring.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()

elif task_option == "返信生成":
    with open(f"prompt/reply.txt","r",encoding="utf-8") as f:
         system_prompt = f.read()

else:
    with open(f"prompt/all.txt","r",encoding="utf-8") as f:
         system_prompt = f.read()

st.title(task_option)

st.sidebar.title('APIキーの入力')

API_KEY = st.sidebar.text_input(
    "APIキーを入力してください"
)

if not (API_KEY or os.environ["OPENAI_API_KEY"]):
    st.write("APIキーを入力してください")
    exit()

def create_agent_chain():
    chat = ChatOpenAI(
            model_name = os.environ["OPENAI_API_MODEL"] ,
            temperature = os.environ["OPENAI_API_TEMPERATURE"],
            api_key = API_KEY,
            streaming = True
    )
    
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name = "memory")],
    }
    
    memory = ConversationBufferMemory(memory_key = "memory", return_messages = True)
    
    tools = load_tools(["wikipedia"])
    return initialize_agent(
        tools, 
        chat, 
        agent = AgentType.OPENAI_FUNCTIONS,
        agent_kwargs = agent_kwargs, 
        memory = memory, 
    )
if "messages" not in st.session_state:
    st.session_state.messages = []


if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())

        response = st.session_state.agent_chain.run(system_prompt + prompt, callbacks = [callback])
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})