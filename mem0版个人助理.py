# final_working_health.py
# 最终工作版 - 基于调试结果的正确实现

import os
import time
from mem0 import Memory
from openai import OpenAI
import streamlit as st

# API配置
API_KEY = "sk-21LnRSoxMhJcF51LwUHcQzUsc9KSbTMUgwBD9db1wfyobpQu"
BASE_URL = "https://api.aiclaude.site/v1"

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_BASE_URL"] = BASE_URL

COLLECTION = f"health_1755158035"

# Mem0配置
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "api_key": API_KEY,
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": API_KEY,
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": COLLECTION,
        }
    }
}

# 健康助理类
class HealthBot:
    def __init__(self, user_id):
        self.user_id = user_id
        self.memory = Memory.from_config(config)
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.conversation_history = []  # 本地对话历史
    
    def extract_memories(self, search_result):
        """从搜索结果中提取记忆文本"""
        memories_text = []
        
        # 处理各种可能的返回格式
        if isinstance(search_result, dict):
            # 如果是字典，查找包含列表的键
            for key, value in search_result.items():
                if isinstance(value, list):
                    # 找到列表，提取内容
                    for item in value[:3]:  # 最多3条
                        if isinstance(item, dict):
                            text = item.get('memory') or item.get('text') or item.get('content') or str(item)
                        else:
                            text = str(item)
                        memories_text.append(text)
                    break
                elif isinstance(value, str):
                    memories_text.append(value)
                    break
        elif isinstance(search_result, list):
            # 如果直接是列表
            for item in search_result[:3]:
                if isinstance(item, dict):
                    text = item.get('memory') or item.get('text') or str(item)
                else:
                    text = str(item)
                memories_text.append(text)
        
        return memories_text
    
    def chat(self, message):
        # 添加到本地历史
        self.conversation_history.append(f"用户: {message}")
        
        # 存储到Mem0
        try:
            self.memory.add(f"用户: {message}", user_id=self.user_id)
        except:
            pass
        
        # 构建上下文 - 使用本地历史 + 记忆搜索
        context = ""
        
        # 1. 最近的对话历史
        if self.conversation_history:
            recent = self.conversation_history[-6:]  # 最近3轮对话
            context = "最近对话:\n" + "\n".join(recent) + "\n\n"
        
        # 2. 尝试搜索相关记忆
        try:
            search_result = self.memory.search(message, user_id=self.user_id, limit=3)
            memories = self.extract_memories(search_result)
            
            if memories:
                context += "相关记录:\n"
                for mem in memories:
                    context += f"- {mem}\n"
        except:
            # 如果搜索失败，仅使用对话历史
            pass
        
        # 调用GPT
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是专业的健康助理。基于对话历史和用户问题，提供专业、温暖的健康建议。回答简洁，100字以内。"},
                {"role": "user", "content": f"{context}\n当前问题: {message}"}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        reply = response.choices[0].message.content.strip()
        
        # 添加到历史
        self.conversation_history.append(f"助理: {reply}")
        
        # 存储到Mem0
        try:
            self.memory.add(f"助理: {reply}", user_id=self.user_id)
        except:
            pass
        
        return reply

# ==================== Streamlit界面 ====================

st.set_page_config(page_title="健康助理", page_icon="🏥", layout="wide")

# 标题栏
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.title("🏥 AI健康助理")
with col2:
    st.caption(f"GPT-4o-mini")
with col3:
    st.caption(f"集合: {COLLECTION[:15]}...")

# 侧边栏
with st.sidebar:
    user_id = st.text_input("用户ID", "user1")
    
    st.divider()
    
    # 快速提问
    st.subheader("💡 快速提问")
    questions = [
        "我最近头痛频繁",
        "血压高要注意什么",
        "如何改善睡眠",
        "感冒了怎么办",
        "缓解压力的方法"
    ]
    
    for q in questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state.pending_question = q
    
    st.divider()
    
    # 操作按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 新对话", use_container_width=True):
            if "bot" in st.session_state:
                st.session_state.bot.conversation_history = []
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("📊 历史", use_container_width=True):
            if "bot" in st.session_state and st.session_state.bot.conversation_history:
                st.info(f"共{len(st.session_state.bot.conversation_history)}条记录")

# 初始化
if "messages" not in st.session_state:
    st.session_state.messages = []

if "bot" not in st.session_state:
    with st.spinner("初始化中..."):
        try:
            st.session_state.bot = HealthBot(user_id)
            st.success("✅ 系统就绪", icon="✅")
        except Exception as e:
            st.error(f"初始化失败: {e}")
            st.stop()

# 欢迎消息
if not st.session_state.messages:
    welcome = """👋 你好！我是你的AI健康助理。

我可以帮你解答健康问题、分析症状、提供生活建议。

请描述你的健康状况或直接点击左侧的快速提问。"""
    st.session_state.messages.append({"role": "assistant", "content": welcome})

# 显示对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 处理快速提问
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question
    
    # 显示问题
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})
    
    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("分析中..."):
            answer = st.session_state.bot.chat(question)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# 聊天输入
if prompt := st.chat_input("请描述你的健康问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 生成回复
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = st.session_state.bot.chat(prompt)
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# 底部提示
st.markdown("---")
st.caption("⚠️ 健康建议仅供参考，严重症状请及时就医")

# 运行: streamlit run final_working_health.py