# enhanced_health_mem0_neo4j.py
# 完善的Mem0健康助理 - 使用Neo4j图数据库和Qdrant向量数据库

import os
import time
from mem0 import Memory
from openai import OpenAI
import streamlit as st
import json

# API配置
API_KEY = "sk-21LnRSoxMhJcF51LwUHcQzUsc9KSbTMUgwBD9db1wfyobpQu"
BASE_URL = "https://api.aiclaude.site/v1"

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_BASE_URL"] = BASE_URL

COLLECTION = f"health_graph_{int(time.time())}"

# 完整的Mem0配置 - 同时使用Neo4j和Qdrant
full_config = {
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
            "api_key": API_KEY
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": COLLECTION,
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "Shucheng6"
        }
    }
}

# 简化配置（仅向量存储）
simple_config = {
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
            "api_key": API_KEY
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

class EnhancedHealthBot:
    """增强版健康助理，正确使用Neo4j图存储"""
    
    def __init__(self, user_id, use_graph_store=True):
        self.user_id = user_id
        self.use_graph_store = use_graph_store
        
        # 根据选择使用不同的配置
        chosen_config = full_config if use_graph_store else simple_config
        
        try:
            self.memory = Memory.from_config(chosen_config)
            
            # 如果使用图存储，启用图功能
            if use_graph_store and hasattr(self.memory, 'enable_graph'):
                self.memory.enable_graph = True
                
            self.config_type = "Neo4j图+Qdrant向量" if use_graph_store else "Qdrant向量"
            st.success(f"✅ 初始化成功 ({self.config_type})")
            
        except Exception as e:
            st.error(f"初始化失败: {e}")
            # 降级到纯向量存储
            self.memory = Memory.from_config(simple_config)
            self.use_graph_store = False
            self.config_type = "Qdrant向量(降级)"
            st.warning("已降级到纯向量存储模式")
        
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.conversation_history = []
    
    def add_health_memory(self, message, category=None):
        """添加健康记忆，包含分类信息"""
        metadata = {"user_id": self.user_id}
        if category:
            metadata["category"] = category
        
        try:
            # 添加到Mem0（会同时存储到向量和图）
            result = self.memory.add(
                message,
                user_id=self.user_id,
                metadata=metadata
            )
            return result
        except Exception as e:
            st.error(f"添加记忆失败: {e}")
            return None
    
    def get_all_memories(self):
        """获取所有记忆（向量存储）"""
        try:
            # 获取向量存储中的记忆
            vector_memories = self.memory.get_all(user_id=self.user_id)
            
            memories_list = []
            if isinstance(vector_memories, dict) and 'results' in vector_memories:
                memories_list = vector_memories['results']
            elif isinstance(vector_memories, list):
                memories_list = vector_memories
            
            return memories_list
        except Exception as e:
            print(f"获取记忆失败: {e}")
            return []
    
    def get_graph_data(self):
        """获取图数据库中的关系数据"""
        if not self.use_graph_store:
            return None
        
        try:
            if hasattr(self.memory, 'graph'):
                graph = self.memory.graph
                
                # 根据测试结果，get_all需要filters参数
                if hasattr(graph, 'get_all'):
                    try:
                        # 尝试获取用户相关的图数据
                        filters = {"user_id": self.user_id}
                        graph_data = graph.get_all(filters)
                        return graph_data
                    except Exception as e:
                        print(f"获取图数据失败: {e}")
                        
                        # 尝试其他方法
                        if hasattr(graph, 'search'):
                            try:
                                # 使用search方法获取图数据
                                graph_data = graph.search(
                                    query="健康",
                                    user_id=self.user_id,
                                    limit=10
                                )
                                return graph_data
                            except Exception as e2:
                                print(f"图搜索失败: {e2}")
                                
        except Exception as e:
            print(f"访问图存储失败: {e}")
        
        return None
    
    def search_memories(self, query, limit=5):
        """搜索相关记忆"""
        try:
            # 使用Mem0的搜索功能（会同时搜索向量和图）
            search_result = self.memory.search(
                query=query,
                user_id=self.user_id,
                limit=limit
            )
            
            memories = []
            if isinstance(search_result, dict) and 'results' in search_result:
                memories = search_result['results']
            elif isinstance(search_result, list):
                memories = search_result
            
            return memories
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def categorize_message(self, message):
        """自动分类健康信息"""
        categories = {
            "症状": ["头痛", "发烧", "疼痛", "不适", "难受", "痛", "晕", "累"],
            "用药": ["药", "吃药", "服用", "医生建议", "处方"],
            "检查": ["血压", "血糖", "体温", "检查", "化验", "指标"],
            "运动": ["跑步", "锻炼", "运动", "健身", "散步"],
            "饮食": ["吃", "饮食", "营养", "食物", "喝"],
            "过敏": ["过敏", "过敏原", "过敏史"],
            "家族史": ["父亲", "母亲", "家族", "遗传", "病史"]
        }
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in message:
                    return category
        return "其他"
    
    def chat(self, message):
        """处理对话"""
        # 添加到对话历史
        self.conversation_history.append(f"用户: {message}")
        
        # 分类并存储
        category = self.categorize_message(message)
        self.add_health_memory(f"用户[{category}]: {message}", category)
        
        # 构建上下文
        context = self._build_context(message)
        
        # 调用GPT生成回复
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": f"""你是专业的健康助理。
存储模式: {self.config_type}
基于用户的健康历史和当前问题，提供专业、温暖的健康建议。
回答简洁明了，100字以内。
如果发现潜在健康风险，温和地提醒就医。"""
                    },
                    {
                        "role": "user", 
                        "content": f"{context}\n\n当前问题: {message}"
                    }
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            reply = response.choices[0].message.content.strip()
            
        except Exception as e:
            reply = f"抱歉，处理请求时出错: {str(e)[:100]}"
        
        # 存储助理回复
        self.conversation_history.append(f"助理: {reply}")
        self.add_health_memory(f"助理回复: {reply}", "建议")
        
        return reply
    
    def _build_context(self, message):
        """构建对话上下文"""
        context_parts = []
        
        # 1. 最近的对话历史
        if self.conversation_history:
            recent = self.conversation_history[-6:]  # 最近3轮
            context_parts.append("=== 最近对话 ===")
            context_parts.extend(recent)
        
        # 2. 搜索相关记忆
        memories = self.search_memories(message, limit=5)
        if memories:
            context_parts.append("\n=== 相关健康记录 ===")
            for memory in memories:
                text = memory.get('memory', str(memory))
                score = memory.get('score', 0)
                context_parts.append(f"- {text} (相关度: {score:.2f})")
        
        # 3. 如果使用图存储，尝试获取关系信息
        if self.use_graph_store:
            graph_data = self.get_graph_data()
            if graph_data:
                context_parts.append("\n=== 健康关系图谱 ===")
                context_parts.append(f"图数据: {str(graph_data)[:200]}...")
        
        return "\n".join(context_parts)
    
    def get_health_summary(self):
        """生成健康摘要"""
        memories = self.get_all_memories()
        
        if not memories:
            return "暂无健康记录"
        
        # 按类别整理记忆
        categories = {}
        for memory in memories:
            text = memory.get('memory', '')
            # 尝试从文本中提取类别
            if '[' in text and ']' in text:
                cat = text[text.find('[')+1:text.find(']')]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(text)
            else:
                if '其他' not in categories:
                    categories['其他'] = []
                categories['其他'].append(text)
        
        # 生成摘要
        summary_parts = ["📋 健康档案摘要\n"]
        for cat, items in categories.items():
            if items:
                summary_parts.append(f"\n**{cat}**")
                for item in items[:3]:  # 每类最多显示3条
                    summary_parts.append(f"• {item}")
        
        return "\n".join(summary_parts)
    
    def clear_all(self):
        """清除所有数据"""
        try:
            # 清除向量存储
            memories = self.get_all_memories()
            for memory in memories:
                if 'id' in memory:
                    self.memory.delete(memory['id'])
            
            # 清除对话历史
            self.conversation_history = []
            
            # 如果有图存储，尝试清除
            if self.use_graph_store and hasattr(self.memory, 'graph'):
                try:
                    if hasattr(self.memory.graph, 'reset'):
                        self.memory.graph.reset()
                except Exception as e:
                    print(f"清除图数据失败: {e}")
            
            return True
        except Exception as e:
            print(f"清除失败: {e}")
            return False

# ==================== Streamlit界面 ====================

st.set_page_config(
    page_title="AI健康助理 - Neo4j增强版",
    page_icon="🏥",
    layout="wide"
)

# 标题栏
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.title("🏥 AI健康助理 (Neo4j增强版)")
with col2:
    st.caption("GPT-4o-mini")
with col3:
    st.caption(f"集合: {COLLECTION[:15]}...")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    
    user_id = st.text_input("用户ID", value="health_user_001")
    
    # 存储模式选择
    st.subheader("🗄️ 存储模式")
    storage_mode = st.radio(
        "选择存储方式",
        ["Neo4j图+Qdrant向量", "仅Qdrant向量"],
        index=0
    )
    use_graph = (storage_mode == "Neo4j图+Qdrant向量")
    
    # 显示存储信息
    if use_graph:
        st.info("""
        🌐 **图数据库模式**
        - Neo4j存储实体关系
        - Qdrant存储向量嵌入
        - 支持复杂关系查询
        """)
    else:
        st.success("""
        📁 **向量数据库模式**
        - 仅使用Qdrant
        - 基于语义相似度搜索
        """)
    
    st.divider()
    
    # 记忆统计
    if "bot" in st.session_state:
        bot = st.session_state.bot
        memories = bot.get_all_memories()
        
        st.subheader("🧠 记忆统计")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("总记忆", len(memories))
        with col2:
            st.metric("对话轮数", len(bot.conversation_history)//2)
        
        # 健康摘要
        with st.expander("📊 健康档案", expanded=False):
            summary = bot.get_health_summary()
            st.markdown(summary)
        
        # 图数据（如果可用）
        if use_graph:
            with st.expander("🌐 图数据", expanded=False):
                graph_data = bot.get_graph_data()
                if graph_data:
                    st.json(graph_data)
                else:
                    st.info("暂无图数据")
    
    st.divider()
    
    # 快速操作
    st.subheader("⚡ 快速操作")
    
    # 预设问题
    quick_questions = {
        "症状记录": [
            "我最近经常头痛",
            "昨天开始发烧38度",
            "胃部不适已经3天了"
        ],
        "健康咨询": [
            "血压140/90需要注意什么",
            "如何改善睡眠质量",
            "感冒了该怎么办"
        ],
        "生活习惯": [
            "我每天跑步5公里",
            "最近开始低盐饮食",
            "戒烟已经一个月了"
        ]
    }
    
    for category, questions in quick_questions.items():
        st.caption(f"**{category}**")
        cols = st.columns(1)
        for q in questions:
            if st.button(q, key=f"q_{q}", use_container_width=True):
                st.session_state.pending_question = q
    
    st.divider()
    
    # 管理按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 新对话", use_container_width=True):
            if "bot" in st.session_state:
                st.session_state.bot.conversation_history = []
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🗑️ 清除所有", use_container_width=True):
            if "bot" in st.session_state:
                if st.session_state.bot.clear_all():
                    st.success("已清除所有数据")
                else:
                    st.warning("清除失败")
                st.session_state.messages = []
                st.rerun()

# 主界面
# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化或重新初始化机器人
need_reinit = False
if "bot" in st.session_state:
    if st.session_state.bot.use_graph_store != use_graph:
        need_reinit = True

if "bot" not in st.session_state or need_reinit:
    with st.spinner("初始化健康助理..."):
        try:
            st.session_state.bot = EnhancedHealthBot(user_id, use_graph)
        except Exception as e:
            st.error(f"初始化失败: {e}")
            st.stop()

# 显示欢迎消息
if not st.session_state.messages:
    welcome_msg = f"""
👋 你好！我是你的AI健康助理。

我使用 **{st.session_state.bot.config_type}** 存储系统来：
• 📝 记录你的健康信息和症状
• 🔍 智能分析健康趋势
• 💊 提供个性化健康建议
• 🏥 提醒及时就医

请告诉我你的健康状况，或点击左侧的快速问题开始。
"""
    st.session_state.messages.append({
        "role": "assistant",
        "content": welcome_msg
    })

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理待处理的问题
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question
    
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    
    # 生成并显示回复
    with st.chat_message("assistant"):
        with st.spinner("分析中..."):
            response = st.session_state.bot.chat(question)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()

# 聊天输入框
if prompt := st.chat_input("请描述你的健康状况或提出问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 生成并显示回复
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = st.session_state.bot.chat(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()

# 底部信息栏
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.caption("⚠️ 健康建议仅供参考，严重症状请及时就医")
with col2:
    if "bot" in st.session_state:
        st.caption(f"💾 {st.session_state.bot.config_type}")
with col3:
    st.caption(f"用户: {user_id}")

# 运行命令：
# streamlit run enhanced_health_mem0_neo4j.py