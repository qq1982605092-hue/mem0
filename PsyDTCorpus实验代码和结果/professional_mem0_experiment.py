"""
专业版Mem0+PsyDTCorpus心理咨询实验
基于CPsyCoun评估框架和REBT专业标准
解决了数据处理和评估指标的问题
"""

import json
import statistics
import time
from collections import defaultdict
from typing import Dict, List, Any
from openai import OpenAI
from mem0 import MemoryClient
from datetime import datetime
import re

# ===================== 配置 =====================
# 使用你提供的API配置
OPENAI_API_KEY = "sk-21LnRSoxMhJcF51LwUHcQzUsc9KSbTMUgwBD9db1wfyobpQu"
OPENAI_BASE_URL = "https://api.aiclaude.site/v1"
MEM0_API_KEY = "m0-zrcMJP7AjsYZ7jzysLN0VMh3XLiaXWX5Ar6xt5bJ"

# 实验参数
TEST_SESSIONS = 1  # 测试会话数量
CONTEXT_LIMIT = 3   # 基线模型的上下文限制
    
# ===================== 数据处理器 =====================
class PsyDTCorpusProcessor:
    """PsyDTCorpus数据处理器 - 正确处理完整会话"""
    
    @staticmethod
    def load_sessions(file_path: str) -> List[Dict]:
        """加载完整会话数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_sessions = []
        for session in data[:TEST_SESSIONS]:
            # 提取系统提示（REBT框架）
            rebt_prompt = ""
            dialogue_turns = []
            
            for msg in session['messages']:
                if msg['role'] == 'system':
                    rebt_prompt = msg['content']
                elif msg['role'] in ['user', 'assistant']:
                    dialogue_turns.append({
                        'role': msg['role'],  # user=患者, assistant=治疗师
                        'content': msg['content'].strip(),
                        'speaker': 'Patient' if msg['role'] == 'user' else 'Therapist'
                    })
            
            if len(dialogue_turns) >= 6:  # 至少3轮完整对话
                processed_sessions.append({
                    'session_id': session['id'],
                    'theme': session['normalizedTag'],
                    'rebt_framework': rebt_prompt,
                    'dialogue_turns': dialogue_turns,
                    'patient_turns': [t for t in dialogue_turns if t['role'] == 'user'],
                    'therapist_turns': [t for t in dialogue_turns if t['role'] == 'assistant']
                })
        
        print(f"✅ 加载了 {len(processed_sessions)} 个有效会话")
        return processed_sessions

class REBTStageAnalyzer:
    """REBT阶段分析器"""
    
    @staticmethod
    def identify_rebt_stage(content: str) -> int:
        """识别REBT治疗阶段 (1-4)"""
        
        # 阶段1关键词：检查非理性信念
        stage1_keywords = ['感受', '情绪', '想法', '发生了什么', '告诉我', '具体']
        
        # 阶段2关键词：与非理性信念辩论
        stage2_keywords = ['为什么', '证据', '合理吗', '真的是', '必须', '应该', '质疑']
        
        # 阶段3关键词：得出合理信念
        stage3_keywords = ['可以', '其他方式', '重新', '理性', '现实', '更好的']
        
        # 阶段4关键词：迁移应用
        stage4_keywords = ['尝试', '练习', '应用', '以后', '日常', '坚持']
        
        content_lower = content.lower()
        
        # 计算各阶段匹配分数
        scores = {
            1: sum(1 for kw in stage1_keywords if kw in content_lower),
            2: sum(1 for kw in stage2_keywords if kw in content_lower),
            3: sum(1 for kw in stage3_keywords if kw in content_lower),
            4: sum(1 for kw in stage4_keywords if kw in content_lower)
        }
        
        # 返回得分最高的阶段
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 1

# ===================== 基线模型（无记忆） =====================
class BaselineREBTModel:
    """基线REBT治疗师模型 - 无长期记忆"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        self.total_calls = 0
        
    def generate_response(self, patient_input: str, context: List[Dict], theme: str) -> str:
        """生成治疗师回复 - 仅基于有限上下文"""
        
        # 构建有限上下文（模拟短期记忆限制）
        context_str = f"咨询主题：{theme}\n\n"
        
        if context:
            # 只使用最近几轮对话
            recent_context = context[-CONTEXT_LIMIT:] if len(context) > CONTEXT_LIMIT else context
            context_str += "最近对话：\n"
            for turn in recent_context:
                speaker = turn['speaker']
                content = turn['content'][:100] + "..." if len(turn['content']) > 100 else turn['content']
                context_str += f"{speaker}: {content}\n"
        
        # REBT治疗师提示
        system_prompt = """你是一位专业的REBT（理情行为疗法）心理咨询师。
你只能看到最近几轮对话，没有患者的完整治疗历史。
请基于有限的信息，运用REBT的ABC理论提供专业的咨询回复。

REBT核心原则：
- A (事件) → B (信念) → C (情绪后果)
- 识别和质疑非理性信念
- 帮助建立理性信念
- 促进情绪和行为的积极改变"""

        user_prompt = f"""
当前情况：
{context_str}

患者刚才说：{patient_input}

请作为REBT治疗师回复（注意：你没有完整的治疗历史）："""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            self.total_calls += 1
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"基线模型回复失败：{str(e)}"

# ===================== Mem0增强模型 =====================
class Mem0REBTModel:
    """Mem0增强REBT治疗师模型 - 具有完整治疗记忆"""
    
    def __init__(self):
        self.memory = MemoryClient(api_key=MEM0_API_KEY)
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        self.total_calls = 0
        self.stored_sessions = set()
        
    def initialize_session_memory(self, session: Dict) -> bool:
        """为会话初始化记忆"""
        session_id = f"session_{session['session_id']}"
        
        if session_id in self.stored_sessions:
            return True
            
        try:
            print(f"🧠 为会话 {session_id} 建立记忆...")
            
            # 存储会话背景
            background = f"""
患者咨询主题：{session['theme']}
REBT治疗框架：理情行为疗法，包含四个阶段
会话总轮次：{len(session['dialogue_turns'])}轮
治疗目标：帮助患者识别和改变非理性信念，建立理性思维模式
"""
            
            self.memory.add(
                messages=[{"role": "user", "content": background}],
                user_id=session_id,
                metadata={"type": "session_background", "theme": session['theme']}
            )
            
            # 存储每轮对话（完整治疗历史）
            for i, turn in enumerate(session['dialogue_turns']):
                speaker = turn['speaker']
                content = turn['content']
                rebt_stage = REBTStageAnalyzer.identify_rebt_stage(content)
                
                memory_content = f"""
{speaker}（第{i+1}轮，REBT阶段{rebt_stage}）：
{content}
"""
                
                # 为患者表述添加特殊标记
                if speaker == 'Patient':
                    memory_content += "\n[关键患者信息 - 需重点关注]"
                    
                self.memory.add(
                    messages=[{"role": "user", "content": memory_content}],
                    user_id=session_id,
                    metadata={
                        "type": "dialogue_turn",
                        "turn_index": i,
                        "speaker": speaker,
                        "rebt_stage": rebt_stage
                    }
                )
                
                if i % 5 == 0:
                    print(f"  📝 已存储 {i+1}/{len(session['dialogue_turns'])} 轮...")
                
                time.sleep(0.2)  # 避免API限制
            
            self.stored_sessions.add(session_id)
            print(f"✅ 会话 {session_id} 记忆建立完成")
            return True
            
        except Exception as e:
            print(f"❌ 记忆建立失败：{str(e)}")
            return False
    
    def generate_response(self, patient_input: str, session_id: int, theme: str, context: List[Dict]) -> str:
        """生成治疗师回复 - 基于完整记忆"""
        
        memory_session_id = f"session_{session_id}"
        
        # 搜索相关记忆
        relevant_memories = []
        try:
            search_results = self.memory.search(
                query=patient_input,
                user_id=memory_session_id,
                limit=8
            )
            relevant_memories = search_results if search_results else []
            print(f"🔍 检索到 {len(relevant_memories)} 条相关记忆")
        except Exception as e:
            print(f"⚠️ 记忆搜索失败：{str(e)}")
        
        # 构建增强上下文
        context_str = f"患者咨询主题：{theme}\n"
        context_str += f"治疗方法：REBT（理情行为疗法）\n\n"
        
        # 添加记忆信息
        if relevant_memories:
            context_str += "📚 相关治疗记忆：\n"
            for i, memory in enumerate(relevant_memories[:6], 1):
                memory_text = memory.get('memory', '').strip()
                context_str += f"{i}. {memory_text}\n"
            context_str += "\n"
        
        # 添加当前上下文
        if context:
            context_str += "💬 当前对话：\n"
            for turn in context[-3:]:  # 最近3轮
                context_str += f"{turn['speaker']}: {turn['content']}\n"
        
        # 增强的REBT治疗师提示
        system_prompt = """你是一位经验丰富的REBT心理咨询师，拥有患者的完整治疗记忆。
你能够：
1. 利用完整的治疗历史理解患者问题
2. 准确应用REBT的ABC理论框架
3. 识别患者的非理性信念模式
4. 提供有针对性的治疗干预
5. 保持治疗的连续性和一致性

REBT四个治疗阶段：
1. 检查非理性信念和自我挫败式思维
2. 与非理性信念辩论
3. 得出合理信念，学会理性思维  
4. 迁移应用治疗收获

请基于完整的治疗记忆提供专业、连贯的咨询回复。"""

        user_prompt = f"""
基于完整的治疗记忆和当前情况：

{context_str}

患者刚才说：{patient_input}

请作为经验丰富的REBT治疗师回复："""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            self.total_calls += 1
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Mem0模型回复失败：{str(e)}"

# ===================== 专业评估系统 =====================
class ProfessionalREBTEvaluator:
    """基于CPsyCoun框架的专业评估器"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        
    def evaluate_comprehensive(self, patient_input: str, baseline_response: str, 
                             mem0_response: str, context: List[Dict]) -> Dict:
        """综合评估两个模型的回复"""
        
        # 1. CPsyCoun四维评估
        cpsy_eval = self.evaluate_cpsy_dimensions(patient_input, baseline_response, mem0_response)
        
        # 2. REBT专业性评估
        rebt_eval = self.evaluate_rebt_competence(patient_input, baseline_response, mem0_response, context)
        
        # 3. 治疗连续性评估
        continuity_eval = self.evaluate_therapeutic_continuity(baseline_response, mem0_response, context)
        
        return {
            'cpsy_dimensions': cpsy_eval,
            'rebt_competence': rebt_eval,
            'therapeutic_continuity': continuity_eval
        }
    
    def evaluate_cpsy_dimensions(self, patient_input: str, baseline_response: str, mem0_response: str) -> Dict:
        """CPsyCoun四维评估：全面性、专业性、真实性、安全性"""
        
        prompt = f"""
作为心理咨询专家，请按CPsyCoun框架评估两个AI治疗师回复的质量。

患者表述：{patient_input}

回复A（基线模型）：{baseline_response}
回复B（记忆增强模型）：{mem0_response}

请从以下四个维度评分（0-5分）：

1. **Comprehensiveness (全面性)**：回复是否全面覆盖患者关切
2. **Professionalism (专业性)**：是否符合心理咨询专业标准
3. **Authenticity (真实性)**：治疗师-患者互动的自然真实程度
4. **Safety (安全性)**：是否遵循心理咨询伦理和安全准则

返回JSON格式：
{{
    "baseline_scores": {{
        "comprehensiveness": 分数,
        "professionalism": 分数,
        "authenticity": 分数,
        "safety": 分数
    }},
    "mem0_scores": {{
        "comprehensiveness": 分数,
        "professionalism": 分数,
        "authenticity": 分数,
        "safety": 分数
    }},
    "dimension_analysis": {{
        "comprehensiveness": "全面性对比分析",
        "professionalism": "专业性对比分析",
        "authenticity": "真实性对比分析",
        "safety": "安全性对比分析"
    }}
}}
"""
        
        return self.get_structured_evaluation(prompt)
    
    def evaluate_rebt_competence(self, patient_input: str, baseline_response: str, 
                                mem0_response: str, context: List[Dict]) -> Dict:
        """REBT专业能力评估"""
        
        context_summary = self.summarize_context(context)
        
        prompt = f"""
作为REBT专家，评估两个AI治疗师的REBT专业能力。

治疗背景：{context_summary}
患者表述：{patient_input}

回复A（基线模型）：{baseline_response}
回复B（记忆增强模型）：{mem0_response}

请评估以下REBT专业维度（0-5分）：

1. **ABC模型应用**：是否正确识别A(事件)-B(信念)-C(后果)
2. **非理性信念识别**：是否准确识别患者的非理性信念
3. **辩论技术使用**：是否有效质疑和挑战非理性信念
4. **理性重建能力**：是否帮助患者建立理性信念
5. **治疗阶段把握**：是否正确把握当前REBT治疗阶段

返回JSON格式：
{{
    "baseline_rebt": {{
        "abc_application": 分数,
        "irrational_identification": 分数,
        "disputation_technique": 分数,
        "rational_reconstruction": 分数,
        "stage_awareness": 分数
    }},
    "mem0_rebt": {{
        "abc_application": 分数,
        "irrational_identification": 分数,
        "disputation_technique": 分数,
        "rational_reconstruction": 分数,
        "stage_awareness": 分数
    }},
    "rebt_analysis": "REBT专业能力详细对比分析"
}}
"""
        
        return self.get_structured_evaluation(prompt)
    
    def evaluate_therapeutic_continuity(self, baseline_response: str, mem0_response: str, context: List[Dict]) -> Dict:
        """治疗连续性评估"""
        
        if len(context) < 2:
            return {"note": "上下文不足，无法评估连续性"}
        
        context_summary = self.summarize_context(context)
        
        prompt = f"""
评估两个治疗师回复的治疗连续性和记忆利用效果。

之前的治疗历史：
{context_summary}

回复A（基线模型，无完整记忆）：{baseline_response}
回复B（记忆增强模型，有完整记忆）：{mem0_response}

请评估以下维度（0-5分）：

1. **治疗目标一致性**：是否与之前确立的治疗目标保持一致
2. **问题理解深度**：是否展现了对患者问题的深层理解
3. **历史信息整合**：是否有效整合了患者的历史信息
4. **治疗进展连贯性**：是否体现了治疗的逐步深入

返回JSON格式：
{{
    "baseline_continuity": {{
        "goal_consistency": 分数,
        "problem_understanding": 分数,
        "history_integration": 分数,
        "progress_coherence": 分数
    }},
    "mem0_continuity": {{
        "goal_consistency": 分数,
        "problem_understanding": 分数,
        "history_integration": 分数,
        "progress_coherence": 分数
    }},
    "memory_advantage": "记忆增强模型的具体优势分析",
    "continuity_comparison": "治疗连续性整体对比"
}}
"""
        
        return self.get_structured_evaluation(prompt)
    
    def get_structured_evaluation(self, prompt: str) -> Dict:
        """获取结构化评估结果"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {"error": f"评估失败：{str(e)}"}
    
    def summarize_context(self, context: List[Dict]) -> str:
        """总结对话上下文"""
        if not context:
            return "无对话历史"
            
        summary = "对话历史摘要：\n"
        for i, turn in enumerate(context[-5:], 1):  # 最近5轮
            content = turn['content'][:80] + "..." if len(turn['content']) > 80 else turn['content']
            summary += f"{i}. {turn['speaker']}: {content}\n"
        
        return summary

# ===================== 实验执行器 =====================
class REBTMemoryExperiment:
    """REBT记忆实验主控制器"""
    
    def __init__(self):
        self.baseline_model = BaselineREBTModel()
        self.mem0_model = Mem0REBTModel()
        self.evaluator = ProfessionalREBTEvaluator()
        self.results = []
        
    def run_experiment(self, sessions: List[Dict]):
        """运行完整实验"""
        print("🧠 开始REBT记忆实验...")
        print(f"📊 总共测试 {len(sessions)} 个会话")
        
        for i, session in enumerate(sessions, 1):
            print(f"\n{'='*60}")
            print(f"🔬 实验进度：{i}/{len(sessions)} - 会话 {session['session_id']} ({session['theme']})")
            print(f"{'='*60}")
            
            try:
                session_result = self.run_single_session(session)
                self.results.append(session_result)
                
                print(f"✅ 会话 {session['session_id']} 实验完成")
                
            except Exception as e:
                print(f"❌ 会话 {session['session_id']} 实验失败：{str(e)}")
                continue
        
        return self.generate_final_report()
    
    def run_single_session(self, session: Dict) -> Dict:
        """运行单个会话实验"""
        
        # 为Mem0模型建立记忆
        memory_success = self.mem0_model.initialize_session_memory(session)
        if not memory_success:
            raise Exception("记忆初始化失败")
        
        time.sleep(2)  # 等待记忆建立完成
        
        session_result = {
            'session_id': session['session_id'],
            'theme': session['theme'],
            'total_turns': len(session['dialogue_turns']),
            'turn_evaluations': []
        }
        
        # 逐轮对比评估
        for i, turn in enumerate(session['dialogue_turns']):
            if turn['role'] == 'user':  # 患者发言轮次
                
                # 获取当前上下文
                current_context = session['dialogue_turns'][:i]
                
                # 基线模型回复
                baseline_response = self.baseline_model.generate_response(
                    patient_input=turn['content'],
                    context=current_context,
                    theme=session['theme']
                )
                
                # Mem0模型回复
                mem0_response = self.mem0_model.generate_response(
                    patient_input=turn['content'],
                    session_id=session['session_id'],
                    theme=session['theme'],
                    context=current_context
                )
                
                # 专业评估
                evaluation = self.evaluator.evaluate_comprehensive(
                    patient_input=turn['content'],
                    baseline_response=baseline_response,
                    mem0_response=mem0_response,
                    context=current_context
                )
                
                turn_result = {
                    'turn_index': i,
                    'patient_input': turn['content'][:100] + "..." if len(turn['content']) > 100 else turn['content'],
                    'baseline_response': baseline_response,
                    'mem0_response': mem0_response,
                    'evaluation': evaluation
                }
                
                session_result['turn_evaluations'].append(turn_result)
                
                print(f"  📝 完成第 {i+1} 轮评估")
                time.sleep(1)  # 避免API限制
        
        return session_result
    
    def generate_final_report(self) -> str:
        """生成最终实验报告"""
        
        if not self.results:
            return "❌ 无有效实验结果"
        
        # 收集所有评估分数
        baseline_scores = defaultdict(list)
        mem0_scores = defaultdict(list)
        
        for session_result in self.results:
            for turn_eval in session_result['turn_evaluations']:
                eval_data = turn_eval['evaluation']
                
                # CPsyCoun维度分数
                if 'cpsy_dimensions' in eval_data:
                    cpsy_data = eval_data['cpsy_dimensions']
                    if 'baseline_scores' in cpsy_data:
                        for dim, score in cpsy_data['baseline_scores'].items():
                            baseline_scores[f'cpsy_{dim}'].append(score)
                    if 'mem0_scores' in cpsy_data:
                        for dim, score in cpsy_data['mem0_scores'].items():
                            mem0_scores[f'cpsy_{dim}'].append(score)
                
                # REBT专业性分数
                if 'rebt_competence' in eval_data:
                    rebt_data = eval_data['rebt_competence']
                    if 'baseline_rebt' in rebt_data:
                        for dim, score in rebt_data['baseline_rebt'].items():
                            baseline_scores[f'rebt_{dim}'].append(score)
                    if 'mem0_rebt' in rebt_data:
                        for dim, score in rebt_data['mem0_rebt'].items():
                            mem0_scores[f'rebt_{dim}'].append(score)
        
        # 生成报告
        report = f"""
# 🧠 Mem0 + PsyDTCorpus REBT心理咨询实验报告

## 📊 实验概况
- **实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **测试会话数**: {len(self.results)}
- **评估轮次数**: {sum(len(r['turn_evaluations']) for r in self.results)}
- **评估框架**: CPsyCoun + REBT专业指标

## 🎯 核心发现

### CPsyCoun四维评估结果

| 维度 | 基线模型 | Mem0模型 | 提升幅度 | 显著性 |
|------|----------|----------|----------|--------|
"""
        
        cpsy_dimensions = ['comprehensiveness', 'professionalism', 'authenticity', 'safety']
        dimension_names = {'comprehensiveness': '全面性', 'professionalism': '专业性', 
                          'authenticity': '真实性', 'safety': '安全性'}
        
        for dim in cpsy_dimensions:
            baseline_key = f'cpsy_{dim}'
            mem0_key = f'cpsy_{dim}'
            
            if baseline_key in baseline_scores and mem0_key in mem0_scores:
                baseline_avg = statistics.mean(baseline_scores[baseline_key])
                mem0_avg = statistics.mean(mem0_scores[mem0_key])
                improvement = ((mem0_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
                significance = "显著" if abs(improvement) > 10 else "一般"
                
                report += f"| {dimension_names[dim]} | {baseline_avg:.3f} | {mem0_avg:.3f} | {improvement:+.1f}% | {significance} |\n"
        
        # REBT专业性评估
        report += f"""
### REBT专业能力评估结果

| REBT维度 | 基线模型 | Mem0模型 | 提升幅度 | 分析 |
|----------|----------|----------|----------|------|
"""
        
        rebt_dimensions = ['abc_application', 'irrational_identification', 'disputation_technique', 
                          'rational_reconstruction', 'stage_awareness']
        rebt_names = {'abc_application': 'ABC模型应用', 'irrational_identification': '非理性信念识别',
                     'disputation_technique': '辩论技术', 'rational_reconstruction': '理性重建',
                     'stage_awareness': '阶段把握'}
        
        overall_improvements = []
        
        for dim in rebt_dimensions:
            baseline_key = f'rebt_{dim}'
            mem0_key = f'rebt_{dim}'
            
            if baseline_key in baseline_scores and mem0_key in mem0_scores:
                baseline_avg = statistics.mean(baseline_scores[baseline_key])
                mem0_avg = statistics.mean(mem0_scores[mem0_key])
                improvement = ((mem0_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
                overall_improvements.append(improvement)
                
                analysis = "优秀" if improvement > 20 else "良好" if improvement > 10 else "一般"
                
                report += f"| {rebt_names[dim]} | {baseline_avg:.3f} | {mem0_avg:.3f} | {improvement:+.1f}% | {analysis} |\n"
        
        # 计算总体提升
        if overall_improvements:
            overall_improvement = statistics.mean(overall_improvements)
            
            report += f"""
## 🚀 核心结论

**整体性能提升：{overall_improvement:+.1f}%**

"""
            
            if overall_improvement > 25:
                report += """🎉 **记忆增强效果显著！**
- Mem0框架在REBT心理咨询中表现优异
- 长期记忆显著提升了治疗连续性和专业性
- 建议进一步优化记忆检索策略"""
            elif overall_improvement > 10:
                report += """✅ **记忆增强效果明显**
- Mem0框架带来了可观的性能提升
- 专业性和连续性有明显改善
- 仍有优化空间，建议针对性改进"""
            else:
                report += """🤔 **记忆增强效果有限**
- 当前记忆策略效果不够显著
- 建议重新设计记忆存储和检索机制
- 需要更深入的专业化改进"""
        
        report += f"""
## 📈 详细统计

**模型调用统计**:
- 基线模型调用：{self.baseline_model.total_calls} 次
- Mem0模型调用：{self.mem0_model.total_calls} 次

**实验覆盖范围**:
- 测试主题：{list(set(r['theme'] for r in self.results))}
- 平均对话轮次：{statistics.mean([r['total_turns'] for r in self.results]):.1f} 轮

## 💡 改进建议

1. **记忆优化策略**
   - 改进REBT阶段感知的记忆权重
   - 增强情绪关键信息的记忆保存
   - 优化ABC框架信息的结构化存储

2. **专业性提升**
   - 加强非理性信念识别的准确性
   - 改进辩论技术的应用效果
   - 提升治疗阶段转换的流畅性

3. **系统集成**
   - 开发实时的REBT阶段检测
   - 集成情绪分析增强记忆索引
   - 建立治疗效果评估反馈机制

---

*本报告展示了Mem0记忆框架在专业心理咨询场景中的应用效果。*
"""
        
        return report

# ===================== 主程序 =====================
def main():
    print("🎯 专业版Mem0+PsyDTCorpus REBT心理咨询实验")
    print("="*80)
    
    # 1. 加载数据

    DATASET_PATH = "PsyDTCorpus/PsyDTCorpus_train_mulit_turn_packing.json"
    data_file = "PsyDTCorpus/PsyDTCorpus_train_mulit_turn_packing.json"  # 替换为实际路径
    
    print("📂 加载数据...")
    sessions = PsyDTCorpusProcessor.load_sessions(data_file)
    
    if not sessions:
        print("❌ 没有找到有效的会话数据")
        return
    
    # 2. 运行实验
    experiment = REBTMemoryExperiment()
    final_report = experiment.run_experiment(sessions)
    
    # 3. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果
    results_file = f"rebt_memory_experiment_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(experiment.results, f, ensure_ascii=False, indent=2)
    
    # 保存报告
    report_file = f"rebt_memory_experiment_report_{timestamp}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(final_report)
    
    print("\n" + "="*80)
    print("🎉 实验完成！")
    print(f"📄 详细报告：{report_file}")
    print(f"📊 原始数据：{results_file}")
    print("="*80)
    
    # 显示报告
    print(final_report)

if __name__ == "__main__":
    main()