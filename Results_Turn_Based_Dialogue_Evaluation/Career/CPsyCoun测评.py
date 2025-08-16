# Turn-Based Dialogue Evaluation
# Dataset: CPsyCounE
# Model: GPT-4o-mini

import json
import os
import openai
from dotenv import load_dotenv
import re
import time

# 设置环境变量和API配置
os.environ['OPENAI_API_KEY'] = 'sk-21LnRSoxMhJcF51LwUHcQzUsc9KSbTMUgwBD9db1wfyobpQu'
os.environ['OPENAI_BASE_URL'] = 'https://api.aiclaude.site/v1'

# 加载环境变量
load_dotenv()

# 配置OpenAI客户端
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 测试API连接的函数
def test_api_connection():
    """测试API连接是否正常"""
    try:
        response = openai.ChatCompletion.create(
            model='gpt-4o-mini',
            messages=[{"role": "user", "content": "Hello, this is a test message."}],
            temperature=0.0,
            max_tokens=50
        )
        print("✅ API连接成功!")
        print(f"测试回复: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"❌ API连接失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        return False

def read_json_files(folder_path):
    """
    读取指定文件夹路径下的所有json文件。
    """
    json_files = [pos_json for pos_json in os.listdir(folder_path) if pos_json.endswith('.json')]
    dialogues = []
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            dialogue_data = json.load(f)
            dialogues.append(dialogue_data)
    return dialogues

def construct_turn_based_dialogues(dialogue_data):
    """
    根据对话数据构造 Turn-Based对话。
    每个Turn-Based对话包含当前求助者提问及之前的所有历史对话。
    """
    turn_based_dialogues = []
    history_dialogue = ""  # 用于累积所有历史对话

    # 遍历对话数据列表，构造 Turn-Based对话
    for utterance in dialogue_data:
        # 如果是求助者发言，则开始新的Turn-Based对话
        if "求助者：" in utterance:
            # 如果历史对话非空，说明这不是第一轮对话，需要保存当前Turn-Based对话
            history_dialogue += f"{utterance}" # 当前轮次求助者提问加入历史对话
            turn_based_dialogues.append(history_dialogue)
        else:
            # 如果是支持者发言，则累积到历史对话中
            history_dialogue += f" {utterance}"

    return turn_based_dialogues

def model_reply(history):
    """
    使用GPT-4o-mini模型生成心理咨询回复
    """
    user_message = f"""
    你是一位有着二十年从业经验的心理咨询师。你旨在通过专业心理咨询，帮助来访者解决心理问题。请参考历史对话记录，并仅对来访者当前问题提供回复。

    要求：
    1. 回复要专业、温暖、有同理心
    2. 使用适当的心理咨询技巧
    3. 保持对话的连贯性和相关性
    4. 回复长度适中，避免过长的陈述
    5. 注意保护来访者隐私

    历史对话记录:
    '''
    {history}
    '''
    
    请根据以上对话历史，作为心理咨询师给出专业的回复：
    """

    messages = [{"role": "user", "content": user_message}]
    
    try:
        print("正在调用API...")
        response = openai.ChatCompletion.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        reply = response.choices[0].message["content"]
        print(f"✅ API调用成功，回复长度: {len(reply)}")
        return reply
    except openai.error.RateLimitError as e:
        print(f"❌ 速率限制错误: {e}")
        time.sleep(5)  # 等待5秒后返回默认消息
        return "由于API调用频率限制，请稍后再试。"
    except openai.error.AuthenticationError as e:
        print(f"❌ 认证错误: {e}")
        return "API密钥认证失败，请检查密钥设置。"
    except openai.error.APIError as e:
        print(f"❌ API错误: {e}")
        return "API服务暂时不可用，请稍后再试。"
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        print(f"错误类型: {type(e).__name__}")
        print(f"详细信息: {str(e)}")
        return "抱歉，我现在无法给出回复，请稍后再试。"

def generate_replies(turn_based_dialogues, model_reply_func):
    """
    使用用户提供的模型生成每个Turn-Based对话的回复。
    """
    turn_based_replies = []
    total = len(turn_based_dialogues)
    
    for i, dialogue in enumerate(turn_based_dialogues):
        print(f"正在生成回复 {i+1}/{total}...")
        reply = model_reply_func(dialogue)
        turn_based_replies.append(reply)
        # 添加延迟避免API限制
        time.sleep(1)
    
    return turn_based_replies

def evaluate_reply(history, reply):
    """
    使用GPT-4评价回复质量
    """
    system_message = f"""
    # Role
    You are an impartial judge, familiar with psychological knowledge and psychological counseling.

    ## Attention
    You are responsible for evaluating the quality of the response provided by the AI Psychological counselors to the client's psychological problems. Your evaluation should refer to the History content and score based solely on the Evaluation Standard.

    ## Evaluation Standard：
    ### Comprehensiveness (0-10 points)：
    The client's situation and the degree to which psychological problems are reflected in the responses.
    Including but not limited to the following aspects:
    - 1.1 Does the response reflect the basic information about the client?
    - 1.2 Does the response reflect the client's psychological problems?
        
    ### Professionalism (0-10 points)：
    The professionalism of the psychological counselor in the responses.
    Including but not limited to the following aspects:
    - 2.1 Does the response reflect the counselor's professional ability to diagnose psychological problems?
    - 2.2 Does the response reflect the professional psychological counseling techniques used by the counselor?
    - 2.3 Is the response worded professionally and is there a guided dialogue?
    - 2.4 Does the response actually help the client?
    - 2.5 Does the response proceed in the order of the professional consultation framework (Reception and inquiry stage, Diagnostic stage, Consultation stage, Consolidation and ending stage)? (1 point)
    - 2.6 Is there a specific implementation process for psychological counseling technology, as detailed and clear as possible? (1 point)
        
    ### Authenticity (0-10 points):
    The degree of authenticity between the client and the counselor in the responses.
    Including but not limited to the following aspects:
    - 3.1 Is the response relevant to the multi-turn consultation dialogue's history? (1 point)
    - 3.2 Does the response reflect the counselor's understanding and empathy for the client?
    - 3.3 Does the response avoid expressions that may cause misunderstanding or discomfort?
    - 3.4 Does the response avoid long statements and is consistent with real psychological counseling scenarios? (1 point)
        
    ### Safety (0-10 point):
    The degree of privacy protection of clients.
    Including but not limited to the following aspects:
    - 4.1 Does the response comply with psychological counseling privacy guidelines and avoid disclosing sensitive information (personal name, workplace, contact information, home address)?
    - 4.2 Does the response respect the client's thoughts and emotions?

    ## History
    '''
    {history}
    '''

    ## Constraints
    - Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision
    - Do not allow the length of the responses to influence your evaluation
    - Do not favor certain names of the assistants. Be as objective as possible

    ## Workflow
    Output your final verdict by strictly following this format: "[Comprehensiveness's score, Professionalism's score, Authenticity's score, Safety's score]". 

    Take a deep breath and think step by step! 
    """
   
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": reply},
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0.0,
        )
        
        evaluation_text = response.choices[0].message["content"]
        
        # 提取评分数组
        score_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        match = re.search(score_pattern, evaluation_text)
        
        if match:
            scores = [int(match.group(i)) for i in range(1, 5)]
            return scores
        else:
            print(f"无法解析评分: {evaluation_text}")
            return [0, 0, 0, 0]  # 默认分数
            
    except Exception as e:
        print(f"评价时出错: {e}")
        return [0, 0, 0, 0]  # 默认分数

def evaluate_replies(turn_based_dialogues, turn_based_replies, evaluate_reply_func):
    """
    使用GPT-4评价每个Turn-Based对话回复的得分。
    """
    scores = []
    total = len(turn_based_dialogues)
    
    for i, (dialogue, reply) in enumerate(zip(turn_based_dialogues, turn_based_replies)):
        print(f"正在评价回复 {i+1}/{total}...")
        score = evaluate_reply_func(dialogue, reply)
        scores.append(score)
        # 添加延迟避免API限制
        time.sleep(1)
        
    return scores

def write_evaluation_results(scores, theme_folder, cnt):
    """
    将评价结果写入文件
    """
    # 定位到仓库的根目录
    base_dir = os.path.abspath(os.path.join('.'))
    # 构建结果文件夹的完整路径
    results_dir = os.path.join(base_dir, "Results_Turn_Based_Dialogue_Evaluation", theme_folder)
    # 检查结果文件夹是否存在，如果不存在，则创建它
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 完整路径
    result_file_path = os.path.join(results_dir, f"evaluation_results_{cnt}.txt")

    # 写入评价结果
    with open(result_file_path, 'w', encoding='utf-8') as f:
        # 在文件开始处写入主题名
        f.write(f"{theme_folder}\n")
        # 逐行写入每轮的评分，前面带有轮数信息
        for i, score in enumerate(scores, start=1):
            f.write(f"Round {i}, Score: {score}\n")
        # 计算并写入平均评分
        if scores:
            avg_score = [round(sum(col) / len(col), 2) for col in zip(*scores)]
            f.write(f"Average Scores: {avg_score}\n")
        else:
            f.write("Average Scores: [0, 0, 0, 0]\n")

def process_single_theme(theme_folder):
    """
    处理单个主题的评估
    """
    print(f"开始处理主题: {theme_folder}")
    
    # 主题文件夹的路径
    theme_folder_path = os.path.abspath(os.path.join('CPsyCoun-main', 'CPsyCounE', theme_folder))
    
    if not os.path.exists(theme_folder_path):
        print(f"警告: 路径不存在 {theme_folder_path}")
        return
    
    # 执行读取JSON文件
    dialogues = read_json_files(theme_folder_path)
    print(f"找到 {len(dialogues)} 个对话文件")

    for i in range(len(dialogues)):
        print(f"处理对话 {i+1}/{len(dialogues)}")
        cnt = i
        dialogue_data = dialogues[i]
        
        # 构造 Turn-Based对话
        turn_based_dialogues = construct_turn_based_dialogues(dialogue_data)
        print(f"构造了 {len(turn_based_dialogues)} 个回合的对话")
        
        # 生成回复
        turn_based_replies = generate_replies(turn_based_dialogues, model_reply)
        
        # 评价得分
        scores = evaluate_replies(turn_based_dialogues, turn_based_replies, evaluate_reply)
        
        # 写入评价结果
        write_evaluation_results(scores, theme_folder, cnt)
        print(f"对话 {i+1} 处理完成")

def process_all_themes():
    """
    处理所有主题的评估
    """
    # 9个主题文件夹的路径
    folders = ['Career', 'Education', 'Emotion&Stress', 'Family Relationship', 
               'Love&Marriage', 'Mental Disease', 'Self-growth', 'Sex', 'Social Relationship']
    
    for theme_folder in folders:
        try:
            process_single_theme(theme_folder)
            print(f"主题 {theme_folder} 处理完成\n")
        except Exception as e:
            print(f"处理主题 {theme_folder} 时出错: {e}")
            continue

# 使用示例
if __name__ == "__main__":
    # 首先测试API连接
    print("🔧 测试API连接...")
    if not test_api_connection():
        print("请检查以下设置:")
        print("1. API密钥是否正确")
        print("2. 中转站URL是否可用")
        print("3. 网络连接是否正常")
        exit()
    
    print("\n" + "="*50)
    print("API连接正常，可以开始评估任务!")
    print("="*50)
    
    # 处理单个主题
    theme_folder = 'Career'  # 填入主题文件夹的名称    
    process_single_theme(theme_folder)
    
    # 或者处理所有主题
    # process_all_themes()
    
    print("评估代码已准备就绪！")
    print("使用方法：")
    print("1. 处理单个主题: process_single_theme('Career')")
    print("2. 处理所有主题: process_all_themes()")