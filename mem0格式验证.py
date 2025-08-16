# mem0_format_test.py
# 测试Mem0返回格式的脚本

import os
import time
from mem0 import Memory

# API配置
API_KEY = "sk-21LnRSoxMhJcF51LwUHcQzUsc9KSbTMUgwBD9db1wfyobpQu"
BASE_URL = "https://api.aiclaude.site/v1"

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_BASE_URL"] = BASE_URL

COLLECTION = f"test_format_{int(time.time())}"

def test_mem0_formats():
    print("=" * 60)
    print("测试Mem0返回格式")
    print("=" * 60)
    
    # 配置
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
    
    try:
        # 初始化Memory
        print("1. 初始化Mem0...")
        memory = Memory.from_config(config)
        print("✅ 初始化成功")
        
        user_id = "test_user_001"
        
        # 添加测试数据
        print("\n2. 添加测试数据...")
        test_messages = [
            "我叫张三，今年25岁",
            "我最近经常头痛",
            "我对花生过敏",
            "我喜欢跑步锻炼",
            "我的血压有点高"
        ]
        
        for i, msg in enumerate(test_messages):
            try:
                result = memory.add(msg, user_id=user_id)
                print(f"   添加 {i+1}: {msg[:20]}... -> {type(result)}")
            except Exception as e:
                print(f"   添加失败 {i+1}: {e}")
        
        print("✅ 数据添加完成")
        
        # 测试get_all方法
        print("\n3. 测试 get_all() 方法...")
        print("-" * 40)
        
        try:
            all_memories = memory.get_all(user_id=user_id)
            
            print(f"📊 get_all() 返回类型: {type(all_memories)}")
            print(f"📊 get_all() 返回值: {all_memories}")
            print()
            
            if isinstance(all_memories, dict):
                print("📁 字典结构分析:")
                for key, value in all_memories.items():
                    print(f"   键: '{key}' -> 类型: {type(value)}")
                    if isinstance(value, list):
                        print(f"      列表长度: {len(value)}")
                        if value:
                            print(f"      第一个元素类型: {type(value[0])}")
                            print(f"      第一个元素内容: {value[0]}")
                    else:
                        print(f"      值: {str(value)[:100]}")
                        
            elif isinstance(all_memories, list):
                print("📁 列表结构分析:")
                print(f"   列表长度: {len(all_memories)}")
                if all_memories:
                    print(f"   第一个元素类型: {type(all_memories[0])}")
                    print(f"   第一个元素内容: {all_memories[0]}")
                    
                    if isinstance(all_memories[0], dict):
                        print("   第一个元素的键:")
                        for key in all_memories[0].keys():
                            print(f"      - {key}")
            else:
                print(f"📁 其他类型: {all_memories}")
                
        except Exception as e:
            print(f"❌ get_all() 失败: {e}")
        
        # 测试search方法
        print("\n4. 测试 search() 方法...")
        print("-" * 40)
        
        try:
            search_result = memory.search("头痛", user_id=user_id, limit=3)
            
            print(f"🔍 search() 返回类型: {type(search_result)}")
            print(f"🔍 search() 返回值: {search_result}")
            print()
            
            if isinstance(search_result, dict):
                print("📁 搜索结果字典结构:")
                for key, value in search_result.items():
                    print(f"   键: '{key}' -> 类型: {type(value)}")
                    if isinstance(value, list):
                        print(f"      列表长度: {len(value)}")
                        if value:
                            print(f"      第一个元素: {value[0]}")
                    else:
                        print(f"      值: {str(value)[:100]}")
                        
            elif isinstance(search_result, list):
                print("📁 搜索结果列表结构:")
                print(f"   列表长度: {len(search_result)}")
                if search_result:
                    print(f"   第一个元素类型: {type(search_result[0])}")
                    print(f"   第一个元素: {search_result[0]}")
                    
        except Exception as e:
            print(f"❌ search() 失败: {e}")
        
        # 测试不同参数的get_all
        print("\n5. 测试不同参数的 get_all()...")
        print("-" * 40)
        
        # 尝试分页参数
        try:
            print("测试分页参数...")
            paged_result = memory.get_all(user_id=user_id, page=1, page_size=3)
            print(f"📄 分页结果类型: {type(paged_result)}")
            print(f"📄 分页结果: {paged_result}")
        except Exception as e:
            print(f"❌ 分页参数失败: {e}")
        
        # 尝试版本参数
        try:
            print("\n测试版本参数...")
            v2_result = memory.get_all(user_id=user_id, version="v2")
            print(f"📄 v2结果类型: {type(v2_result)}")
            print(f"📄 v2结果: {v2_result}")
        except Exception as e:
            print(f"❌ v2参数失败: {e}")
            
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成！请将以上输出发送给我进行分析。")
    print("=" * 60)

if __name__ == "__main__":
    test_mem0_formats()