#!/usr/bin/env python3
"""
BERTopic主题标签生成器 - 简洁实现
===============================
自动为BERTopic主题生成中文标签
"""

# =============================================================================
# 🔧 用户配置区域 - 请根据需要修改以下配置
# =============================================================================

# API配置
API_KEY = "sk-oS1mPWSBCvf5czQQ1CyRXRGWOWQ4CFWis8qNV0UlEqbQYzoH"  # 请替换为您的API密钥
API_URL = "https://openai.sharkmagic.com.cn/v1/chat/completions"  # API代理地址
MODEL_NAME = "[官自-0.7]gemini-2-5-flash"  # 使用的模型

# 提示词配置
PROMPT_TEMPLATE = """基于关键词生成中文议题名称，要求10-15个汉字，输出JSON格式：
关键词：{keywords}
输出：{{"topic_label": "议题名称"}}"""

# 处理配置
MAX_RETRIES = 3  # API调用最大重试次数
REQUEST_DELAY = 0.5  # 请求间隔（秒）
TIMEOUT = 60  # 请求超时时间（秒）

# 文件路径配置（基于脚本位置）
SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "results/topics_summary.csv"  # 输入文件路径
OUTPUT_FILE = SCRIPT_DIR / "results/topics_summary_labeled.csv"  # 输出文件路径

# =============================================================================
# 📦 导入依赖库
# =============================================================================

import os
import sys
import yaml
import json
import requests
import pandas as pd
from tqdm import tqdm
import time


def load_config():
    """加载配置文件"""
    config_path = SCRIPT_DIR / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_topics_summary(file_path) -> pd.DataFrame:
    """加载主题摘要CSV文件"""
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        print("请先运行 main.py 生成 topics_summary.csv 文件")
        sys.exit(1)
    
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"✅ 加载 {len(df)} 个主题")
    return df


def build_prompt(keywords: str) -> str:
    """构建API请求的提示词"""
    return PROMPT_TEMPLATE.format(keywords=keywords)


def call_llm_api(prompt: str, api_key: str) -> str:
    """调用大语言模型API - 参考社媒评论项目"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"}
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=data,
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # 解析JSON
                try:
                    json_result = json.loads(content)
                    return json_result.get('topic_label', '')
                except:
                    # 提取中文文本
                    import re
                    chinese_match = re.search(r'[\u4e00-\u9fff]+', content)
                    return chinese_match.group(0) if chinese_match else content
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                return f"[API错误: {response.status_code}]"
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # 指数退避
                continue
            return f"[API调用失败: {str(e)[:50]}]"
    
    return "[API调用失败: 超过最大重试次数]"


def process_topics(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """为所有主题生成标签"""
    print("🤖 开始生成主题标签...")
    
    df['AI_Generated_Label'] = ''
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="生成标签"):
        keywords = row['Top_Words']
        prompt = build_prompt(keywords)
        label = call_llm_api(prompt, api_key)
        df.at[idx, 'AI_Generated_Label'] = label
        time.sleep(REQUEST_DELAY)  # 避免API限制
    
    return df


def save_results(df: pd.DataFrame, output_path):
    """保存结果"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ 结果已保存: {output_path}")


def main():
    """主执行流程"""
    print("="*50)
    print("🏷️  BERTopic主题标签生成器")
    print("="*50)
    
    # 加载配置
    config = load_config()
    
    # 加载数据
    input_file = SCRIPT_DIR / config['results_paths']['summary_file']
    df = load_topics_summary(input_file)
    
    # 生成标签
    df_labeled = process_topics(df, API_KEY)
    
    # 保存结果
    save_results(df_labeled, OUTPUT_FILE)
    
    print("\n✨ 完成！")
    print(f"📊 处理了 {len(df)} 个主题")
    print(f"📁 结果: {OUTPUT_FILE}")
    print("="*50)


if __name__ == "__main__":
    main()
