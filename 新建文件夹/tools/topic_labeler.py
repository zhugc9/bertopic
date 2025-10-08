#!/usr/bin/env python3
"""
BERTopic主题标签生成器 - 配置驱动版
===============================
从config.yaml读取所有配置，自动为BERTopic主题生成中文标签
"""

# =============================================================================
# 📦 导入依赖库
# =============================================================================

import os
import sys
import yaml
import json
import requests
import pandas as pd
from pathlib import Path
from config_loader import load_runtime_config
from tqdm import tqdm
import time

# 获取项目根目录
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_config():
    """加载配置文件"""
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    return load_runtime_config(config_path)


def load_topics_summary(file_path) -> pd.DataFrame:
    """加载主题摘要CSV文件"""
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        print("请先运行 main.py 生成 1-主题摘要表.csv 文件")
        sys.exit(1)
    
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"✅ 加载 {len(df)} 个主题")
    return df


def build_prompt(keywords: str, config: dict) -> str:
    """构建API请求的提示词"""
    # 从新的配置结构读取设置
    ai_advanced = config.get('ai_labeling_advanced', {})
    label_settings = ai_advanced.get('label_settings', {})
    
    # 获取提示词风格和模板
    prompt_style = ai_advanced.get('prompt_style', 'academic')
    prompt_templates = ai_advanced.get('prompt_templates', {})
    
    # 根据选择的风格获取对应模板
    if prompt_style in prompt_templates:
        template = prompt_templates[prompt_style]
    else:
        # 回退到默认模板
        template = prompt_templates.get('academic', '基于关键词：{keywords}，生成{length}的{style}中文标签。输出：{{"topic_label": "标签"}}')
    
    # 获取标签参数
    length = label_settings.get('length', '8-12个汉字')
    style = label_settings.get('style', '学术化简洁')
    
    # 替换模板中的变量
    return template.format(
        keywords=keywords,
        length=length,
        style=style
    )


def call_llm_api(prompt: str, config: dict) -> str:
    """调用大语言模型API"""
    # 从新的配置结构读取API设置
    ai_advanced = config.get('ai_labeling_advanced', {})
    api_config = ai_advanced.get('api_config', {})
    
    # 安全读取API密钥：支持环境变量
    api_key_raw = api_config['API_KEYS'][0] if api_config['API_KEYS'] else ""
    if api_key_raw.startswith('${') and api_key_raw.endswith('}'):
        # 从环境变量读取
        env_var = api_key_raw[2:-1]  # 移除${}
        api_key = os.getenv(env_var, "")
        if not api_key:
            raise ValueError(f"环境变量 {env_var} 未设置")
    else:
        api_key = api_key_raw
    base_url = api_config['BASE_URL']
    model = api_config['MODEL']
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"}
    }
    
    # 从配置文件读取重试参数
    label_settings = ai_advanced.get('label_settings', {})
    max_retries = label_settings.get('max_retries', 3)
    timeout = label_settings.get('timeout', 60)
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                base_url,
                headers=headers,
                json=data,
                timeout=timeout
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
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                return f"[API错误: {response.status_code}]"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
                continue
            return f"[API调用失败: {str(e)[:50]}]"
    
    return "[API调用失败: 超过最大重试次数]"


def process_topics(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """为所有主题生成标签"""
    print("🤖 开始生成主题标签...")
    
    # 从配置文件读取请求间隔
    ai_advanced = config.get('ai_labeling_advanced', {})
    label_settings = ai_advanced.get('label_settings', {})
    request_delay = label_settings.get('request_delay', 0.5)
    
    df['AI_Generated_Label'] = ''
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="生成标签"):
        keywords = row['Top_Words']
        prompt = build_prompt(keywords, config)
        label = call_llm_api(prompt, config)
        df.at[idx, 'AI_Generated_Label'] = label
        time.sleep(request_delay)  # 从配置文件读取的API请求间隔
    
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
    
    # 检查AI标签功能是否启用
    ai_config = config.get('ai_labeling', {})
    if not ai_config.get('enable', False):
        print("❌ AI主题标签功能未启用")
        print("请在config.yaml中设置 ai_labeling.enable: true")
        return
    
    # 检查API密钥
    ai_advanced = config.get('ai_labeling_advanced', {})
    api_config = ai_advanced.get('api_config', {})
    api_keys = api_config.get('API_KEYS', [])
    if not api_keys or not api_keys[0] or api_keys[0] == "YOUR_API_KEY_HERE":
        print("❌ 请在config.yaml中设置您的OpenAI API密钥")
        print("位置: ai_labeling_advanced.api_config.API_KEYS")
        return
    
    # 加载数据
    results_cfg = config.get('output_settings', {})
    results_dir = PROJECT_ROOT / results_cfg.get('folder', 'results')
    summary_filename = results_cfg.get('names', {}).get('topic_summary', '主题摘要表.csv')
    input_file = results_dir / summary_filename

    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        print("请先运行 main.py 生成主题分析结果")
        return

    df = load_topics_summary(input_file)

    # 生成标签
    df_labeled = process_topics(df, config)

    # 保存结果
    labeled_filename = summary_filename.replace('.csv', '_带AI标签.csv') if summary_filename.endswith('.csv') else "主题摘要表_带AI标签.csv"
    output_file = results_dir / labeled_filename
    save_results(df_labeled, output_file)
    
    print("\n✨ 完成！")
    print(f"📊 处理了 {len(df)} 个主题")
    print(f"📁 结果: {output_file}")
    print("="*50)


if __name__ == "__main__":
    main()
