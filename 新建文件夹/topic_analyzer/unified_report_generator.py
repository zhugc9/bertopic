#!/usr/bin/env python3
# 统一分析报告生成器 - 简化版
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

class UnifiedReportGenerator:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def generate_comprehensive_report(self, topic_model, documents, config, input_files, metadata=None):
        print(" 生成统一分析报告...")
        analysis_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 基础信息
        topic_info = topic_model.get_topic_info()
        total_topics = len(topic_info) - 1
        
        # 生成简化的HTML报告
        html_content = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>BERTopic分析报告</title></head>
<body><h1>BERTopic主题分析报告</h1>
<h2>分析概览</h2><p>分析时间: {datetime.now()}</p>
<p>总文档数: {len(documents)}</p><p>发现主题数: {total_topics}</p>
<h2>输入文件</h2><ul>"""
        
        for file_path in input_files:
            html_content += f"<li>{file_path}</li>"
            
        html_content += "</ul><h2>配置参数</h2><pre>" + json.dumps(config, indent=2, ensure_ascii=False) + "</pre></body></html>"
        
        # 保存报告
        report_path = self.results_dir / f"综合分析报告_{analysis_id}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f" 报告已生成: {report_path}")
        return str(report_path)

if __name__ == "__main__":
    print(" 统一报告生成器已准备就绪")
