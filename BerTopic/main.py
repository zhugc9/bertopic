#!/usr/bin/env python3
"""
BERTopic议题挖掘与分析系统 - 主程序
=====================================
遵循SOTA和KISS原则的简洁实现
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from topic_analyzer.data_loader import DataLoader
from topic_analyzer.model import TopicAnalyzer

# 配置日志
log_file = Path(__file__).parent / 'bertopic_analysis.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ 配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ 配置文件加载失败: {e}")
        sys.exit(1)


def create_directories(config: dict):
    """创建必要的目录结构"""
    directories = [
        config['results_paths']['output_dir'],
        config['results_paths']['model_dir']
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ 目录结构创建完成")


def main():
    """主执行流程"""
    print("="*60)
    print("🚀 BERTopic议题挖掘与分析系统")
    print("="*60)
    
    start_time = datetime.now()
    
    try:
        # Step 1: 加载配置
        logger.info("📁 Step 1: 加载配置文件...")
        config = load_config()
        
        # Step 2: 创建目录
        logger.info("📂 Step 2: 创建项目目录...")
        create_directories(config)
        
        # Step 3: 加载和准备数据
        logger.info("📊 Step 3: 加载和准备数据...")
        data_loader = DataLoader(config)
        documents, metadata_df = data_loader.load_and_prepare_data()
        logger.info(f"✅ 成功加载 {len(documents)} 个文档")
        
        # Step 4: 训练BERTopic模型
        logger.info("🤖 Step 4: 训练BERTopic模型...")
        analyzer = TopicAnalyzer(config)
        topic_model, topics = analyzer.train_bertopic_model(documents)
        
        # 获取主题统计
        topic_info = topic_model.get_topic_info()
        n_topics = len(topic_info) - 1  # 减去离群点主题(-1)
        logger.info(f"✅ 成功识别 {n_topics} 个主题")
        
        # Step 5: 生成增强分析结果
        logger.info("📈 Step 5: 生成增强分析结果与可视化...")
        
        # 增强主题表示
        enhanced_topics = analyzer.expert_extractor.enhance_topic_representation(
            topic_model, documents
        )
        
        analyzer.generate_enhanced_results(
            topic_model=topic_model,
            documents=documents,
            topics=topics,
            metadata_df=metadata_df,
            enhanced_topics=enhanced_topics
        )
        
        # 计算执行时间
        elapsed_time = datetime.now() - start_time
        
        # 打印完成信息
        print("\n" + "="*60)
        print("✨ 分析完成！")
        print("="*60)
        print(f"📊 识别主题数: {n_topics}")
        print(f"📄 分析文档数: {len(documents)}")
        print(f"⏱️  执行时间: {elapsed_time}")
        print(f"📁 结果保存在: {config['results_paths']['output_dir']}/")
        print("\n主要输出文件:")
        print(f"  • 增强主题摘要: {config['results_paths']['summary_enhanced']}")
        print(f"  • 跨语言分析: {config['results_paths']['cross_lingual_file']}")
        print(f"  • 动态演化分析: {config['results_paths']['evolution_file']}")
        print(f"  • 主题可视化: {config['results_paths']['viz_file']}")
        print(f"  • 时间演化: {config['results_paths']['timeline_analysis']}")
        print(f"  • 框架热力图: {config['results_paths']['frame_heatmap']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()