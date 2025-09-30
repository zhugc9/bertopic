"""
BERTopic工作流管理器
==================
简化的工作流程管理
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from config_loader import load_runtime_config


class WorkflowManager:
    """简化的工作流管理器"""
    
    def __init__(self, config_path: str):
        """初始化工作流管理器"""
        self.config_path = Path(config_path)
        self.config = load_runtime_config(self.config_path)
        self.results_dir = Path(self.config['results_paths']['output_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化状态
        self.documents = None
        self.metadata_df = None
        self.topic_model = None
        self.topics = None
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.results_dir / 'bertopic_analysis.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> bool:
        """加载数据"""
        self.logger.info("📁 开始加载数据...")
        
        try:
            from topic_analyzer.pipeline import AnalysisPipeline

            pipeline = AnalysisPipeline(self.config_path)
            result = pipeline.run_analysis()
            self.documents = result.documents
            self.metadata_df = result.metadata_df
            self.topic_model = result.topic_model
            self.topics = result.topics

            self.logger.info(f"✅ 数据加载完成：{len(self.documents)} 个文档")

            if self.metadata_df is not None:
                self.logger.info(f"  元数据维度: {self.metadata_df.shape}")

            return True
            
        except Exception as e:
            self.logger.error(f"❌ 数据加载失败: {e}")
            return False
    
    def run_hyperparameter_optimization(self) -> Optional[Dict]:
        """运行超参数优化"""
        opt_config = self.config.get('hyperparameter_optimization', {})
        
        if not opt_config.get('enable', False):
            self.logger.info("⏭️ 超参数优化已禁用，跳过")
            return None
        
        self.logger.info("🔍 开始超参数优化...")
        
        try:
            from topic_analyzer.pipeline import AnalysisPipeline

            pipeline = AnalysisPipeline(self.config_path)
            optimization_results = pipeline.run_hyperparameter_search()
            
            if optimization_results:
                self.logger.info("✅ 超参数优化完成")
                return optimization_results
            else:
                self.logger.warning("⚠️ 超参数优化未返回结果")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 超参数优化失败: {e}")
            return None
    
    def train_topic_model(self, use_optimized_params: bool = False) -> bool:
        """训练主题模型"""
        self.logger.info("🤖 开始训练主题模型...")
        
        try:
            from topic_analyzer.pipeline import AnalysisPipeline

            pipeline = AnalysisPipeline(self.config_path)
            result = pipeline.run_analysis()
            self.topic_model = result.topic_model
            self.topics = result.topics
            
            # 模型统计
            topic_info = self.topic_model.get_topic_info()
            n_topics = len(topic_info) - 1  # 排除噪声主题
            
            self.logger.info(f"✅ 模型训练完成")
            self.logger.info(f"  发现主题数: {n_topics}")
            self.logger.info(f"  总文档数: {len(self.documents)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 模型训练失败: {e}")
            return False
    
    def generate_results(self) -> bool:
        """生成分析结果"""
        if not self.topic_model:
            self.logger.error("模型未训练，无法生成结果")
            return False
        
        self.logger.info("📊 生成分析结果...")
        
        try:
            from topic_analyzer.pipeline import AnalysisPipeline

            pipeline = AnalysisPipeline(self.config_path)
            result = pipeline.run_analysis()
            pipeline.generate_results(result)
            self.topic_model = result.topic_model
            self.documents = result.documents
            self.topics = result.topics
            self.metadata_df = result.metadata_df
            
            self.logger.info("✅ 分析结果生成完成")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 结果生成失败: {e}")
            return False
    
    def run_complete_analysis(self, optimize_params: bool = False) -> bool:
        """运行完整分析流程"""
        self.logger.info("🚀 开始完整主题分析流程...")
        
        # 1. 加载数据
        if not self.load_data():
            return False
        
        # 2. 超参数优化（可选）
        optimization_results = None
        if optimize_params:
            optimization_results = self.run_hyperparameter_optimization()
        
        # 3. 训练模型
        use_optimized = optimization_results is not None
        if not self.train_topic_model(use_optimized_params=use_optimized):
            return False
        
        # 4. 生成结果
        if not self.generate_results():
            return False
        
        self.logger.info("🎉 完整分析流程成功完成！")
        return True
