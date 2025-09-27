"""
BERTopic工作流管理器
==================
简化的工作流程管理
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class WorkflowManager:
    """简化的工作流管理器"""
    
    def __init__(self, config_path: str):
        """初始化工作流管理器"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results_dir = Path(self.config['results_paths']['output_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化状态
        self.documents = None
        self.metadata_df = None
        self.topic_model = None
        self.topics = None
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
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
            from topic_analyzer.data_loader import DataLoader
            
            data_loader = DataLoader(self.config)
            self.documents, self.metadata_df = data_loader.load_and_prepare_data()
            
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
            from topic_analyzer.model import TopicAnalyzer
            
            analyzer = TopicAnalyzer(self.config)
            optimization_results = analyzer.optimize_hyperparameters(self.documents)
            
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
            from topic_analyzer.model import TopicAnalyzer
            
            analyzer = TopicAnalyzer(self.config)
            
            if use_optimized_params:
                self.topic_model, self.topics = analyzer.train_with_optimized_parameters(self.documents)
                self.logger.info("  使用优化参数训练")
            else:
                self.topic_model, self.topics = analyzer.train_bertopic_model(self.documents)
                self.logger.info("  使用默认配置训练")
            
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
            from topic_analyzer.model import TopicAnalyzer
            
            analyzer = TopicAnalyzer(self.config)
            
            # 1. 基础结果
            self.logger.info("  生成基础结果...")
            analyzer.generate_results(
                self.topic_model, 
                self.documents, 
                self.topics, 
                self.metadata_df
            )
            
            # 2. 高级可视化
            viz_config = self.config.get('visualization', {}).get('sota_charts', {})
            if viz_config.get('enable', True):
                self.logger.info("  生成高级可视化...")
                charts = analyzer.generate_sota_visualizations(
                    self.topic_model,
                    self.documents, 
                    self.topics,
                    self.metadata_df
                )
                
                if charts:
                    self.logger.info(f"  ✓ 生成图表: {len(charts)} 个")
                else:
                    self.logger.warning("  ⚠ 图表生成失败")
            else:
                self.logger.info("  ⏭️ 高级可视化已禁用")
            
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
