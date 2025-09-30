"""
BERTopic模型训练与分析模块
==========================
SOTA实现：使用最新的BERTopic特性和优化技术
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# BERTopic核心
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

# 专家级关键词提取
from .expert_keywords import ExpertKeywordExtractor

# 子模块导入
from .visualizations import VisualizationGenerator
from .dynamic_evolution import DynamicTopicEvolution
from .cross_lingual import CrossLingualAnalyzer
from .hyperparameter_optimizer import OptunaBERTopicOptimizer
from .multilingual_preprocessor import EnhancedMultilingualVectorizer
from .results_generator import ResultsGenerator

# 降维和聚类
from umap import UMAP
from hdbscan import HDBSCAN

# 嵌入模型
from sentence_transformers import SentenceTransformer

# 向量化
from sklearn.feature_extraction.text import CountVectorizer

# 可视化
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class TopicAnalyzer:
    """主题分析器类 - SOTA实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化主题分析器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model_params = config['bertopic_params']
        self.viz_params = config['visualization']
        self.results_paths = config['results_paths']
        
        # 初始化专家级关键词提取器
        self.expert_extractor = ExpertKeywordExtractor(config)
        
        # 初始化可视化生成器
        self.visualizer = VisualizationGenerator(config)
        
        # 初始化动态演化分析器
        self.evolution_analyzer = DynamicTopicEvolution(config)
        
        # 初始化跨语言分析器
        self.cross_lingual_analyzer = CrossLingualAnalyzer(config)
        
        # 初始化超参数优化器
        self.hyperparameter_optimizer = OptunaBERTopicOptimizer(config)
        
        # 初始化多语言预处理器
        self.multilingual_vectorizer = EnhancedMultilingualVectorizer(config)
        
        # 初始化结果生成器
        self.results_generator = ResultsGenerator(config)

        self.random_seed = self.config.get('system', {}).get('random_seed', 42)
        
        # 设置matplotlib中文支持
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def train_bertopic_model(self, 
                           documents: List[str]) -> Tuple[BERTopic, List[int]]:
        """
        训练BERTopic模型
        
        Args:
            documents: 文档列表
            
        Returns:
            topic_model: 训练好的模型
            topics: 每个文档的主题编号
        """
        logger.info("初始化BERTopic组件...")
        
        # 1. 初始化嵌入模型（SOTA：多语言支持）
        embedding_model = self._create_embedding_model()
        
        # 2. 初始化UMAP降维（SOTA优化参数）
        umap_model = self._create_umap_model()
        
        # 3. 初始化HDBSCAN聚类（SOTA优化参数）
        hdbscan_model = self._create_hdbscan_model()
        
        # 4. 初始化向量化器（支持中文）
        vectorizer_model = self._create_vectorizer()
        
        # 5. 初始化表示模型（SOTA：组合多种表示方法）
        representation_model = self._create_representation_model()
        
        # 6. 创建BERTopic模型
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            nr_topics=self._get_nr_topics(),
            top_n_words=10,
            verbose=self.config['system']['verbose'],
            calculate_probabilities=True  # SOTA：计算概率分布
        )
        
        # 7. 训练模型
        logger.info(f"开始训练模型 (文档数: {len(documents)})...")
        topics, probs = topic_model.fit_transform(documents)
        
        # 8. 保存模型
        self._save_model(topic_model)
        
        # 统计信息
        topic_info = topic_model.get_topic_info()
        n_topics = len(topic_info) - 1  # 减去离群点
        logger.info(f"✅ 模型训练完成: 发现 {n_topics} 个主题")
        
        return topic_model, topics
    
    def analyze_document_languages(self, documents: List[str]) -> Dict[str, Any]:
        """
        分析文档的语言分布
        
        Args:
            documents: 文档列表
            
        Returns:
            语言分析结果
        """
        logger.info("🌍 分析文档语言分布...")
        
        language_stats = self.multilingual_vectorizer.get_language_statistics(documents)
        
        # 显示分析结果
        logger.info("📊 语言分布统计:")
        for lang, stats in language_stats['language_distribution'].items():
            lang_names = {'zh': '中文', 'en': '英文', 'ru': '俄文', 'unknown': '未知'}
            lang_name = lang_names.get(lang, lang)
            logger.info(f"  {lang_name}: {stats['count']} 个文档 ({stats['percentage']:.1f}%)")
        
        return language_stats
    
    def generate_sota_visualizations(self, 
                                    topic_model: BERTopic,
                                    documents: List[str],
                                    topics: List[int],
                                    metadata_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """
        生成SOTA级学术可视化图表
        
        Args:
            topic_model: 训练好的模型
            documents: 文档列表
            topics: 主题分配
            metadata_df: 元数据DataFrame
            
        Returns:
            生成的图表文件路径字典
        """
        logger.info("🎨 生成SOTA级学术可视化图表...")
        
        # 准备时间数据
        timestamps = None
        if metadata_df is not None:
            # 尝试从元数据中提取时间信息
            date_columns = ['日期', 'Date', 'date', 'timestamp', 'time']
            for col in date_columns:
                if col in metadata_df.columns:
                    try:
                        timestamps = pd.to_datetime(metadata_df[col], errors='coerce').tolist()
                        break
                    except:
                        continue
        
        # 生成所有SOTA图表
        charts = self.visualizer.generate_all_visualizations(
            topic_model=topic_model,
            documents=documents,
            topics=topics,
            metadata_df=metadata_df,
            timestamps=timestamps
        )
        
        # 保存图表路径信息
        results_dir = Path(self.results_paths['output_dir'])
        results_dir.mkdir(exist_ok=True)
        
        chart_summary_file = Path(self.results_paths['charts_summary'])
        chart_summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(chart_summary_file, 'w', encoding='utf-8') as f:
            f.write("学术级可视化图表生成报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"生成图表数量: {len(charts)}\n\n")
            
            for chart_name, chart_path in charts.items():
                f.write(f"• {chart_name}: {chart_path}\n")
        
        logger.info(f"🎉 学术级图表生成完成，共 {len(charts)} 个图表")
        
        return charts
    
    def optimize_hyperparameters(self, documents: List[str]) -> Dict[str, Any]:
        """
        执行超参数优化
        
        Args:
            documents: 文档列表
            
        Returns:
            优化结果字典
        """
        # 检查是否启用超参数优化
        opt_config = self.config.get('hyperparameter_optimization', {})
        if not opt_config.get('enable', False):
            logger.info("超参数优化未启用，跳过优化步骤")
            return None
        
        logger.info("🔍 开始超参数优化...")
        
        # 获取优化配置
        n_trials = opt_config.get('n_trials', 50)
        study_name = opt_config.get('study_name', 'auto')
        if study_name == 'auto':
            study_name = f"bertopic_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 执行优化
        optimization_results = self.hyperparameter_optimizer.optimize_hyperparameters(
            documents=documents,
            n_trials=n_trials,
            study_name=study_name,
            save_results=opt_config.get('save_intermediate', True)
        )
        
        # 显示优化结果
        self._display_optimization_results(optimization_results)
        
        # 询问用户是否使用最佳参数
        if opt_config.get('auto_apply_best', False):
            logger.info("🚀 自动应用最佳参数进行分析")
            self._apply_optimized_parameters(optimization_results['best_params'])
        else:
            logger.info("💡 超参数优化完成，可通过配置启用最佳参数")
        
        return optimization_results
    
    def train_with_optimized_parameters(self, 
                                      documents: List[str],
                                      optimized_params: Optional[Dict[str, Any]] = None) -> Tuple[BERTopic, List[int]]:
        """
        使用优化的参数训练模型
        
        Args:
            documents: 文档列表
            optimized_params: 优化的参数字典，如果为None则尝试加载已保存的最佳参数
            
        Returns:
            topic_model: 训练好的模型
            topics: 每个文档的主题编号
        """
        if optimized_params is None:
            # 尝试加载已保存的最佳参数
            optimized_params = self.hyperparameter_optimizer.load_best_parameters()
            if optimized_params is None:
                logger.warning("未找到优化参数，使用默认配置")
                return self.train_bertopic_model(documents)
        
        logger.info("🎯 使用优化参数训练模型...")
        
        # 临时应用优化参数
        # 临时应用优化参数
        backup_config = self._backup_current_config()
        self._apply_optimized_parameters(optimized_params)

        try:
            topic_model, topics = self.train_bertopic_model(documents)
            return topic_model, topics
        finally:
            self._restore_config(backup_config)
    
    def _display_optimization_results(self, results: Dict[str, Any]) -> None:
        """显示优化结果"""
        logger.info("📊 超参数优化结果:")
        logger.info(f"   🏆 最佳分数: {results['best_score']:.4f}")
        logger.info(f"   ⏱️  优化时间: {results['optimization_time']:.1f} 秒")
        logger.info(f"   🔬 试验次数: {results['n_trials']}")
        
        logger.info("🔝 前5名参数组合:")
        for result in results['top_5_results']:
            logger.info(f"   #{result['rank']}: 分数={result['score']:.4f}")
            logger.info(f"        min_cluster_size={result['params']['min_cluster_size']}, "
                       f"n_neighbors={result['params']['n_neighbors']}")
    
    def _apply_optimized_parameters(self, optimized_params: Dict[str, Any]) -> None:
        """应用优化的参数到配置中"""
        # 更新UMAP参数
        if 'n_neighbors' in optimized_params:
            self.config['bertopic_params']['umap_params']['n_neighbors'] = optimized_params['n_neighbors']
        if 'n_components' in optimized_params:
            self.config['bertopic_params']['umap_params']['n_components'] = optimized_params['n_components']
        if 'min_dist' in optimized_params:
            self.config['bertopic_params']['umap_params']['min_dist'] = optimized_params['min_dist']
        
        # 更新HDBSCAN参数
        if 'min_cluster_size' in optimized_params:
            self.config['bertopic_params']['hdbscan_params']['min_cluster_size'] = optimized_params['min_cluster_size']
            self.config['bertopic_params']['min_topic_size'] = optimized_params['min_cluster_size']
        if 'min_samples' in optimized_params:
            self.config['bertopic_params']['hdbscan_params']['min_samples'] = optimized_params['min_samples']
        
        # 更新N-gram参数
        if 'ngram_range_start' in optimized_params and 'ngram_range_end' in optimized_params:
            self.config['bertopic_params']['n_gram_range'] = [
                optimized_params['ngram_range_start'], 
                optimized_params['ngram_range_end']
            ]
        
        logger.info("✅ 优化参数已应用到配置中")
    
    def _backup_current_config(self) -> Dict[str, Any]:
        """备份当前配置"""
        import copy
        return copy.deepcopy(self.config)
    
    def _restore_config(self, backup_config: Dict[str, Any]) -> None:
        """恢复配置"""
        self.config = backup_config
        
    def generate_results(self, 
                        topic_model: BERTopic,
                        documents: List[str],
                        topics: List[int],
                        metadata_df: pd.DataFrame):
        """
        生成所有分析结果
        
        Args:
            topic_model: 训练好的模型
            documents: 文档列表
            topics: 主题列表
            metadata_df: 元数据
        """
        logger.info("生成分析结果...")
        
        # 1. 将主题添加到元数据
        metadata_df['topic'] = topics
        metadata_df['topic_label'] = [
            f"Topic {t}" if t != -1 else "Outlier" 
            for t in topics
        ]
        
        # 2. 生成并保存主题摘要
        self.results_generator.save_topic_summary(topic_model)
        
        # 3. 生成主题可视化
        self.results_generator.generate_topic_visualization(topic_model, documents)
        
        # 4. 生成来源分析（如果有Source列）
        if 'Source' in metadata_df.columns:
            self.results_generator.generate_source_analysis(metadata_df, topic_model)
        
        # 5. 生成时间演化分析（如果有日期列）
        if self.config['analysis']['time_analysis']['enable']:
            if '日期' in metadata_df.columns:
                self.results_generator.generate_timeline_analysis(metadata_df, topic_model, documents)
        
        # 6. 生成框架热力图（如果有框架列）
        if self.config['analysis']['frame_analysis']['enable']:
            self.results_generator.generate_frame_heatmap(metadata_df)
        
        logger.info("✅ 所有分析结果已生成")
    
    def generate_enhanced_results(self, 
                                topic_model: BERTopic,
                                documents: List[str],
                                topics: List[int],
                                metadata_df: pd.DataFrame,
                                enhanced_topics: Optional[Dict] = None):
        """
        生成增强的分析结果，集成所有新模块
        
        Args:
            topic_model: 训练好的模型
            documents: 文档列表
            topics: 主题列表
            metadata_df: 元数据
            enhanced_topics: 增强的主题表示
        """
        logger.info("🚀 生成增强分析结果...")
        
        # 1. 生成基础结果
        self.generate_results(topic_model, documents, topics, metadata_df)
        
        # 2. 跨语言分析
        logger.info("🌍 执行跨语言主题成分分析...")
        cross_lingual_results = self.cross_lingual_analyzer.run_full_cross_lingual_analysis(
            documents, topics
        )
        
        # 3. 学术级图表生成
        logger.info("🎨 生成学术级图表...")
        academic_charts = self.visualizer.generate_all_visualizations(
            topic_model=topic_model,
            documents=documents,
            topics=topics,
            metadata_df=metadata_df
        )
        
        # 4. 动态演化分析（如果有时间数据）
        evolution_results = {}
        timestamps = None
        if '日期' in metadata_df.columns and self.config['analysis']['time_analysis']['enable']:
            logger.info("🕐 执行动态主题演化分析...")
            
            # 准备时间戳数据
            try:
                import pandas as pd
                timestamps = pd.to_datetime(metadata_df['日期'], errors='coerce').dropna().tolist()
                logger.info(f"  ✓ 准备时间戳数据: {len(timestamps)} 个有效时间点")
            except Exception as e:
                logger.warning(f"  ⚠ 时间戳数据准备失败: {e}")
                timestamps = None
            
            evolution_results = self.evolution_analyzer.run_full_evolution_analysis(
                topic_model, documents, metadata_df
            )
            
            # 生成演化图表
            if evolution_results.get('topics_over_time') is not None:
                topics_over_time = evolution_results['topics_over_time']
                if not topics_over_time.empty:
                    # 使用可视化生成器的演化图表功能
                    evolution_charts = self.visualizer.generate_all_visualizations(
                        topic_model=topic_model,
                        documents=documents,
                        topics=topics,
                        metadata_df=metadata_df,
                        timestamps=timestamps
                    )
                    academic_charts.update(evolution_charts)
        
        # 5. 更新主题摘要文件，集成增强信息
        self._save_enhanced_topic_summary(
            topic_model, enhanced_topics, cross_lingual_results.get('composition_df')
        )
        
        # 6. 生成综合分析报告
        self._generate_comprehensive_report(
            topic_model, cross_lingual_results, evolution_results, academic_charts
        )
        
        logger.info("✨ 增强分析结果生成完成！")
    
    def _save_enhanced_topic_summary(self,
                                   topic_model: BERTopic,
                                   enhanced_topics: Optional[Dict] = None,
                                   composition_df: Optional[pd.DataFrame] = None):
        """保存增强的主题摘要"""
        topic_info = topic_model.get_topic_info()
        
        # 添加增强的关键词
        if enhanced_topics:
            topic_info['Enhanced_Keywords'] = topic_info['Topic'].apply(
                lambda x: ', '.join([word for word, _ in enhanced_topics.get(x, [])[:5]])
                if x in enhanced_topics else ''
            )
        
        # 添加语言构成信息
        if composition_df is not None and not composition_df.empty:
            # 创建语言信息字典
            lang_info = {}
            for _, row in composition_df.iterrows():
                topic_id = row['Topic']
                lang_info[topic_id] = {
                    'Dominant_Language': row.get('Dominant_Language', ''),
                    'Topic_Type': row.get('Topic_Type', ''),
                    'Chinese_Percentage': row.get('Chinese_Percentage', 0),
                    'Russian_Percentage': row.get('Russian_Percentage', 0),
                    'English_Percentage': row.get('English_Percentage', 0)
                }
            
            # 添加到主题信息中
            topic_info['Dominant_Language'] = topic_info['Topic'].apply(
                lambda x: lang_info.get(x, {}).get('Dominant_Language', '')
            )
            topic_info['Topic_Type'] = topic_info['Topic'].apply(
                lambda x: lang_info.get(x, {}).get('Topic_Type', '')
            )
            topic_info['Chinese_Pct'] = topic_info['Topic'].apply(
                lambda x: lang_info.get(x, {}).get('Chinese_Percentage', 0)
            )
            topic_info['Russian_Pct'] = topic_info['Topic'].apply(
                lambda x: lang_info.get(x, {}).get('Russian_Percentage', 0)
            )
            topic_info['English_Pct'] = topic_info['Topic'].apply(
                lambda x: lang_info.get(x, {}).get('English_Percentage', 0)
            )
        
        # 添加原始关键词（用于对比）
        topic_info['Original_Keywords'] = topic_info['Topic'].apply(
            lambda x: ', '.join([word for word, _ in topic_model.get_topic(x)[:5]])
            if x != -1 else 'Outlier'
        )
        
        # 保存增强的摘要文件
        enhanced_summary_path = str(Path(self.results_paths['summary_file']).with_name(Path(self.results_paths['summary_file']).stem + '_enhanced.csv'))
        topic_info.to_csv(enhanced_summary_path, index=False, encoding='utf-8-sig')
        logger.info(f"  ✓ 增强主题摘要已保存: {enhanced_summary_path}")
    
    def _generate_comprehensive_report(self,
                                     topic_model: BERTopic,
                                     cross_lingual_results: Dict,
                                     evolution_results: Dict,
                                     academic_charts: Dict):
        """生成综合分析报告"""
        report_path = Path(self.results_paths.get('analysis_report', str(Path(self.results_paths['output_dir']) / '主题分析报告.txt')))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("BERTopic 增强分析综合报告\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 基础统计
                topic_info = topic_model.get_topic_info()
                n_topics = len(topic_info) - 1
                f.write(f"📊 基础统计信息:\n")
                f.write(f"  识别主题数量: {n_topics} 个\n")
                f.write(f"  总文档数量: {topic_info['Count'].sum()} 个\n\n")
                
                # 跨语言分析结果
                if cross_lingual_results.get('summary'):
                    summary = cross_lingual_results['summary']
                    f.write(f"🌍 跨语言分析结果:\n")
                    f.write(f"  总文档数: {summary['total_documents']}\n")
                    f.write(f"  分析主题数: {summary['total_topics']}\n")
                    
                    lang_dist = summary['language_distribution']
                    f.write(f"  语言分布:\n")
                    f.write(f"    中文文档: {lang_dist['chinese']} ({lang_dist['chinese']/summary['total_documents']*100:.1f}%)\n")
                    f.write(f"    俄文文档: {lang_dist['russian']} ({lang_dist['russian']/summary['total_documents']*100:.1f}%)\n")
                    f.write(f"    英文文档: {lang_dist['english']} ({lang_dist['english']/summary['total_documents']*100:.1f}%)\n")
                    
                    f.write(f"  关键洞察:\n")
                    for insight in summary['key_insights']:
                        f.write(f"    • {insight}\n")
                    f.write("\n")
                
                # 动态演化分析结果
                if evolution_results.get('summary'):
                    evo_summary = evolution_results['summary']
                    f.write(f"🕐 动态演化分析结果:\n")
                    f.write(f"  分析时间段: {evo_summary['analysis_period']}\n")
                    f.write(f"  时间点数量: {evo_summary['time_points']}\n")
                    f.write(f"  演化主题数: {evo_summary['total_topics']}\n")
                    
                    if evolution_results.get('evolution_patterns'):
                        patterns = evolution_results['evolution_patterns']
                        f.write(f"  演化模式:\n")
                        f.write(f"    上升趋势主题: {len(patterns.get('rising_topics', []))} 个\n")
                        f.write(f"    下降趋势主题: {len(patterns.get('declining_topics', []))} 个\n")
                        f.write(f"    稳定主题: {len(patterns.get('stable_topics', []))} 个\n")
                        f.write(f"    波动主题: {len(patterns.get('volatile_topics', []))} 个\n")
                    f.write("\n")
                
                # 生成的图表文件
                if academic_charts:
                    f.write(f"🎨 生成的学术级图表:\n")
                    for chart_name, chart_path in academic_charts.items():
                        if chart_path:
                            f.write(f"  {chart_name}: {chart_path}\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("报告结束\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"  ✓ 综合分析报告已保存: {report_path}")
            
        except Exception as e:
            logger.error(f"  ✗ 生成综合报告失败: {e}")
    
    def _create_embedding_model(self) -> SentenceTransformer:
        """创建嵌入模型"""
        model_name = self.model_params['embedding_model']
        logger.info(f"  • 加载嵌入模型: {model_name}")
        return SentenceTransformer(model_name)
    
    def _create_umap_model(self) -> UMAP:
        """创建UMAP降维模型"""
        params = self.model_params['umap_params']
        return UMAP(
            n_neighbors=params['n_neighbors'],
            n_components=params['n_components'],
            min_dist=params['min_dist'],
            metric=params['metric'],
            random_state=params.get('random_state', self.random_seed)
        )
    
    def _create_hdbscan_model(self) -> HDBSCAN:
        """创建HDBSCAN聚类模型"""
        params = self.model_params['hdbscan_params']
        return HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            metric=params['metric'],
            cluster_selection_method=params['cluster_selection_method'],
            prediction_data=params['prediction_data']
        )
    
    def _create_vectorizer(self) -> CountVectorizer:
        """创建增强的向量化器，集成多语言处理和专家级关键词提取"""
        # 检查配置，决定使用哪种向量化器
        advanced_preprocessing = self.config.get('bertopic_params', {}).get('expert_keyword_extraction', {})
        use_multilingual = advanced_preprocessing.get('enable_multilingual_preprocessing', True)
        
        if use_multilingual:
            # 使用新的多语言向量化器
            enhanced_vectorizer = self.multilingual_vectorizer.create_vectorizer()
            logger.info("  ✓ 创建多语言增强向量化器")
        else:
            # 使用原有的专家级关键词提取器
            enhanced_vectorizer = self.expert_extractor.create_enhanced_vectorizer()
            logger.info("  ✓ 创建专家级增强向量化器")
        
        return enhanced_vectorizer
    
    def _create_representation_model(self):
        """创建表示模型（SOTA：组合多种方法）"""
        # KeyBERT风格的关键词提取
        keybert = KeyBERTInspired()
        
        # 最大边际相关性
        mmr = MaximalMarginalRelevance(diversity=0.3)
        
        # 组合多种表示方法
        return [keybert, mmr]
    
    def _save_model(self, topic_model: BERTopic) -> None:
        """保存训练好的模型"""
        model_dir = Path(self.results_paths['model_dir'])
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model_dir / 'bertopic_model'
        topic_model.save(str(model_path), serialization="safetensors", save_ctfidf=True)
        logger.info(f"  ✓ 模型已保存: {model_path}")
    
    def _get_nr_topics(self) -> Optional[int]:
        """获取主题数量设置"""
        nr_topics = self.model_params['nr_topics']
        if nr_topics == "auto" or nr_topics is None:
            return None
        return int(nr_topics)
    
    
    def _generate_source_analysis(self, 
                                 metadata_df: pd.DataFrame,
                                 topic_model: BERTopic):
        """生成来源分析图表"""
        try:
            # 准备数据
            source_topic_df = metadata_df.groupby(['Source', 'topic_label']).size().reset_index(name='count')
            
            # 过滤掉离群点
            source_topic_df = source_topic_df[source_topic_df['topic_label'] != 'Outlier']
            
            # 创建堆叠柱状图
            fig = px.bar(
                source_topic_df, 
                x='Source', 
                y='count',
                color='topic_label',
                title='议题在不同来源中的分布',
                labels={'count': '文档数量', 'Source': '来源', 'topic_label': '主题'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # 保存
            source_path = self.results_paths['source_analysis']
            fig.write_html(source_path)
            logger.info(f"  ✓ 来源分析已保存: {source_path}")
            
        except Exception as e:
            logger.warning(f"  ⚠ 来源分析失败: {e}")
    
    def _generate_temporal_analysis(self,
                                   topic_model: BERTopic,
                                   documents: List[str],
                                   metadata_df: pd.DataFrame):
        """生成时间演化分析"""
        try:
            time_column = self.config['analysis']['time_analysis']['time_column']
            
            # 确保日期格式正确
            timestamps = pd.to_datetime(metadata_df[time_column])
            
            # 生成主题随时间变化
            topics_over_time = topic_model.topics_over_time(
                documents,
                timestamps=timestamps,
                nr_bins=self.config['analysis']['time_analysis']['bins']
            )
            
            # 创建可视化
            fig = topic_model.visualize_topics_over_time(topics_over_time)
            
            # 保存
            timeline_path = self.results_paths['timeline_analysis']
            fig.write_html(timeline_path)
            logger.info(f"  ✓ 时间演化分析已保存: {timeline_path}")
            
        except Exception as e:
            logger.warning(f"  ⚠ 时间演化分析失败: {e}")
    
    def _generate_frame_heatmap(self, 
                               metadata_df: pd.DataFrame,
                               topic_model: BERTopic):
        """生成框架热力图"""
        try:
            # 查找所有框架列
            frame_columns = [col for col in metadata_df.columns 
                           if col.startswith('Frame_') and col.endswith('_Present')]
            
            if not frame_columns:
                logger.warning("  ⚠ 未找到框架列，跳过框架分析")
                return
            
            # 计算每个主题的框架使用频率
            topic_frame_matrix = []
            topic_labels = []
            
            for topic in sorted(metadata_df['topic'].unique()):
                if topic == -1:  # 跳过离群点
                    continue
                    
                topic_data = metadata_df[metadata_df['topic'] == topic]
                frame_freq = topic_data[frame_columns].mean()
                topic_frame_matrix.append(frame_freq.values)
                topic_labels.append(f"Topic {topic}")
            
            # 创建热力图数据
            frame_names = [col.replace('Frame_', '').replace('_Present', '') 
                          for col in frame_columns]
            
            # 使用plotly创建热力图
            fig = go.Figure(data=go.Heatmap(
                z=topic_frame_matrix,
                x=frame_names,
                y=topic_labels,
                colorscale='RdBu_r',
                text=np.round(topic_frame_matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="频率")
            ))
            
            fig.update_layout(
                title='主题-框架关联热力图',
                xaxis_title='叙事框架',
                yaxis_title='主题',
                height=max(400, len(topic_labels) * 30),
                xaxis={'tickangle': -45}
            )
            
            # 保存
            heatmap_path = self.results_paths['frame_heatmap']
            fig.write_html(heatmap_path)
            logger.info(f"  ✓ 框架热力图已保存: {heatmap_path}")
            
        except Exception as e:
            logger.warning(f"  ⚠ 框架热力图生成失败: {e}")