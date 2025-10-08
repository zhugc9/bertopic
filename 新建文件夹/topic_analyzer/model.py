"""
BERTopic模型训练与分析模块
==========================
SOTA实现：使用最新的BERTopic特性和优化技术
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import time
import re
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

        self._console_prefix = "[TopicAnalyzer]"
        
        # 初始化超参数优化器
        self.hyperparameter_optimizer = OptunaBERTopicOptimizer(config)
        
        # 初始化多语言预处理器
        self.multilingual_vectorizer = EnhancedMultilingualVectorizer(config)
        
        # 初始化结果生成器
        self.results_generator = ResultsGenerator(config)

        self.random_seed = self.config.get('system', {}).get('random_seed', 42)
        
        # 设置matplotlib多语言支持（中文、俄文、英文）
        plt.rcParams.update({
            'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial', 'DejaVu Sans', 'sans-serif'],
            'font.family': 'sans-serif',
            'axes.unicode_minus': False,
            'pdf.fonttype': 42,  # TrueType字体嵌入PDF
            'ps.fonttype': 42    # PostScript字体嵌入PS
        })
        
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
        self._announce("[2/3] 开始核心模型训练")
        logger.info(f"开始训练模型 (文档数: {len(documents)})...")
        topics, probs = topic_model.fit_transform(documents)
        self._announce("[2/3] 核心模型训练完成")
        
        # 8. 保存模型
        self._save_model(topic_model)
        
        # 统计信息
        topic_info = topic_model.get_topic_info()
        n_topics = len(topic_info) - 1  # 减去离群点
        logger.info(f"✅ 模型训练完成: 发现 {n_topics} 个主题")
        
        return topic_model, topics

    def _announce(self, message: str, level: str = "info") -> None:
        console_message = f"{self._console_prefix} {message}"
        print(console_message, flush=True)
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.error(message)
    
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
        import traceback
        logger.info("🎨 生成SOTA级学术可视化图表...")
        
        try:
            # 准备时间数据
            timestamps = None
            if metadata_df is not None:
                # 尝试从元数据中提取时间信息
                date_columns = ['日期', 'Date', 'date', 'timestamp', 'time']
                for col in date_columns:
                    if col in metadata_df.columns:
                        try:
                            timestamps = pd.to_datetime(metadata_df[col], errors='coerce').tolist()
                            logger.info(f"✓ 从'{col}'列提取了时间戳数据")
                            break
                        except Exception as e:
                            logger.warning(f"从'{col}'列提取时间戳失败: {e}")
                            continue
            
            # 生成所有SOTA图表
            logger.info("→ 调用可视化生成器...")
            charts = self.visualizer.generate_all_visualizations(
                topic_model=topic_model,
                documents=documents,
                topics=topics,
                metadata_df=metadata_df,
                timestamps=timestamps
            )
            
            logger.info(f"✅ 学术级图表生成完成，共 {len(charts)} 个图表")
            return charts
            
        except Exception as e:
            logger.error(f"❌ SOTA可视化生成失败: {type(e).__name__}: {e}")
            logger.error(f"详细错误:\n{traceback.format_exc()}")
            return {}
    
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
            logger.info("自动应用最佳参数进行分析")
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
        best_score = results.get('best_score')
        optimization_time = results.get('optimization_time')
        n_trials = results.get('n_trials')
        best_score_str = f"{best_score:.4f}" if isinstance(best_score, (int, float)) else str(best_score)
        optimization_time_str = f"{optimization_time:.1f}" if isinstance(optimization_time, (int, float)) else str(optimization_time)
        n_trials_str = str(n_trials) if n_trials is not None else "N/A"
        logger.info("📊 超参数优化结果:")
        logger.info(f"   🏆 最佳分数: {best_score_str}")
        logger.info(f"   ⏱️  优化时间: {optimization_time_str}")
        logger.info(f"   🔬 试验次数: {n_trials_str}")

        top_results = results.get('top_5_results', [])

        logger.info("🔝 前5名参数组合:")
        for result in top_results:
            params = result.get('params', {})
            score = result.get('score')
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            logger.info(f"   #{result.get('rank', 'N/A')}: 分数={score_str}")
            logger.info(f"        min_cluster_size={params.get('min_cluster_size')}, "
                       f"n_neighbors={params.get('n_neighbors')}")
    
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
        self._generate_results_enhanced(topic_model, documents, topics, metadata_df)

    def _generate_results_enhanced(self,
                                   topic_model: BERTopic,
                                   documents: List[str],
                                   topics: List[int],
                                   metadata_df: pd.DataFrame) -> None:
        import traceback
        logger.info("生成增强版分析结果...")

        try:
            metadata_df['topic'] = topics
            metadata_df['topic_label'] = [
                f"主题 {t}" if t != -1 else "离群点"
                for t in topics
            ]
            logger.info("✓ 元数据准备完成")
        except Exception as e:
            logger.error(f"❌ 元数据准备失败: {e}\n{traceback.format_exc()}")
            raise

        # 基础结果
        try:
            logger.info("→ 保存主题摘要表...")
            self.results_generator.save_topic_summary(topic_model)
            logger.info("✓ 主题摘要表已保存")
        except Exception as e:
            logger.error(f"❌ 保存主题摘要表失败: {e}\n{traceback.format_exc()}")
            raise

        try:
            logger.info("→ 保存文档主题映射...")
            self.results_generator.save_document_topic_mapping(documents, topics, metadata_df)
            logger.info("✓ 文档主题映射已保存")
        except Exception as e:
            logger.error(f"❌ 保存文档主题映射失败: {e}\n{traceback.format_exc()}")
            raise

        try:
            logger.info("→ 生成主题可视化...")
            self.results_generator.generate_topic_visualization(topic_model, documents)
            logger.info("✓ 主题可视化已生成")
        except Exception as e:
            logger.error(f"❌ 生成主题可视化失败: {e}\n{traceback.format_exc()}")
            # 可视化失败不应该中断整个流程
            logger.warning("⚠ 可视化失败，继续其他步骤...")

        try:
            if 'Source' in metadata_df.columns:
                logger.info("→ 生成来源分析...")
                self.results_generator.generate_source_analysis(metadata_df, topic_model)
                logger.info("✓ 来源分析已生成")
        except Exception as e:
            logger.error(f"❌ 生成来源分析失败: {e}\n{traceback.format_exc()}")

        try:
            if self.config['analysis']['time_analysis']['enable'] and '日期' in metadata_df.columns:
                logger.info("→ 生成时间线分析...")
                self.results_generator.generate_timeline_analysis(metadata_df, topic_model, documents)
                logger.info("✓ 时间线分析已生成")
        except Exception as e:
            logger.error(f"❌ 生成时间线分析失败: {e}\n{traceback.format_exc()}")

        try:
            if self.config['analysis']['frame_analysis']['enable']:
                logger.info("→ 生成框架热力图...")
                self.results_generator.generate_frame_heatmap(metadata_df)
                logger.info("✓ 框架热力图已生成")
        except Exception as e:
            logger.error(f"❌ 生成框架热力图失败: {e}\n{traceback.format_exc()}")

        # 增强模块
        outputs_config = self.config.get('outputs', {})

        cross_lingual_results = {}
        try:
            if outputs_config.get('cross_lingual', True):
                logger.info("→ 运行跨语言分析...")
                cross_lingual_results = self.cross_lingual_analyzer.run_full_cross_lingual_analysis(documents, topics)
                logger.info("✓ 跨语言分析完成")
        except Exception as e:
            logger.error(f"❌ 跨语言分析失败: {e}\n{traceback.format_exc()}")

        evolution_results = {}
        try:
            if (outputs_config.get('comprehensive_report', True) or outputs_config.get('cross_lingual', True)) and \
                    self.config['analysis']['time_analysis']['enable'] and '日期' in metadata_df.columns:
                logger.info("→ 运行主题演化分析...")
                evolution_results = self.evolution_analyzer.run_full_evolution_analysis(topic_model, documents, metadata_df)
                logger.info("✓ 主题演化分析完成")
        except Exception as e:
            logger.error(f"❌ 主题演化分析失败: {e}\n{traceback.format_exc()}")

        try:
            if outputs_config.get('enhanced_keywords', True):
                logger.info("→ 增强关键词提取...")
                enhanced_topics = self.expert_extractor.enhance_topic_representation(topic_model, documents)
                logger.info("✓ 增强关键词提取完成")
            else:
                enhanced_topics = None
        except Exception as e:
            logger.error(f"❌ 增强关键词提取失败: {e}\n{traceback.format_exc()}")
            enhanced_topics = None

        try:
            if outputs_config.get('enhanced_summary', True):
                logger.info("→ 生成增强主题摘要...")
                self._save_enhanced_topic_summary(
                    topic_model, enhanced_topics, cross_lingual_results.get('composition_df') if cross_lingual_results else None
                )
                logger.info("✓ 增强主题摘要已生成")
        except Exception as e:
            logger.error(f"❌ 生成增强主题摘要失败: {e}\n{traceback.format_exc()}")

        try:
            if outputs_config.get('comprehensive_report', True):
                logger.info("→ 生成综合报告...")
                self._generate_comprehensive_report(
                    topic_model,
                    cross_lingual_results if cross_lingual_results else {},
                    evolution_results,
                    {}
                )
                logger.info("✓ 综合报告已生成")
        except Exception as e:
            logger.error(f"❌ 生成综合报告失败: {e}\n{traceback.format_exc()}")

        try:
            logger.info("→ 生成AI主题标签...")
            self._generate_ai_topic_labels(topic_model)
            logger.info("✓ AI主题标签生成流程完成")
        except Exception as e:
            logger.error(f"❌ AI主题标签生成失败: {type(e).__name__}: {e}")
            logger.error(f"详细错误信息:\n{traceback.format_exc()}")
            # 确保错误信息也输出到控制台
            print(f"AI 主题标签生成失败: {type(e).__name__}: {e}")
            print(f"详细错误:\n{traceback.format_exc()}")
        
        logger.info("✨ 分析结果已生成")

    def _generate_ai_topic_labels(self, topic_model: BERTopic) -> None:
        """在分析流程中尝试生成AI主题标签"""
        ai_config = self.config.get('ai_labeling', {})
        if not ai_config.get('enable', False):
            return

        api_cfg = self.config.get('ai_labeling_advanced', {}).get('api_config', {})
        api_keys = api_cfg.get('API_KEYS', [])
        base_url = api_cfg.get('BASE_URL')
        model_name = api_cfg.get('MODEL')

        if not api_keys or not api_keys[0] or api_keys[0] == 'YOUR_API_KEY_HERE':
            logger.warning("AI 主题标签未生成：API key 未配置")
            return
        if not base_url or not model_name:
            logger.warning("AI 主题标签未生成：API base 或 model 未配置")
            return

        try:
            # 直接读取主题摘要表
            summary_path = Path(self.results_paths['summary_file'])
            if not summary_path.exists():
                logger.warning("AI 主题标签未生成：未找到主题摘要表")
                return

            df = pd.read_csv(summary_path, encoding='utf-8')
            if 'Top_Words' not in df.columns:
                logger.warning("AI 主题标签未生成：摘要表中缺少Top_Words列")
                return
                
            labels = self._call_ai_labeler(df)
            if labels is None:
                logger.warning("AI 主题标签生成失败：API返回空结果")
                return

            # 分离标签和详细分析
            label_list = []
            meaning_list = []
            discourse_list = []
            uniqueness_list = []
            
            for label_data in labels:
                if isinstance(label_data, dict):
                    label_list.append(label_data.get('topic_label', ''))
                    meaning_list.append(label_data.get('core_meaning', ''))
                    discourse = label_data.get('typical_discourse', [])
                    if isinstance(discourse, list):
                        discourse_list.append('; '.join(discourse))
                    else:
                        discourse_list.append(str(discourse) if discourse else '')
                    uniqueness_list.append(label_data.get('uniqueness', ''))
                else:
                    # 兼容旧格式（纯字符串）
                    label_list.append(str(label_data) if label_data else '')
                    meaning_list.append('')
                    discourse_list.append('')
                    uniqueness_list.append('')
            
            df['AI_Label'] = label_list
            df['AI_CoreMeaning'] = meaning_list
            df['AI_TypicalDiscourse'] = discourse_list
            df['AI_Uniqueness'] = uniqueness_list
            
            output_path = summary_path.with_name(summary_path.stem + '_带AI标签.csv')
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"AI 主题标签已生成: {output_path}")
            logger.info(f"  包含字段：标签、核心内涵、话语特征、独特性")
        except Exception as exc:
            import traceback
            logger.warning(f"AI 主题标签生成失败: {type(exc).__name__}: {exc}")
            logger.debug(f"详细错误:\n{traceback.format_exc()}")

    def _call_ai_labeler(self, df: pd.DataFrame) -> Optional[List[str]]:
        """调用语言模型为主题生成标签"""
        try:
            import openai
        except ImportError:
            logger.warning("请先安装 openai 库以启用主题命名：pip install openai")
            return None

        ai_config = self.config.get('ai_labeling', {})
        ai_advanced = self.config.get('ai_labeling_advanced', {})
        api_cfg = ai_advanced.get('api_config', {})
        label_settings = ai_advanced.get('label_settings', {})
        prompt_style = ai_advanced.get('prompt_style', 'academic')
        prompt_templates = ai_advanced.get('prompt_templates', {})

        prompt_template = prompt_templates.get(
            prompt_style,
            '基于关键词：{keywords}，生成{length}的{style}中文标签。输出：{"topic_label": "标签"}'
        )

        length = label_settings.get('length', '8-12个汉字')
        style = label_settings.get('style', '学术化简洁')
        request_delay = label_settings.get('request_delay', 0.5)
        max_retries = label_settings.get('max_retries', 3)
        timeout = label_settings.get('timeout', 60)

        api_key_raw = api_cfg.get('API_KEYS', [''])[0]
        if api_key_raw.startswith('${') and api_key_raw.endswith('}'):  # 环境变量
            env_var = api_key_raw[2:-1]
            api_key = os.getenv(env_var, '')
        else:
            api_key = api_key_raw

        if not api_key:
            logger.warning("AI 主题标签未生成：未找到有效的 API key")
            return None

        base_url = api_cfg.get('BASE_URL')
        model_name = api_cfg.get('MODEL')
        
        # 打印配置信息（用于诊断）
        logger.info("🤖 开始AI主题标签生成...")
        logger.info(f"  API地址: {base_url}")
        logger.info(f"  模型: {model_name}")
        logger.info(f"  主题数量: {len(df)}")
        logger.info(f"  重试次数: {max_retries}")
        logger.info(f"  超时设置: {timeout}秒")
        
        # 同时输出到控制台
        print(f"\n🤖 开始AI主题标签生成...")
        print(f"  API: {base_url}")
        print(f"  模型: {model_name}")
        print(f"  主题数量: {len(df)}")
        print(f"  将为每个主题生成中文标签...\n")
        
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        labels: List[str] = []
        for idx, row in df.iterrows():
            try:
                # 读取所有可用字段
                top_words = row.get('Top_Words', '')
                representation = row.get('Representation', '')
                enhanced_keywords = row.get('Enhanced_Keywords', '')
                representative_docs = row.get('Representative_Docs', '')
                
                # 确保是字符串
                if not isinstance(top_words, str):
                    top_words = str(top_words)
                if not isinstance(representation, str):
                    representation = str(representation)
                if not isinstance(enhanced_keywords, str):
                    enhanced_keywords = str(enhanced_keywords)
                if not isinstance(representative_docs, str):
                    representative_docs = str(representative_docs)
                
                # 限制 representative_docs 长度（取前2000字）
                if len(representative_docs) > 2000:
                    representative_docs = representative_docs[:2000] + "..."

                topic_id = row.get('Topic', idx)
                logger.info(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                logger.info(f"  【开始处理】主题 {topic_id} ({idx + 1}/{len(df)})")
                logger.info(f"  【关键词】: {top_words[:100]}")
                
                # 使用新的参数格式
                prompt = prompt_template.format(
                    top_words=top_words,
                    representation=representation,
                    enhanced_keywords=enhanced_keywords,
                    representative_docs=representative_docs,
                    length=length,
                    style=style
                )
                logger.info(f"  【提示词长度】: {len(prompt)} 字符")
                
                label = self._request_label(client, model_name, prompt, max_retries, timeout)
                
                if label:
                    # 显示标签预览
                    if isinstance(label, dict):
                        preview = label.get('topic_label', str(label)[:50])
                    else:
                        preview = str(label)[:50]
                    logger.info(f"  ✅ 【成功】主题 {topic_id} 标签: {preview}")
                else:
                    logger.warning(f"  ❌ 【失败】主题 {topic_id} 标签生成失败，将使用空字典")
                
                labels.append(label if label else {})
                time.sleep(request_delay)
                
            except Exception as e:
                logger.error(f"  ❌ 【异常】处理主题 {topic_id} 时发生错误: {type(e).__name__}: {e}")
                import traceback
                logger.error(f"  【堆栈】:\n{traceback.format_exc()}")
                labels.append("")  # 失败时添加空字符串
                continue

        # 统计成功数量
        success_count = sum(1 for l in labels if l and (isinstance(l, dict) or isinstance(l, str)))
        logger.info(f"🎉 AI标签生成完成: {success_count}/{len(labels)} 个成功")
        
        # 统计完整分析数量（包含核心内涵等字段）
        full_analysis_count = sum(1 for l in labels if isinstance(l, dict) and 'core_meaning' in l)
        if full_analysis_count > 0:
            logger.info(f"  其中 {full_analysis_count} 个包含详细分析（核心内涵、话语特征等）")
        
        return labels

    def _request_label(self, client, model_name: str, prompt: str, max_retries: int, timeout: int) -> str:
        """调用 OpenAI API 获取单个主题标签"""
        for attempt in range(max_retries):
            try:
                # 直接调用，不强制JSON格式（让API自然返回）
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout
                )
                content = response.choices[0].message.content.strip()
                
                # 详细日志：显示完整的API返回内容
                logger.info(f"  【API原始返回】: {content}")
                print(f"  【API原始返回】: {content}")  # 同时输出到控制台
                
                try:
                    # 先尝试清理可能的格式问题
                    content_cleaned = content.strip()
                    
                    # 清理可能的markdown代码块标记
                    if content_cleaned.startswith('```json'):
                        content_cleaned = content_cleaned[7:]
                    elif content_cleaned.startswith('```'):
                        content_cleaned = content_cleaned[3:]
                    if content_cleaned.endswith('```'):
                        content_cleaned = content_cleaned[:-3]
                    content_cleaned = content_cleaned.strip()
                    
                    logger.info(f"  【清理后内容】: {content_cleaned[:200]}...")
                    
                    # 如果返回内容不是JSON格式，直接提取中文/俄文
                    if not (content_cleaned.startswith('{') and content_cleaned.endswith('}')):
                        logger.info(f"  【解析策略】: 非JSON格式，直接提取文本")
                        # 提取中文或俄文
                        match = re.search(r'[\u4e00-\u9fff\u0400-\u04FF]+', content_cleaned)
                        if match:
                            extracted = match.group(0)
                            logger.info(f"  【提取成功】: {extracted}")
                            return extracted
                        # 如果没有中俄文，返回清理后的内容
                        cleaned = re.sub(r'["\'{}\[\]:,]', '', content_cleaned).strip()
                        if cleaned:
                            logger.info(f"  【清理后结果】: {cleaned[:50]}")
                            return cleaned[:50]  # 限制长度
                        logger.warning(f"  【警告】: 无法从非JSON内容中提取有效文本")
                        return ""
                    
                    # 尝试解析JSON
                    logger.info(f"  【解析策略】: 尝试JSON解析")
                    data = json.loads(content_cleaned)
                    logger.info(f"  【JSON解析成功】: keys={list(data.keys())}")
                    
                    # 如果是完整的主题分析JSON，直接返回整个字典
                    if 'topic_label' in data or 'core_meaning' in data or 'typical_discourse' in data:
                        logger.info(f"  【返回完整JSON】: 包含{len(data)}个字段")
                        return data
                    
                    # 否则尝试多种可能的键名提取标签
                    for key in ['topic_label', 'label', '标签', '主题标签', 'name', 'title']:
                        if key in data:
                            label = data[key]
                            if label and isinstance(label, str) and label.strip():
                                logger.info(f"  【标签提取成功】: key='{key}', value='{label.strip()}'")
                                return label.strip()
                    
                    # 如果找不到预期的键，尝试获取任何字符串值
                    logger.info(f"  【尝试备用提取】: 从所有值中查找")
                    for k, v in data.items():
                        if isinstance(v, str) and v.strip():
                            logger.info(f"  【使用备用key】: key='{k}', value='{v.strip()}'")
                            return v.strip()
                    
                    # JSON解析成功但没有找到标签
                    logger.warning(f"  【警告】: JSON中未找到有效标签")
                    logger.warning(f"  【JSON内容】: {data}")
                    
                except json.JSONDecodeError as je:
                    logger.warning(f"  【JSON解析失败】: {str(je)}")
                    logger.warning(f"  【原始内容】: {content}")
                    # JSON解析失败，尝试直接提取中文/俄文
                    logger.info(f"  【降级策略】: 直接提取中文/俄文字符")
                    match = re.search(r'[\u4e00-\u9fff\u0400-\u04FF]+', content)
                    if match:
                        extracted = match.group(0)
                        logger.info(f"  【提取成功】: {extracted}")
                        return extracted
                    
                except KeyError as ke:
                    logger.error(f"  【KeyError异常】: {str(ke)}")
                    logger.error(f"  【异常详情】: key={repr(ke.args[0])}")
                    logger.error(f"  【JSON数据】: {data if 'data' in locals() else 'N/A'}")
                    logger.error(f"  【原始内容】: {content}")
                    
                except Exception as parse_err:
                    logger.warning(f"  【解析异常】: {type(parse_err).__name__}: {parse_err}")
                    logger.warning(f"  【原始内容】: {content}")
                    import traceback
                    logger.debug(f"  【异常堆栈】:\n{traceback.format_exc()}")

                # 最后的降级策略：直接从文本中提取
                logger.info(f"  【最终降级】: 尝试提取任何有效文本")
                # 提取中文
                match = re.search(r'[\u4e00-\u9fff]+', content)
                if match:
                    extracted = match.group(0)
                    logger.info(f"  【中文提取】: {extracted}")
                    return extracted
                
                # 提取俄文
                match = re.search(r'[\u0400-\u04FF]+', content)
                if match:
                    extracted = match.group(0)
                    logger.info(f"  【俄文提取】: {extracted}")
                    return extracted
                
                # 返回清理后的内容
                if content:
                    cleaned = re.sub(r'["\'{}\[\]:,]', '', content).strip()
                    if cleaned:
                        logger.info(f"  【清理后返回】: {cleaned[:50]}")
                        return cleaned[:50]
                
                logger.warning(f"  【失败】: 无法从返回内容中提取任何有效标签")
                return ""
                
            except Exception as exc:
                # 详细错误信息
                error_type = type(exc).__name__
                error_msg = str(exc)
                
                logger.error(f"  【API请求异常】 (尝试 {attempt + 1}/{max_retries})")
                logger.error(f"     类型: {error_type}")
                logger.error(f"     信息: {error_msg}")
                
                # 打印完整堆栈
                import traceback
                logger.debug(f"     堆栈:\n{traceback.format_exc()}")
                
                # 特殊错误提示
                if "429" in error_msg or "rate" in error_msg.lower():
                    logger.warning(f"     → API速率限制，等待后重试")
                elif "401" in error_msg or "auth" in error_msg.lower():
                    logger.error(f"     → API认证失败，检查API_KEY")
                elif "timeout" in error_msg.lower():
                    logger.warning(f"     → 请求超时，考虑增加timeout设置")
                
                if attempt == max_retries - 1:
                    logger.error(f"  【最终失败】: AI标签请求失败 (已重试{max_retries}次)")
                    return ""
                
                wait_time = 2 ** attempt
                logger.info(f"     等待 {wait_time}秒 后重试...")
                time.sleep(wait_time)

        return ""
    
    def _save_enhanced_topic_summary(self,
                                   topic_model: BERTopic,
                                   enhanced_topics: Optional[Dict] = None,
                                   composition_df: Optional[pd.DataFrame] = None):
        """保存增强的主题摘要（直接保存为主题摘要表.csv）"""
        topic_info = topic_model.get_topic_info()
        
        # 添加基础关键词（Top_Words列，用于AI标签和基本查看）
        topic_info['Top_Words'] = topic_info['Topic'].apply(
            lambda x: ', '.join([word for word, _ in topic_model.get_topic(x)[:5]])
            if x != -1 else 'Outlier'
        )
        
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
        
        # 直接保存为主题摘要表（覆盖基础版）
        summary_path = Path(self.results_paths['summary_file'])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        topic_info.to_csv(summary_path, index=False, encoding='utf-8-sig')
        logger.info(f"  ✓ 增强主题摘要已保存: {summary_path}")
    
    def _generate_comprehensive_report(self,
                                     topic_model: BERTopic,
                                     cross_lingual_results: Dict,
                                     evolution_results: Dict,
                                     academic_charts: Dict):
        """生成综合分析报告"""
        report_path = Path(self.results_paths.get('analysis_report', str(Path(self.results_paths['output_dir']) / '4-主题分析报告.txt')))
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
        # 方案A：expert_keywords（深度词性分析，高质量，适合学术）
        # 方案B：multilingual_preprocessor（轻量分词，快速）
        advanced_config = self.config.get('bertopic_params', {})
        use_expert = advanced_config.get('use_expert_keywords', True)
        
        if use_expert:
            # 方案A：使用专家级关键词提取器（基于spaCy词性标注）
            enhanced_vectorizer = self.expert_extractor.create_enhanced_vectorizer()
            logger.info("  ✓ 创建专家级增强向量化器（词性标注模式）")
        else:
            # 方案B：使用轻量级多语言向量化器
            enhanced_vectorizer = self.multilingual_vectorizer.create_vectorizer()
            logger.info("  ✓ 创建轻量级多语言向量化器")
        
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
        import traceback
        model_dir = Path(self.results_paths['model_dir'])
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model_dir / 'bertopic_model'
        
        try:
            # 尝试完整保存（包括c-TF-IDF配置）
            topic_model.save(str(model_path), serialization="safetensors", save_ctfidf=True)
            logger.info(f"  ✓ 模型已保存: {model_path}")
        except TypeError as e:
            # 如果遇到JSON序列化错误（numpy类型），尝试不保存c-TF-IDF配置
            if "is not JSON serializable" in str(e):
                logger.warning(f"  ⚠ 保存c-TF-IDF配置失败（numpy类型问题），尝试简化保存...")
                try:
                    topic_model.save(str(model_path), serialization="safetensors", save_ctfidf=False)
                    logger.info(f"  ✓ 模型已保存（不含c-TF-IDF配置）: {model_path}")
                except Exception as e2:
                    logger.warning(f"  ⚠ 模型保存失败，继续分析流程: {e2}")
                    logger.debug(f"详细错误:\n{traceback.format_exc()}")
            else:
                logger.warning(f"  ⚠ 模型保存失败: {e}")
                logger.debug(f"详细错误:\n{traceback.format_exc()}")
        except Exception as e:
            logger.warning(f"  ⚠ 模型保存失败，继续分析流程: {type(e).__name__}: {e}")
            logger.debug(f"详细错误:\n{traceback.format_exc()}")
    
    def _get_nr_topics(self) -> Optional[int]:
        """获取主题数量设置"""
        nr_topics = self.model_params['nr_topics']
        if nr_topics == "auto" or nr_topics is None:
            return None
        return int(nr_topics)