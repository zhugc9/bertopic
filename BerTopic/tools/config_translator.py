"""
配置翻译器
===========
将用户友好的config.yaml转换为技术参数
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigTranslator:
    """配置翻译器 - 将用户配置转换为技术参数"""
    
    def __init__(self, user_config_path: str):
        """
        初始化配置翻译器
        
        Args:
            user_config_path: 用户配置文件路径
        """
        self.user_config_path = Path(user_config_path)
        self.user_config = self._load_user_config()
    
    def _load_user_config(self) -> Dict[str, Any]:
        """加载用户配置"""
        if not self.user_config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.user_config_path}")
        
        with open(self.user_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info("✅ 用户配置加载成功")
        return config
    
    def translate_to_technical_config(self) -> Dict[str, Any]:
        """
        将用户友好配置转换为技术配置
        
        Returns:
            技术配置字典
        """
        logger.info("🔄 开始配置转换...")
        
        # 基础技术配置框架
        tech_config = {
            'data_paths': self._translate_data_paths(),
            'results_paths': self._translate_results_paths(),
            'bertopic_params': self._translate_bertopic_params(),
            'visualization': self._translate_visualization(),
            'analysis': self._translate_analysis(),
            'hyperparameter_optimization': self._translate_optimization(),
            'system': self._translate_system(),
            'data_processing': self._translate_data_processing()
        }
        
        logger.info("✅ 配置转换完成")
        return tech_config
    
    def _translate_data_paths(self) -> Dict[str, Any]:
        """转换数据路径配置"""
        # 新配置结构
        user_paths = self.user_config.get('data_files', {})
        
        return {
            'media_data': user_paths.get('traditional_media'),
            'social_media_data': user_paths.get('social_media')
        }
    
    def _translate_results_paths(self) -> Dict[str, Any]:
        """转换结果路径配置"""
        output_dir = self.user_config.get('output_settings', {}).get('results_folder', 'results')
        
        return {
            'output_dir': output_dir,
            'model_dir': f"{output_dir}/trained_model",
            'summary_file': f"{output_dir}/topics_summary.csv",
            'summary_enhanced': f"{output_dir}/topics_summary_enhanced.csv",
            'cross_lingual_file': f"{output_dir}/cross_lingual_composition.csv",
            'evolution_file': f"{output_dir}/dynamic_evolution_analysis.csv",
            'viz_file': f"{output_dir}/topic_visualization.html",
            'source_analysis': f"{output_dir}/topic_by_source.html",
            'timeline_analysis': f"{output_dir}/topics_over_time.html",
            'frame_heatmap': f"{output_dir}/topic_frame_heatmap.html"
        }
    
    def _translate_bertopic_params(self) -> Dict[str, Any]:
        """转换BERTopic参数"""
        # 新配置结构
        topic_config = self.user_config.get('topic_settings', {})
        
        # 根据用户选择的精度级别设置参数
        advanced = topic_config.get('advanced', {})
        precision = advanced.get('keyword_extraction', 'standard')
        if precision == 'basic':
            ngram_range = [1, 2]
            max_features = 3000
        elif precision == 'enhanced':
            ngram_range = [1, 4]
            max_features = 10000
        else:  # standard
            ngram_range = [1, 3]
            max_features = 5000
        
        # 根据语言模式选择embedding模型
        language_mode = topic_config.get('text_language', 'chinese')
        if language_mode == 'chinese':
            embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        elif language_mode == 'english':
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        elif language_mode == 'multilingual':
            embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        else:  # 默认中文
            embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        # 获取参数（调参模式用默认值，分析模式用选中的参数）
        if self.is_tuning_mode():
            params = self._get_default_parameters()
        else:
            params = self.get_selected_parameters()
        
        advanced = topic_config.get('advanced', {})
        
        return {
            'language': language_mode,
            'embedding_model': embedding_model,
            'min_topic_size': params.get('min_topic_size', topic_config.get('min_documents_per_topic', 15)),
            'nr_topics': topic_config.get('expected_topics', 'auto'),
            'n_gram_range': advanced.get('ngram_range', ngram_range),
            
            # 多语言预处理配置
            'expert_keyword_extraction': {
                'enable_pos_patterns': True,
                'enable_multilingual_preprocessing': True,
                'pos_patterns': {
                    'zh': '<n.*|a.*>*<n.*>+',
                    'en': '<JJ.*>*<NN.*>+',
                    'ru': '<A.*>*<N.*>+'
                },
                'custom_stopwords_path': 'stopwords/politics_stopwords.txt',
                'use_custom_stopwords': True,
                'pos_language_detection': True
            },
            
            # UMAP参数
            'umap_params': {
                'n_neighbors': params.get('n_neighbors', 15),
                'n_components': params.get('n_components', 5),
                'min_dist': advanced.get('min_dist', 0.0),
                'metric': 'cosine',
                'random_state': 42
            },
            
            # HDBSCAN参数
            'hdbscan_params': {
                'min_cluster_size': params.get('min_cluster_size', 15),
                'min_samples': params.get('min_samples', 5),
                'metric': 'euclidean',
                'cluster_selection_method': 'eom',
                'prediction_data': True
            }
        }
    
    def _translate_visualization(self) -> Dict[str, Any]:
        """转换可视化配置"""
        viz_config = self.user_config.get('visualization', {})
        
        # 根据图表质量设置DPI
        quality = viz_config.get('chart_quality', 'high')
        if quality == 'standard':
            dpi = 150
        elif quality == 'publication':
            dpi = 600
        else:  # high
            dpi = 300
        
        return {
            'figsize': [viz_config.get('figure_width', 12), viz_config.get('figure_height', 8)],
            'dpi': viz_config.get('dpi', dpi),
            'style': 'matplotlib',
            'color_scheme': 'viridis',
            'save_format': 'png',
            
            'sota_charts': {
                'enable': self.user_config.get('analysis_mode', {}).get('enable_academic_visualizations', True),
                'high_dpi': dpi,
                'formats': viz_config.get('output_formats', ['png', 'svg']),
                'font_family': 'Times New Roman' if viz_config.get('font_style') == 'serif' else 'Arial',
                'color_palette': viz_config.get('color_theme', 'academic'),
                'enable_annotations': True,
                'annotation_examples': []
            }
        }
    
    def _translate_analysis(self) -> Dict[str, Any]:
        """转换分析功能配置"""
        temporal = self.user_config.get('temporal_analysis', {})
        source = self.user_config.get('source_analysis', {})
        frame = self.user_config.get('frame_analysis', {})
        
        return {
            'time_analysis': {
                'enable': temporal.get('enable', True),
                'time_column': temporal.get('date_column', '日期'),
                'bins': temporal.get('time_periods', 10)
            },
            
            'source_analysis': {
                'enable': source.get('enable', True),
                'source_column': source.get('source_column', 'Source')
            },
            
            'frame_analysis': {
                'enable': frame.get('enable', True),
                'threshold': frame.get('frame_threshold', 0.1)
            }
        }
    
    def _translate_optimization(self) -> Dict[str, Any]:
        """转换超参数优化配置"""
        enable_opt = self.is_tuning_mode()
        tuning_config = self.user_config.get('tuning_settings', {})
        search_ranges = tuning_config.get('search_ranges', {})
        
        return {
            'enable': enable_opt,
            'n_trials': tuning_config.get('optimization_trials', 100),
            
            'search_space': {
                'n_neighbors': search_ranges.get('n_neighbors', [5, 50]),
                'n_components': search_ranges.get('n_components', [2, 10]),
                'min_dist': [0.0, 0.5],
                'min_cluster_size': search_ranges.get('min_cluster_size', [5, 100]),
                'min_samples': search_ranges.get('min_samples', [1, 15]),
                'ngram_range': [1, 4],
                'max_features': [1000, 10000]
            },
            
            'coherence_type': 'c_v',
            'save_top_n': tuning_config.get('save_top_candidates', 5),
            'auto_apply_best': False,
            'show_progress': True,
            'save_intermediate': True,
            'study_name': 'auto'
        }
    
    def _translate_system(self) -> Dict[str, Any]:
        """转换系统配置"""
        system_config = self.user_config.get('advanced_settings', {}).get('system', {})
        
        return {
            'random_seed': system_config.get('random_seed', 42),
            'verbose': system_config.get('verbose', True),
            'save_intermediate': False,
            'use_gpu': False
        }
    
    def _translate_data_processing(self) -> Dict[str, Any]:
        """转换数据处理配置"""
        return {
            'text_column': 'Unit_Text',
            'merge_strategy': 'concat',
            'metadata_columns': [
                '序号', '日期', '标题', '链接', 'Token数', 'text', 'token数',
                'Unit_ID', 'Source', 'Macro_Chunk_ID', 'speaker', 'Unit_Text',
                'seed_sentence', 'expansion_logic', 'Unit_Hash', 'processing_status',
                'Incident', 'Valence',
                
                # Frame分析列
                'Frame_ProblemDefinition', 'Frame_ProblemDefinition_Present',
                'Frame_ResponsibilityAttribution', 'Frame_ResponsibilityAttribution_Present',
                'Frame_MoralEvaluation', 'Frame_MoralEvaluation_Present',
                'Frame_SolutionRecommendation', 'Frame_TreatmentRecommendation_Present',
                'Frame_ActionStatement', 'Frame_ConflictAttribution_Present',
                'Frame_CausalExplanation', 'Frame_CausalInterpretation_Present',
                
                # 分析维度列
                'Evidence_Type', 'Attribution_Level', 'Temporal_Focus',
                'Primary_Actor_Type', 'Geographic_Scope',
                'Relationship_Model_Definition', 'Discourse_Type'
            ]
        }
    
    def get_analysis_mode(self) -> str:
        """获取分析模式"""
        # 新配置结构
        mode = self.user_config.get('analysis_mode', 'analyze')
        
        # 两阶段模式映射
        if mode == 'tune':
            return 'tune'  # 调参模式
        else:
            return 'analyze'  # 分析模式
    
    def is_tuning_mode(self) -> bool:
        """是否为调参模式"""
        return self.get_analysis_mode() == 'tune'
    
    def get_selected_parameters(self) -> Dict[str, Any]:
        """获取选中的候选参数"""
        # 新配置结构
        candidate_config = self.user_config.get('parameter_selection', {})
        
        if candidate_config.get('manual_override', {}).get('enable', False):
            # 使用手动参数
            manual = candidate_config['manual_override']
            return {
                'min_topic_size': manual.get('min_documents_per_topic', 15),
                'n_neighbors': 15,  # 默认值
                'n_components': 5,  # 默认值
                'min_cluster_size': manual.get('min_documents_per_topic', 15),
                'min_samples': 5  # 默认值
            }
        else:
            # 使用候选参数
            selected_num = candidate_config.get('use_candidate', 1)
            candidates = candidate_config.get('candidates', {})
            candidate_key = f'candidate_{selected_num}'
            
            if candidate_key in candidates:
                candidate = candidates[candidate_key]
                return {
                    'min_topic_size': candidate.get('min_topic_size', 15),
                    'n_neighbors': candidate.get('n_neighbors', 15),
                    'n_components': candidate.get('n_components', 5),
                    'min_cluster_size': candidate.get('min_cluster_size', 15),
                    'min_samples': candidate.get('min_samples', 5)
                }
            else:
                # 回退到默认参数
                return self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        topic_config = self.user_config.get('topic_parameters', {})
        advanced = topic_config.get('advanced', {})
        
        return {
            'min_topic_size': topic_config.get('min_topic_size', 15),
            'n_neighbors': advanced.get('n_neighbors', 15),
            'n_components': advanced.get('n_components', 5),
            'min_cluster_size': advanced.get('min_cluster_size', 15),
            'min_samples': advanced.get('min_samples', 5)
        }
    
    def save_technical_config(self, output_path: str = None) -> str:
        """
        保存技术配置到文件
        
        Args:
            output_path: 输出路径
            
        Returns:
            保存的文件路径
        """
        if output_path is None:
            output_path = self.user_config_path.parent / 'config_technical.yaml'
        
        tech_config = self.translate_to_technical_config()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(tech_config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        logger.info(f"✅ 技术配置已保存: {output_path}")
        return str(output_path)


def translate_config(user_config_path: str) -> Dict[str, Any]:
    """
    便捷函数：直接转换配置
    
    Args:
        user_config_path: 用户配置文件路径
        
    Returns:
        技术配置字典
    """
    translator = ConfigTranslator(user_config_path)
    return translator.translate_to_technical_config()


if __name__ == "__main__":
    # 测试配置转换
    translator = ConfigTranslator('config.yaml')
    tech_config = translator.translate_to_technical_config()
    translator.save_technical_config('config_technical.yaml')
    
    print("🎉 配置转换完成！")
    print(f"分析模式: {translator.get_analysis_mode()}")
