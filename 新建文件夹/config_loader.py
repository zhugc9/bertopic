"""配置加载与默认值处理"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


METADATA_COLUMNS = [
    '序号', '日期', '标题', '链接', 'Token数', 'text', 'token数',
    'Unit_ID', 'Source', 'Macro_Chunk_ID', 'speaker', 'Unit_Text',
    'seed_sentence', 'expansion_logic', 'Unit_Hash', 'processing_status',
    'Incident', 'Valence',
    'Frame_ProblemDefinition', 'Frame_ProblemDefinition_Present',
    'Frame_ResponsibilityAttribution', 'Frame_ResponsibilityAttribution_Present',
    'Frame_MoralEvaluation', 'Frame_MoralEvaluation_Present',
    'Frame_SolutionRecommendation', 'Frame_TreatmentRecommendation_Present',
    'Frame_ActionStatement', 'Frame_ConflictAttribution_Present',
    'Frame_CausalExplanation', 'Frame_CausalInterpretation_Present',
    'Evidence_Type', 'Attribution_Level', 'Temporal_Focus',
    'Primary_Actor_Type', 'Geographic_Scope',
    'Relationship_Model_Definition', 'Discourse_Type'
]


DEFAULT_USER_CONFIG: Dict[str, Any] = {
    'data': {
        'files': {
            'traditional_media': None,
            'social_media': None,
        }
    },
    'analysis': {
        'mode': 'analyze',
        'save_candidates': True,
    },
    'topic': {
        'min_documents_per_topic': 15,
        'expected_topics': 'auto',
        'text_language': 'chinese',
        'advanced': {
            'keyword_extraction': 'standard',
            'ngram_range': [1, 3],
            'min_dist': 0.0,
            'n_neighbors': 15,
            'n_components': 5,
            'min_samples': 5,
            'min_cluster_size': 15,
        }
    },
    'advanced': {
        'auto_tuning': {
            'trials': 100,
            'save_best': 5,
        },
        'auto_tuning_advanced': {
            'search_space': {
                'topic_granularity': [5, 50],
                'clustering_sensitivity': [2, 15],
            }
        },
        'graphs': {
            'academic_charts': True,
            'formats': ['pdf'],
            'dpi': 300,
            'figure_width': 12,
            'figure_height': 8,
        },
        'sota_visuals': True,
        'system': {
            'random_seed': 42,
            'max_memory': 'auto',
            'use_gpu': False,
        }
    },
    'features': {
        'time_evolution': {
            'enable': True,
            'time_periods': 10,
            'date_column': '日期',
        },
        'source_comparison': {
            'enable': True,
            'source_column': 'Source',
        },
        'frame_analysis': {
            'enable': False,
            'frame_threshold': 0.1,
        }
    },
    'output_settings': {
        'folder': 'results',
        'names': {
            'topic_summary': '主题摘要表.csv',
            'document_mapping': '文档主题分布表.csv',
            'keyword_table': '主题关键词表.csv',
            'analysis_report': '主题分析报告.txt',
        }
    },
    'outputs': {
        'topic_summary': True,
        'document_mapping': True,
        'keyword_table': True,
        'analysis_report': True,
        'interactive_charts': True,
    },
    'ai_labeling': {
        'enable': False,
    },
    'candidate_parameters': {}
}


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典"""
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_user_config(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return copy.deepcopy(DEFAULT_USER_CONFIG)
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return deep_merge(DEFAULT_USER_CONFIG, data)


def _build_results_paths(user_config: Dict[str, Any]) -> Dict[str, str]:
    output_cfg = user_config.get('output_settings') or user_config.get('output', {})
    results_dir = output_cfg.get('folder', 'results')
    file_names = output_cfg.get('names', {})

    def path_for(name: str, default: str) -> str:
        return str(Path(results_dir) / file_names.get(name, default))

    return {
        'output_dir': str(Path(results_dir)),
        'model_dir': str(Path(results_dir) / '训练模型'),
        'summary_file': path_for('topic_summary', '主题摘要表.csv'),
        'summary_enhanced': path_for('keyword_table', '增强版主题摘要.csv'),
        'cross_lingual_file': str(Path(results_dir) / '跨语言构成报告.csv'),
        'evolution_file': str(Path(results_dir) / '主题演化分析.csv'),
        'viz_file': str(Path(results_dir) / '主题可视化.html'),
        'source_analysis': str(Path(results_dir) / '来源主题热力图.html'),
        'timeline_analysis': str(Path(results_dir) / '主题时间演化图.html'),
        'frame_heatmap': str(Path(results_dir) / '框架热力图.html'),
        'analysis_summary': str(Path(results_dir) / '分析结果摘要.txt'),
        'charts_summary': str(Path(results_dir) / '图表清单.txt'),
        'document_topic_mapping': path_for('document_mapping', '文档主题分布表.csv'),
        'keyword_table': path_for('keyword_table', '主题关键词表.csv'),
        'analysis_report': path_for('analysis_report', '主题分析报告.txt'),
    }


LANGUAGE_EMBEDDINGS = {
    'chinese': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'english': 'sentence-transformers/all-MiniLM-L6-v2',
    'multilingual': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
}


def build_runtime_config(user_config: Dict[str, Any]) -> Dict[str, Any]:
    runtime = copy.deepcopy(user_config)

    # 数据路径
    data_cfg = user_config.get('data', {})
    data_files = data_cfg.get('files', {}) or user_config.get('data_files', {})
    runtime['data_paths'] = {
        'media_data': data_files.get('traditional_media'),
        'social_media_data': data_files.get('social_media'),
    }

    # 结果路径
    runtime['results_paths'] = _build_results_paths(user_config)

    # BERTopic参数
    topic_settings = user_config.get('topic', {})
    advanced = topic_settings.get('advanced', {})
    system_settings = user_config.get('advanced', {}).get('system', {})
    language = topic_settings.get('text_language', 'chinese')
    embedding_model = LANGUAGE_EMBEDDINGS.get(language, LANGUAGE_EMBEDDINGS['chinese'])
    ngram_range = advanced.get('ngram_range')
    if not ngram_range:
        extraction_mode = advanced.get('keyword_extraction', 'standard')
        if extraction_mode == 'basic':
            ngram_range = [1, 2]
        elif extraction_mode == 'enhanced':
            ngram_range = [1, 4]
        else:
            ngram_range = [1, 3]

    random_seed = system_settings.get('random_seed', 42)

    runtime['bertopic_params'] = {
        'language': language,
        'embedding_model': embedding_model,
            'min_topic_size': topic_settings.get('min_documents_per_topic', 15),
            'nr_topics': topic_settings.get('expected_topics', 'auto'),
        'n_gram_range': ngram_range,
        'expert_keyword_extraction': {
            'enable_pos_patterns': True,
            'enable_multilingual_preprocessing': True,
            'pos_patterns': {
                'zh': '<n.*|a.*>*<n.*>+',
                'en': '<JJ.*>*<NN.*>+',
                'ru': '<A.*>*<N.*>+',
            },
            'custom_stopwords_path': 'stopwords/politics_stopwords.txt',
            'use_custom_stopwords': True,
            'pos_language_detection': True,
        },
        'umap_params': {
            'n_neighbors': advanced.get('n_neighbors', 15),
            'n_components': advanced.get('n_components', 5),
            'min_dist': advanced.get('min_dist', 0.0),
            'metric': 'cosine',
            'random_state': random_seed,
        },
        'hdbscan_params': {
            'min_cluster_size': advanced.get('min_cluster_size', topic_settings.get('min_documents_per_topic', 15)),
            'min_samples': advanced.get('min_samples', 5),
            'metric': 'euclidean',
            'cluster_selection_method': 'eom',
            'prediction_data': True,
        },
    }

    # 可视化设置
    charts_cfg = user_config.get('advanced', {}).get('graphs', {})
    runtime['visualization'] = {
        'figsize': [charts_cfg.get('figure_width', 12), charts_cfg.get('figure_height', 8)],
        'dpi': charts_cfg.get('dpi', 300),
        'style': 'matplotlib',
        'color_scheme': 'viridis',
        'save_format': 'png',
        'sota_charts': {
            'enable': charts_cfg.get('enable', True),
            'high_dpi': charts_cfg.get('dpi', 300),
            'formats': charts_cfg.get('formats', ['pdf']),
            'font_family': 'Arial',
            'color_palette': 'academic',
            'enable_annotations': True,
            'annotation_examples': [],
        },
    }

    # 分析设置
    features = user_config.get('features', {})
    runtime['analysis'] = {
        'time_analysis': {
            'enable': features.get('time_evolution', {}).get('enable', True),
            'time_column': features.get('time_evolution', {}).get('date_column', '日期'),
            'bins': features.get('time_evolution', {}).get('time_periods', 10),
        },
        'source_analysis': {
            'enable': features.get('source_comparison', {}).get('enable', True),
            'source_column': features.get('source_comparison', {}).get('source_column', 'Source'),
        },
        'frame_analysis': {
            'enable': features.get('frame_analysis', {}).get('enable', False),
            'threshold': features.get('frame_analysis', {}).get('frame_threshold', 0.1),
        },
    }

    # 超参数优化
    auto_tuning = user_config.get('advanced', {}).get('auto_tuning', {})
    search_space = user_config.get('advanced', {}).get('auto_tuning_advanced', {}).get('search_space', {})
    mode = user_config.get('analysis', {}).get('mode', 'analyze')
    runtime['hyperparameter_optimization'] = {
        'mode': mode,
        'enable': mode == 'tune',
        'n_trials': auto_tuning.get('trials', 100),
        'save_top_n': auto_tuning.get('save_best', 5),
        'coherence_type': 'c_v',
        'auto_apply_best': False,
        'show_progress': True,
        'save_intermediate': True,
        'study_name': 'auto',
        'sample_size': auto_tuning.get('sample_size'),
        'sample_strategy': auto_tuning.get('sample_strategy', 'random'),
        'search_space': {
            'n_neighbors': search_space.get('topic_granularity', [5, 50]),
            'n_components': [2, 10],
            'min_dist': [0.0, 0.5],
            'min_cluster_size': search_space.get('topic_granularity', [5, 50]),
            'min_samples': search_space.get('clustering_sensitivity', [2, 15]),
            'ngram_range': [1, 4],
            'max_features': [1000, 10000],
        },
    }

    # 系统设置
    runtime['system'] = {
        'random_seed': random_seed,
        'verbose': True,
        'save_intermediate': False,
        'use_gpu': system_settings.get('use_gpu', False),
    }

    # 数据处理
    data_processing_cfg = user_config.get('data_processing', {})
    runtime['data_processing'] = {
        'text_column': data_processing_cfg.get('text_column', 'Unit_Text'),
        'merge_strategy': data_processing_cfg.get('merge_strategy', 'concat'),
        'metadata_columns': data_processing_cfg.get('metadata_columns', METADATA_COLUMNS),
    }

    runtime['candidate_parameters'] = user_config.get('candidate_parameters', {})
    runtime.setdefault('analysis', {})
    runtime['analysis'].setdefault('mode', user_config.get('analysis', {}).get('mode', 'analyze'))
    runtime['outputs'] = user_config.get('outputs', {})


    return runtime


def load_runtime_config(path: Path) -> Dict[str, Any]:
    user_cfg = load_user_config(path)
    return build_runtime_config(user_cfg)

