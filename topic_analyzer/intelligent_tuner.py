"""
智能参数调优器
==============
基于数据特征分析自动选择最优BERTopic参数
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List, Any
from pathlib import Path
import re
from collections import Counter
import jieba
from langdetect import detect
import statistics


class DataAnalyzer:
    """数据特征分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_text_data(self, documents: List[str]) -> Dict[str, Any]:
        """
        分析文本数据特征
        
        Args:
            documents: 文档列表
            
        Returns:
            数据特征字典
        """
        features = {}
        
        # 基础统计
        features['total_docs'] = len(documents)
        
        # 文本长度分析
        text_lengths = [len(doc) if doc else 0 for doc in documents]
        features['avg_text_length'] = np.mean(text_lengths)
        features['median_text_length'] = np.median(text_lengths)
        features['text_length_std'] = np.std(text_lengths)
        features['min_text_length'] = min(text_lengths)
        features['max_text_length'] = max(text_lengths)
        
        # 语言分布分析
        features['language_distribution'] = self._analyze_languages(documents)
        features['dominant_language'] = max(features['language_distribution'].items(), 
                                          key=lambda x: x[1])[0]
        
        # 词汇丰富度分析
        features['vocabulary_diversity'] = self._calculate_vocabulary_diversity(documents)
        
        # 重复文本分析
        features['duplicate_ratio'] = self._calculate_duplicate_ratio(documents)
        
        # 文本密度分析（词汇密度）
        features['lexical_density'] = self._calculate_lexical_density(documents)
        
        # 主题复杂度预估
        features['estimated_complexity'] = self._estimate_topic_complexity(features)
        
        self.logger.info(f"数据分析完成: {features['total_docs']} 个文档")
        return features
    
    def _analyze_languages(self, documents: List[str]) -> Dict[str, int]:
        """分析语言分布"""
        language_counts = Counter()
        
        # 采样分析（大数据集只分析前1000个）
        sample_docs = documents[:1000] if len(documents) > 1000 else documents
        
        for doc in sample_docs:
            if not doc or len(doc.strip()) < 10:
                continue
                
            try:
                # 检测语言
                lang = detect(doc[:200])  # 只检测前200字符
                language_counts[lang] += 1
            except:
                # 如果检测失败，根据字符特征判断
                if re.search(r'[\u4e00-\u9fff]', doc):
                    language_counts['zh'] += 1
                elif re.search(r'[а-яё]', doc, re.IGNORECASE):
                    language_counts['ru'] += 1
                else:
                    language_counts['en'] += 1
        
        return dict(language_counts)
    
    def _calculate_vocabulary_diversity(self, documents: List[str]) -> float:
        """计算词汇多样性（TTR - Type-Token Ratio）"""
        all_words = []
        
        for doc in documents[:500]:  # 采样分析
            if not doc:
                continue
                
            # 根据语言选择分词方式
            if re.search(r'[\u4e00-\u9fff]', doc):
                # 中文分词
                words = list(jieba.cut(doc))
            else:
                # 英文/俄文简单分词
                words = re.findall(r'\b\w+\b', doc.lower())
            
            all_words.extend([w for w in words if len(w) > 1])
        
        if not all_words:
            return 0.0
            
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        return unique_words / total_words if total_words > 0 else 0.0
    
    def _calculate_duplicate_ratio(self, documents: List[str]) -> float:
        """计算重复文档比例"""
        if len(documents) < 2:
            return 0.0
            
        # 标准化文档用于比较
        normalized_docs = []
        for doc in documents:
            if doc:
                # 去除空白字符和标点，转为小写
                normalized = re.sub(r'[^\w\s]', '', doc.lower().strip())
                normalized_docs.append(normalized)
            else:
                normalized_docs.append('')
        
        # 计算重复比例
        unique_docs = len(set(normalized_docs))
        total_docs = len(normalized_docs)
        
        return 1 - (unique_docs / total_docs) if total_docs > 0 else 0.0
    
    def _calculate_lexical_density(self, documents: List[str]) -> float:
        """计算词汇密度（内容词vs功能词比例）"""
        # 功能词列表（中英俄）
        function_words = {
            'zh': {'的', '了', '在', '是', '和', '与', '或', '但', '而', '为', '对', '从', '到'},
            'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'},
            'ru': {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'за', 'при', 'о', 'об', 'что', 'как'}
        }
        
        content_words = 0
        function_word_count = 0
        
        for doc in documents[:200]:  # 采样分析
            if not doc:
                continue
                
            if re.search(r'[\u4e00-\u9fff]', doc):
                # 中文处理
                words = list(jieba.cut(doc))
                for word in words:
                    if len(word) > 1:
                        if word in function_words['zh']:
                            function_word_count += 1
                        else:
                            content_words += 1
            else:
                # 英文/俄文处理
                words = re.findall(r'\b\w+\b', doc.lower())
                for word in words:
                    if word in function_words['en'] or word in function_words['ru']:
                        function_word_count += 1
                    else:
                        content_words += 1
        
        total_words = content_words + function_word_count
        return content_words / total_words if total_words > 0 else 0.5
    
    def _estimate_topic_complexity(self, features: Dict[str, Any]) -> str:
        """基于特征估计主题复杂度"""
        complexity_score = 0
        
        # 基于文档数量
        if features['total_docs'] > 5000:
            complexity_score += 2
        elif features['total_docs'] > 1000:
            complexity_score += 1
        
        # 基于文本长度变异
        if features['text_length_std'] > features['avg_text_length'] * 0.5:
            complexity_score += 1
        
        # 基于词汇多样性
        if features['vocabulary_diversity'] > 0.6:
            complexity_score += 2
        elif features['vocabulary_diversity'] > 0.4:
            complexity_score += 1
        
        # 基于重复度
        if features['duplicate_ratio'] > 0.3:
            complexity_score -= 1
        
        # 基于语言多样性
        if len(features['language_distribution']) > 1:
            complexity_score += 1
        
        if complexity_score >= 4:
            return 'high'
        elif complexity_score >= 2:
            return 'medium'
        else:
            return 'low'


class ParameterOptimizer:
    """智能参数优化器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 参数优化规则库
        self.optimization_rules = {
            'min_topic_size': self._optimize_min_topic_size,
            'umap_n_neighbors': self._optimize_umap_neighbors,
            'umap_n_components': self._optimize_umap_components,
            'hdbscan_min_cluster_size': self._optimize_hdbscan_min_cluster,
            'hdbscan_min_samples': self._optimize_hdbscan_min_samples,
            'n_gram_range': self._optimize_ngram_range,
            'nr_topics': self._optimize_nr_topics
        }
    
    def optimize_parameters(self, data_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于数据特征优化参数
        
        Args:
            data_features: 数据特征字典
            
        Returns:
            优化后的参数字典
        """
        optimized_params = {}
        
        self.logger.info(f"开始智能参数优化，数据复杂度: {data_features['estimated_complexity']}")
        
        # 应用所有优化规则
        for param_name, optimizer_func in self.optimization_rules.items():
            try:
                optimized_params[param_name] = optimizer_func(data_features)
                self.logger.debug(f"优化参数 {param_name}: {optimized_params[param_name]}")
            except Exception as e:
                self.logger.warning(f"参数 {param_name} 优化失败: {e}")
        
        # 参数一致性检查和调整
        optimized_params = self._ensure_parameter_consistency(optimized_params, data_features)
        
        # 添加推荐理由
        optimized_params['optimization_reasoning'] = self._generate_reasoning(data_features, optimized_params)
        
        self.logger.info("参数优化完成")
        return optimized_params
    
    def _optimize_min_topic_size(self, features: Dict[str, Any]) -> int:
        """优化最小主题大小 - 海量数据专用算法"""
        total_docs = features['total_docs']
        complexity = features['estimated_complexity']
        duplicate_ratio = features['duplicate_ratio']
        dominant_lang = features.get('dominant_language', 'unknown')
        
        # 海量数据分级处理
        if total_docs >= 100000:
            # 超大规模：10万+文档
            base_percentage = 0.0005  # 0.05%
            min_size = 50
        elif total_docs >= 50000:
            # 大规模：5-10万文档
            base_percentage = 0.001   # 0.1%
            min_size = 30
        elif total_docs >= 10000:
            # 中大规模：1-5万文档
            base_percentage = 0.002   # 0.2%
            min_size = 20
        elif total_docs >= 5000:
            # 中等规模：5千-1万文档
            base_percentage = 0.004   # 0.4%
            min_size = 15
        else:
            # 小规模：5千以下文档
            base_percentage = 0.008   # 0.8%
            min_size = 8
        
        base_size = max(min_size, int(total_docs * base_percentage))
        
        # 语言特定优化
        if dominant_lang == 'ru':
            # 俄文词汇变化丰富，需要更小的主题捕获语义差异
            base_size = max(min_size // 2, int(base_size * 0.7))
        elif dominant_lang == 'zh':
            # 中文语义密度高，可以用稍大的主题
            base_size = int(base_size * 1.1)
        
        # 复杂度调整
        if complexity == 'high':
            base_size = max(min_size // 2, int(base_size * 0.8))
        elif complexity == 'low' and duplicate_ratio > 0.2:
            base_size = int(base_size * 1.3)
        
        # 严格边界限制
        return max(min_size // 2, min(min_size * 3, base_size))
    
    def _optimize_umap_neighbors(self, features: Dict[str, Any]) -> int:
        """优化UMAP邻居数 - 海量数据专用算法"""
        total_docs = features['total_docs']
        complexity = features['estimated_complexity']
        vocab_diversity = features['vocabulary_diversity']
        dominant_lang = features.get('dominant_language', 'unknown')
        
        # 海量数据分级处理 - 使用对数缩放
        if total_docs >= 100000:
            base_neighbors = 50  # 超大规模需要更多邻居保持全局结构
        elif total_docs >= 50000:
            base_neighbors = 40
        elif total_docs >= 10000:
            base_neighbors = 30
        elif total_docs >= 5000:
            base_neighbors = 25
        elif total_docs >= 1000:
            base_neighbors = 20
        else:
            base_neighbors = 15
        
        # 语言特定优化
        if dominant_lang == 'ru':
            # 俄文形态变化丰富，需要更多邻居捕获语义相似性
            base_neighbors = int(base_neighbors * 1.2)
        elif dominant_lang == 'zh':
            # 中文词汇密度高，可以用稍少邻居
            base_neighbors = int(base_neighbors * 0.9)
        
        # 复杂度和多样性调整
        if complexity == 'high' and vocab_diversity > 0.6:
            # 高复杂度高多样性：需要更多邻居保持结构
            base_neighbors = int(base_neighbors * 1.3)
        elif complexity == 'low' and vocab_diversity < 0.3:
            # 低复杂度低多样性：减少邻居避免过度平滑
            base_neighbors = int(base_neighbors * 0.8)
        
        # 海量数据的严格边界限制
        min_neighbors = max(10, int(np.log10(total_docs)) * 3)
        max_neighbors = min(100, int(np.sqrt(total_docs / 100)))
        
        return max(min_neighbors, min(max_neighbors, base_neighbors))
    
    def _optimize_umap_components(self, features: Dict[str, Any]) -> int:
        """优化UMAP降维维度"""
        total_docs = features['total_docs']
        complexity = features['estimated_complexity']
        vocab_diversity = features['vocabulary_diversity']
        
        # 基础维度选择
        if complexity == 'high' and vocab_diversity > 0.5:
            # 高复杂度高多样性：保留更多维度
            base_components = 7
        elif complexity == 'low' or total_docs > 10000:
            # 低复杂度或大数据集：降低维度提高速度
            base_components = 3
        else:
            # 中等情况
            base_components = 5
        
        return max(2, min(10, base_components))
    
    def _optimize_hdbscan_min_cluster(self, features: Dict[str, Any]) -> int:
        """优化HDBSCAN最小聚类大小"""
        # 通常与min_topic_size保持一致
        return self._optimize_min_topic_size(features)
    
    def _optimize_hdbscan_min_samples(self, features: Dict[str, Any]) -> int:
        """优化HDBSCAN最小样本数 - 海量数据专用算法"""
        min_cluster_size = self._optimize_hdbscan_min_cluster(features)
        duplicate_ratio = features['duplicate_ratio']
        total_docs = features['total_docs']
        dominant_lang = features.get('dominant_language', 'unknown')
        
        # 海量数据的基础值计算
        if total_docs >= 100000:
            # 超大规模：更严格的样本要求
            base_samples = max(3, min_cluster_size // 4)
        elif total_docs >= 10000:
            # 大规模：平衡精度和召回
            base_samples = max(2, min_cluster_size // 5)
        else:
            # 中小规模：传统方法
            base_samples = max(1, min_cluster_size // 3)
        
        # 根据重复度调整
        if duplicate_ratio > 0.4:
            # 高重复度：增加最小样本数提高稳定性
            base_samples = max(base_samples, min_cluster_size // 2)
        
        return max(1, min(20, base_samples))
    
    def _optimize_ngram_range(self, features: Dict[str, Any]) -> List[int]:
        """优化N-gram范围"""
        avg_length = features['avg_text_length']
        dominant_lang = features['dominant_language']
        lexical_density = features['lexical_density']
        
        # 基于语言特征
        if dominant_lang == 'zh':
            # 中文：较短的n-gram
            if avg_length > 200:
                return [1, 3]
            else:
                return [1, 2]
        else:
            # 英文/俄文：根据文本长度和词汇密度
            if avg_length > 500 and lexical_density > 0.6:
                return [1, 3]
            elif avg_length > 100:
                return [1, 2]
            else:
                return [1, 2]
    
    def _optimize_nr_topics(self, features: Dict[str, Any]) -> str:
        """优化主题数量"""
        total_docs = features['total_docs']
        complexity = features['estimated_complexity']
        
        # 大多数情况下使用自动确定
        if complexity == 'low' and total_docs < 1000:
            # 小规模低复杂度数据：限制主题数避免过拟合
            return min(15, max(5, total_docs // 50))
        else:
            # 其他情况：自动确定
            return 'auto'
    
    def _ensure_parameter_consistency(self, params: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """确保参数之间的一致性"""
        # 确保min_cluster_size = min_topic_size
        if 'min_topic_size' in params and 'hdbscan_min_cluster_size' in params:
            params['hdbscan_min_cluster_size'] = params['min_topic_size']
        
        # 确保min_samples <= min_cluster_size
        if 'hdbscan_min_samples' in params and 'hdbscan_min_cluster_size' in params:
            params['hdbscan_min_samples'] = min(
                params['hdbscan_min_samples'], 
                params['hdbscan_min_cluster_size']
            )
        
        # 确保components <= neighbors
        if 'umap_n_components' in params and 'umap_n_neighbors' in params:
            if params['umap_n_components'] >= params['umap_n_neighbors']:
                params['umap_n_components'] = max(2, params['umap_n_neighbors'] - 1)
        
        return params
    
    def _generate_reasoning(self, features: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
        """生成参数选择的推理说明"""
        reasoning = []
        
        # 数据规模分析
        total_docs = features['total_docs']
        if total_docs > 5000:
            reasoning.append(f"📊 大规模数据集({total_docs}文档)，采用平衡策略保证效率")
        elif total_docs < 500:
            reasoning.append(f"📊 小规模数据集({total_docs}文档)，采用精细化分析策略")
        else:
            reasoning.append(f"📊 中等规模数据集({total_docs}文档)，采用标准分析策略")
        
        # 复杂度分析
        complexity = features['estimated_complexity']
        if complexity == 'high':
            reasoning.append("🔍 检测到高复杂度文本，使用细粒度主题发现")
        elif complexity == 'low':
            reasoning.append("📝 检测到低复杂度文本，使用概括性主题聚合")
        else:
            reasoning.append("⚖️ 中等复杂度文本，平衡细节与概括")
        
        # 语言特征
        dominant_lang = features['dominant_language']
        lang_names = {'zh': '中文', 'en': '英文', 'ru': '俄文'}
        reasoning.append(f"🌐 主要语言: {lang_names.get(dominant_lang, dominant_lang)}，优化分词策略")
        
        # 参数解释
        if 'min_topic_size' in params:
            reasoning.append(f"🎯 主题大小设定为{params['min_topic_size']}，平衡主题质量与数量")
        
        return reasoning


class IntelligentTuner:
    """智能参数调优器主类"""
    
    def __init__(self):
        self.data_analyzer = DataAnalyzer()
        self.parameter_optimizer = ParameterOptimizer()
        self.logger = logging.getLogger(__name__)
    
    def auto_tune(self, documents: List[str]) -> Dict[str, Any]:
        """
        自动调优主函数
        
        Args:
            documents: 输入文档列表
            
        Returns:
            优化的参数配置和分析报告
        """
        self.logger.info("开始智能参数调优")
        
        # Step 1: 分析数据特征
        self.logger.info("第1步：分析数据特征...")
        data_features = self.data_analyzer.analyze_text_data(documents)
        
        # Step 2: 优化参数
        self.logger.info("第2步：智能参数优化...")
        optimized_params = self.parameter_optimizer.optimize_parameters(data_features)
        
        # Step 3: 生成完整报告
        tuning_report = {
            'data_features': data_features,
            'optimized_parameters': optimized_params,
            'tuning_summary': self._generate_tuning_summary(data_features, optimized_params)
        }
        
        self.logger.info("智能参数调优完成")
        return tuning_report
    
    def _generate_tuning_summary(self, features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, str]:
        """生成调优摘要"""
        return {
            'data_scale': f"{features['total_docs']} 个文档",
            'complexity_level': features['estimated_complexity'],
            'primary_language': features['dominant_language'],
            'optimization_strategy': f"最小主题大小: {params.get('min_topic_size', 'N/A')}",
            'expected_performance': self._predict_performance(features, params)
        }
    
    def _predict_performance(self, features: Dict[str, Any], params: Dict[str, Any]) -> str:
        """预测性能表现"""
        total_docs = features['total_docs']
        complexity = features['estimated_complexity']
        
        if total_docs > 10000 and complexity == 'high':
            return "预计运行时间较长，但结果详细准确"
        elif total_docs < 1000:
            return "快速分析，适合探索性研究"
        else:
            return "平衡的性能和质量表现"
