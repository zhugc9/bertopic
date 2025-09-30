"""
BERTopic超参数自动优化模块
============================
基于Optuna的贝叶斯超参数优化 - SOTA & KISS实现
"""

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import time
import json
from datetime import datetime

# BERTopic和相关库
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from .multilingual_preprocessor import MultilingualTokenizer

# 主题一致性评估 - 延迟导入避免版本冲突
GENSIM_AVAILABLE = False
CoherenceModel = None
Dictionary = None
gensim = None

logger = logging.getLogger(__name__)


def _ensure_gensim_loaded():
    """按需加载gensim，避免启动时的版本冲突"""
    global GENSIM_AVAILABLE, CoherenceModel, Dictionary, gensim
    
    if not GENSIM_AVAILABLE:
        try:
            from gensim.models import CoherenceModel as CM
            from gensim.corpora import Dictionary as Dict
            import gensim as gs
            
            CoherenceModel = CM
            Dictionary = Dict
            gensim = gs
            GENSIM_AVAILABLE = True
            
        except ImportError as e:
            logger.warning(f"gensim不可用，将禁用主题一致性评估: {e}")
            GENSIM_AVAILABLE = False
        except Exception as e:
            logger.warning(f"gensim导入错误，将禁用主题一致性评估: {e}")
            GENSIM_AVAILABLE = False
    
    return GENSIM_AVAILABLE


class TopicCoherenceEvaluator:
    """主题一致性评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = MultilingualTokenizer(config)
    
    def calculate_coherence_score(self, 
                                topic_model: BERTopic, 
                                documents: List[str],
                                coherence_type: str = 'c_v') -> float:
        """
        计算主题一致性分数
        
        Args:
            topic_model: 训练好的BERTopic模型
            documents: 原始文档列表
            coherence_type: 一致性指标类型 ('c_v', 'c_npmi', 'c_uci', 'u_mass')
            
        Returns:
            主题一致性分数
        """
        # 如果gensim不可用，使用简化的评估方法
        if not _ensure_gensim_loaded():
            self.logger.warning("Gensim不可用，使用简化的主题质量评估")
            return self._simple_topic_quality_score(topic_model, documents)
        
        try:
            # 获取主题信息
            topic_info = topic_model.get_topic_info()
            topics = topic_model.topics_
            
            # 过滤噪声主题（-1）
            valid_topics = [t for t in set(topics) if t != -1]
            
            if len(valid_topics) < 2:
                self.logger.warning("有效主题数量小于2，返回默认分数")
                return 0.1
            
            # 准备主题词
            topic_words = []
            for topic_id in valid_topics:
                words = topic_model.get_topic(topic_id)
                if words:
                    # 只取前10个关键词
                    topic_words.append([word for word, _ in words[:10]])
            
            if not topic_words:
                return 0.1
            
            # 准备文档用于计算一致性
            # 简单分词（这里可以后续用更好的tokenizer替换）
            texts = []
            for doc in documents[:1000]:  # 采样1000个文档以提高速度
                if doc and isinstance(doc, str):
                    tokens = self.tokenizer.tokenize_text(doc)
                    texts.append([word for word in tokens if len(word) > 2])
            
            if not texts:
                return 0.1
            
            # 创建词典和语料库
            dictionary = Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            
            # 计算一致性
            coherence_model = CoherenceModel(
                topics=topic_words,
                texts=texts,
                dictionary=dictionary,
                coherence=coherence_type
            )
            
            coherence_score = coherence_model.get_coherence()
            self.logger.info(f"一致性分数 ({coherence_type}): {coherence_score:.4f}")
            
            return coherence_score if coherence_score is not None else 0.1
            
        except Exception as e:
            self.logger.error(f"计算主题一致性失败: {e}")
            return 0.1
    
    def _simple_topic_quality_score(self, topic_model: BERTopic, documents: List[str]) -> float:
        """
        简化的主题质量评估（当gensim不可用时）
        
        Args:
            topic_model: 训练好的BERTopic模型
            documents: 文档列表
            
        Returns:
            质量评分（0-1之间）
        """
        try:
            topic_info = topic_model.get_topic_info()
            topics = topic_model.topics_
            
            # 有效主题数量
            valid_topics = [t for t in set(topics) if t != -1]
            n_topics = len(valid_topics)
            
            if n_topics < 2:
                return 0.1
            
            # 噪声比例
            noise_ratio = sum(1 for t in topics if t == -1) / len(topics) if topics else 1.0
            
            # 主题大小均匀性
            if len(topic_info) > 1:
                sizes = topic_info['Count'].values[1:]  # 排除噪声主题
                if len(sizes) > 1:
                    size_std = np.std(sizes)
                    size_mean = np.mean(sizes)
                    cv = size_std / size_mean if size_mean > 0 else 1.0
                    uniformity = 1.0 / (1.0 + cv)
                else:
                    uniformity = 1.0
            else:
                uniformity = 0.1
            
            # 合成评分
            quality_score = 0.4 * (1.0 - noise_ratio) + 0.3 * uniformity + 0.3 * min(n_topics / 20.0, 1.0)
            
            return max(0.1, min(1.0, quality_score))
            
        except Exception as e:
            self.logger.error(f"简化质量评估失败: {e}")
            return 0.1


class OptunaBERTopicOptimizer:
    """基于Optuna的BERTopic超参数优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化优化器
        
        Args:
            config: 配置字典
        """
        # 检查依赖
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna未安装。请运行: pip install optuna\n"
                "这是超参数优化功能所必需的。"
            )
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.coherence_evaluator = TopicCoherenceEvaluator(self.config)
        
        # 缓存数据以避免重复计算
        self.documents = None
        self.embeddings = None
        self.embedding_model = None
        
    def create_objective_function(self, 
                                documents: List[str],
                                study_name: str = "bertopic_optimization") -> callable:
        """
        创建Optuna目标函数
        
        Args:
            documents: 文档列表
            study_name: 研究名称
            
        Returns:
            Optuna目标函数
        """
        # 缓存文档和预计算嵌入以提高效率
        self.documents = documents
        self.logger.info(f"预计算文档嵌入向量（{len(documents)}个文档）...")
        
        # 初始化嵌入模型
        embedding_model_name = self.config['bertopic_params']['embedding_model']
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # 预计算嵌入向量（这是最耗时的部分，只做一次）
        start_time = time.time()
        self.embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        embed_time = time.time() - start_time
        self.logger.info(f"嵌入计算完成，用时 {embed_time:.2f} 秒")
        
        def objective(trial) -> float:
            """
            Optuna目标函数
            
            Args:
                trial: Optuna试验对象
                
            Returns:
                需要最大化的目标值（主题一致性分数）
            """
            try:
                # 定义超参数搜索空间
                hyperparams = self._define_hyperparameter_space(trial)
                
                # 使用建议的超参数训练BERTopic模型
                topic_model = self._create_bertopic_model(hyperparams)
                
                # 训练模型（使用预计算的嵌入）
                topics, _ = topic_model.fit_transform(documents, self.embeddings)
                
                # 计算主题一致性分数
                coherence_score = self.coherence_evaluator.calculate_coherence_score(
                    topic_model, documents, 'c_v'
                )
                
                # 计算额外的质量指标
                quality_metrics = self._calculate_quality_metrics(topic_model, topics)
                
                # 组合得分（主要是一致性，加上质量奖励）
                final_score = coherence_score + quality_metrics['quality_bonus']
                
                # 记录试验信息
                self.logger.info(f"试验 {trial.number}: 一致性={coherence_score:.4f}, "
                               f"最终得分={final_score:.4f}, "
                               f"参数={hyperparams}")
                
                return final_score
                
            except Exception as e:
                self.logger.error(f"试验 {trial.number} 失败: {e}")
                # 返回很低的分数以便Optuna知道这个参数组合不好
                return 0.0
        
        return objective
    
    def _define_hyperparameter_space(self, trial) -> Dict[str, Any]:
        """
        定义超参数搜索空间
        
        Args:
            trial: Optuna试验对象
            
        Returns:
            超参数字典
        """
        hyperparams = {}
        
        # UMAP参数
        hyperparams['n_neighbors'] = trial.suggest_int('n_neighbors', 5, 50)
        hyperparams['n_components'] = trial.suggest_int('n_components', 2, 10)
        hyperparams['min_dist'] = trial.suggest_float('min_dist', 0.0, 0.5)
        hyperparams['metric'] = trial.suggest_categorical('metric', ['cosine', 'euclidean'])
        
        # HDBSCAN参数
        hyperparams['min_cluster_size'] = trial.suggest_int('min_cluster_size', 10, 150)
        hyperparams['min_samples'] = trial.suggest_int('min_samples', 1, 15)
        hyperparams['cluster_selection_method'] = trial.suggest_categorical(
            'cluster_selection_method', ['eom', 'leaf']
        )
        
        # BERTopic特定参数
        hyperparams['min_topic_size'] = hyperparams['min_cluster_size']  # 保持一致
        
        # 向量化参数
        hyperparams['ngram_range_start'] = trial.suggest_int('ngram_range_start', 1, 2)
        hyperparams['ngram_range_end'] = trial.suggest_int('ngram_range_end', 2, 4)
        hyperparams['max_features'] = trial.suggest_categorical(
            'max_features', [None, 1000, 5000, 10000]
        )
        
        return hyperparams
    
    def _create_bertopic_model(self, hyperparams: Dict[str, Any]) -> BERTopic:
        """
        基于超参数创建BERTopic模型
        
        Args:
            hyperparams: 超参数字典
            
        Returns:
            配置好的BERTopic模型
        """
        # UMAP降维模型
        umap_model = UMAP(
            n_neighbors=hyperparams['n_neighbors'],
            n_components=hyperparams['n_components'],
            min_dist=hyperparams['min_dist'],
            metric=hyperparams['metric'],
            random_state=42
        )
        
        # HDBSCAN聚类模型
        hdbscan_model = HDBSCAN(
            min_cluster_size=hyperparams['min_cluster_size'],
            min_samples=hyperparams['min_samples'],
            metric='euclidean',
            cluster_selection_method=hyperparams['cluster_selection_method'],
            prediction_data=True
        )
        
        # 向量化模型
        vectorizer_model = CountVectorizer(
            ngram_range=(hyperparams['ngram_range_start'], hyperparams['ngram_range_end']),
            stop_words=None,  # 我们在其他地方处理停用词
            max_features=hyperparams['max_features'],
            min_df=2,
            max_df=0.95
        )
        
        # 表示模型（用于提高关键词质量）
        representation_model = KeyBERTInspired()
        
        # 创建BERTopic模型
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            min_topic_size=hyperparams['min_topic_size'],
            nr_topics=None,  # 自动确定
            calculate_probabilities=False,  # 提高速度
            verbose=False
        )
        
        return topic_model
    
    def _calculate_quality_metrics(self, topic_model: BERTopic, topics: List[int]) -> Dict[str, float]:
        """
        计算额外的质量指标
        
        Args:
            topic_model: 训练好的模型
            topics: 主题分配
            
        Returns:
            质量指标字典
        """
        metrics = {}
        
        # 有效主题数量
        valid_topics = len([t for t in set(topics) if t != -1])
        metrics['n_topics'] = valid_topics
        
        # 噪声文档比例
        noise_ratio = sum(1 for t in topics if t == -1) / len(topics) if topics else 0
        metrics['noise_ratio'] = noise_ratio
        
        # 主题大小分布的均匀性
        topic_sizes = topic_model.get_topic_info()
        if len(topic_sizes) > 1:
            sizes = topic_sizes['Count'].values[1:]  # 排除噪声主题
            if len(sizes) > 1:
                size_std = np.std(sizes)
                size_mean = np.mean(sizes)
                cv = size_std / size_mean if size_mean > 0 else 1.0
                metrics['size_uniformity'] = 1.0 / (1.0 + cv)  # 越均匀分数越高
            else:
                metrics['size_uniformity'] = 1.0
        else:
            metrics['size_uniformity'] = 0.1
        
        # 计算质量奖励
        quality_bonus = 0.0
        
        # 合理的主题数量奖励（5-50个主题较理想）
        if 5 <= valid_topics <= 50:
            quality_bonus += 0.05
        elif valid_topics < 5:
            quality_bonus -= 0.1  # 主题太少扣分
        elif valid_topics > 100:
            quality_bonus -= 0.05  # 主题太多轻微扣分
        
        # 低噪声比例奖励
        if noise_ratio < 0.1:
            quality_bonus += 0.03
        elif noise_ratio > 0.3:
            quality_bonus -= 0.05
        
        # 主题大小均匀性奖励
        quality_bonus += metrics['size_uniformity'] * 0.02
        
        metrics['quality_bonus'] = quality_bonus
        
        return metrics
    
    def optimize_hyperparameters(self, 
                                documents: List[str],
                                n_trials: int = 50,
                                study_name: Optional[str] = None,
                                save_results: bool = True) -> Dict[str, Any]:
        """
        执行超参数优化
        
        Args:
            documents: 文档列表
            n_trials: 试验次数
            study_name: 研究名称
            save_results: 是否保存结果
            
        Returns:
            优化结果字典
        """
        if study_name is None:
            study_name = f"bertopic_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"开始超参数优化：{n_trials} 次试验")
        
        # 创建Optuna研究
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # 创建目标函数
        objective_function = self.create_objective_function(documents, study_name)
        
        # 执行优化
        start_time = time.time()
        study.optimize(objective_function, n_trials=n_trials)
        optimization_time = time.time() - start_time
        
        # 获取结果
        best_params = study.best_params
        best_score = study.best_value
        
        # 获取前5名的参数组合
        top_trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:5]
        top_results = []
        
        for i, trial in enumerate(top_trials):
            if trial.value is not None:
                top_results.append({
                    'rank': i + 1,
                    'score': trial.value,
                    'params': trial.params
                })
        
        # 构建结果字典
        optimization_results = {
            'best_params': best_params,
            'best_score': best_score,
            'top_5_results': top_results,
            'n_trials': n_trials,
            'optimization_time': optimization_time,
            'study_name': study_name,
            'optimization_summary': self._generate_optimization_summary(
                best_params, best_score, n_trials, optimization_time
            )
        }
        
        # 保存结果
        if save_results:
            self._save_optimization_results(optimization_results)
        
        self.logger.info(f"超参数优化完成！最佳分数: {best_score:.4f}")
        self.logger.info(f"最佳参数: {best_params}")
        
        return optimization_results
    
    def _generate_optimization_summary(self, 
                                     best_params: Dict[str, Any], 
                                     best_score: float,
                                     n_trials: int,
                                     optimization_time: float) -> Dict[str, str]:
        """生成优化摘要"""
        return {
            'optimization_status': '优化成功完成',
            'best_coherence_score': f"{best_score:.4f}",
            'trials_completed': f"{n_trials} 次试验",
            'optimization_duration': f"{optimization_time:.1f} 秒",
            'recommended_action': "使用最佳参数进行正式分析" if best_score > 0.3 else "建议检查数据质量或增加试验次数"
        }
    
    def _save_optimization_results(self, results: Dict[str, Any]) -> None:
        """保存优化结果"""
        try:
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # 保存详细结果
            results_file = results_dir / f"hyperparameter_optimization_{results['study_name']}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 保存简化的最佳参数
            best_params_file = results_dir / "best_hyperparameters.json"
            with open(best_params_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'best_params': results['best_params'],
                    'best_score': results['best_score'],
                    'optimization_date': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"优化结果已保存到: {results_file}")
            
        except Exception as e:
            self.logger.error(f"保存优化结果失败: {e}")
    
    def load_best_parameters(self) -> Optional[Dict[str, Any]]:
        """加载最佳参数"""
        try:
            best_params_file = Path("results/best_hyperparameters.json")
            if best_params_file.exists():
                with open(best_params_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data['best_params']
            else:
                self.logger.warning("未找到已保存的最佳参数文件")
                return None
        except Exception as e:
            self.logger.error(f"加载最佳参数失败: {e}")
            return None


# 便捷函数
def quick_optimize(documents: List[str], 
                  config: Dict[str, Any],
                  n_trials: int = 50) -> Dict[str, Any]:
    """
    快速超参数优化函数
    
    Args:
        documents: 文档列表
        config: 配置字典
        n_trials: 试验次数
        
    Returns:
        优化结果
    """
    optimizer = OptunaBERTopicOptimizer(config)
    return optimizer.optimize_hyperparameters(documents, n_trials=n_trials)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 示例配置
    test_config = {
        'bertopic_params': {
            'embedding_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        }
    }
    
    # 示例文档
    test_documents = [
        "人工智能正在改变我们的世界",
        "机器学习算法在医疗诊断中发挥重要作用",
        "深度学习技术推动自动驾驶汽车发展",
        "Natural language processing enables better human-computer interaction",
        "Computer vision applications are revolutionizing industries",
        "Machine learning models require large amounts of training data"
    ] * 10  # 重复以获得足够的数据
    
    print("开始测试超参数优化...")
    results = quick_optimize(test_documents, test_config, n_trials=5)
    print(f"测试完成！最佳分数: {results['best_score']:.4f}")
