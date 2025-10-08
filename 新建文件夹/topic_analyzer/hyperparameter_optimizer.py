"""
BERTopic超参数自动优化模块
============================
基于Optuna的贝叶斯超参数优化 - SOTA & KISS实现
"""

try:
    import optuna
    OPTUNA_AVAILABLE = True
    try:
        from optuna.integration import TqdmCallback
    except ImportError:
        TqdmCallback = None
except ImportError:
    OPTUNA_AVAILABLE = False
    TqdmCallback = None

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional, Callable
from pathlib import Path
import time
import json
import threading
from datetime import datetime
import math
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from tqdm import tqdm

# BERTopic和相关库
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from .multilingual_preprocessor import MultilingualTokenizer, EnhancedMultilingualVectorizer

logger = logging.getLogger(__name__)

DEFAULT_TQDM_NCOLS = 100
DEFAULT_TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
DEFAULT_STAGE1_REPEATS = 1
DEFAULT_STAGE1_RATIO = 0.1
DEFAULT_STAGE1_MIN_SAMPLES = 300
DEFAULT_STAGE1_SCORE_WEIGHTS = {
    'diversity_weight': 0.4,
    'noise_weight': 0.4,
    'uniformity_weight': 0.2,
}


def _sample_documents(documents: List[str], ratio: float, min_samples: int, seed: int) -> Tuple[List[str], List[int]]:
    total_docs = len(documents)
    if total_docs == 0:
        return [], []
    ratio = max(0.0, min(1.0, ratio or 0.0))
    min_samples = max(0, min_samples or 0)
    sample_count = max(min_samples, int(math.ceil(total_docs * ratio)))
    sample_count = min(sample_count, total_docs)
    if sample_count >= total_docs:
        indices = list(range(total_docs))
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(total_docs), sample_count))
    sampled_docs = [documents[i] for i in indices]
    return sampled_docs, indices


def compute_topic_diversity(topic_model: BERTopic, top_k: int = 25) -> float:
    try:
        unique_words = set()
        total_words = 0
        for topic_id, words in topic_model.get_topics().items():
            if topic_id == -1 or not words:
                continue
            top_words = [word for word, _ in words[:top_k] if isinstance(word, str)]
            total_words += len(top_words)
            unique_words.update(top_words)
        if total_words == 0:
            return 0.0
        return len(unique_words) / float(total_words)
    except Exception:
        return 0.0


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
        self.random_seed = self.config.get('system', {}).get('random_seed', 42)
        hpo_cfg = self.config.get('hyperparameter_optimization', {})
        self.stage1_ratio = hpo_cfg.get('sample_ratio_stage1', DEFAULT_STAGE1_RATIO)
        self.stage1_min_samples = hpo_cfg.get('min_samples_stage1', DEFAULT_STAGE1_MIN_SAMPLES)
        self.stage1_repeats = hpo_cfg.get('stage1_repeats', DEFAULT_STAGE1_REPEATS)
        
        # 读取得分权重（可配置）
        self.stage1_score_weights = hpo_cfg.get('score_weights', DEFAULT_STAGE1_SCORE_WEIGHTS.copy())
        
        # 读取搜索空间配置
        self.search_space = hpo_cfg.get('search_space', {})
        
        # 缓存数据以避免重复计算
        self.documents = None
        self.embeddings = None
        self.embedding_model = None
        self.total_trials = hpo_cfg.get('n_trials', 0)

    def _announce(self, message: str, level: str = "info") -> None:
        """同时输出到控制台和日志，确保用户能看到进展"""
        console_message = f"[Optuna] {message}"
        print(console_message, flush=True)
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        else:
            self.logger.error(message)
        
    def _prepare_embeddings(self, documents: List[str]) -> None:
        self._announce("[Stage 1] 开始预计算句向量")
        embedding_model_name = self.config['bertopic_params']['embedding_model']
        self.embedding_model = SentenceTransformer(embedding_model_name)
        start_time = time.time()
        self.embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        embed_time = time.time() - start_time
        self._announce(f"[Stage 1] 句向量预计算完成，用时 {embed_time:.1f} 秒")

    def _calculate_stage1_score(self, quality_metrics: Dict[str, float]) -> float:
        noise_ratio = quality_metrics.get('noise_ratio', 1.0)
        size_uniformity = quality_metrics.get('size_uniformity', 0.0)
        diversity = quality_metrics.get('topic_diversity', 0.0)
        noise_score = max(0.0, 1.0 - noise_ratio)
        uniformity_score = size_uniformity
        diversity_score = diversity
        score = (
            self.stage1_score_weights['noise_weight'] * noise_score +
            self.stage1_score_weights['uniformity_weight'] * uniformity_score +
            self.stage1_score_weights['diversity_weight'] * diversity_score
        )
        return max(0.0, min(1.0, score))

    def _run_stage1_trials(self, documents: List[str]) -> Dict[str, Any]:
        total_docs = len(documents)
        total_trials = self.config.get('hyperparameter_optimization', {}).get('n_trials', 50)
        repeats = max(1, int(self.stage1_repeats))
        overall_trials = max(1, total_trials * repeats)
        self._announce(
            f"[Stage 1] 计划运行 {overall_trials} 次试验（{repeats} 轮，每轮 {total_trials} 次），总文档 {total_docs} 条"
        )

        stage1_results = []

        for repeat_idx in range(repeats):
            seed = self.random_seed + repeat_idx
            sampled_docs, indices = _sample_documents(
                documents,
                ratio=self.stage1_ratio,
                min_samples=self.stage1_min_samples,
                seed=seed,
            )
            self._announce(
                f"[Stage 1] 第 {repeat_idx + 1}/{repeats} 轮抽样完成，样本量 {len(sampled_docs)} 条"
            )
            self._prepare_embeddings(sampled_docs)

            trial_offset = repeat_idx * total_trials

            def objective(trial: optuna.trial.Trial) -> float:
                global_index = trial_offset + trial.number + 1
                progress_prefix = f"[Stage 1] 试验 {global_index}/{overall_trials}"
                start = time.time()
                try:
                    hyperparams = self._define_hyperparameter_space(trial)
                    trial.set_user_attr('min_topic_size', hyperparams.get('min_topic_size'))
                    self._announce(f"{progress_prefix} 开始训练模型")
                    topic_model = self._create_bertopic_model(hyperparams)
                    topics, _ = topic_model.fit_transform(sampled_docs, self.embeddings)
                    quality_metrics = self._calculate_quality_metrics(topic_model, topics)
                    quality_metrics['topic_diversity'] = compute_topic_diversity(topic_model)
                    score = self._calculate_stage1_score(quality_metrics)
                    elapsed = time.time() - start
                    self._announce(
                        f"{progress_prefix} 完成，得分 {score:.4f}，用时 {elapsed:.1f} 秒"
                    )
                    self.logger.info(
                        f"Stage1 Trial {global_index}: score={score:.4f}, metrics={quality_metrics}, params={hyperparams}"
                    )
                    return score
                except Exception as e:
                    self.logger.error(f"Stage1 Trial {global_index} failed: {e}")
                    return 0.0

            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, interval_steps=1)
            study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
            study.optimize(objective, n_trials=total_trials)
            pruned_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
            self._announce(
                f"[Stage 1] 第 {repeat_idx + 1}/{repeats} 轮完成，最佳得分 {study.best_value:.4f}，提前停止 {pruned_trials} 次"
            )

            stage1_results.append({
                'study_name': study.study_name,
                'best_params': study.best_params,
                'best_value': study.best_value,
                'trials': [
                    {
                        'number': t.number,
                        'value': t.value,
                        'params': t.params,
                        'state': str(t.state),
                        'user_attrs': t.user_attrs,
                    }
                    for t in study.trials
                ],
                'sampled_indices': indices,
                'sample_count': len(sampled_docs),
                'seed': seed,
            })

        best_run = max(stage1_results, key=lambda r: r['best_value']) if stage1_results else None
        if best_run:
            self._announce(
                f"[Stage 1] 全部完成，最优得分 {best_run['best_value']:.4f}，准备进入 Stage 2 复核"
            )

        return {
            'stage1_runs': stage1_results,
            'total_docs': total_docs,
            'stage1_ratio': self.stage1_ratio,
            'stage1_min_samples': self.stage1_min_samples,
            'repeats': repeats,
            'n_trials': total_trials,
            'overall_trials': overall_trials,
        }

    def _define_hyperparameter_space(self, trial) -> Dict[str, Any]:
        """
        定义超参数搜索空间（从config读取范围）
        
        Args:
            trial: Optuna试验对象
            
        Returns:
            超参数字典
        """
        hyperparams = {}
        
        # UMAP参数（从config读取）
        n_neighbors_range = self.search_space.get('n_neighbors', [5, 50])
        hyperparams['n_neighbors'] = trial.suggest_int('n_neighbors', n_neighbors_range[0], n_neighbors_range[1])
        
        n_components_range = self.search_space.get('n_components', [2, 10])
        hyperparams['n_components'] = trial.suggest_int('n_components', n_components_range[0], n_components_range[1])
        
        min_dist_range = self.search_space.get('min_dist', [0.0, 0.5])
        hyperparams['min_dist'] = trial.suggest_float('min_dist', min_dist_range[0], min_dist_range[1])
        
        metric_options = self.search_space.get('metric', ['cosine', 'euclidean'])
        hyperparams['metric'] = trial.suggest_categorical('metric', metric_options)
        
        # HDBSCAN参数（从config读取）
        min_cluster_size_range = self.search_space.get('min_cluster_size', [10, 150])
        hyperparams['min_cluster_size'] = trial.suggest_int('min_cluster_size', min_cluster_size_range[0], min_cluster_size_range[1])
        
        min_samples_range = self.search_space.get('min_samples', [1, 15])
        hyperparams['min_samples'] = trial.suggest_int('min_samples', min_samples_range[0], min_samples_range[1])
        
        cluster_method_options = self.search_space.get('cluster_selection_method', ['eom', 'leaf'])
        hyperparams['cluster_selection_method'] = trial.suggest_categorical('cluster_selection_method', cluster_method_options)
        
        # BERTopic特定参数
        hyperparams['min_topic_size'] = hyperparams['min_cluster_size']  # 保持一致
        
        # 向量化参数（从config读取，不在这里探索）
        # 注：ngram_range 等参数由 EnhancedMultilingualVectorizer 从 config.topic.advanced 读取
        max_features_options = self.search_space.get('max_features', [None, 1000, 5000, 10000])
        hyperparams['max_features'] = trial.suggest_categorical('max_features', max_features_options)
        
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
            random_state=self.random_seed
        )
        
        # HDBSCAN聚类模型
        hdbscan_model = HDBSCAN(
            min_cluster_size=hyperparams['min_cluster_size'],
            min_samples=hyperparams['min_samples'],
            metric='euclidean',
            cluster_selection_method=hyperparams['cluster_selection_method'],
            prediction_data=True
        )
        
        # 向量化模型（与正式分析保持一致）
        vectorizer_enhancer = EnhancedMultilingualVectorizer(self.config)
        vectorizer_model = vectorizer_enhancer.create_vectorizer()
        
        # 表示模型（用于提高关键词质量）
        representation_model = KeyBERTInspired()
        
        # 创建BERTopic模型
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
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
        执行超参数优化（仅Stage 1）
        
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
        self._announce(f"[Stage 1] 即将运行 {n_trials} 次试验进行参数筛选")

        optimization_start = time.time()
        stage1_summary = self._run_stage1_trials(documents)
        total_optimization_time = time.time() - optimization_start

        # 直接使用 Stage 1 结果
        runs = stage1_summary.get('stage1_runs', [])
        if runs:
            best_run = max(runs, key=lambda r: r['best_value'])
            best_params = best_run['best_params']
            best_score = best_run['best_value']
        else:
            raise RuntimeError("无法找到任何候选参数")

        # 提取前5名候选
        all_trials = []
        for run in runs:
            all_trials.extend(run.get('trials', []))
        all_trials.sort(key=lambda t: t.get('value', 0), reverse=True)
        
        save_top_n = self.config.get('hyperparameter_optimization', {}).get('save_top_n', 5)
        top_candidates = []
        for idx, trial in enumerate(all_trials[:save_top_n], start=1):
            # 合并params和user_attrs，方便后续读取
            combined_params = trial.get('params', {}).copy()
            combined_params.update(trial.get('user_attrs', {}))
            
            top_candidates.append({
                'rank': idx,
                'params': combined_params,
                'score': trial.get('value', 0),
                'final_score': trial.get('value', 0)
            })

        optimization_results = {
            'stage1': stage1_summary,
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': stage1_summary.get('overall_trials', n_trials),
            'optimization_time': total_optimization_time,
            'study_name': study_name,
            'top_5_results': top_candidates,
            'optimization_summary': self._generate_optimization_summary(
                best_params,
                best_score,
                stage1_summary.get('overall_trials', n_trials),
                total_optimization_time
            )
        }
        
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
        best_score_str = f"{best_score:.4f}" if isinstance(best_score, (int, float)) else "N/A"
        duration_str = f"{optimization_time:.1f} 秒" if isinstance(optimization_time, (int, float)) else "N/A"

        return {
            'optimization_status': '优化成功完成',
            'best_quality_score': best_score_str,
            'trials_completed': f"{n_trials} 次试验",
            'optimization_duration': duration_str,
            'recommended_action': "使用最佳参数进行正式分析" if isinstance(best_score, (int, float)) and best_score > 0.3 else "建议检查数据质量或增加试验次数"
        }
    
    def _save_optimization_results(self, results: Dict[str, Any]) -> None:
        """保存优化结果（由TuningManager统一管理）"""
        # 优化结果由TuningManager的candidate_parameters.yaml统一保存
        # 不再生成冗余的best_hyperparameters.json和stage1日志文件
        pass
    
    def load_best_parameters(self) -> Optional[Dict[str, Any]]:
        """加载最佳参数（从候选参数.yaml读取）"""
        try:
            candidates_file = Path("results/候选参数.yaml")
            if candidates_file.exists():
                import yaml
                with open(candidates_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                # 返回candidate_1（最佳参数）
                candidate_1 = data.get('candidates', {}).get('candidate_1', {})
                if candidate_1:
                    # 移除非参数字段
                    params = {k: v for k, v in candidate_1.items() 
                             if k not in ['quality_score', 'rank']}
                    return params
            self.logger.warning("未找到候选参数文件")
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
