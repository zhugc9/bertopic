"""
学术级可视化模块
================
集成出版级图表生成和高级可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns
from pathlib import Path
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

# 科学计算和机器学习
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform, pdist

# 网络图（可选）
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# 中文字体支持
import matplotlib.font_manager as fm

logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """学术级可视化生成器 - 统一实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化可视化生成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.results_paths = config.get('results_paths', {})
        self.viz_config = config.get('visualization', {}).get('sota_charts', {})
        self.random_seed = config.get('system', {}).get('random_seed', 42)
        
        # 学术级配色方案
        self.academic_colors = {
            'primary': '#2E86AB',      # 深蓝
            'secondary': '#A23B72',    # 深紫红
            'accent': '#F18F01',       # 橙色
            'success': '#C73E1D',      # 深红
            'info': '#5DADE2',         # 浅蓝
            'warning': '#F7DC6F',      # 黄色
            'light': '#ECF0F1',        # 浅灰
            'dark': '#2C3E50'          # 深灰
        }
        
        # 设置学术样式
        self._setup_academic_style()
        
    def _setup_academic_style(self):
        """设置学术论文级样式"""
        # 获取配置的DPI设置
        high_dpi = self.viz_config.get('high_dpi', 300)
        
        # 字体优先级：Microsoft YaHei > SimHei > Arial > 系统默认
        multilingual_fonts = ['Microsoft YaHei', 'SimHei', 'Arial', 'DejaVu Sans', 'sans-serif']
        
        # 设置matplotlib参数（包含字体和PDF嵌入设置）
        plt.rcParams.update({
            # 字体设置（支持中文、俄文、英文）
            'font.sans-serif': multilingual_fonts,
            'font.family': 'sans-serif',
            'axes.unicode_minus': False,
            # PDF字体嵌入设置（确保PDF中文字正常显示）
            'pdf.fonttype': 42,  # TrueType字体
            'ps.fonttype': 42,   # PostScript字体
            # 图表尺寸和DPI
            'figure.figsize': (12, 8),
            'figure.dpi': high_dpi,
            'savefig.dpi': high_dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.2,
            # 字体大小
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            # 线条样式
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2
        })
        
        # 设置seaborn样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        logger.info(f"✅ 学术级样式设置完成 (字体: {multilingual_fonts[0]}, DPI: {high_dpi})")
    
    def _save_chart(self, fig: plt.Figure, filename_prefix: str, chinese_name: str = None) -> Dict[str, str]:
        """
        保存图表到多种格式
        
        Args:
            fig: matplotlib图形对象
            filename_prefix: 文件名前缀（英文）
            chinese_name: 中文图表名称
            
        Returns:
            保存的文件路径字典
        """
        results_dir = Path(self.results_paths.get('output_dir', 'results'))
        results_dir.mkdir(exist_ok=True)
        
        # 获取中文名称映射
        chart_names = self.config.get('output_settings', {}).get('chart_names', {})
        chinese_filename = chart_names.get(filename_prefix, chinese_name or filename_prefix)
        
        # 获取配置的输出格式
        formats = self.viz_config.get('formats', ['png', 'svg'])
        saved_files = {}
        
        for format_type in formats:
            # 使用中文文件名
            file_path = results_dir / f"{chinese_filename}.{format_type}"
            
            try:
                fig.savefig(
                    file_path,
                    format=format_type,
                    bbox_inches='tight',
                    pad_inches=0.2,
                    facecolor='white',
                    edgecolor='none'
                )
                saved_files[format_type] = str(file_path)
                logger.info(f"  ✓ 保存{format_type.upper()}格式: {file_path.name}")
                
            except Exception as e:
                logger.warning(f"  ⚠ {format_type.upper()}格式保存失败: {e}")
        
        return saved_files
    
    def _add_annotations(self, ax: plt.Axes):
        """添加自定义标注"""
        if not self.viz_config.get('enable_annotations', False):
            return
        
        # 获取标注配置
        annotations = self.viz_config.get('annotation_examples', [])
        
        for annotation in annotations:
            text = annotation.get('text', '')
            xy = annotation.get('xy', [0.5, 0.8])
            style = annotation.get('style', 'default')
            
            if text:
                # 应用不同的标注样式
                if style == 'highlight':
                    bbox_props = dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7)
                elif style == 'important':
                    bbox_props = dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7)
                else:
                    bbox_props = dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7)
                
                ax.annotate(
                    text,
                    xy=xy,
                    xycoords='axes fraction',
                    bbox=bbox_props,
                    fontsize=10,
                    ha='center'
                )
    
    def plot_topic_distance_map(self, topic_model, embeddings=None, filename_prefix: str = "topic_distance_map") -> Dict[str, str]:
        """
        生成主题距离散点图
        
        Args:
            topic_model: BERTopic模型
            embeddings: 文档嵌入（可选）
            filename_prefix: 文件名前缀
            
        Returns:
            保存的文件路径字典
        """
        logger.info("  生成主题距离散点图...")
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            topic_info = topic_model.get_topic_info()
            topic_info = topic_info[topic_info.Topic != -1]

            topic_embeddings = None
            if hasattr(topic_model, 'topic_embeddings_'):
                embeddings = topic_model.topic_embeddings_
                if embeddings is not None and len(embeddings) >= len(topic_info):
                    topic_embeddings = embeddings[:len(topic_info)]

            topic_info_reduced = topic_info.reset_index(drop=True)
            if topic_embeddings is None or len(topic_info_reduced) < 2:
                logger.warning("  ⚠ 有效主题不足或缺少嵌入，使用 TF-IDF + SVD 生成占位图")
                topic_embeddings = self._build_fallback_embeddings(topic_model, topic_info_reduced)

            if topic_embeddings is None or len(topic_embeddings) == 0:
                ax.text(0.5, 0.5, '可视化暂不可用（缺少主题嵌入）',
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)
            else:
                embeddings_2d = self._reduce_to_2d(topic_embeddings)

                scatter = ax.scatter(
                    embeddings_2d[:, 0],
                    embeddings_2d[:, 1],
                    s=np.clip(topic_info_reduced['Count'].values * 2, 20, 800),
                    c=range(len(embeddings_2d)),
                    cmap='viridis',
                    alpha=0.75,
                    edgecolors='black',
                    linewidth=0.5
                )

                for idx, (x, y) in enumerate(embeddings_2d):
                    rep = topic_info_reduced.iloc[idx]['Representation']
                    if isinstance(rep, list):
                        label = ', '.join(rep[:3])
                    else:
                        label = str(rep)[:20]
                    ax.annotate(
                        f"主题 {topic_info_reduced.iloc[idx]['Topic']}: {label}",
                        (x, y),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.85
                    )

                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('主题索引', rotation=270, labelpad=15)

                ax.set_title('主题距离分布图', fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('降维轴 1', fontsize=12)
                ax.set_ylabel('降维轴 2', fontsize=12)
                ax.grid(True, alpha=0.3)

            self._add_annotations(ax)

            plt.tight_layout()
            saved_files = self._save_chart(fig, filename_prefix)
            plt.close(fig)

            return saved_files

        except Exception as e:
            logger.error(f"主题距离图生成失败: {e}")
            plt.close('all')
            return {}

    def _build_fallback_embeddings(self, topic_model, topic_info: pd.DataFrame) -> Optional[np.ndarray]:
        if topic_info.empty:
            return None

        rep_docs = topic_model.get_representative_docs()
        documents = []
        for topic_id in topic_info['Topic']:
            docs = rep_docs.get(topic_id) or []
            documents.extend(docs[:5])

        if not documents:
            return None
        
        # 根据文档数量动态调整参数，避免min_df/max_df冲突
        n_docs = len(documents)
        min_df = 1 if n_docs < 10 else 2
        max_df = 0.95 if n_docs >= 10 else 1.0

        vectorizer = CountVectorizer(max_features=1000, min_df=min_df, max_df=max_df)
        doc_vectors = vectorizer.fit_transform(documents)

        if doc_vectors.shape[0] < 2:
            return doc_vectors.toarray()

        svd = TruncatedSVD(n_components=min(10, doc_vectors.shape[1], doc_vectors.shape[0] - 1), random_state=self.random_seed)
        reduced = svd.fit_transform(doc_vectors)

        topic_embeddings = []
        offset = 0
        for topic_id in topic_info['Topic']:
            docs = rep_docs.get(topic_id) or []
            count = min(5, len(docs))
            if count == 0:
                topic_embeddings.append(np.zeros(reduced.shape[1]))
            else:
                topic_embeddings.append(np.mean(reduced[offset:offset + count], axis=0))
            offset += count

        return np.array(topic_embeddings)

    def _reduce_to_2d(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.shape[0] < 2:
            padded = np.vstack([embeddings, embeddings + 1e-3])
            tsne = TSNE(n_components=2, random_state=self.random_seed, perplexity=2)
            reduced = tsne.fit_transform(padded)
            return reduced[:1]

        try:
            tsne = TSNE(n_components=2, random_state=self.random_seed, perplexity=min(30, embeddings.shape[0] - 1))
            return tsne.fit_transform(embeddings)
        except Exception:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=2, random_state=self.random_seed)
            return svd.fit_transform(embeddings)
    
    def plot_hierarchical_dendrogram(self, topic_model, filename_prefix: str = "hierarchical_dendrogram") -> Dict[str, str]:
        """
        生成层次聚类树状图
        
        Args:
            topic_model: BERTopic模型
            filename_prefix: 文件名前缀
            
        Returns:
            保存的文件路径字典
        """
        logger.info("  生成层次聚类树状图...")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 12))
            
            # 获取主题嵌入并计算距离
            if hasattr(topic_model, 'topic_embeddings_') and topic_model.topic_embeddings_ is not None:
                topic_embeddings = topic_model.topic_embeddings_
                
                # 计算距离矩阵
                distance_matrix = pdist(topic_embeddings, metric='cosine')
                
                # 执行层次聚类
                linkage_matrix = linkage(distance_matrix, method='ward')
                
                # 获取主题标签
                topic_info = topic_model.get_topic_info()
                topic_info = topic_info[topic_info.Topic != -1]
                
                topic_labels = []
                for i in range(len(topic_embeddings)):
                    if i < len(topic_info):
                        words = topic_info.iloc[i]['Representation']
                        if isinstance(words, list):
                            label = f"主题{i}: {', '.join(words[:2])}"
                        else:
                            label = f"主题{i}: {str(words)[:15]}"
                        topic_labels.append(label)
                    else:
                        topic_labels.append(f"主题{i}")
                
                # 绘制树状图（横向）
                dendrogram(
                    linkage_matrix,
                    labels=topic_labels,
                    ax=ax,
                    orientation='left',
                    leaf_rotation=0,
                    leaf_font_size=10,
                    color_threshold=0.7 * max(linkage_matrix[:, 2])
                )
                
                ax.set_title('主题层次聚类树状图', fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('距离', fontsize=12)
                ax.set_ylabel('主题', fontsize=12)
                ax.grid(True, alpha=0.3, axis='x')
                
            else:
                ax.text(0.5, 0.5, '无可用的主题嵌入数据', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
            
            # 添加自定义标注
            self._add_annotations(ax)
            
            plt.tight_layout()
            saved_files = self._save_chart(fig, filename_prefix)
            plt.close(fig)
            
            return saved_files
            
        except Exception as e:
            logger.error(f"层次聚类图生成失败: {e}")
            plt.close('all')
            return {}
    
    def plot_topics_over_time(self, topic_model, documents, timestamps=None, filename_prefix: str = "topics_over_time") -> Dict[str, str]:
        """
        生成主题随时间演化图
        
        Args:
            topic_model: BERTopic模型
            documents: 文档列表
            timestamps: 时间戳列表
            filename_prefix: 文件名前缀
            
        Returns:
            保存的文件路径字典
        """
        logger.info("  生成主题演化趋势图...")
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if timestamps is not None and len(timestamps) == len(documents):
                # 使用提供的时间戳
                try:
                    topics_over_time = topic_model.topics_over_time(documents, timestamps)
                    
                    # 获取前几个主要主题
                    top_topics = topic_model.get_topic_info()
                    top_topics = top_topics[top_topics.Topic != -1].head(5)
                    
                    # 为每个主题绘制趋势线
                    for topic_id in top_topics['Topic']:
                        topic_data = topics_over_time[topics_over_time.Topic == topic_id]
                        
                        if not topic_data.empty:
                            ax.plot(
                                topic_data['Timestamp'], 
                                topic_data['Frequency'],
                                marker='o',
                                markersize=6,
                                linewidth=2,
                                label=f"主题{topic_id}",
                                alpha=0.8
                            )
                    
                    ax.set_title('主题随时间演化趋势', fontsize=16, fontweight='bold', pad=20)
                    ax.set_xlabel('时间', fontsize=12)
                    ax.set_ylabel('主题频率', fontsize=12)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                    
                    # 旋转x轴标签
                    plt.xticks(rotation=45)
                    
                except Exception as e:
                    logger.warning(f"时间序列分析失败: {e}")
                    ax.text(0.5, 0.5, '时间序列数据处理失败', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
            else:
                ax.text(0.5, 0.5, '缺少时间戳数据', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
            
            # 添加自定义标注
            self._add_annotations(ax)
            
            plt.tight_layout()
            saved_files = self._save_chart(fig, filename_prefix)
            plt.close(fig)
            
            return saved_files
            
        except Exception as e:
            logger.error(f"主题演化图生成失败: {e}")
            plt.close('all')
            return {}
    
    def plot_topic_sizes(self, topic_model, filename_prefix: str = "topic_sizes") -> Dict[str, str]:
        """
        生成主题规模条形图
        
        Args:
            topic_model: BERTopic模型
            filename_prefix: 文件名前缀
            
        Returns:
            保存的文件路径字典
        """
        logger.info("  生成主题规模条形图...")
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 获取主题信息
            topic_info = topic_model.get_topic_info()
            topic_info = topic_info[topic_info.Topic != -1]  # 排除噪声主题
            
            if not topic_info.empty:
                # 选择前15个主题
                top_topics = topic_info.head(15)
                
                # 创建主题标签
                topic_labels = []
                for _, row in top_topics.iterrows():
                    words = row['Representation']
                    if isinstance(words, list):
                        label = ', '.join(words[:3])
                    else:
                        label = str(words)[:20]
                    topic_labels.append(f"主题{row['Topic']}: {label}")
                
                # 创建条形图
                bars = ax.barh(
                    range(len(top_topics)), 
                    top_topics['Count'],
                    color=plt.cm.viridis(np.linspace(0, 1, len(top_topics))),
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5
                )
                
                # 设置y轴标签
                ax.set_yticks(range(len(top_topics)))
                ax.set_yticklabels(topic_labels, fontsize=10)
                
                # 在条形末端添加数值标签
                for i, (bar, count) in enumerate(zip(bars, top_topics['Count'])):
                    ax.text(
                        bar.get_width() + max(top_topics['Count']) * 0.01,
                        bar.get_y() + bar.get_height()/2,
                        str(count),
                        va='center',
                        fontsize=10
                    )
                
                ax.set_title('主题规模分布', fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('文档数量', fontsize=12)
                ax.grid(True, alpha=0.3, axis='x')
                
                # 反转y轴使最大的主题在顶部
                ax.invert_yaxis()
                
            else:
                ax.text(0.5, 0.5, '无主题数据可显示', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
            
            # 添加自定义标注
            self._add_annotations(ax)
            
            plt.tight_layout()
            saved_files = self._save_chart(fig, filename_prefix)
            plt.close(fig)
            
            return saved_files
            
        except Exception as e:
            logger.error(f"主题规模图生成失败: {e}")
            plt.close('all')
            return {}
    
    def plot_topic_similarity_heatmap(self, topic_model, filename_prefix: str = "topic_similarity_heatmap") -> Dict[str, str]:
        """
        生成主题相似度热力图
        
        Args:
            topic_model: BERTopic模型
            filename_prefix: 文件名前缀
            
        Returns:
            保存的文件路径字典
        """
        logger.info("  生成主题相似度热力图...")
        
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # 获取主题嵌入
            if hasattr(topic_model, 'topic_embeddings_') and topic_model.topic_embeddings_ is not None:
                topic_embeddings = topic_model.topic_embeddings_
                
                # 计算相似度矩阵
                similarity_matrix = cosine_similarity(topic_embeddings)
                
                # 获取主题标签
                topic_info = topic_model.get_topic_info()
                topic_info = topic_info[topic_info.Topic != -1]
                
                topic_labels = []
                for i in range(min(len(topic_embeddings), len(topic_info))):
                    words = topic_info.iloc[i]['Representation']
                    if isinstance(words, list):
                        label = f"主题{i}: {words[0]}"
                    else:
                        label = f"主题{i}: {str(words)[:10]}"
                    topic_labels.append(label)
                
                # 如果主题太多，只显示前20个
                if len(similarity_matrix) > 20:
                    similarity_matrix = similarity_matrix[:20, :20]
                    topic_labels = topic_labels[:20]
                
                # 创建热力图
                sns.heatmap(
                    similarity_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlBu_r',
                    center=0,
                    square=True,
                    xticklabels=topic_labels,
                    yticklabels=topic_labels,
                    cbar_kws={'label': '相似度'},
                    ax=ax
                )
                
                ax.set_title('主题相似度矩阵', fontsize=16, fontweight='bold', pad=20)
                
                # 旋转标签
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                
            else:
                ax.text(0.5, 0.5, '无可用的主题嵌入数据', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
            
            # 添加自定义标注
            self._add_annotations(ax)
            
            plt.tight_layout()
            saved_files = self._save_chart(fig, filename_prefix)
            plt.close(fig)
            
            return saved_files
            
        except Exception as e:
            logger.error(f"相似度热力图生成失败: {e}")
            plt.close('all')
            return {}
    
    def plot_topic_word_scores(self, topic_model, max_topics: int = 20, top_n_words: int = 8, 
                               filename_prefix: str = "topic_word_scores") -> Dict[str, str]:
        """
        生成主题关键词得分多子图（类似期望图片中的Topic Word Scores）
        
        Args:
            topic_model: BERTopic模型
            max_topics: 最多显示多少个主题
            top_n_words: 每个主题显示多少个关键词
            filename_prefix: 文件名前缀
            
        Returns:
            保存的文件路径字典
        """
        logger.info("  生成主题关键词得分多子图...")
        
        try:
            # 获取主题信息
            topic_info = topic_model.get_topic_info()
            topic_info = topic_info[topic_info.Topic != -1]  # 排除噪声主题
            
            if topic_info.empty:
                logger.warning("  ⚠ 无有效主题，跳过生成")
                return {}
            
            # 限制主题数量
            n_topics = min(len(topic_info), max_topics)
            
            # 计算网格布局：尽量接近正方形
            n_cols = int(np.ceil(np.sqrt(n_topics)))
            n_rows = int(np.ceil(n_topics / n_cols))
            
            # 创建子图
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
            fig.suptitle('Topic Word Scores', fontsize=18, fontweight='bold', y=0.995)
            
            # 确保axes是2D数组
            if n_topics == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            # 为每个主题生成子图
            for idx in range(n_topics):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col]
                
                topic_id = topic_info.iloc[idx]['Topic']
                
                # 获取该主题的关键词和得分
                topic_words = topic_model.get_topic(topic_id)
                
                if topic_words and len(topic_words) > 0:
                    # 取前N个词
                    words = [word for word, _ in topic_words[:top_n_words]]
                    scores = [score for _, score in topic_words[:top_n_words]]
                    
                    # 反转顺序，使得最高分在顶部
                    words = words[::-1]
                    scores = scores[::-1]
                    
                    # 创建条形图
                    colors = plt.cm.Set3(np.linspace(0, 1, len(words)))
                    bars = ax.barh(range(len(words)), scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                    
                    # 设置标签
                    ax.set_yticks(range(len(words)))
                    ax.set_yticklabels(words, fontsize=9)
                    ax.set_xlabel('c-TF-IDF Score', fontsize=8)
                    ax.set_title(f'Topic {topic_id}', fontsize=10, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # 设置x轴范围
                    if scores:
                        ax.set_xlim(0, max(scores) * 1.1)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            
            # 隐藏多余的子图
            for idx in range(n_topics, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            saved_files = self._save_chart(fig, filename_prefix)
            plt.close(fig)
            
            return saved_files
            
        except Exception as e:
            logger.error(f"主题关键词得分图生成失败: {e}")
            plt.close('all')
            return {}
    
    def plot_ctfidf_decay(self, topic_model, max_topics: int = 50, max_words: int = 10,
                          filename_prefix: str = "ctfidf_decay") -> Dict[str, str]:
        """
        生成c-TF-IDF权重衰减曲线图（类似期望图片中的特征词权重下降趋势图）
        
        Args:
            topic_model: BERTopic模型
            max_topics: 最多显示多少个主题的曲线
            max_words: 每个主题显示多少个词的权重
            filename_prefix: 文件名前缀
            
        Returns:
            保存的文件路径字典
        """
        logger.info("  生成c-TF-IDF权重衰减曲线图...")
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 获取主题信息
            topic_info = topic_model.get_topic_info()
            topic_info = topic_info[topic_info.Topic != -1]  # 排除噪声主题
            
            if topic_info.empty:
                logger.warning("  ⚠ 无有效主题，跳过生成")
                return {}
            
            # 限制主题数量
            n_topics = min(len(topic_info), max_topics)
            
            # 为每个主题绘制衰减曲线
            colors = plt.cm.tab20(np.linspace(0, 1, n_topics))
            
            for idx in range(n_topics):
                topic_id = topic_info.iloc[idx]['Topic']
                topic_words = topic_model.get_topic(topic_id)
                
                if topic_words and len(topic_words) > 0:
                    # 提取得分
                    scores = [score for _, score in topic_words[:max_words]]
                    
                    # 绘制曲线（半透明，避免过于杂乱）
                    ax.plot(range(1, len(scores) + 1), scores, 
                           color=colors[idx], alpha=0.3, linewidth=1.5)
            
            # 标注典型主题（根据主题数量动态选择）
            # 如果主题少于10个，标注所有；如果多，标注代表性主题
            if n_topics <= 10:
                # 少量主题：全部标注
                labeled_indices = list(range(n_topics))
            else:
                # 大量主题：标注首、中、末各几个
                labeled_indices = [0, 1, n_topics//4, n_topics//2, 3*n_topics//4, n_topics-2, n_topics-1]
                labeled_indices = sorted(set([i for i in labeled_indices if i < n_topics]))
            
            for idx in labeled_indices:
                topic_id = topic_info.iloc[idx]['Topic']
                topic_words = topic_model.get_topic(topic_id)
                if topic_words:
                    scores = [score for _, score in topic_words[:max_words]]
                    ax.plot(range(1, len(scores) + 1), scores,
                           color=colors[idx], alpha=1.0, linewidth=2.5,
                           label=f'Topic {topic_id}')
            
            ax.set_title('c-TF-IDF权重衰减趋势图', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Term Rank', fontsize=12)
            ax.set_ylabel('c-TF-IDF Score', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=10)
            
            # 添加自定义标注
            self._add_annotations(ax)
            
            plt.tight_layout()
            saved_files = self._save_chart(fig, filename_prefix)
            plt.close(fig)
            
            return saved_files
            
        except Exception as e:
            logger.error(f"c-TF-IDF衰减图生成失败: {e}")
            plt.close('all')
            return {}
    
    def plot_topic_impact_frequency(self, topic_model, filename_prefix: str = "topic_impact_frequency") -> Dict[str, str]:
        """
        生成主题影响力-频率分析图（TRIFA四象限图）
        
        Args:
            topic_model: BERTopic模型
            filename_prefix: 文件名前缀
            
        Returns:
            保存的文件路径字典
        """
        logger.info("  生成主题影响力-频率分析图...")
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 获取主题信息
            topic_info = topic_model.get_topic_info()
            topic_info = topic_info[topic_info.Topic != -1]  # 排除噪声主题
            
            if topic_info.empty:
                logger.warning("  ⚠ 无有效主题，跳过生成")
                return {}
            
            # 计算频率（文档数量，取对数）
            frequencies = topic_info['Count'].values
            log_frequencies = np.log10(frequencies + 1)  # +1避免log(0)
            
            # 计算影响力（c-TF-IDF平均分）
            impacts = []
            for topic_id in topic_info['Topic']:
                topic_words = topic_model.get_topic(topic_id)
                if topic_words and len(topic_words) > 0:
                    # 取前5个词的平均得分作为影响力
                    avg_score = np.mean([score for _, score in topic_words[:5]])
                    impacts.append(avg_score)
                else:
                    impacts.append(0)
            
            impacts = np.array(impacts)
            
            # 标准化影响力到[-1, 1]范围（使其居中在0）
            if len(impacts) > 1:
                impacts = (impacts - impacts.mean()) / (impacts.std() + 1e-10)
            
            # 创建散点图
            scatter = ax.scatter(
                log_frequencies,
                impacts,
                s=frequencies * 2,  # 点的大小也反映频率
                c=range(len(topic_info)),
                cmap='viridis',
                alpha=0.6,
                edgecolors='black',
                linewidth=1
            )
            
            # 添加主题标签（只标注一些关键主题）
            for idx in range(min(10, len(topic_info))):
                words = topic_info.iloc[idx]['Representation']
                if isinstance(words, list):
                    label = words[0]
                else:
                    label = str(words)[:15]
                
                ax.annotate(
                    f"T{topic_info.iloc[idx]['Topic']}",
                    (log_frequencies[idx], impacts[idx]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )
            
            # 添加四象限分隔线
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(x=np.median(log_frequencies), color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            # 添加象限标注
            x_range = log_frequencies.max() - log_frequencies.min()
            y_range = impacts.max() - impacts.min()
            
            ax.text(log_frequencies.max() - 0.1 * x_range, impacts.max() - 0.1 * y_range,
                   'Quadrant II\nHigh Impact\nHigh Frequency',
                   ha='right', va='top', fontsize=10, alpha=0.5, style='italic')
            
            ax.text(log_frequencies.min() + 0.1 * x_range, impacts.max() - 0.1 * y_range,
                   'Quadrant I\nHigh Impact\nLow Frequency',
                   ha='left', va='top', fontsize=10, alpha=0.5, style='italic')
            
            ax.text(log_frequencies.min() + 0.1 * x_range, impacts.min() + 0.1 * y_range,
                   'Quadrant III\nLow Impact\nLow Frequency',
                   ha='left', va='bottom', fontsize=10, alpha=0.5, style='italic')
            
            ax.text(log_frequencies.max() - 0.1 * x_range, impacts.min() + 0.1 * y_range,
                   'Quadrant IV\nLow Impact\nHigh Frequency',
                   ha='right', va='bottom', fontsize=10, alpha=0.5, style='italic')
            
            ax.set_title('主题影响力-频率分析图 (TRIFA)', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Topic Frequency (log)', fontsize=12)
            ax.set_ylabel('Topic Rating Impact', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Topic Index', rotation=270, labelpad=15)
            
            # 添加自定义标注
            self._add_annotations(ax)
            
            plt.tight_layout()
            saved_files = self._save_chart(fig, filename_prefix)
            plt.close(fig)
            
            return saved_files
            
        except Exception as e:
            logger.error(f"主题影响力-频率分析图生成失败: {e}")
            plt.close('all')
            return {}
    
    def generate_all_visualizations(self, 
                                   topic_model,
                                   documents: List[str],
                                   topics: List[int],
                                   metadata_df: Optional[pd.DataFrame] = None,
                                   timestamps: Optional[List[datetime]] = None) -> Dict[str, str]:
        """
        生成所有可视化图表
        
        Args:
            topic_model: BERTopic模型
            documents: 文档列表
            topics: 主题列表
            metadata_df: 元数据DataFrame
            timestamps: 时间戳列表
            
        Returns:
            生成的图表文件路径字典
        """
        logger.info("🎨 开始生成所有可视化图表...")
        
        all_charts = {}
        
        # 读取用户配置的图表生成开关
        sota_charts_config = self.config.get('outputs', {}).get('sota_charts', {})
        
        # 图表生成函数列表
        chart_functions = [
            ('topic_distance_map', self.plot_topic_distance_map),
            ('hierarchical_dendrogram', self.plot_hierarchical_dendrogram),
            ('topic_sizes', self.plot_topic_sizes),
            ('topic_similarity_heatmap', self.plot_topic_similarity_heatmap),
            ('topic_word_scores', self.plot_topic_word_scores),
            ('ctfidf_decay', self.plot_ctfidf_decay),
            ('topic_impact_frequency', self.plot_topic_impact_frequency),
        ]
        
        # 根据配置过滤图表列表
        if sota_charts_config:
            chart_functions = [
                (name, func) for name, func in chart_functions
                if sota_charts_config.get(name, True)  # 默认为 True（生成）
            ]
            logger.info(f"  根据配置，将生成 {len(chart_functions)} 个图表")
        
        # 生成基础图表
        for chart_name, chart_func in tqdm(
            chart_functions,
            desc="[3/3] 生成可视化图表",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            try:
                logger.info(f"  → 生成图表: {chart_name}")
                if chart_name == 'topic_distance_map':
                    chart_files = chart_func(topic_model, filename_prefix=chart_name)
                else:
                    chart_files = chart_func(topic_model, filename_prefix=chart_name)
                
                for format_type, file_path in chart_files.items():
                    all_charts[f"{chart_name}_{format_type}"] = file_path
                logger.info(f"  ✓ 图表 {chart_name} 生成成功")
                    
            except Exception as e:
                import traceback
                logger.warning(f"  ❌ 图表 {chart_name} 生成失败: {type(e).__name__}: {e}")
                logger.debug(f"详细错误:\n{traceback.format_exc()}")
        
        # 生成时间序列图表（如果有时间数据且用户启用）
        if timestamps and sota_charts_config.get('topics_over_time', True):
            try:
                logger.info("  → 生成图表: topics_over_time")
                chart_files = self.plot_topics_over_time(
                    topic_model, documents, timestamps, 'topics_over_time'
                )
                for format_type, file_path in chart_files.items():
                    all_charts[f"topics_over_time_{format_type}"] = file_path
                logger.info("  ✓ 图表 topics_over_time 生成成功")
            except Exception as e:
                import traceback
                logger.warning(f"  ❌ 时间序列图表生成失败: {type(e).__name__}: {e}")
                logger.debug(f"详细错误:\n{traceback.format_exc()}")
        
        logger.info(f"✅ 可视化图表生成完成，共 {len(all_charts)} 个文件")
        
        return all_charts
