"""
出版级学术图表生成模块
====================
生成符合学术论文发表标准的高质量静态图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
import logging
from typing import List, Dict, Any, Tuple, Optional
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AcademicChartGenerator:
    """出版级学术图表生成器 - SOTA & KISS实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化学术图表生成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.results_paths = config['results_paths']
        
        # 设置学术级图表样式
        self._setup_academic_style()
        
    def _setup_academic_style(self):
        """设置学术论文级的图表样式"""
        # 设置全局参数
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,  # 高分辨率
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.2,
            'patch.linewidth': 0.5,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.4,
            'ytick.minor.width': 0.4,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': False,
            'legend.frameon': False,
            'text.usetex': False,  # 不使用LaTeX以确保兼容性
        })
        
        # 学术论文配色方案
        self.academic_colors = [
            '#2E4057',  # 深蓝
            '#048A81',  # 青绿
            '#54C6EB',  # 天蓝
            '#F18F01',  # 橙色
            '#C73E1D',  # 红色
            '#7209B7',  # 紫色
            '#2D5016',  # 深绿
            '#F72585',  # 粉红
            '#4361EE',  # 蓝紫
            '#F77F00',  # 深橙
        ]
    
    def generate_topic_distribution_chart(self, 
                                        topic_model,
                                        documents: List[str],
                                        topics: List[int],
                                        enhanced_topics: Optional[Dict] = None,
                                        output_path: str = None) -> str:
        """
        生成带注释的二维主题分布图
        
        Args:
            topic_model: 训练好的BERTopic模型
            documents: 文档列表
            topics: 主题列表
            enhanced_topics: 增强的主题表示
            output_path: 输出路径
            
        Returns:
            生成的文件路径
        """
        logger.info("生成学术级主题分布图...")
        
        # 获取嵌入向量并降维到2D
        embeddings = topic_model._extract_embeddings(documents, method="document")
        
        # 使用t-SNE降维（比UMAP在2D可视化中效果更好）
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(documents)//4))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 获取主题信息
        topic_info = topic_model.get_topic_info()
        unique_topics = sorted([t for t in set(topics) if t != -1])
        
        # 绘制每个主题的点
        for i, topic_id in enumerate(unique_topics):
            # 获取该主题的文档索引
            topic_mask = np.array(topics) == topic_id
            topic_embeddings = embeddings_2d[topic_mask]
            
            if len(topic_embeddings) == 0:
                continue
                
            # 选择颜色
            color = self.academic_colors[i % len(self.academic_colors)]
            
            # 绘制散点，使用半透明
            ax.scatter(topic_embeddings[:, 0], topic_embeddings[:, 1], 
                      c=color, alpha=0.6, s=20, edgecolors='white', linewidth=0.5)
            
            # 计算主题中心点
            center_x = np.mean(topic_embeddings[:, 0])
            center_y = np.mean(topic_embeddings[:, 1])
            
            # 获取主题标签
            if enhanced_topics and topic_id in enhanced_topics:
                # 使用增强的主题表示
                top_words = [word for word, _ in enhanced_topics[topic_id][:3]]
            else:
                # 使用原始主题表示
                topic_words = topic_model.get_topic(topic_id)
                top_words = [word for word, _ in topic_words[:3]]
            
            # 创建主题标签
            topic_label = f"Topic {topic_id}: {', '.join(top_words)}"
            
            # 添加标签，使用白色背景框
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white', 
                            edgecolor=color, alpha=0.9, linewidth=1)
            
            ax.annotate(topic_label, 
                       xy=(center_x, center_y),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=bbox_props,
                       fontsize=9, ha='left',
                       arrowprops=dict(arrowstyle='->', color=color, lw=1))
        
        # 处理离群点
        outlier_mask = np.array(topics) == -1
        if np.any(outlier_mask):
            outlier_embeddings = embeddings_2d[outlier_mask]
            ax.scatter(outlier_embeddings[:, 0], outlier_embeddings[:, 1], 
                      c='lightgray', alpha=0.3, s=10, edgecolors='none')
        
        # 设置图表样式
        ax.set_title('Topic Distribution in 2D Space', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        
        # 移除坐标轴刻度
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 设置背景为纯白
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # 设置边框
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if output_path is None:
            output_path = Path(self.results_paths['output_dir']) / 'academic_topic_distribution.png'
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为PNG和PDF两种格式
        png_path = output_path.with_suffix('.png')
        pdf_path = output_path.with_suffix('.pdf')
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"  ✓ 学术级主题分布图已保存: {png_path}")
        logger.info(f"  ✓ PDF版本已保存: {pdf_path}")
        
        return str(png_path)
    
    def generate_topic_size_chart(self,
                                 topic_model,
                                 output_path: str = None) -> str:
        """
        生成主题规模分布图（南丁格尔玫瑰图风格）
        
        Args:
            topic_model: 训练好的BERTopic模型
            output_path: 输出路径
            
        Returns:
            生成的文件路径
        """
        logger.info("生成学术级主题规模分布图...")
        
        # 获取主题信息
        topic_info = topic_model.get_topic_info()
        topic_info = topic_info[topic_info['Topic'] != -1]  # 排除离群点
        
        if len(topic_info) == 0:
            logger.warning("  ⚠ 没有有效主题，跳过规模分布图")
            return ""
        
        # 准备数据
        topics = topic_info['Topic'].tolist()
        counts = topic_info['Count'].tolist()
        
        # 创建极坐标图
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # 计算角度
        theta = np.linspace(0, 2 * np.pi, len(topics), endpoint=False)
        
        # 归一化计数以用作半径
        max_count = max(counts)
        radii = [count / max_count for count in counts]
        
        # 计算条形宽度
        width = 2 * np.pi / len(topics) * 0.8
        
        # 绘制条形
        bars = ax.bar(theta, radii, width=width, alpha=0.7)
        
        # 为每个条形设置颜色
        for i, bar in enumerate(bars):
            color = self.academic_colors[i % len(self.academic_colors)]
            bar.set_facecolor(color)
            bar.set_edgecolor('white')
            bar.set_linewidth(1)
        
        # 添加标签
        for i, (angle, radius, topic_id, count) in enumerate(zip(theta, radii, topics, counts)):
            # 获取主题关键词
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                label = f"T{topic_id}\n({count})"
                
                # 调整标签位置
                label_radius = radius + 0.1
                ax.text(angle, label_radius, label, 
                       ha='center', va='center', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # 设置样式
        ax.set_title('Topic Size Distribution\n(Nightingale Rose Chart)', 
                    fontsize=14, fontweight='bold', pad=30)
        ax.set_ylim(0, 1.3)
        ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_rlabel_position(0)
        ax.grid(True, alpha=0.3)
        
        # 移除角度标签
        ax.set_thetagrids([])
        
        # 设置背景
        fig.patch.set_facecolor('white')
        
        # 保存图表
        if output_path is None:
            output_path = Path(self.results_paths['output_dir']) / 'academic_topic_sizes.png'
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为PNG和PDF
        png_path = output_path.with_suffix('.png')
        pdf_path = output_path.with_suffix('.pdf')
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"  ✓ 学术级主题规模图已保存: {png_path}")
        logger.info(f"  ✓ PDF版本已保存: {pdf_path}")
        
        return str(png_path)
    
    def generate_topic_evolution_chart(self,
                                     topics_over_time: pd.DataFrame,
                                     output_path: str = None) -> str:
        """
        生成学术级主题演化图
        
        Args:
            topics_over_time: 主题时间演化数据
            output_path: 输出路径
            
        Returns:
            生成的文件路径
        """
        logger.info("生成学术级主题演化图...")
        
        if topics_over_time.empty:
            logger.warning("  ⚠ 没有时间演化数据，跳过演化图")
            return ""
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 获取主题列表
        topics = topics_over_time['Topic'].unique()
        topics = [t for t in topics if t != -1]  # 排除离群点
        
        # 为每个主题绘制线条
        for i, topic in enumerate(topics):
            topic_data = topics_over_time[topics_over_time['Topic'] == topic]
            color = self.academic_colors[i % len(self.academic_colors)]
            
            ax.plot(topic_data['Timestamp'], topic_data['Frequency'], 
                   color=color, linewidth=2, alpha=0.8, 
                   marker='o', markersize=4, label=f'Topic {topic}')
        
        # 设置样式
        ax.set_title('Topic Evolution Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Topic Frequency', fontsize=12)
        
        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 旋转x轴标签
        plt.xticks(rotation=45)
        
        # 设置图例
        if len(topics) <= 10:  # 只有在主题数不太多时才显示图例
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # 设置背景
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if output_path is None:
            output_path = Path(self.results_paths['output_dir']) / 'academic_topic_evolution.png'
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为PNG和PDF
        png_path = output_path.with_suffix('.png')
        pdf_path = output_path.with_suffix('.pdf')
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"  ✓ 学术级主题演化图已保存: {png_path}")
        logger.info(f"  ✓ PDF版本已保存: {pdf_path}")
        
        return str(png_path)
    
    def generate_heatmap_chart(self,
                              data: pd.DataFrame,
                              title: str,
                              output_path: str = None) -> str:
        """
        生成学术级热力图
        
        Args:
            data: 热力图数据
            title: 图表标题
            output_path: 输出路径
            
        Returns:
            生成的文件路径
        """
        logger.info(f"生成学术级热力图: {title}")
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 生成热力图
        sns.heatmap(data, 
                   annot=True, 
                   fmt='.2f',
                   cmap='RdYlBu_r',
                   center=0,
                   square=False,
                   cbar_kws={'shrink': 0.8},
                   annot_kws={'fontsize': 9},
                   ax=ax)
        
        # 设置样式
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
        
        # 旋转标签
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 设置背景
        fig.patch.set_facecolor('white')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if output_path is None:
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_path = Path(self.results_paths['output_dir']) / f'academic_{safe_title.replace(" ", "_").lower()}.png'
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为PNG和PDF
        png_path = output_path.with_suffix('.png')
        pdf_path = output_path.with_suffix('.pdf')
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"  ✓ 学术级热力图已保存: {png_path}")
        logger.info(f"  ✓ PDF版本已保存: {pdf_path}")
        
        return str(png_path)
    
    def generate_all_academic_charts(self,
                                   topic_model,
                                   documents: List[str],
                                   topics: List[int],
                                   metadata_df: pd.DataFrame = None,
                                   enhanced_topics: Optional[Dict] = None) -> Dict[str, str]:
        """
        生成所有学术级图表
        
        Args:
            topic_model: 训练好的BERTopic模型
            documents: 文档列表
            topics: 主题列表
            metadata_df: 元数据
            enhanced_topics: 增强的主题表示
            
        Returns:
            生成的文件路径字典
        """
        logger.info("🎨 生成所有学术级图表...")
        
        generated_charts = {}
        
        # 1. 主题分布图
        try:
            chart_path = self.generate_topic_distribution_chart(
                topic_model, documents, topics, enhanced_topics
            )
            generated_charts['topic_distribution'] = chart_path
        except Exception as e:
            logger.error(f"  ✗ 主题分布图生成失败: {e}")
        
        # 2. 主题规模图
        try:
            chart_path = self.generate_topic_size_chart(topic_model)
            generated_charts['topic_sizes'] = chart_path
        except Exception as e:
            logger.error(f"  ✗ 主题规模图生成失败: {e}")
        
        # 3. 如果有时间数据，生成演化图
        if metadata_df is not None and '日期' in metadata_df.columns:
            try:
                # 这里需要先计算topics_over_time数据
                # 简化实现，实际应该调用BERTopic的topics_over_time方法
                logger.info("  → 主题演化图需要在主流程中生成")
            except Exception as e:
                logger.error(f"  ✗ 主题演化图生成失败: {e}")
        
        logger.info(f"✅ 学术级图表生成完成，共生成 {len(generated_charts)} 个图表")
        return generated_charts
