"""
BERTopic结果生成器
================
专门负责生成和保存分析结果
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
from bertopic import BERTopic

logger = logging.getLogger(__name__)


class ResultsGenerator:
    """分析结果生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化结果生成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.results_paths = config['results_paths']
    
    def save_model(self, topic_model: BERTopic):
        """保存模型"""
        model_path = Path(self.results_paths['model_dir'])
        model_path.mkdir(parents=True, exist_ok=True)
        
        save_path = model_path / 'bertopic_model'
        topic_model.save(str(save_path), serialization="safetensors", save_ctfidf=True)
        logger.info(f"  ✓ 模型已保存到: {save_path}")
    
    def save_topic_summary(self, topic_model: BERTopic):
        """保存主题摘要"""
        topic_info = topic_model.get_topic_info()
        
        # 添加更多有用信息
        topic_info['Top_Words'] = topic_info['Topic'].apply(
            lambda x: ', '.join([word for word, _ in topic_model.get_topic(x)[:5]])
            if x != -1 else 'Outlier'
        )
        
        # 保存为CSV
        summary_path = Path(self.results_paths['summary_file'])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        topic_info.to_csv(summary_path, index=False, encoding='utf-8-sig')
        logger.info(f"  ✓ 主题摘要已保存: {summary_path}")
    
    def generate_topic_visualization(self, topic_model: BERTopic, documents: List[str]):
        """生成主题可视化"""
        try:
            # 生成主题距离图
            fig = topic_model.visualize_topics()
            
            # 保存为HTML
            viz_path = Path(self.results_paths['viz_file'])
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(viz_path)
            logger.info(f"  ✓ 主题可视化已保存: {viz_path}")
            
        except Exception as e:
            logger.warning(f"  ⚠ 主题可视化失败: {e}")
    
    def generate_source_analysis(self, metadata_df: pd.DataFrame, topic_model: BERTopic):
        """生成来源分析图表"""
        try:
            if 'Source' not in metadata_df.columns:
                logger.warning("  ⚠ 缺少Source列，跳过来源分析")
                return
            
            # 计算每个来源的主题分布
            source_topic_counts = metadata_df.groupby(['Source', 'topic']).size().unstack(fill_value=0)
            
            # 创建热力图
            fig = go.Figure(data=go.Heatmap(
                z=source_topic_counts.values,
                x=[f"Topic {i}" for i in source_topic_counts.columns],
                y=source_topic_counts.index,
                colorscale='Blues',
                text=source_topic_counts.values,
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title='不同来源的主题分布',
                xaxis_title='主题',
                yaxis_title='来源',
                height=max(400, len(source_topic_counts) * 50)
            )
            
            # 保存
            source_path = Path(self.results_paths['source_analysis'])
            source_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(source_path)
            logger.info(f"  ✓ 来源分析已保存: {source_path}")
            
        except Exception as e:
            logger.warning(f"  ⚠ 来源分析失败: {e} (检查是否已生成 '来源主题热力图.html')")
    
    def generate_timeline_analysis(self, metadata_df: pd.DataFrame, topic_model: BERTopic, documents: List[str] = None):
        """生成时间线分析"""
        try:
            date_columns = ['日期', 'Date', 'date', 'timestamp', 'time']
            date_col = None
            
            for col in date_columns:
                if col in metadata_df.columns:
                    date_col = col
                    break
            
            if date_col is None:
                logger.warning("  ⚠ 缺少日期列，跳过时间线分析")
                return
            
            # 转换日期并计算每个时间段的主题分布
            metadata_df[date_col] = pd.to_datetime(metadata_df[date_col], errors='coerce')
            metadata_df = metadata_df.dropna(subset=[date_col])
            
            # 按月分组
            metadata_df['year_month'] = metadata_df[date_col].dt.to_period('M')
            
            # 计算主题随时间的变化 - 使用正确的文档数据
            if documents is None:
                # 如果没有传入documents，尝试从元数据中获取文本列
                text_columns = ['Unit_Text', 'text', 'content', 'document']
                doc_col = None
                for col in text_columns:
                    if col in metadata_df.columns:
                        doc_col = col
                        break
                
                if doc_col is None:
                    logger.warning("  ⚠ 缺少文档文本列，跳过时间线分析")
                    return
                
                documents_for_timeline = metadata_df[doc_col].tolist()
            else:
                documents_for_timeline = documents
            
            if len(documents_for_timeline) != len(metadata_df[date_col]):
                logger.warning("  ⚠ 文档与时间戳数量不一致，跳过时间线分析")
                return

            topics_over_time = topic_model.topics_over_time(
                documents_for_timeline,
                metadata_df[date_col],
                nr_bins=10
            )

            # 创建时间线图
            fig = topic_model.visualize_topics_over_time(topics_over_time)

            # 保存
            timeline_path = Path(self.results_paths['timeline_analysis'])
            timeline_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(timeline_path)
            logger.info(f"  ✓ 时间线分析已保存: {timeline_path}")
            
        except Exception as e:
            logger.warning(f"  ⚠ 时间线分析失败: {e}")
    
    def generate_frame_heatmap(self, metadata_df: pd.DataFrame):
        """生成框架热力图"""
        try:
            # 查找框架相关列
            frame_columns = [col for col in metadata_df.columns 
                           if col.startswith('Frame_') and col.endswith('_Present')]
            
            if not frame_columns:
                logger.warning("  ⚠ 缺少框架列，跳过框架分析")
                return
            
            # 计算每个主题的框架频率
            topic_frame_matrix = []
            topic_labels = []
            
            for topic in sorted(metadata_df['topic'].unique()):
                if topic == -1:  # 跳过噪声主题
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
            heatmap_path = Path(self.results_paths['frame_heatmap'])
            heatmap_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(heatmap_path)
            logger.info(f"  ✓ 框架热力图已保存: {heatmap_path}")
            
        except Exception as e:
            logger.warning(f"  ⚠ 框架热力图生成失败: {e}")
    
    def generate_enhanced_summary(self, topic_model: BERTopic, documents: List[str], metadata_df: Optional[pd.DataFrame] = None):
        """生成增强版主题摘要"""
        try:
            topic_info = topic_model.get_topic_info()
            
            # 添加文档数量统计
            if metadata_df is not None:
                topic_counts = metadata_df['topic'].value_counts().sort_index()
                topic_info['Document_Count'] = topic_info['Topic'].map(topic_counts).fillna(0)
            
            # 添加关键词信息
            topic_info['Top_Keywords'] = topic_info['Topic'].apply(
                lambda x: [word for word, _ in topic_model.get_topic(x)[:10]]
                if x != -1 else []
            )
            
            # 添加c-TF-IDF分数
            topic_info['Keyword_Scores'] = topic_info['Topic'].apply(
                lambda x: [score for _, score in topic_model.get_topic(x)[:10]]
                if x != -1 else []
            )
            
            # 保存增强版摘要
            enhanced_path = Path(self.results_paths['summary_enhanced'])
            enhanced_path.parent.mkdir(parents=True, exist_ok=True)
            topic_info.to_csv(enhanced_path, index=False, encoding='utf-8-sig')
            logger.info(f"  ✓ 增强版主题摘要已保存: {enhanced_path}")
            
        except Exception as e:
            logger.warning(f"  ⚠ 增强版摘要生成失败: {e}")
