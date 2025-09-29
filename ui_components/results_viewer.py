"""
结果查看组件 - 简洁展示
=====================
高效的结果展示和下载
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
import pickle
import numpy as np

from tools.config_translator import ConfigTranslator

CONFIG = ConfigTranslator("config.yaml").translate_to_technical_config()


class ResultsViewer:
    """结果查看器"""
    
    def render_results_interface(self):
        """渲染结果界面"""
        if not self._has_results():
            st.info("⏳ 分析完成后，结果将显示在这里")
            return
        
        results = st.session_state.analysis_results
        
        st.markdown("#### 📊 分析结果")
        
        # 结果摘要
        self._show_summary(results)
        
        # 可视化结果
        self._show_visualizations()
        
        # 详细结果
        self._show_details(results)
    
    def _has_results(self) -> bool:
        """检查是否有结果"""
        return (
            hasattr(st.session_state, 'analysis_results') and 
            st.session_state.analysis_results is not None
        )
    
    def _show_summary(self, results: Dict[str, Any]):
        """显示结果摘要"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "发现主题", 
                results.get('n_topics', 0),
                help="识别出的主要主题数量"
            )
        
        with col2:
            st.metric(
                "分析文档", 
                results.get('n_documents', 0),
                help="参与分析的文档总数"
            )
        
        with col3:
            preset = results.get('preset_used', '未知')
            st.metric(
                "使用模式", 
                preset,
                help="使用的分析模式"
            )
    
    def _show_details(self, results: Dict[str, Any]):
        """显示详细结果"""
        st.markdown("---")
        
        # 分析信息
        with st.expander("📋 分析详情", expanded=True):
            timestamp = results.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                    st.write(f"**完成时间:** {formatted_time}")
                except:
                    st.write(f"**完成时间:** {timestamp}")
            
            st.write(f"**主题数量:** {results.get('n_topics', 0)}")
            st.write(f"**文档数量:** {results.get('n_documents', 0)}")
            st.write(f"**分析模式:** {results.get('preset_used', '未知')}")
        
        # 下载选项
        with st.expander("📥 导出结果", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 下载分析报告", use_container_width=True):
                    self._download_report(results)
            
            with col2:
                if st.button("📊 下载详细数据", use_container_width=True):
                    self._download_data(results)
        
        # 成功提示
        st.success("🎉 分析完成！您可以导出结果或开始新的分析。")
    
    def _download_report(self, results: Dict[str, Any]):
        """下载分析报告"""
        report = self._generate_report(results)
        
        st.download_button(
            label="📄 下载报告",
            data=report,
            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    def _download_data(self, results: Dict[str, Any]):
        """下载详细数据"""
        import json
        
        data = json.dumps(results, ensure_ascii=False, indent=2)
        
        st.download_button(
            label="📊 下载数据",
            data=data,
            file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def _generate_report(self, results: Dict[str, Any]) -> str:
        """生成分析报告"""
        timestamp = results.get('timestamp', datetime.now().isoformat())
        
        report = f"""
BERTopic 主题分析报告
==================

分析时间: {timestamp}
分析模式: {results.get('preset_used', '未知')}

结果摘要:
--------
- 识别主题数: {results.get('n_topics', 0)}
- 分析文档数: {results.get('n_documents', 0)}

分析说明:
--------
本次分析使用BERTopic算法进行主题建模，通过BERT嵌入和UMAP降维技术
识别文本中的潜在主题结构。

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report
    
    def _show_visualizations(self):
        """显示可视化结果"""
        st.markdown("---")
        st.markdown("#### 📈 可视化分析")
        
        # 检查是否有必要的文件
        results_dir = Path(CONFIG['results_paths']['output_dir'])
        model_dir = Path(CONFIG['results_paths']['model_dir'])
        mapping_path = Path(CONFIG['results_paths']['document_topic_mapping'])
        model_file = model_dir / "bertopic_model.pkl"
        
        if not mapping_path.exists():
            st.warning("⚠️ 未找到文档映射文件(文档主题分布表.csv)，无法生成可视化")
            return
        
        try:
            # 加载数据
            df_mapping = pd.read_csv(mapping_path)
            
            # 可视化选项
            viz_tabs = st.tabs(["📊 主题分布", "📈 主题大小", "🎯 Outlier分析", "📝 关键词云"])
            
            with viz_tabs[0]:
                self._plot_topic_distribution(df_mapping)
            
            with viz_tabs[1]:
                self._plot_topic_sizes(df_mapping)
            
            with viz_tabs[2]:
                self._plot_outlier_analysis(df_mapping)
            
            with viz_tabs[3]:
                self._plot_keyword_analysis(model_file)
                
        except Exception as e:
            st.error(f"可视化生成失败: {e}")
    
    def _plot_topic_distribution(self, df_mapping: pd.DataFrame):
        """绘制主题分布图"""
        st.markdown("##### 📊 主题文档分布")
        
        # 统计每个主题的文档数
        topic_counts = df_mapping['topic'].value_counts().sort_index()
        
        # 分离Outlier和正常主题
        outlier_count = topic_counts.get(-1, 0)
        normal_topics = topic_counts[topic_counts.index != -1]
        
        # 创建柱状图
        fig = go.Figure()
        
        # 正常主题
        fig.add_trace(go.Bar(
            x=[f"主题 {i}" for i in normal_topics.index],
            y=normal_topics.values,
            name="正常主题",
            marker_color='lightblue',
            text=normal_topics.values,
            textposition='auto'
        ))
        
        # Outlier
        if outlier_count > 0:
            fig.add_trace(go.Bar(
                x=["Outlier"],
                y=[outlier_count],
                name="异常值",
                marker_color='red',
                text=[outlier_count],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="主题文档数量分布",
            xaxis_title="主题",
            yaxis_title="文档数量",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 统计信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总主题数", len(normal_topics))
        with col2:
            st.metric("Outlier数量", outlier_count)
        with col3:
            outlier_pct = (outlier_count / len(df_mapping)) * 100 if len(df_mapping) > 0 else 0
            st.metric("Outlier比例", f"{outlier_pct:.1f}%")
    
    def _plot_topic_sizes(self, df_mapping: pd.DataFrame):
        """绘制主题大小分析"""
        st.markdown("##### 📈 主题大小分析")
        
        topic_counts = df_mapping['topic'].value_counts()
        normal_topics = topic_counts[topic_counts.index != -1].sort_values(ascending=False)
        
        # 主题大小分布
        fig = px.histogram(
            x=normal_topics.values,
            nbins=10,
            title="主题大小分布直方图",
            labels={'x': '主题大小（文档数）', 'y': '主题数量'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 主题大小统计
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最大主题", normal_topics.max() if len(normal_topics) > 0 else 0)
        with col2:
            st.metric("最小主题", normal_topics.min() if len(normal_topics) > 0 else 0)
        with col3:
            st.metric("平均大小", f"{normal_topics.mean():.1f}" if len(normal_topics) > 0 else "0")
        with col4:
            st.metric("中位数", f"{normal_topics.median():.1f}" if len(normal_topics) > 0 else "0")
        
        # 前10大主题
        if len(normal_topics) > 0:
            st.markdown("**📋 主题排行榜（前10）**")
            top_topics = normal_topics.head(10)
            for i, (topic_id, count) in enumerate(top_topics.items(), 1):
                st.write(f"{i}. 主题 {topic_id}: {count} 个文档")
    
    def _plot_outlier_analysis(self, df_mapping: pd.DataFrame):
        """绘制Outlier分析"""
        st.markdown("##### 🎯 异常值分析")
        
        total_docs = len(df_mapping)
        outlier_docs = len(df_mapping[df_mapping['topic'] == -1])
        normal_docs = total_docs - outlier_docs
        
        # 饼图
        fig = go.Figure(data=[go.Pie(
            labels=['正常文档', '异常文档'],
            values=[normal_docs, outlier_docs],
            marker_colors=['lightgreen', 'lightcoral'],
            textinfo='label+percent+value'
        )])
        
        fig.update_layout(title="文档分类分布")
        st.plotly_chart(fig, use_container_width=True)
        
        # Outlier质量评估
        outlier_pct = (outlier_docs / total_docs) * 100 if total_docs > 0 else 0
        
        if outlier_pct < 5:
            st.success(f"✅ Outlier比例 {outlier_pct:.1f}% - 优秀")
        elif outlier_pct < 10:
            st.info(f"ℹ️ Outlier比例 {outlier_pct:.1f}% - 良好")
        elif outlier_pct < 15:
            st.warning(f"⚠️ Outlier比例 {outlier_pct:.1f}% - 需要优化")
        else:
            st.error(f"❌ Outlier比例 {outlier_pct:.1f}% - 严重问题，建议调参")
        
        # 调参建议
        if outlier_pct > 10:
            st.markdown("**🛠 调参建议:**")
            st.write("- 减小 `min_topic_size` 参数")
            st.write("- 降低 `min_samples` 参数") 
            st.write("- 增加 `n_neighbors` 参数")
    
    def _plot_keyword_analysis(self, model_file: Path):
        """绘制关键词分析"""
        st.markdown("##### 📝 主题关键词分析")
        
        if not model_file.exists():
            st.warning("⚠️ 未找到训练模型，无法显示关键词分析")
            return
        
        try:
            # 加载模型
            with open(model_file, 'rb') as f:
                topic_model = pickle.load(f)
            
            # 获取主题信息
            topic_info = topic_model.get_topic_info()
            
            # 显示前几个主题的关键词
            st.markdown("**🔑 主要主题关键词:**")
            
            for idx, row in topic_info.head(8).iterrows():
                if row['Topic'] != -1:  # 排除Outlier
                    topic_id = row['Topic']
                    count = row['Count']
                    
                    # 获取关键词
                    keywords = topic_model.get_topic(topic_id)
                    if keywords:
                        keyword_text = ", ".join([word for word, _ in keywords[:8]])
                        
                        with st.expander(f"主题 {topic_id} ({count} 文档)", expanded=False):
                            st.write(f"**关键词:** {keyword_text}")
                            
                            # 关键词权重图
                            if len(keywords) > 0:
                                words, scores = zip(*keywords[:8])
                                fig = go.Figure(data=[go.Bar(
                                    x=list(scores),
                                    y=list(words),
                                    orientation='h',
                                    marker_color='skyblue'
                                )])
                                fig.update_layout(
                                    title=f"主题 {topic_id} 关键词权重",
                                    xaxis_title="权重",
                                    yaxis_title="关键词",
                                    height=300
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
        except Exception as e:
            st.error(f"关键词分析失败: {e}")
            st.write("请确保模型文件完整且格式正确")
