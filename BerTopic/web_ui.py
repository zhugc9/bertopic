"""
交互式Web应用界面
================
使用Streamlit创建用户友好的图形界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import logging
import tempfile
import zipfile
from pathlib import Path
from io import StringIO, BytesIO
import sys
import subprocess
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# 确保能找到项目模块
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from topic_analyzer.data_loader import DataLoader
from topic_analyzer.model import TopicAnalyzer

# 配置页面
st.set_page_config(
    page_title="BERTopic 议题分析系统",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .module-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class BERTopicWebUI:
    """BERTopic Web界面类"""
    
    def __init__(self):
        """初始化Web界面"""
        self.config = self._load_default_config()
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = {}
    
    def _load_default_config(self) -> dict:
        """加载默认配置"""
        default_config = {
            'data_paths': {
                'media_data': 'temp/media_data.xlsx',
                'social_media_data': 'temp/social_media_data.xlsx'
            },
            'results_paths': {
                'output_dir': 'temp/results',
                'model_dir': 'temp/results/trained_model',
                'summary_file': 'temp/results/topics_summary.csv',
                'viz_file': 'temp/results/topic_visualization.html',
                'source_analysis': 'temp/results/topic_by_source.html',
                'timeline_analysis': 'temp/results/topics_over_time.html',
                'frame_heatmap': 'temp/results/topic_frame_heatmap.html',
                'academic_charts': {
                    'topic_distribution': 'temp/results/academic_topic_distribution.png',
                    'topic_sizes': 'temp/results/academic_topic_sizes.png',
                    'topic_evolution': 'temp/results/academic_topic_evolution.png',
                    'cross_lingual': 'temp/results/academic_cross_lingual.png'
                }
            },
            'data_processing': {
                'text_column': 'Incident',
                'merge_strategy': 'concat',
                'metadata_columns': ['Source', '日期', 'speaker', 'Valence']
            },
            'bertopic_params': {
                'language': 'multilingual',
                'embedding_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'min_topic_size': 15,
                'nr_topics': 'auto',
                'n_gram_range': [1, 3],
                'expert_keyword_extraction': {
                    'enable_pos_patterns': True,
                    'pos_patterns': {
                        'zh': '<n.*|a.*>*<n.*>+',
                        'en': '<JJ.*>*<NN.*>+',
                        'ru': '<A.*>*<N.*>+'
                    },
                    'custom_stopwords_path': 'stopwords/politics_stopwords.txt',
                    'use_custom_stopwords': True,
                    'pos_language_detection': True
                },
                'umap_params': {
                    'n_neighbors': 15,
                    'n_components': 5,
                    'min_dist': 0.0,
                    'metric': 'cosine',
                    'random_state': 42
                },
                'hdbscan_params': {
                    'min_cluster_size': 15,
                    'min_samples': 5,
                    'metric': 'euclidean',
                    'cluster_selection_method': 'eom',
                    'prediction_data': True
                }
            },
            'visualization': {
                'figsize': [12, 8],
                'dpi': 100,
                'style': 'plotly',
                'color_scheme': 'viridis',
                'save_format': 'html'
            },
            'analysis': {
                'time_analysis': {
                    'enable': True,
                    'time_column': '日期',
                    'bins': 10
                },
                'source_analysis': {
                    'enable': True,
                    'source_column': 'Source'
                },
                'frame_analysis': {
                    'enable': True,
                    'threshold': 0.1
                }
            },
            'system': {
                'random_seed': 42,
                'verbose': True,
                'save_intermediate': False,
                'use_gpu': False
            }
        }
        return default_config
    
    def render_main_page(self):
        """渲染主页面"""
        # 主标题
        st.markdown('<div class="main-header">🚀 BERTopic 议题分析系统</div>', unsafe_allow_html=True)
        
        # 侧边栏
        self._render_sidebar()
        
        # 主内容区域
        tab1, tab2, tab3, tab4 = st.tabs(["📁 数据上传", "⚙️ 参数配置", "🚀 运行分析", "📊 结果查看"])
        
        with tab1:
            self._render_data_upload()
        
        with tab2:
            self._render_parameter_config()
        
        with tab3:
            self._render_analysis_runner()
        
        with tab4:
            self._render_results_viewer()
    
    def _render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.markdown("### 🎯 快速导航")
        
        # 系统状态
        st.sidebar.markdown("#### 📊 系统状态")
        
        # 检查数据文件状态
        media_uploaded = 'media_data' in st.session_state.uploaded_files
        social_uploaded = 'social_media_data' in st.session_state.uploaded_files
        
        st.sidebar.write(f"媒体数据: {'✅' if media_uploaded else '❌'}")
        st.sidebar.write(f"社交媒体数据: {'✅' if social_uploaded else '❌'}")
        st.sidebar.write(f"分析结果: {'✅' if st.session_state.analysis_results else '❌'}")
        
        # 快速操作
        st.sidebar.markdown("#### ⚡ 快速操作")
        
        if st.sidebar.button("🔄 重置所有数据"):
            st.session_state.uploaded_files = {}
            st.session_state.analysis_results = None
            st.success("数据已重置")
        
        if st.sidebar.button("📥 下载配置模板"):
            self._download_config_template()
        
        # 帮助信息
        st.sidebar.markdown("#### ❓ 帮助信息")
        st.sidebar.info("""
        **使用步骤:**
        1. 上传Excel数据文件
        2. 调整分析参数
        3. 运行分析
        4. 查看和下载结果
        
        **支持的文件格式:**
        - Excel (.xlsx)
        - 需要包含文本列
        """)
    
    def _render_data_upload(self):
        """渲染数据上传界面"""
        st.markdown('<div class="module-header">📁 数据文件上传</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("媒体数据文件")
            media_file = st.file_uploader(
                "上传媒体数据Excel文件",
                type=['xlsx'],
                key="media_uploader",
                help="请上传包含媒体文本数据的Excel文件"
            )
            
            if media_file is not None:
                # 保存文件到临时目录
                temp_path = self._save_uploaded_file(media_file, "media_data")
                st.session_state.uploaded_files['media_data'] = temp_path
                
                # 预览数据
                try:
                    df = pd.read_excel(temp_path)
                    st.success(f"✅ 文件上传成功! 包含 {len(df)} 行数据")
                    
                    with st.expander("📋 数据预览"):
                        st.dataframe(df.head())
                        st.write(f"**列名:** {', '.join(df.columns.tolist())}")
                    
                except Exception as e:
                    st.error(f"❌ 文件读取失败: {e}")
        
        with col2:
            st.subheader("社交媒体数据文件")
            social_file = st.file_uploader(
                "上传社交媒体数据Excel文件",
                type=['xlsx'],
                key="social_uploader",
                help="请上传包含社交媒体文本数据的Excel文件"
            )
            
            if social_file is not None:
                # 保存文件到临时目录
                temp_path = self._save_uploaded_file(social_file, "social_media_data")
                st.session_state.uploaded_files['social_media_data'] = temp_path
                
                # 预览数据
                try:
                    df = pd.read_excel(temp_path)
                    st.success(f"✅ 文件上传成功! 包含 {len(df)} 行数据")
                    
                    with st.expander("📋 数据预览"):
                        st.dataframe(df.head())
                        st.write(f"**列名:** {', '.join(df.columns.tolist())}")
                    
                except Exception as e:
                    st.error(f"❌ 文件读取失败: {e}")
        
        # 数据验证
        if len(st.session_state.uploaded_files) > 0:
            st.markdown("---")
            st.subheader("🔍 数据验证")
            self._validate_uploaded_data()
    
    def _render_parameter_config(self):
        """渲染参数配置界面"""
        st.markdown('<div class="module-header">⚙️ 分析参数配置</div>', unsafe_allow_html=True)
        
        # 基础参数配置
        st.subheader("📊 基础参数")
        col1, col2 = st.columns(2)
        
        with col1:
            self.config['data_processing']['text_column'] = st.selectbox(
                "文本列名",
                options=['Incident', 'Unit_Text', 'Text', 'Content'],
                index=0,
                help="选择包含要分析文本的列名"
            )
            
            self.config['bertopic_params']['min_topic_size'] = st.slider(
                "最小主题大小",
                min_value=5,
                max_value=100,
                value=15,
                help="较小值产生更多细粒度主题，较大值产生更少粗粒度主题"
            )
        
        with col2:
            nr_topics_auto = st.checkbox("自动确定主题数量", value=True)
            if nr_topics_auto:
                self.config['bertopic_params']['nr_topics'] = 'auto'
            else:
                self.config['bertopic_params']['nr_topics'] = st.slider(
                    "目标主题数量",
                    min_value=2,
                    max_value=50,
                    value=20
                )
            
            self.config['bertopic_params']['n_gram_range'] = [
                st.selectbox("N-gram 最小值", [1, 2], index=0),
                st.selectbox("N-gram 最大值", [2, 3, 4], index=1)
            ]
        
        # 高级参数配置
        st.markdown("---")
        st.subheader("🔧 高级参数")
        
        with st.expander("🎯 专家级关键词提取"):
            col1, col2 = st.columns(2)
            with col1:
                self.config['bertopic_params']['expert_keyword_extraction']['enable_pos_patterns'] = st.checkbox(
                    "启用词性标注模式", value=True
                )
                self.config['bertopic_params']['expert_keyword_extraction']['use_custom_stopwords'] = st.checkbox(
                    "使用自定义停用词", value=True
                )
            with col2:
                self.config['bertopic_params']['expert_keyword_extraction']['pos_language_detection'] = st.checkbox(
                    "自动语言检测", value=True
                )
        
        with st.expander("📈 UMAP降维参数"):
            col1, col2 = st.columns(2)
            with col1:
                self.config['bertopic_params']['umap_params']['n_neighbors'] = st.slider(
                    "邻居数量", 5, 50, 15
                )
                self.config['bertopic_params']['umap_params']['n_components'] = st.slider(
                    "降维维度", 2, 10, 5
                )
            with col2:
                self.config['bertopic_params']['umap_params']['min_dist'] = st.slider(
                    "最小距离", 0.0, 1.0, 0.0, 0.1
                )
                self.config['bertopic_params']['umap_params']['metric'] = st.selectbox(
                    "距离度量", ['cosine', 'euclidean', 'manhattan'], index=0
                )
        
        with st.expander("🔍 HDBSCAN聚类参数"):
            col1, col2 = st.columns(2)
            with col1:
                self.config['bertopic_params']['hdbscan_params']['min_cluster_size'] = st.slider(
                    "最小聚类大小", 5, 100, 15
                )
                self.config['bertopic_params']['hdbscan_params']['min_samples'] = st.slider(
                    "最小样本数", 1, 20, 5
                )
            with col2:
                self.config['bertopic_params']['hdbscan_params']['metric'] = st.selectbox(
                    "聚类距离", ['euclidean', 'manhattan'], index=0
                )
                self.config['bertopic_params']['hdbscan_params']['cluster_selection_method'] = st.selectbox(
                    "聚类选择方法", ['eom', 'leaf'], index=0
                )
        
        # 分析功能开关
        st.markdown("---")
        st.subheader("🔬 分析功能")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self.config['analysis']['time_analysis']['enable'] = st.checkbox(
                "时间演化分析", value=True
            )
        with col2:
            self.config['analysis']['source_analysis']['enable'] = st.checkbox(
                "来源对比分析", value=True
            )
        with col3:
            self.config['analysis']['frame_analysis']['enable'] = st.checkbox(
                "框架分析", value=True
            )
    
    def _render_analysis_runner(self):
        """渲染分析运行界面"""
        st.markdown('<div class="module-header">🚀 运行分析</div>', unsafe_allow_html=True)
        
        # 检查前置条件
        can_run = self._check_prerequisites()
        
        if not can_run:
            st.warning("⚠️ 请先完成数据上传和参数配置")
            return
        
        # 显示当前配置摘要
        with st.expander("📋 当前配置摘要"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**基础参数:**")
                st.write(f"- 文本列: {self.config['data_processing']['text_column']}")
                st.write(f"- 最小主题大小: {self.config['bertopic_params']['min_topic_size']}")
                st.write(f"- 主题数量: {self.config['bertopic_params']['nr_topics']}")
            with col2:
                st.write("**启用功能:**")
                st.write(f"- 词性标注: {'✅' if self.config['bertopic_params']['expert_keyword_extraction']['enable_pos_patterns'] else '❌'}")
                st.write(f"- 时间分析: {'✅' if self.config['analysis']['time_analysis']['enable'] else '❌'}")
                st.write(f"- 跨语言分析: ✅")
        
        # 运行按钮
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 开始分析", type="primary", use_container_width=True):
                self._run_analysis()
    
    def _render_results_viewer(self):
        """渲染结果查看界面"""
        st.markdown('<div class="module-header">📊 分析结果查看</div>', unsafe_allow_html=True)
        
        if st.session_state.analysis_results is None:
            st.info("🔍 还没有分析结果，请先运行分析")
            return
        
        results = st.session_state.analysis_results
        
        # 结果摘要
        st.subheader("📈 分析摘要")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("识别主题数", results.get('n_topics', 0))
        with col2:
            st.metric("分析文档数", results.get('n_documents', 0))
        with col3:
            st.metric("处理时间", f"{results.get('elapsed_time', 0):.1f}秒")
        with col4:
            st.metric("生成文件数", len(results.get('generated_files', [])))
        
        # 结果文件下载
        st.markdown("---")
        st.subheader("📥 结果文件下载")
        
        if results.get('generated_files'):
            # 创建下载包
            zip_buffer = self._create_results_zip(results['generated_files'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📦 下载所有结果",
                    data=zip_buffer,
                    file_name=f"bertopic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
            
            with col2:
                st.info(f"包含 {len(results['generated_files'])} 个结果文件")
        
        # 在线预览
        st.markdown("---")
        st.subheader("👁️ 结果预览")
        
        # 显示主题摘要表格
        if 'topics_summary' in results:
            st.subheader("📊 主题摘要")
            st.dataframe(results['topics_summary'])
        
        # 显示图表
        if 'charts' in results:
            for chart_name, chart_path in results['charts'].items():
                if chart_path and Path(chart_path).exists():
                    st.subheader(f"📈 {chart_name}")
                    if chart_path.endswith('.png'):
                        st.image(chart_path)
                    elif chart_path.endswith('.html'):
                        with open(chart_path, 'r', encoding='utf-8') as f:
                            st.components.v1.html(f.read(), height=500)
    
    def _save_uploaded_file(self, uploaded_file, file_key: str) -> str:
        """保存上传的文件到临时目录"""
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / f"{file_key}.xlsx"
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    
    def _validate_uploaded_data(self):
        """验证上传的数据"""
        all_valid = True
        
        for file_key, file_path in st.session_state.uploaded_files.items():
            try:
                df = pd.read_excel(file_path)
                text_column = self.config['data_processing']['text_column']
                
                if text_column not in df.columns:
                    st.error(f"❌ {file_key}: 未找到文本列 '{text_column}'")
                    all_valid = False
                else:
                    valid_texts = df[text_column].dropna()
                    st.success(f"✅ {file_key}: {len(valid_texts)} 个有效文本")
                    
            except Exception as e:
                st.error(f"❌ {file_key}: 文件验证失败 - {e}")
                all_valid = False
        
        if all_valid and len(st.session_state.uploaded_files) >= 2:
            st.success("🎉 所有数据文件验证通过，可以开始分析！")
    
    def _check_prerequisites(self) -> bool:
        """检查运行分析的前置条件"""
        if len(st.session_state.uploaded_files) < 2:
            return False
        
        # 检查文件是否可读
        for file_path in st.session_state.uploaded_files.values():
            if not Path(file_path).exists():
                return False
        
        return True
    
    def _run_analysis(self):
        """运行分析"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 更新配置中的文件路径
            self.config['data_paths']['media_data'] = st.session_state.uploaded_files['media_data']
            self.config['data_paths']['social_media_data'] = st.session_state.uploaded_files['social_media_data']
            
            # 创建输出目录
            output_dir = Path(self.config['results_paths']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            status_text.text("🔄 正在加载数据...")
            progress_bar.progress(20)
            
            # 加载数据
            data_loader = DataLoader(self.config)
            documents, metadata_df = data_loader.load_and_prepare_data()
            
            status_text.text("🤖 正在训练模型...")
            progress_bar.progress(50)
            
            # 训练模型
            analyzer = TopicAnalyzer(self.config)
            topic_model, topics = analyzer.train_bertopic_model(documents)
            
            status_text.text("📊 正在生成结果...")
            progress_bar.progress(80)
            
            # 生成增强结果
            enhanced_topics = analyzer.expert_extractor.enhance_topic_representation(
                topic_model, documents
            )
            
            analyzer.generate_enhanced_results(
                topic_model=topic_model,
                documents=documents,
                topics=topics,
                metadata_df=metadata_df,
                enhanced_topics=enhanced_topics
            )
            
            status_text.text("✅ 分析完成！")
            progress_bar.progress(100)
            
            # 保存结果到session state
            results = self._collect_results(topic_model, documents, output_dir)
            st.session_state.analysis_results = results
            
            st.success("🎉 分析完成！请查看结果标签页。")
            
        except Exception as e:
            st.error(f"❌ 分析失败: {e}")
            st.exception(e)
    
    def _collect_results(self, topic_model, documents, output_dir: Path) -> dict:
        """收集分析结果"""
        results = {
            'n_topics': len(topic_model.get_topic_info()) - 1,
            'n_documents': len(documents),
            'elapsed_time': 0,  # TODO: 实际计算时间
            'generated_files': [],
            'charts': {},
            'topics_summary': None
        }
        
        # 收集生成的文件
        for file_path in output_dir.glob("**/*"):
            if file_path.is_file():
                results['generated_files'].append(str(file_path))
                
                # 分类图表文件
                if file_path.suffix in ['.png', '.html', '.pdf']:
                    results['charts'][file_path.stem] = str(file_path)
        
        # 读取主题摘要
        summary_files = list(output_dir.glob("*summary*.csv"))
        if summary_files:
            try:
                results['topics_summary'] = pd.read_csv(summary_files[0])
            except:
                pass
        
        return results
    
    def _create_results_zip(self, file_paths: list) -> bytes:
        """创建结果文件的zip包"""
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in file_paths:
                if Path(file_path).exists():
                    zip_file.write(file_path, Path(file_path).name)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    def _download_config_template(self):
        """下载配置模板"""
        config_yaml = yaml.dump(self.config, default_flow_style=False, allow_unicode=True)
        st.sidebar.download_button(
            label="📄 config.yaml",
            data=config_yaml,
            file_name="config_template.yaml",
            mime="text/yaml"
        )


def main():
    """主函数"""
    try:
        web_ui = BERTopicWebUI()
        web_ui.render_main_page()
    except Exception as e:
        st.error(f"应用启动失败: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
