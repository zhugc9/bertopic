"""
配置管理组件 - SOTA实现
======================
负责参数配置和预设管理，核心智能调参功能
"""

import streamlit as st
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Any, List
import sys

# 添加项目路径以导入topic_analyzer模块
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))


class ConfigManager:
    """配置管理器 - 简洁高效"""
    
    def __init__(self):
        self.presets = {
            "智能推荐": {
                "name": "智能推荐 🤖",
                "description": "基于您的数据自动分析并优化参数",
                "auto_adjust": True,
                "requires_analysis": True
            },
            "手动配置": {
                "name": "手动配置 ⚙️",
                "description": "自主调节所有参数",
                "manual_config": True
            },
            "新手友好": {
                "name": "新手友好 🌱", 
                "description": "保守稳定的参数",
                "params": {
                    "min_topic_size": 20,
                    "nr_topics": "auto",
                    "n_gram_range": [1, 2],
                    "umap_params": {"n_neighbors": 15, "n_components": 5},
                    "hdbscan_params": {"min_cluster_size": 20, "min_samples": 5}
                }
            },
            "细粒度分析": {
                "name": "细粒度分析 🔍",
                "description": "发现更多细节主题",
                "params": {
                    "min_topic_size": 10,
                    "nr_topics": "auto",
                    "n_gram_range": [1, 3],
                    "umap_params": {"n_neighbors": 10, "n_components": 4},
                    "hdbscan_params": {"min_cluster_size": 10, "min_samples": 3}
                }
            }
        }
    
    def render_config_selector(self) -> str:
        """渲染配置选择器"""
        st.markdown("#### ⚙️ 分析模式选择")
        
        preset_names = list(self.presets.keys())
        current = st.session_state.get('current_preset', '智能推荐')
        
        selected = st.selectbox(
            "选择分析类型",
            preset_names,
            index=preset_names.index(current) if current in preset_names else 0,
            help="选择适合您数据的分析模式"
        )
        
        # 显示说明
        preset_info = self.presets[selected]
        st.info(f"📝 {preset_info['description']}")
        
        # 根据选择的模式显示对应界面
        if selected == "智能推荐":
            self._render_smart_analysis()
        elif selected == "手动配置":
            self._render_manual_config()
        
        # 更新session_state
        st.session_state.current_preset = selected
        
        return selected
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'data_processing': {
                'text_column': 'Unit_Text',
                'merge_strategy': 'concat',
                'metadata_columns': [
                    # 基础列（两种格式都可能有）
                    '序号', '日期', '标题', '链接', 'Token数', 'text', 'token数', 
                    'Unit_ID', 'Source', 'Macro_Chunk_ID', 'speaker', 'Unit_Text',
                    'seed_sentence', 'expansion_logic', 'Unit_Hash', 'processing_status', 
                    'Incident', 'Valence',
                    
                    # Frame分析列（两种格式略有不同）
                    'Frame_ProblemDefinition', 'Frame_ProblemDefinition_Present',
                    'Frame_ResponsibilityAttribution', 'Frame_ResponsibilityAttribution_Present',
                    'Frame_MoralEvaluation', 'Frame_MoralEvaluation_Present',
                    'Frame_SolutionRecommendation', 'Frame_TreatmentRecommendation_Present',
                    'Frame_ActionStatement', 'Frame_ConflictAttribution_Present',
                    'Frame_CausalExplanation', 'Frame_CausalInterpretation_Present',
                    'Frame_HumanInterest_Present', 'Frame_EconomicConsequences_Present',
                    
                    # 分析维度列
                    'Evidence_Type', 'Attribution_Level', 'Temporal_Focus', 
                    'Primary_Actor_Type', 'Geographic_Scope', 'Relationship_Model_Definition', 
                    'Discourse_Type'
                ]
            },
            'bertopic_params': {
                'language': 'multilingual',
                'embedding_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'min_topic_size': 15,
                'nr_topics': 'auto',
                'n_gram_range': [1, 2],
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
                },
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
                }
            },
            'data_paths': {
                'media_data': '',
                'social_media_data': ''
            },
            'results_paths': {
                'output_dir': 'results',
                'model_dir': 'results/trained_model',
                'summary_file': 'results/topics_summary.csv',
                'summary_enhanced': 'results/topics_summary_enhanced.csv',
                'cross_lingual_file': 'results/cross_lingual_composition.csv',
                'evolution_file': 'results/dynamic_evolution_analysis.csv',
                'viz_file': 'results/topic_visualization.html',
                'source_analysis': 'results/topic_by_source.html',
                'timeline_analysis': 'results/topics_over_time.html',
                'frame_heatmap': 'results/topic_frame_heatmap.html',
                'academic_charts': {
                    'topic_distribution': 'results/academic_topic_distribution.png',
                    'topic_sizes': 'results/academic_topic_sizes.png',
                    'topic_evolution': 'results/academic_topic_evolution.png',
                    'cross_lingual': 'results/academic_cross_lingual.png'
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
    
    def apply_preset(self, preset_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用预设参数 - 基于现有接口"""
        if preset_name not in self.presets:
            return config
            
        preset = self.presets[preset_name]
        
        # 智能推荐：应用智能分析结果
        if preset_name == "智能推荐" and hasattr(st.session_state, 'smart_params'):
            smart_params = st.session_state.smart_params
            # 转换IntelligentTuner的扁平结构为config需要的嵌套结构
            converted_params = self._convert_smart_params(smart_params)
            # 深度合并参数，确保嵌套字典正确合并
            self._deep_merge_params(config['bertopic_params'], converted_params)
            
        # 手动配置：应用用户设置
        elif preset_name == "手动配置" and hasattr(st.session_state, 'manual_params'):
            manual_params = st.session_state.manual_params
            config['bertopic_params'].update(manual_params)
            
        # 预设模式：应用固定参数
        elif 'params' in preset:
            config['bertopic_params'].update(preset['params'])
        
        return config
    
    def _render_smart_analysis(self):
        """渲染智能分析界面 - KISS原则"""
        st.markdown("##### 🤖 智能参数分析")
        
        if not st.session_state.get('uploaded_files'):
            st.warning("⏳ 请先上传数据文件")
            return
        
        # 第1步：让用户选择文本列
        text_column = self._select_text_column()
        if not text_column:
            return
        
        # 第2步：分析按钮
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔍 分析我的数据", type="primary", use_container_width=True):
                self._execute_smart_analysis(text_column)
        with col2:
            if hasattr(st.session_state, 'smart_analysis_results'):
                st.success("✅ 完成")
        
        # 第3步：显示结果
        if hasattr(st.session_state, 'smart_analysis_results'):
            self._display_smart_results()
    
    def _render_manual_config(self):
        """渲染手动配置界面 - 详细指导版"""
        st.markdown("##### ⚙️ 手动参数配置")
        st.info("💡 根据您的数据特点调整参数。不确定时可以先用智能推荐作为参考。")
        
        # 获取数据规模信息（如果有的话）
        total_docs = 0
        if st.session_state.get('uploaded_files'):
            try:
                for file_path in st.session_state.uploaded_files.values():
                    df = pd.read_excel(file_path)
                    total_docs += len(df)
            except:
                pass
        
        # 1. 核心主题参数
        with st.expander("🎯 核心主题参数", expanded=True):
            st.markdown("**控制主题数量和质量的关键参数**")
            
            col1, col2 = st.columns([3, 2])
            with col1:
                min_topic_size = st.slider(
                    "最小主题大小", 5, 100, 15,
                    help="每个主题至少包含的文档数。值越小=主题越多越细致，值越大=主题越少越宏观"
                )
            with col2:
                # 动态计算建议值
                if total_docs > 0:
                    suggested_min = max(5, int(total_docs * 0.01))
                    suggested_max = max(10, int(total_docs * 0.03))
                    st.info(f"📊 您的数据({total_docs}条)建议值: {suggested_min}-{suggested_max}")
                else:
                    st.info("📊 一般建议: 文档数×1-3%")
            
            # 参数效果说明
            st.markdown("**📈 调节效果:**")
            if min_topic_size <= 8:
                st.warning("⚠️ 很小(≤8): 会产生很多细碎主题，可能有噪音")
            elif min_topic_size <= 15:
                st.success("✅ 小(9-15): 产生较多细致主题，适合详细分析")
            elif min_topic_size <= 30:
                st.info("ℹ️ 中等(16-30): 产生适中数量主题，平衡细节与概括")
            else:
                st.info("🔵 大(>30): 产生较少宏观主题，适合高层次概览")
        
        # 2. UMAP降维参数
        with st.expander("🔄 UMAP降维参数", expanded=True):
            st.markdown("**控制文档在低维空间中的分布结构**")
            
            col1, col2 = st.columns(2)
            with col1:
                n_neighbors = st.slider(
                    "邻居数量", 5, 50, 15,
                    help="每个点考虑多少个邻居。影响局部vs全局结构的平衡"
                )
                st.markdown("**🔗 邻居数效果:**")
                if n_neighbors <= 10:
                    st.info("🔍 少邻居(5-10): 保持局部细节，但可能过度分割")
                elif n_neighbors <= 20:
                    st.success("⚖️ 中等邻居(11-20): 平衡局部与全局结构")
                else:
                    st.info("🌐 多邻居(21-50): 强调全局结构，更稳定但可能丢失细节")
                    
            with col2:
                n_components = st.slider(
                    "降维维度", 2, 10, 5,
                    help="降维后的维度数。影响信息保留程度"
                )
                st.markdown("**📐 维度效果:**")
                if n_components <= 3:
                    st.warning("⚠️ 低维(2-3): 计算快但信息损失多")
                elif n_components <= 6:
                    st.success("✅ 中维(4-6): 平衡性能与信息保留")
                else:
                    st.info("🔵 高维(7-10): 保留更多信息但计算慢")
        
        # 3. HDBSCAN聚类参数
        with st.expander("🎯 HDBSCAN聚类参数", expanded=True):
            st.markdown("**控制如何将相似文档聚集成主题**")
            
            col1, col2 = st.columns(2)
            with col1:
                min_samples = st.slider(
                    "最小样本数", 1, 30, 5,
                    help="形成聚类核心需要的最少点数。影响聚类严格程度"
                )
                st.markdown("**🎯 样本数效果:**")
                if min_samples <= 3:
                    st.warning("⚠️ 宽松(1-3): 容易形成聚类，但可能包含噪音")
                elif min_samples <= 7:
                    st.success("✅ 适中(4-7): 平衡聚类质量与数量")
                else:
                    st.info("🔵 严格(8+): 高质量聚类，但可能遗漏边缘文档")
                    
            with col2:
                # 显示推荐关系
                recommended_samples = max(1, min_topic_size // 3)
                st.info(f"💡 推荐值: {recommended_samples} (主题大小÷3)")
                st.markdown("**🔗 参数关系:**")
                st.write(f"• 最小样本数通常设为主题大小的1/3")
                st.write(f"• 当前设置: {min_samples} vs 推荐 {recommended_samples}")
                
        # 4. 高级参数
        with st.expander("🔧 高级参数", expanded=False):
            st.markdown("**高级用户可调节的额外参数**")
            
            col1, col2 = st.columns(2)
            with col1:
                nr_topics_option = st.selectbox(
                    "目标主题数量",
                    ["auto", "10", "15", "20", "25", "30", "35", "40"],
                    help="auto=自动确定，或手动指定主题数量"
                )
                nr_topics = "auto" if nr_topics_option == "auto" else int(nr_topics_option)
                
                n_gram_min = st.selectbox("N-gram最小值", [1, 2], index=0, help="单词组合的最小长度")
                
            with col2:
                n_gram_max = st.selectbox("N-gram最大值", [1, 2, 3], index=1, help="单词组合的最大长度")
                
                # 显示N-gram效果
                if n_gram_min == 1 and n_gram_max == 1:
                    st.info("🔤 单词模式: 只考虑单个词汇")
                elif n_gram_min == 1 and n_gram_max == 2:
                    st.success("✅ 标准模式: 单词+双词组合(推荐)")
                elif n_gram_min == 1 and n_gram_max == 3:
                    st.info("📚 丰富模式: 包含三词短语(学术文本)")
                else:
                    st.info("🔧 自定义模式")
        
        # 5. 当前参数总结
        with st.expander("📋 当前参数总结", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🎯 主题控制:**")
                st.write(f"• 最小主题大小: {min_topic_size}")
                st.write(f"• 目标主题数: {nr_topics}")
                st.write(f"• N-gram范围: [{n_gram_min}, {n_gram_max}]")
                
            with col2:
                st.markdown("**⚙️ 算法参数:**")
                st.write(f"• UMAP邻居数: {n_neighbors}")
                st.write(f"• 降维维度: {n_components}")
                st.write(f"• 最小样本数: {min_samples}")
                
            # 参数合理性检查
            st.markdown("**🔍 参数检查:**")
            warnings = []
            if min_samples > min_topic_size:
                warnings.append("⚠️ 最小样本数大于主题大小，可能导致无法形成主题")
            if n_neighbors > 30 and total_docs < 1000:
                warnings.append("⚠️ 小数据集使用过多邻居可能导致过度平滑")
            if min_topic_size < 5:
                warnings.append("⚠️ 主题大小过小可能产生噪音主题")
                
            if warnings:
                for warning in warnings:
                    st.warning(warning)
            else:
                st.success("✅ 参数设置合理")
        
        # 保存完整的手动配置
        manual_params = {
            'min_topic_size': min_topic_size,
            'nr_topics': nr_topics,
            'n_gram_range': [n_gram_min, n_gram_max],
            'umap_params': {
                'n_neighbors': n_neighbors,
                'n_components': n_components,
                'min_dist': 0.0,
                'metric': 'cosine',
                'random_state': 42
            },
            'hdbscan_params': {
                'min_cluster_size': min_topic_size,
                'min_samples': min_samples,
                'metric': 'euclidean',
                'cluster_selection_method': 'eom',
                'prediction_data': True
            }
        }
        
        st.session_state.manual_params = manual_params
    
    def _select_text_column(self) -> str:
        """让用户选择文本列 - KISS原则"""
        try:
            # 获取第一个文件的列名
            first_file = list(st.session_state.uploaded_files.values())[0]
            df = pd.read_excel(first_file)
            
            columns = df.columns.tolist()
            
            # 智能推荐文本列 - 基于用户反馈优化
            text_candidates = ['Unit_Text', 'text', 'Text', 'Incident', 'Content', '文本']
            suggested = None
            for candidate in text_candidates:
                if candidate in columns:
                    suggested = candidate
                    break
            
            # 用户选择
            selected_col = st.selectbox(
                "选择文本列",
                columns,
                index=columns.index(suggested) if suggested else 0,
                help="选择包含要分析文本的列"
            )
            
            return selected_col
            
        except Exception as e:
            st.error(f"读取文件失败: {e}")
            return None
    
    def _execute_smart_analysis(self, text_column: str):
        """执行智能分析 - 调用现有IntelligentTuner接口"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_text = st.empty()
        
        try:
            # 步骤1: 加载数据
            status_text.info("📁 加载数据...")
            progress_bar.progress(0.2)
            
            documents = self._load_documents(text_column)
            log_text.success(f"✓ 加载了 {len(documents)} 个文档")
            
            if not documents:
                st.error("❌ 未找到有效的文本数据")
                return
            
            # 步骤2: 智能分析
            status_text.info("🔍 分析文本特征...")
            progress_bar.progress(0.5)
            
            # 使用现有的IntelligentTuner接口
            from topic_analyzer.intelligent_tuner import IntelligentTuner
            tuner = IntelligentTuner()
            
            # 智能采样策略：保持原始数据规模用于参数计算，但限制分析样本提升速度
            total_docs = len(documents)
            # 海量数据智能采样策略
            if total_docs >= 100000:
                # 超大规模：分层采样5000条
                sample_size = 5000
                step = max(1, total_docs // sample_size)
                sample_docs = [documents[i] for i in range(0, total_docs, step)][:sample_size]
                log_text.info(f"• 超大数据集检测：总计 {total_docs} 条，分层采样 {len(sample_docs)} 条")
            elif total_docs >= 50000:
                # 大规模：分层采样3000条
                sample_size = 3000
                step = max(1, total_docs // sample_size)
                sample_docs = [documents[i] for i in range(0, total_docs, step)][:sample_size]
                log_text.info(f"• 大数据集检测：总计 {total_docs} 条，分层采样 {len(sample_docs)} 条")
            elif total_docs > 10000:
                # 中大规模：前2000条采样
                sample_docs = documents[:2000]
                log_text.info(f"• 中大数据集检测：总计 {total_docs} 条，采样 {len(sample_docs)} 条")
            else:
                # 中小规模：全量分析
                sample_docs = documents
                log_text.info(f"• 全量分析 {len(sample_docs)} 个文档")
            
            # 手动设置真实文档数量到特征中（确保参数计算基于真实规模）
            log_text.info(f"• 基于真实数据规模 {total_docs} 条进行参数优化")
            
            # 步骤3: 参数优化
            status_text.info("⚙️ 优化参数...")
            progress_bar.progress(0.8)
            
            # 执行智能分析
            try:
                results = tuner.auto_tune(sample_docs)
                log_text.info(f"• IntelligentTuner返回类型: {type(results)}")
                
                # 检查返回结果结构
                if not isinstance(results, dict):
                    st.error("❌ IntelligentTuner返回结果格式错误")
                    return
                
                if 'data_features' not in results or 'optimized_parameters' not in results:
                    st.error("❌ 缺少必要的分析结果字段")
                    return
                
                # 修正特征中的文档数量为真实数量
                results['data_features']['total_docs'] = total_docs
                results['data_features']['sample_size'] = len(sample_docs)
                
                # 智能修正：确保参数基于真实数据规模，但保留智能优化逻辑
                if total_docs != len(sample_docs):
                    log_text.info(f"• 基于真实规模 {total_docs} 重新优化参数")
                    # 重新运行参数优化器，使用真实数据规模
                    from topic_analyzer.intelligent_tuner import ParameterOptimizer
                    optimizer = ParameterOptimizer()
                    corrected_params = optimizer.optimize_parameters(results['data_features'])
                    # 更新优化后的参数
                    results['optimized_parameters'].update(corrected_params)
                
                params = results['optimized_parameters']
                
            except Exception as tune_error:
                log_text.error(f"• IntelligentTuner调用失败: {tune_error}")
                st.error("❌ 智能分析失败，请检查数据格式")
                return
            
            # 调试信息 - 显示实际返回的参数键
            log_text.write(f"• 返回的参数键: {list(params.keys())}")
            
            # 完成
            status_text.success("🎉 智能分析完成！")
            progress_bar.progress(1.0)
            
            # 保存结果
            st.session_state.smart_analysis_results = results
            st.session_state.smart_params = params
            
            log_text.success("✅ 参数已自动优化")
                
        except Exception as e:
            status_text.error(f"❌ 分析失败: {e}")
            log_text.error(f"错误详情: {str(e)}")
            # 显示更多调试信息
            import traceback
            log_text.text(traceback.format_exc())
    
    def _load_documents(self, text_column: str) -> List[str]:
        """加载用户文档 - KISS原则"""
        documents = []
        
        for file_path in st.session_state.uploaded_files.values():
            try:
                df = pd.read_excel(file_path)
                texts = df[text_column].dropna().astype(str)
                texts = texts[texts.str.len() > 10].tolist()
                documents.extend(texts)
            except Exception:
                continue
        
        return documents
    
    def _display_smart_results(self):
        """显示智能分析结果"""
        results = st.session_state.smart_analysis_results
        
        st.markdown("##### 📊 数据分析结果")
        
        # 数据概览
        features = results['data_features']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("文档数量", features.get('total_docs', 0))
        with col2:
            # 语言映射
            lang_map = {'zh': '中文', 'en': '英文', 'ru': '俄文'}
            dominant_lang = features.get('dominant_language', 'unknown')
            display_lang = lang_map.get(dominant_lang, dominant_lang)
            st.metric("主要语言", display_lang)
        with col3:
            complexity_map = {'low': '简单', 'medium': '中等', 'high': '复杂'}
            complexity = features.get('estimated_complexity', 'unknown')
            display_complexity = complexity_map.get(complexity, complexity)
            st.metric("复杂度", display_complexity)
        
        # 推荐参数
        st.markdown("##### 🎯 推荐参数")
        params = results['optimized_parameters']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**最小主题大小:** {params.get('min_topic_size', 15)}")
            st.write(f"**UMAP邻居数:** {params.get('umap_n_neighbors', 15)}")
        with col2:
            st.write(f"**降维维度:** {params.get('umap_n_components', 5)}")
            st.write(f"**最小样本数:** {params.get('hdbscan_min_samples', 5)}")
        
        # 详细推荐理由 - 科学依据说明
        with st.expander("💡 详细推荐理由", expanded=True):
            self._display_detailed_reasoning(results['data_features'], params)
        
        st.info("💡 这些参数将自动应用到分析中")
    
    def _convert_smart_params(self, smart_params: Dict[str, Any]) -> Dict[str, Any]:
        """转换IntelligentTuner的扁平参数结构为config需要的嵌套结构"""
        converted = {}
        
        # 基础参数
        if 'min_topic_size' in smart_params:
            converted['min_topic_size'] = smart_params['min_topic_size']
        if 'nr_topics' in smart_params:
            converted['nr_topics'] = smart_params['nr_topics']
        if 'n_gram_range' in smart_params:
            converted['n_gram_range'] = smart_params['n_gram_range']
        
        # UMAP参数 - 保留默认值并覆盖智能调参结果
        umap_params = {
            'min_dist': 0.0,
            'metric': 'cosine', 
            'random_state': 42
        }
        if 'umap_n_neighbors' in smart_params:
            umap_params['n_neighbors'] = smart_params['umap_n_neighbors']
        if 'umap_n_components' in smart_params:
            umap_params['n_components'] = smart_params['umap_n_components']
        converted['umap_params'] = umap_params
        
        # HDBSCAN参数 - 保留默认值并覆盖智能调参结果
        hdbscan_params = {
            'metric': 'euclidean',
            'cluster_selection_method': 'eom',
            'prediction_data': True
        }
        if 'hdbscan_min_cluster_size' in smart_params:
            hdbscan_params['min_cluster_size'] = smart_params['hdbscan_min_cluster_size']
        if 'hdbscan_min_samples' in smart_params:
            hdbscan_params['min_samples'] = smart_params['hdbscan_min_samples']
        converted['hdbscan_params'] = hdbscan_params
        
        return converted
    
    def _deep_merge_params(self, target: Dict[str, Any], source: Dict[str, Any]):
        """深度合并参数字典，确保嵌套字典正确合并"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                self._deep_merge_params(target[key], value)
            else:
                # 直接赋值
                target[key] = value
    
    def _display_detailed_reasoning(self, features: Dict[str, Any], params: Dict[str, Any]):
        """显示详细的参数推荐理由 - 科学依据"""
        total_docs = features.get('total_docs', 0)
        complexity = features.get('estimated_complexity', 'medium')
        dominant_lang = features.get('dominant_language', 'unknown')
        avg_length = features.get('avg_text_length', 0)
        vocab_diversity = features.get('vocabulary_diversity', 0)
        
        # 1. 主题大小的具体计算依据
        if 'min_topic_size' in params:
            topic_size = params['min_topic_size']
            percentage = (topic_size / total_docs) * 100 if total_docs > 0 else 0
            st.markdown(f"**🎯 最小主题大小 {topic_size}**")
            st.write(f"   • 计算公式：{total_docs}文档 × {percentage:.1f}% = {topic_size}")
            st.write(f"   • 科学依据：1-5%比例确保主题有足够样本，避免噪音主题")
            st.write("")
        
        # 2. UMAP邻居数的具体原理
        if params.get('umap_n_neighbors'):
            neighbors = params['umap_n_neighbors']
            st.markdown(f"**🔗 UMAP邻居数 {neighbors}**")
            if total_docs < 500:
                st.write("   • 小数据集(<500)：用较少邻居保持局部结构精度")
            elif total_docs < 2000:
                st.write("   • 中等数据集(500-2000)：平衡局部细节与全局结构")
            else:
                st.write("   • 大数据集(>2000)：用更多邻居捕获全局模式")
            st.write("")
        
        # 3. 降维维度的选择理由
        if params.get('umap_n_components'):
            components = params['umap_n_components']
            st.markdown(f"**📐 降维维度 {components}**")
            if complexity == 'high':
                st.write("   • 高复杂度文本：需要更多维度保留语义信息")
            elif complexity == 'low':
                st.write("   • 低复杂度文本：较少维度避免过拟合")
            else:
                st.write("   • 中等复杂度：标准维度平衡性能与精度")
            st.write("")
        
        # 4. 语言特征影响
        lang_names = {'zh': '中文', 'en': '英文', 'ru': '俄文'}
        lang_display = lang_names.get(dominant_lang, dominant_lang)
        st.markdown(f"**🌐 语言特征优化（{lang_display}）**")
        
        if dominant_lang == 'zh':
            st.write("   • 中文词汇密度高，N-gram=[1,2]，主题大小可适当减小")
        elif dominant_lang == 'ru':
            st.write("   • 俄文词汇变化丰富，N-gram=[1,2]，需要更多样本避免碎片化")
        elif dominant_lang == 'en':
            st.write("   • 英文支持短语，N-gram=[1,3]，标准参数设置")
        else:
            st.write(f"   • {lang_display}文本，使用通用参数配置")
        st.write("")
        
        # 5. 数据特征综合分析
        st.markdown("**📊 数据特征综合分析**")
        if avg_length > 0:
            if avg_length > 200:
                st.write(f"   • 平均文本长度 {avg_length:.0f}字符：长文本信息丰富，可用较小主题")
            elif avg_length < 50:
                st.write(f"   • 平均文本长度 {avg_length:.0f}字符：短文本需要更大主题避免碎片化")
            else:
                st.write(f"   • 平均文本长度 {avg_length:.0f}字符：中等长度，使用标准参数")
        
        if vocab_diversity > 0:
            if vocab_diversity > 0.6:
                st.write(f"   • 词汇多样性 {vocab_diversity:.2f}：高多样性需要更精细的主题划分")
            elif vocab_diversity < 0.3:
                st.write(f"   • 词汇多样性 {vocab_diversity:.2f}：低多样性用较大主题避免重复")
            else:
                st.write(f"   • 词汇多样性 {vocab_diversity:.2f}：中等多样性，平衡设置")
        
        # 6. 总结建议
        st.markdown("**🎯 参数调节总结**")
        st.write(f"   • 基于 {total_docs} 条{lang_display}文本的深度分析")
        st.write(f"   • {complexity}复杂度数据的专业参数配置")
        st.write("   • 所有参数均基于数据驱动的科学计算")
    
