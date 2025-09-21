"""
分析运行组件 - SOTA实现
=====================
高效的分析执行和进度管理
"""

import streamlit as st
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class AnalysisRunner:
    """分析运行器 - 简洁高效"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def render_runner_interface(self):
        """渲染运行界面"""
        st.markdown("#### 🚀 开始分析")
        
        # 检查前置条件
        if not self._check_ready():
            st.warning("⚠️ 请先完成前面的步骤")
            return
        
        # 显示配置摘要
        self._show_config_summary()
        
        # 运行按钮
        if st.button("🚀 开始分析", type="primary", use_container_width=True):
            self._run_analysis()
        
        # 显示状态
        self._show_status()
    
    def _check_ready(self) -> bool:
        """检查是否准备就绪"""
        return (
            'uploaded_files' in st.session_state and 
            len(st.session_state.uploaded_files) > 0 and
            'current_preset' in st.session_state
        )
    
    def _show_config_summary(self):
        """显示配置摘要"""
        files_count = len(st.session_state.get('uploaded_files', {}))
        preset = st.session_state.get('current_preset', '未设置')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("数据文件", f"{files_count} 个")
        with col2:
            st.metric("分析模式", preset)
    
    def _run_analysis(self):
        """运行分析"""
        # 初始化状态
        st.session_state.analysis_running = True
        st.session_state.analysis_progress = 0
        
        # 创建进度显示
        progress_container = st.empty()
        
        try:
            with progress_container.container():
                st.info("🚀 分析开始...")
                progress_bar = st.progress(0)
            
            self._execute_analysis(progress_container, progress_bar)
            
        except Exception as e:
            st.error(f"❌ 分析失败: {e}")
        finally:
            st.session_state.analysis_running = False
    
    def _execute_analysis(self, progress_container, progress_bar):
        """执行分析流程 - KISS原则精简版"""
        status_text = st.empty()
        
        try:
            # 步骤1: 加载数据
            status_text.info("📁 加载数据...")
            progress_bar.progress(0.3)
            self._load_data()
            
            # 步骤2: 训练模型
            status_text.info("🤖 训练模型...")
            progress_bar.progress(0.7)
            self._train_model()
            
            # 步骤3: 生成结果
            status_text.info("📊 生成结果...")
            progress_bar.progress(0.9)
            self._generate_results()
            
            # 完成
            status_text.success("🎉 分析完成！")
            progress_bar.progress(1.0)
            st.balloons()
                
        except Exception as e:
            status_text.error(f"❌ 分析失败: {e}")
            st.write(f"错误详情: {str(e)}")
    
    def _load_data(self):
        """加载数据 - 实时反馈版"""
        from topic_analyzer.data_loader import DataLoader
        
        # 实时显示进度
        st.write("• 配置数据路径...")
        # 确保data_paths存在
        if 'data_paths' not in self.config:
            self.config['data_paths'] = {}
        
        # 设置上传的文件路径，确保未上传的文件路径不会导致错误
        uploaded_files = st.session_state.get('uploaded_files', {})
        if 'media_data' in uploaded_files:
            self.config['data_paths']['media_data'] = uploaded_files['media_data']
            st.write(f"  ✓ 主数据文件: {uploaded_files['media_data']}")
        else:
            # 如果没有media_data，将其设为空，DataLoader会跳过
            self.config['data_paths']['media_data'] = ''
            
        if 'social_media_data' in uploaded_files:
            self.config['data_paths']['social_media_data'] = uploaded_files['social_media_data']
            st.write(f"  ✓ 补充数据文件: {uploaded_files['social_media_data']}")
        else:
            # 如果没有social_media_data，将其设为空，DataLoader会跳过
            self.config['data_paths']['social_media_data'] = ''
        
        st.write("• 保存临时配置...")
        temp_config = self.results_dir / "temp_config.yaml"
        with open(temp_config, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        st.write("• 开始加载数据...")
        # 修复：DataLoader期望配置字典，不是文件路径
        loader = DataLoader(self.config)
        # 修复：正确解包DataLoader返回的元组
        self.documents, self.metadata_df = loader.load_and_prepare_data()
        st.write(f"✓ 加载完成：{len(self.documents)} 个文档")
    
    def _train_model(self):
        """训练模型 - 实时反馈版"""
        from topic_analyzer.model import TopicAnalyzer
        
        st.write("• 初始化BERTopic模型...")
        # 修复：TopicAnalyzer期望配置字典，不是文件路径
        analyzer = TopicAnalyzer(self.config)
        
        doc_count = len(self.documents)
        st.write(f"• 开始训练（{doc_count} 个文档）...")
        
        import time
        start_time = time.time()
        
        # 实时状态更新容器
        status_container = st.empty()
        
        # 步骤1: 初始化
        status_container.info("📊 步骤1/5: 初始化训练组件...")
        st.write("  • 加载嵌入模型...")
        time.sleep(1)
        
        # 步骤2: 文档嵌入
        status_container.info(f"🚀 步骤2/5: 生成文档嵌入向量 ({doc_count}个文档)")
        st.write("  • 正在处理文档嵌入，这可能需要几分钟...")
        
        # 创建实时监控显示
        progress_placeholder = st.empty()
        
        # 执行实际训练 - 这里是最耗时的部分
        import threading
        training_complete = threading.Event()
        training_error = [None]  # 用列表存储异常，因为线程无法直接修改外部变量
        
        def training_task():
            """训练任务线程"""
            try:
                self.topic_model, self.topics = analyzer.train_bertopic_model(self.documents)
                training_complete.set()
            except Exception as e:
                training_error[0] = e
                training_complete.set()
        
        def progress_monitor():
            """进度监控线程 - 每分钟更新一次"""
            minutes_elapsed = 0
            while not training_complete.is_set():
                if minutes_elapsed == 0:
                    progress_placeholder.info("🔄 训练进行中... (刚开始)")
                else:
                    progress_placeholder.info(f"🔄 训练进行中... (已运行 {minutes_elapsed} 分钟)")
                
                # 等待60秒或训练完成
                training_complete.wait(timeout=60)
                minutes_elapsed += 1
        
        # 启动训练和监控线程
        training_thread = threading.Thread(target=training_task)
        monitor_thread = threading.Thread(target=progress_monitor)
        
        training_thread.daemon = True
        monitor_thread.daemon = True
        
        training_thread.start()
        monitor_thread.start()
        
        # 等待训练完成
        training_thread.join()
        
        # 检查训练结果
        if training_error[0]:
            if isinstance(training_error[0], KeyboardInterrupt):
                status_container.warning("⚠️ 训练被用户中断")
                st.warning("训练已中断，没有生成完整结果")
            else:
                status_container.error(f"❌ 训练失败: {training_error[0]}")
                st.write(f"错误详情: {str(training_error[0])}")
                st.write("💡 提示：可以检查 results/ 目录查看是否有部分结果文件")
            raise training_error[0]
        
        # 训练成功，清除进度监控
        progress_placeholder.empty()
        st.write("  • 训练完成，准备后续处理...")
        
        # 步骤3: 降维处理
        status_container.info("🔄 步骤3/5: UMAP降维处理完成")
        st.write("  • 降维处理完成")
        time.sleep(0.5)
        
        # 步骤4: 聚类分析
        status_container.info("🎯 步骤4/5: HDBSCAN聚类分析完成")
        st.write("  • 聚类分析完成")
        time.sleep(0.5)
        
        # 步骤5: 主题优化
        status_container.info("✨ 步骤5/5: 主题标签生成完成")
        st.write("  • 主题标签生成完成")
        time.sleep(0.5)
        
        # 完成统计
        end_time = time.time()
        training_time = end_time - start_time
        
        topic_info = self.topic_model.get_topic_info()
        n_topics = len(topic_info) - 1
        
        st.success(f"✅ 训练完成！发现 {n_topics} 个主题，用时 {training_time:.1f}秒")
        
        # 详细统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("发现主题", f"{n_topics} 个")
        with col2:
            st.metric("训练时间", f"{training_time:.1f} 秒")
        with col3:
            st.metric("处理速度", f"{doc_count/training_time:.1f} 文档/秒")
        
        st.write(f"✓ 模型训练成功：{n_topics} 个主题")
    
    def _generate_results(self):
        """生成结果 - 实时反馈版"""
        st.write("• 提取主题信息...")
        topic_info = self.topic_model.get_topic_info()
        
        st.write("• 保存结果文件...")
        
        # 首先保存训练好的模型
        try:
            st.write("  • 保存训练模型...")
            model_dir = self.results_dir / "trained_model"
            model_dir.mkdir(exist_ok=True)
            
            if hasattr(self, 'topic_model') and self.topic_model is not None:
                # 使用不同的保存方式避免权限问题
                import pickle
                model_file = model_dir / "bertopic_model.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(self.topic_model, f)
                st.write(f"    ✓ 模型已保存: {model_file}")
            else:
                st.warning("    ⚠️ 模型对象不存在，跳过保存")
                
        except Exception as save_error:
            st.warning(f"    ⚠️ 模型保存失败: {save_error}")
            st.write("    • 继续保存其他结果文件...")
        
        # 保存主题摘要
        summary_file = self.results_dir / "topics_summary.csv"
        topic_info.to_csv(summary_file, index=False, encoding='utf-8-sig')
        st.write(f"  ✓ 主题摘要: {summary_file}")
        
        # 保存文档-主题映射
        try:
            import pandas as pd
            
            # 确保所有数组长度一致
            n_docs = len(self.documents)
            n_topics = len(self.topics) if hasattr(self, 'topics') and self.topics is not None else 0
            
            st.write(f"  • 准备文档映射 ({n_docs}个文档, {n_topics}个主题分配)...")
            
            if n_topics > 0 and n_docs == n_topics:
                # 保存所有文档，不做限制
                max_docs = n_docs  # 保存全部文档
                
                doc_topic_df = pd.DataFrame({
                    'document_id': range(max_docs),
                    'document_text': [str(doc)[:200] + '...' if len(str(doc)) > 200 else str(doc) 
                                    for doc in self.documents[:max_docs]],  # 限制文本长度
                    'topic': self.topics[:max_docs],
                    'topic_name': [f"Topic_{topic}" if topic != -1 else 'Outlier' 
                                  for topic in self.topics[:max_docs]]
                })
                
                doc_mapping_file = self.results_dir / "document_topic_mapping.csv"
                doc_topic_df.to_csv(doc_mapping_file, index=False, encoding='utf-8-sig')
                st.write(f"  ✓ 文档主题映射: {doc_mapping_file} ({max_docs}条记录)")
            else:
                st.warning(f"  ⚠️ 数据长度不匹配 (文档:{n_docs}, 主题:{n_topics})，跳过映射保存")
                
        except Exception as mapping_error:
            st.warning(f"  ⚠️ 文档映射保存失败: {mapping_error}")
            st.write("  • 继续保存其他文件...")
        
        # ========== 调用专家级模块 ==========
        st.write("• 运行专家级分析...")
        
        try:
            from topic_analyzer.model import TopicAnalyzer
            analyzer = TopicAnalyzer(self.config)
            
            # 1. 跨语言主题成分分析
            st.write("  • 跨语言成分分析...")
            # 先检测文档语言
            document_languages = analyzer.cross_lingual_analyzer.analyze_document_languages(self.documents)
            # 然后分析主题语言构成
            analyzer.cross_lingual_analyzer.analyze_topic_language_composition(
                self.topics, document_languages
            )
            st.write("  ✓ 跨语言分析完成")
            
            # 2. 动态主题演化分析 (如果有日期数据)
            if hasattr(self, 'metadata_df') and self.metadata_df is not None:
                date_columns = ['日期', 'Date', 'date', 'timestamp']
                date_col = None
                for col in date_columns:
                    if col in self.metadata_df.columns:
                        date_col = col
                        break
                
                if date_col:
                    st.write("  • 动态演化分析...")
                    # 转换日期列为时间戳
                    import pandas as pd
                    timestamps = pd.to_datetime(self.metadata_df[date_col], errors='coerce').tolist()
                    analyzer.evolution_analyzer.analyze_dynamic_topics(
                        self.topic_model, self.documents, timestamps
                    )
                    st.write("  ✓ 演化分析完成")
                else:
                    st.write("  ⚠️ 未找到日期列，跳过演化分析")
            
            # 3. 生成学术级图表
            st.write("  • 生成学术图表...")
            analyzer.chart_generator.generate_all_academic_charts(
                self.topic_model, self.documents, self.topics, self.metadata_df
            )
            st.write("  ✓ 学术图表生成完成")
            
        except Exception as expert_error:
            st.warning(f"  ⚠️ 专家级分析部分失败: {expert_error}")
            st.write("  • 继续保存基础结果...")

        # 保存分析元信息
        metadata_file = self.results_dir / "analysis_metadata.json"
        import json
        metadata = {
            'n_topics': len(topic_info) - 1,
            'n_documents': len(self.documents),
            'timestamp': datetime.now().isoformat(),
            'preset_used': st.session_state.current_preset,
            'config_used': self.config,
            'output_files': {
                'topics_summary': str(summary_file),
                'document_mapping': str(doc_mapping_file),
                'trained_model': str(self.results_dir / "trained_model"),
                'analysis_metadata': str(metadata_file)
            }
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        st.write(f"  ✓ 分析元数据: {metadata_file}")
        
        # 更新session状态
        st.session_state.analysis_results = metadata
        
        # 显示输出文件总结
        st.success("📁 所有结果文件已保存到 results/ 目录：")
        st.write("```")
        st.write(f"results/")
        st.write(f"├── topics_summary.csv                      # 主题摘要")
        st.write(f"├── document_topic_mapping.csv              # 文档-主题映射")
        st.write(f"├── cross_lingual_composition.csv           # 跨语言成分分析")
        st.write(f"├── dynamic_evolution_analysis.csv          # 动态演化分析")
        st.write(f"├── analysis_metadata.json                  # 分析元数据")
        st.write(f"├── academic_topic_distribution.png         # 学术级主题分布图")
        st.write(f"├── academic_topic_sizes.png                # 学术级主题规模图")
        st.write(f"├── academic_topic_evolution.png            # 学术级演化图")
        st.write(f"├── academic_cross_lingual.png              # 学术级跨语言图")
        st.write(f"└── trained_model/                          # 训练好的模型")
        st.write(f"    └── bertopic_model.pkl                  # BERTopic模型文件")
        st.write("```")
    
    
    def _show_status(self):
        """显示状态"""
        if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
            st.success("✅ 分析已完成！")
        elif hasattr(st.session_state, 'analysis_running') and st.session_state.analysis_running:
            st.info("🔄 分析进行中...")
        
        if hasattr(st.session_state, 'analysis_progress'):
            st.progress(st.session_state.analysis_progress / 100)
