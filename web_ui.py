"""
BERTopic Web UI - SOTA & KISS 重构版
===================================
简洁、高效、模块化的Web界面
"""

import streamlit as st
import sys
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入模块化组件
from ui_components.config_manager import ConfigManager
from ui_components.file_uploader import FileUploader
from ui_components.analysis_runner import AnalysisRunner
from ui_components.results_viewer import ResultsViewer


class BERTopicWebUI:
    """BERTopic Web界面 - 重构版"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.file_uploader = FileUploader()
        self.results_viewer = ResultsViewer()
        
        # 初始化session_state
        self._init_session_state()
    
    def _init_session_state(self):
        """初始化会话状态"""
        defaults = {
            'uploaded_files': {},
            'current_preset': '智能推荐',
            'analysis_results': None,
            'analysis_running': False,
            'analysis_progress': 0
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """运行主界面"""
        # 页面配置
        st.set_page_config(
            page_title="BERTopic 主题分析",
            page_icon="🚀",
            layout="wide"
        )
        
        # 样式
        self._apply_styles()
        
        # 主标题
        st.markdown("# 🚀 BERTopic 主题分析系统")
        st.markdown("**简洁 · 高效 · 智能**")
        
        # 侧边栏
        self._render_sidebar()
        
        # 主要步骤
        self._render_main_steps()
    
    def _apply_styles(self):
        """应用样式"""
        st.markdown("""
        <style>
        .step-container {
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            background-color: #f8f9fa;
        }
        .step-title {
            font-size: 1.2rem;
        font-weight: bold;
            color: #1f77b4;
            margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.markdown("## 📋 状态面板")
        
        # 当前步骤
        current_step = self._get_current_step()
        st.sidebar.info(f"**当前步骤:** {current_step}")
        
        # 文件状态
        files_count = len(st.session_state.uploaded_files)
        st.sidebar.metric("已上传文件", f"{files_count} 个")
        
        # 分析状态
        if st.session_state.analysis_results:
            st.sidebar.success("✅ 分析已完成")
        elif st.session_state.analysis_running:
            st.sidebar.info("🔄 分析进行中")
        else:
            st.sidebar.info("⏳ 等待开始")
        
        # 快速操作
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 快速操作")
        
        if st.sidebar.button("🔄 重置全部"):
            for key in ['uploaded_files', 'analysis_results', 'analysis_running']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    def _get_current_step(self) -> str:
        """获取当前步骤"""
        if not st.session_state.uploaded_files:
            return "第1步：上传文件"
        elif not st.session_state.current_preset:
            return "第2步：选择模式"
        elif not st.session_state.analysis_results:
            return "第3步：开始分析"
        else:
            return "第4步：查看结果"
    
    def _render_main_steps(self):
        """渲染主要步骤"""
        # 第1步：文件上传
        with st.container():
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            step1_title = "📁 第1步：上传数据文件"
            if st.session_state.uploaded_files:
                step1_title += " ✅"
            st.markdown(f'<div class="step-title">{step1_title}</div>', unsafe_allow_html=True)
            
            if st.session_state.uploaded_files:
                summary = self.file_uploader.get_data_summary()
                st.success(f"✅ 已上传 {summary['files']} 个文件，共 {summary['total_records']} 条记录")
                
                with st.expander("📋 查看文件详情"):
                    for key, path in st.session_state.uploaded_files.items():
                        st.write(f"- **{key}**: `{path}`")
            else:
                uploaded_files = self.file_uploader.render_upload_interface()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 第2步：配置选择
        with st.container():
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            step2_title = "⚙️ 第2步：选择分析模式"
            if st.session_state.uploaded_files:
                step2_title += " ✅"
            st.markdown(f'<div class="step-title">{step2_title}</div>', unsafe_allow_html=True)
            
            if st.session_state.uploaded_files:
                selected_preset = self.config_manager.render_config_selector()
            else:
                st.info("⏳ 请先上传数据文件")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 第3步：运行分析
        with st.container():
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            step3_title = "🚀 第3步：开始分析"
            if st.session_state.analysis_results:
                step3_title += " ✅"
            st.markdown(f'<div class="step-title">{step3_title}</div>', unsafe_allow_html=True)
            
            # 创建分析运行器
            config = self.config_manager.get_default_config()
            config = self.config_manager.apply_preset(st.session_state.current_preset, config)
            
            analysis_runner = AnalysisRunner(config)
            analysis_runner.render_runner_interface()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 第4步：查看结果
        with st.container():
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            step4_title = "📊 第4步：查看结果"
            if st.session_state.analysis_results:
                step4_title += " ✅"
            st.markdown(f'<div class="step-title">{step4_title}</div>', unsafe_allow_html=True)
            
            self.results_viewer.render_results_interface()
            
            st.markdown('</div>', unsafe_allow_html=True)


def main():
    """主函数"""
    try:
        app = BERTopicWebUI()
        app.run()
    except Exception as e:
        st.error(f"应用启动失败: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
