"""
文件上传组件 - KISS原则
=====================
简洁的文件上传和管理
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, List


class FileUploader:
    """文件上传管理器"""
    
    def __init__(self):
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def render_upload_interface(self) -> Dict[str, str]:
        """渲染上传界面"""
        st.markdown("#### 📁 上传数据文件")
        st.info("支持Excel文件(.xlsx)，需包含文本列")
        
        uploaded_files = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📰 主数据文件**")
            file1 = st.file_uploader(
                "上传第一个文件",
                type=['xlsx'],
                key="file1_uploader",
                help="必需的主数据文件"
            )
            
            if file1:
                path = self._save_file(file1, "media_data")
                uploaded_files['media_data'] = path
                self._show_file_info(file1, path)
        
        with col2:
            st.markdown("**📱 补充数据文件 (可选)**")
            file2 = st.file_uploader(
                "上传第二个文件",
                type=['xlsx'],
                key="file2_uploader",
                help="可选的补充数据文件"
            )
            
            if file2:
                path = self._save_file(file2, "social_media_data")
                uploaded_files['social_media_data'] = path
                self._show_file_info(file2, path)
        
        # 更新session_state
        if uploaded_files:
            if 'uploaded_files' not in st.session_state:
                st.session_state.uploaded_files = {}
            st.session_state.uploaded_files.update(uploaded_files)
            
            # 自动设置preset
            if 'current_preset' not in st.session_state:
                st.session_state.current_preset = "智能推荐"
        
        return st.session_state.get('uploaded_files', {})
    
    def _save_file(self, uploaded_file, file_key: str) -> str:
        """保存上传的文件"""
        file_path = self.temp_dir / f"{file_key}.xlsx"
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    
    def _show_file_info(self, uploaded_file, file_path: str):
        """显示文件信息"""
        try:
            df = pd.read_excel(file_path)
            st.success(f"✅ {len(df)} 条记录")
            
            with st.expander("📋 预览", expanded=False):
                st.dataframe(df.head(3))
                st.write(f"**列名:** {', '.join(df.columns.tolist()[:5])}")
                
        except Exception as e:
            st.error(f"❌ 读取失败: {e}")
    
    def get_data_summary(self) -> Dict[str, int]:
        """获取数据摘要"""
        summary = {'total_records': 0, 'files': 0}
        
        if 'uploaded_files' in st.session_state:
            for file_path in st.session_state.uploaded_files.values():
                try:
                    df = pd.read_excel(file_path)
                    summary['total_records'] += len(df)
                    summary['files'] += 1
                except:
                    pass
        
        return summary
