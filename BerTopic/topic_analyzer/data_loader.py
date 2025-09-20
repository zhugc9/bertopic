"""
数据加载与预处理模块
====================
负责读取Excel数据并准备用于BERTopic分析的文档
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器类 - KISS原则的简洁实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据加载器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.media_path = config['data_paths']['media_data']
        self.social_media_path = config['data_paths']['social_media_data']
        self.text_column = config['data_processing']['text_column']
        self.metadata_columns = config['data_processing']['metadata_columns']
        
    def load_and_prepare_data(self) -> Tuple[List[str], pd.DataFrame]:
        """
        加载并准备数据
        
        Returns:
            documents: 文本列表
            metadata_df: 元数据DataFrame
        """
        logger.info("开始加载数据...")
        
        # 1. 读取数据文件
        media_df = self._load_excel(self.media_path, "媒体数据")
        social_df = self._load_excel(self.social_media_path, "社交媒体数据")
        
        # 2. 合并数据
        combined_df = self._merge_dataframes(media_df, social_df)
        logger.info(f"合并后总数据量: {len(combined_df)} 条")
        
        # 3. 数据清洗
        combined_df = self._clean_data(combined_df)
        
        # 4. 提取文档和元数据
        documents, metadata_df = self._extract_documents_and_metadata(combined_df)
        
        logger.info(f"✅ 数据准备完成: {len(documents)} 个有效文档")
        return documents, metadata_df
    
    def _load_excel(self, file_path: str, data_name: str) -> pd.DataFrame:
        """
        读取Excel文件
        
        Args:
            file_path: 文件路径
            data_name: 数据名称（用于日志）
            
        Returns:
            DataFrame
        """
        try:
            df = pd.read_excel(file_path)
            logger.info(f"  ✓ {data_name}: {len(df)} 条记录")
            return df
        except FileNotFoundError:
            logger.error(f"  ✗ 找不到文件: {file_path}")
            raise
        except Exception as e:
            logger.error(f"  ✗ 读取{data_name}失败: {e}")
            raise
    
    def _merge_dataframes(self, media_df: pd.DataFrame, 
                         social_df: pd.DataFrame) -> pd.DataFrame:
        """
        合并两个数据框
        
        Args:
            media_df: 媒体数据
            social_df: 社交媒体数据
            
        Returns:
            合并后的DataFrame
        """
        # 确保列名一致性
        media_df = self._standardize_columns(media_df, "媒体")
        social_df = self._standardize_columns(social_df, "社交媒体")
        
        # 根据配置选择合并策略
        merge_strategy = self.config['data_processing'].get('merge_strategy', 'concat')
        
        if merge_strategy == 'concat':
            # 简单拼接（KISS原则）
            combined_df = pd.concat([media_df, social_df], 
                                   ignore_index=True, 
                                   sort=False)
        else:
            # 其他合并策略可在此扩展
            combined_df = pd.concat([media_df, social_df], 
                                   ignore_index=True, 
                                   sort=False)
        
        return combined_df
    
    def _standardize_columns(self, df: pd.DataFrame, source_type: str) -> pd.DataFrame:
        """
        标准化列名并添加来源标识
        
        Args:
            df: 原始DataFrame
            source_type: 来源类型
            
        Returns:
            标准化后的DataFrame
        """
        # 如果没有Source列，添加来源标识
        if 'Source' not in df.columns:
            df['Source_Type'] = source_type
        
        # 确保必要的列存在
        required_columns = [self.text_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"  ⚠ {source_type}数据缺少列: {missing_columns}")
            # 尝试查找替代列
            if self.text_column not in df.columns:
                # 查找可能的文本列
                text_candidates = ['Unit_Text', 'Text', 'Content', 'incident', 'text']
                for candidate in text_candidates:
                    if candidate in df.columns:
                        df[self.text_column] = df[candidate]
                        logger.info(f"  → 使用 '{candidate}' 作为文本列")
                        break
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            df: 原始DataFrame
            
        Returns:
            清洗后的DataFrame
        """
        logger.info("执行数据清洗...")
        
        # 1. 删除文本列为空的行
        initial_count = len(df)
        df = df.dropna(subset=[self.text_column])
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.info(f"  • 删除 {removed_count} 条空文本记录")
        
        # 2. 文本清洗
        df[self.text_column] = df[self.text_column].astype(str).str.strip()
        
        # 3. 删除过短的文本（少于10个字符）
        df = df[df[self.text_column].str.len() >= 10]
        
        # 4. 处理日期列（如果存在）
        if '日期' in df.columns:
            try:
                df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
                df = df.dropna(subset=['日期'])  # 删除无效日期
            except:
                logger.warning("  ⚠ 日期格式转换失败，保留原始格式")
        
        # 5. 重置索引
        df = df.reset_index(drop=True)
        
        logger.info(f"  ✓ 清洗后保留 {len(df)} 条有效记录")
        return df
    
    def _extract_documents_and_metadata(self, 
                                       df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """
        提取文档列表和元数据
        
        Args:
            df: 清洗后的DataFrame
            
        Returns:
            documents: 文本列表
            metadata_df: 元数据DataFrame
        """
        # 提取文档
        documents = df[self.text_column].tolist()
        
        # 准备元数据
        # 只保留配置中指定的列，且这些列在数据中实际存在
        available_columns = [col for col in self.metadata_columns if col in df.columns]
        missing_columns = [col for col in self.metadata_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"  ⚠ 以下元数据列不存在: {missing_columns}")
        
        metadata_df = df[available_columns].copy()
        
        # 添加文档ID
        metadata_df['doc_id'] = range(len(metadata_df))
        
        # 添加原始文本（用于后续分析）
        metadata_df['original_text'] = documents
        
        logger.info(f"  ✓ 保留 {len(available_columns)} 个元数据列")
        
        return documents, metadata_df