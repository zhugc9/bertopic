"""
动态主题演化分析模块
==================
实现BERTopic的官方动态主题建模功能，分析主题随时间的演化
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DynamicTopicEvolution:
    """动态主题演化分析器 - SOTA & KISS实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化动态主题演化分析器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.analysis_config = config['analysis']['time_analysis']
        self.results_paths = config['results_paths']
        
    def prepare_temporal_data(self, 
                            documents: List[str],
                            metadata_df: pd.DataFrame) -> Tuple[List[str], List[datetime]]:
        """
        准备时间序列数据
        
        Args:
            documents: 文档列表
            metadata_df: 元数据DataFrame
            
        Returns:
            (文档列表, 时间戳列表)
        """
        time_column = self.analysis_config['time_column']
        
        if time_column not in metadata_df.columns:
            logger.warning(f"  ⚠ 时间列 '{time_column}' 不存在")
            return documents, []
        
        # 确保时间列是datetime格式
        try:
            timestamps = pd.to_datetime(metadata_df[time_column])
            # 过滤掉无效时间戳
            valid_mask = timestamps.notna()
            valid_documents = [doc for i, doc in enumerate(documents) if valid_mask.iloc[i]]
            valid_timestamps = timestamps[valid_mask].tolist()
            
            logger.info(f"  ✓ 准备时间序列数据: {len(valid_documents)} 个有效文档")
            logger.info(f"  ✓ 时间范围: {min(valid_timestamps)} ~ {max(valid_timestamps)}")
            
            return valid_documents, valid_timestamps
            
        except Exception as e:
            logger.error(f"  ✗ 时间数据预处理失败: {e}")
            return documents, []
    
    def analyze_dynamic_topics(self,
                             topic_model,
                             documents: List[str],
                             timestamps: List[datetime]) -> pd.DataFrame:
        """
        执行动态主题分析
        
        Args:
            topic_model: 训练好的BERTopic模型
            documents: 文档列表
            timestamps: 时间戳列表
            
        Returns:
            主题时间演化DataFrame
        """
        if not timestamps:
            logger.warning("  ⚠ 没有有效时间戳，跳过动态分析")
            return pd.DataFrame()
        
        logger.info("🕐 执行动态主题演化分析...")
        
        try:
            # 使用BERTopic的topics_over_time方法
            bins = self.analysis_config.get('bins', 10)
            topics_over_time = topic_model.topics_over_time(
                
                timestamps=timestamps,
                nr_bins=bins,
                datetime_format=None,  # 已经是datetime对象
                evolution_tuning=True,  # 启用演化调优
                global_tuning=True     # 启用全局调优
            )
            
            logger.info(f"  ✓ 动态分析完成: {len(topics_over_time)} 个时间点")
            return topics_over_time
            
        except Exception as e:
            logger.error(f"  ✗ 动态主题分析失败: {e}")
            return pd.DataFrame()
    
    def analyze_topic_birth_death(self,
                                 topics_over_time: pd.DataFrame) -> Dict[str, Any]:
        """
        分析主题的诞生和消亡
        
        Args:
            topics_over_time: 主题时间演化数据
            
        Returns:
            主题诞生消亡分析结果
        """
        if topics_over_time.empty:
            return {}
        
        logger.info("📊 分析主题诞生和消亡...")
        
        results = {
            'topic_births': {},
            'topic_deaths': {},
            'persistent_topics': [],
            'ephemeral_topics': []
        }
        
        try:
            # 获取所有主题
            topics = topics_over_time['Topic'].unique()
            topics = [t for t in topics if t != -1]  # 排除离群点
            
            for topic in topics:
                topic_data = topics_over_time[topics_over_time['Topic'] == topic]
                topic_data = topic_data.sort_values('Timestamp')
                
                # 找到主题首次出现和最后出现的时间
                first_appearance = topic_data.iloc[0]['Timestamp']
                last_appearance = topic_data.iloc[-1]['Timestamp']
                
                # 计算主题活跃度（非零频率的时间点比例）
                activity_ratio = len(topic_data[topic_data['Frequency'] > 0]) / len(topic_data)
                
                # 记录诞生时间
                results['topic_births'][topic] = first_appearance
                
                # 判断是否"死亡"（最近几个时间点频率为0）
                recent_data = topic_data.tail(3)
                if all(recent_data['Frequency'] == 0):
                    results['topic_deaths'][topic] = last_appearance
                
                # 分类持续性话题和短暂话题
                if activity_ratio > 0.5:
                    results['persistent_topics'].append(topic)
                else:
                    results['ephemeral_topics'].append(topic)
            
            logger.info(f"  ✓ 发现 {len(results['topic_births'])} 个主题诞生")
            logger.info(f"  ✓ 发现 {len(results['topic_deaths'])} 个主题消亡")
            logger.info(f"  ✓ 持续性主题: {len(results['persistent_topics'])} 个")
            logger.info(f"  ✓ 短暂性主题: {len(results['ephemeral_topics'])} 个")
            
        except Exception as e:
            logger.error(f"  ✗ 主题诞生消亡分析失败: {e}")
        
        return results
    
    def detect_topic_evolution_patterns(self,
                                      topics_over_time: pd.DataFrame) -> Dict[str, Any]:
        """
        检测主题演化模式
        
        Args:
            topics_over_time: 主题时间演化数据
            
        Returns:
            演化模式分析结果
        """
        if topics_over_time.empty:
            return {}
        
        logger.info("🔍 检测主题演化模式...")
        
        patterns = {
            'rising_topics': [],      # 上升趋势主题
            'declining_topics': [],   # 下降趋势主题
            'stable_topics': [],      # 稳定主题
            'volatile_topics': [],    # 波动主题
            'seasonal_topics': []     # 季节性主题
        }
        
        try:
            topics = topics_over_time['Topic'].unique()
            topics = [t for t in topics if t != -1]
            
            for topic in topics:
                topic_data = topics_over_time[topics_over_time['Topic'] == topic]
                topic_data = topic_data.sort_values('Timestamp')
                frequencies = topic_data['Frequency'].values
                
                if len(frequencies) < 3:
                    continue
                
                # 计算趋势
                x = np.arange(len(frequencies))
                slope = np.polyfit(x, frequencies, 1)[0]
                
                # 计算波动性（标准差）
                volatility = np.std(frequencies)
                mean_freq = np.mean(frequencies)
                cv = volatility / (mean_freq + 1e-6)  # 变异系数
                
                # 分类演化模式
                if slope > 0.01:
                    patterns['rising_topics'].append((topic, slope))
                elif slope < -0.01:
                    patterns['declining_topics'].append((topic, slope))
                elif cv < 0.3:
                    patterns['stable_topics'].append((topic, cv))
                elif cv > 0.8:
                    patterns['volatile_topics'].append((topic, cv))
                
                # 检测季节性（简单实现：检查周期性峰值）
                if len(frequencies) >= 6:
                    autocorr = np.corrcoef(frequencies[:-3], frequencies[3:])[0, 1]
                    if autocorr > 0.6:
                        patterns['seasonal_topics'].append((topic, autocorr))
            
            # 排序结果
            patterns['rising_topics'].sort(key=lambda x: x[1], reverse=True)
            patterns['declining_topics'].sort(key=lambda x: x[1])
            patterns['stable_topics'].sort(key=lambda x: x[1])
            patterns['volatile_topics'].sort(key=lambda x: x[1], reverse=True)
            patterns['seasonal_topics'].sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"  ✓ 上升趋势主题: {len(patterns['rising_topics'])} 个")
            logger.info(f"  ✓ 下降趋势主题: {len(patterns['declining_topics'])} 个")
            logger.info(f"  ✓ 稳定主题: {len(patterns['stable_topics'])} 个")
            logger.info(f"  ✓ 波动主题: {len(patterns['volatile_topics'])} 个")
            logger.info(f"  ✓ 季节性主题: {len(patterns['seasonal_topics'])} 个")
            
        except Exception as e:
            logger.error(f"  ✗ 演化模式检测失败: {e}")
        
        return patterns
    
    def save_evolution_analysis(self,
                               topics_over_time: pd.DataFrame,
                               birth_death_analysis: Dict[str, Any],
                               evolution_patterns: Dict[str, Any]) -> str:
        """
        保存演化分析结果
        
        Args:
            topics_over_time: 主题时间演化数据
            birth_death_analysis: 诞生消亡分析
            evolution_patterns: 演化模式分析
            
        Returns:
            保存的文件路径
        """
        output_path = Path(self.results_paths['output_dir']) / 'dynamic_evolution_analysis.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 创建综合分析结果
            analysis_results = []
            
            if not topics_over_time.empty:
                topics = topics_over_time['Topic'].unique()
                topics = [t for t in topics if t != -1]
                
                for topic in topics:
                    topic_data = topics_over_time[topics_over_time['Topic'] == topic]
                    
                    result = {
                        'Topic': topic,
                        'First_Appearance': birth_death_analysis.get('topic_births', {}).get(topic, ''),
                        'Last_Appearance': topic_data['Timestamp'].max(),
                        'Total_Frequency': topic_data['Frequency'].sum(),
                        'Average_Frequency': topic_data['Frequency'].mean(),
                        'Peak_Frequency': topic_data['Frequency'].max(),
                        'Activity_Duration': len(topic_data[topic_data['Frequency'] > 0]),
                        'Is_Persistent': topic in birth_death_analysis.get('persistent_topics', []),
                        'Evolution_Pattern': self._get_topic_pattern(topic, evolution_patterns)
                    }
                    analysis_results.append(result)
            
            # 保存为CSV
            if analysis_results:
                df = pd.DataFrame(analysis_results)
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                logger.info(f"  ✓ 演化分析结果已保存: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"  ✗ 保存演化分析失败: {e}")
            return ""
    
    def _get_topic_pattern(self, topic: int, patterns: Dict[str, Any]) -> str:
        """获取主题的演化模式标签"""
        for pattern_name, topic_list in patterns.items():
            if any(t[0] == topic for t in topic_list if isinstance(t, tuple)):
                return pattern_name.replace('_topics', '')
            elif topic in topic_list:
                return pattern_name.replace('_topics', '')
        return 'unknown'
    
    def run_full_evolution_analysis(self,
                                   topic_model,
                                   documents: List[str],
                                   metadata_df: pd.DataFrame) -> Dict[str, Any]:
        """
        运行完整的动态演化分析
        
        Args:
            topic_model: 训练好的BERTopic模型
            documents: 文档列表
            metadata_df: 元数据
            
        Returns:
            完整的演化分析结果
        """
        logger.info("🚀 开始完整动态主题演化分析...")
        
        # 1. 准备时间序列数据
        temporal_docs, timestamps = self.prepare_temporal_data(documents, metadata_df)
        
        if not timestamps:
            logger.warning("  ⚠ 无法进行动态分析：缺少有效时间数据")
            return {}
        
        # 2. 执行动态主题分析
        topics_over_time = self.analyze_dynamic_topics(topic_model, temporal_docs, timestamps)
        
        if topics_over_time.empty:
            logger.warning("  ⚠ 动态分析结果为空")
            return {}
        
        # 3. 分析主题诞生和消亡
        birth_death_analysis = self.analyze_topic_birth_death(topics_over_time)
        
        # 4. 检测演化模式
        evolution_patterns = self.detect_topic_evolution_patterns(topics_over_time)
        
        # 5. 保存分析结果
        saved_path = self.save_evolution_analysis(
            topics_over_time, birth_death_analysis, evolution_patterns
        )
        
        # 6. 组织返回结果
        results = {
            'topics_over_time': topics_over_time,
            'birth_death_analysis': birth_death_analysis,
            'evolution_patterns': evolution_patterns,
            'saved_path': saved_path,
            'summary': {
                'total_topics': len(topics_over_time['Topic'].unique()) - 1,  # 排除-1
                'time_points': len(topics_over_time['Timestamp'].unique()),
                'analysis_period': f"{min(timestamps)} to {max(timestamps)}"
            }
        }
        
        logger.info("✅ 动态主题演化分析完成")
        return results
