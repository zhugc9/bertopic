"""
跨语言主题成分分析模块
====================
分析每个主题内部的语言构成和跨语言特性
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Tuple
from langdetect import detect
import re
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CrossLingualAnalyzer:
    """跨语言主题成分分析器 - SOTA & KISS实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化跨语言分析器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.results_paths = config['results_paths']
        
        # 语言检测和分类规则
        self.language_patterns = {
            'zh': re.compile(r'[\u4e00-\u9fff]'),      # 中文字符
            'ru': re.compile(r'[\u0400-\u04ff]'),      # 俄文字符  
            'en': re.compile(r'^[a-zA-Z\s\d\.,!?\-\'\"]*$')  # 英文字符（相对宽松）
        }
        
    def detect_document_language(self, text: str) -> str:
        """
        检测单个文档的主要语言
        
        Args:
            text: 文档文本
            
        Returns:
            语言代码 ('zh', 'ru', 'en', 'mixed', 'unknown')
        """
        if not text or len(text.strip()) < 3:
            return 'unknown'
        
        text = text.strip()
        
        # 统计各语言字符数量
        zh_chars = len(self.language_patterns['zh'].findall(text))
        ru_chars = len(self.language_patterns['ru'].findall(text))
        
        # 总字符数（排除空格和标点）
        total_chars = len(re.sub(r'[\s\d\.,!?\-\'\"()【】《》""''；：]', '', text))
        
        if total_chars == 0:
            return 'unknown'
        
        # 计算各语言字符比例
        zh_ratio = zh_chars / total_chars
        ru_ratio = ru_chars / total_chars
        other_ratio = 1 - zh_ratio - ru_ratio
        
        # 判断主要语言
        if zh_ratio > 0.3:
            if ru_ratio > 0.1 or other_ratio > 0.3:
                return 'mixed'  # 混合语言
            return 'zh'
        elif ru_ratio > 0.3:
            if zh_ratio > 0.1 or other_ratio > 0.3:
                return 'mixed'
            return 'ru'
        elif other_ratio > 0.8:
            return 'en'  # 假设其他字符主要是英文
        else:
            return 'mixed'
    
    def analyze_document_languages(self, documents: List[str]) -> List[str]:
        """
        批量分析文档语言
        
        Args:
            documents: 文档列表
            
        Returns:
            语言标签列表
        """
        logger.info("🌍 分析文档语言组成...")
        
        languages = []
        language_counts = {'zh': 0, 'ru': 0, 'en': 0, 'mixed': 0, 'unknown': 0}
        
        for i, doc in enumerate(documents):
            lang = self.detect_document_language(doc)
            languages.append(lang)
            language_counts[lang] += 1
            
            if i % 1000 == 0 and i > 0:
                logger.info(f"  → 已处理 {i}/{len(documents)} 个文档")
        
        # 统计结果
        total = len(documents)
        logger.info("  ✓ 语言分布统计:")
        for lang, count in language_counts.items():
            percentage = (count / total) * 100
            lang_name = {'zh': '中文', 'ru': '俄文', 'en': '英文', 'mixed': '混合', 'unknown': '未知'}[lang]
            logger.info(f"    {lang_name}: {count} ({percentage:.1f}%)")
        
        return languages
    
    def analyze_topic_language_composition(self,
                                         topics: List[int],
                                         document_languages: List[str]) -> pd.DataFrame:
        """
        分析每个主题的语言构成
        
        Args:
            topics: 主题标签列表
            document_languages: 文档语言列表
            
        Returns:
            主题语言构成DataFrame
        """
        logger.info("📊 分析主题语言构成...")
        
        # 创建数据框
        df = pd.DataFrame({
            'topic': topics,
            'language': document_languages
        })
        
        # 排除离群点
        df = df[df['topic'] != -1]
        
        if df.empty:
            logger.warning("  ⚠ 没有有效主题数据")
            return pd.DataFrame()
        
        # 计算每个主题的语言构成
        composition_data = []
        
        for topic_id in sorted(df['topic'].unique()):
            topic_docs = df[df['topic'] == topic_id]
            total_docs = len(topic_docs)
            
            if total_docs == 0:
                continue
            
            # 统计各语言文档数量
            lang_counts = topic_docs['language'].value_counts()
            
            composition = {
                'Topic': topic_id,
                'Total_Documents': total_docs,
                'Chinese_Count': lang_counts.get('zh', 0),
                'Russian_Count': lang_counts.get('ru', 0),
                'English_Count': lang_counts.get('en', 0),
                'Mixed_Count': lang_counts.get('mixed', 0),
                'Unknown_Count': lang_counts.get('unknown', 0),
            }
            
            # 计算百分比
            composition.update({
                'Chinese_Percentage': (composition['Chinese_Count'] / total_docs) * 100,
                'Russian_Percentage': (composition['Russian_Count'] / total_docs) * 100,
                'English_Percentage': (composition['English_Count'] / total_docs) * 100,
                'Mixed_Percentage': (composition['Mixed_Count'] / total_docs) * 100,
                'Unknown_Percentage': (composition['Unknown_Count'] / total_docs) * 100,
            })
            
            # 主要语言标签
            max_lang = max(lang_counts.items(), key=lambda x: x[1])
            composition['Dominant_Language'] = max_lang[0]
            composition['Dominance_Percentage'] = (max_lang[1] / total_docs) * 100
            
            # 分类主题类型
            zh_pct = composition['Chinese_Percentage']
            ru_pct = composition['Russian_Percentage']
            en_pct = composition['English_Percentage']
            
            if zh_pct > 70:
                topic_type = 'Chinese-Dominant'
            elif ru_pct > 70:
                topic_type = 'Russian-Dominant'
            elif en_pct > 70:
                topic_type = 'English-Dominant'
            elif zh_pct + ru_pct > 80:
                topic_type = 'Sino-Russian'
            elif abs(zh_pct - ru_pct) < 20 and zh_pct + ru_pct > 50:
                topic_type = 'Balanced-Multilingual'
            else:
                topic_type = 'Other-Multilingual'
            
            composition['Topic_Type'] = topic_type
            composition_data.append(composition)
        
        composition_df = pd.DataFrame(composition_data)
        
        if not composition_df.empty:
            logger.info(f"  ✓ 分析了 {len(composition_df)} 个主题的语言构成")
            
            # 统计主题类型分布
            type_counts = composition_df['Topic_Type'].value_counts()
            logger.info("  ✓ 主题类型分布:")
            for topic_type, count in type_counts.items():
                logger.info(f"    {topic_type}: {count} 个主题")
        
        return composition_df
    
    def generate_language_distribution_chart(self,
                                           composition_df: pd.DataFrame,
                                           output_path: str = None) -> str:
        """
        生成语言分布可视化图表
        
        Args:
            composition_df: 主题语言构成数据
            output_path: 输出路径
            
        Returns:
            生成的文件路径
        """
        if composition_df.empty:
            logger.warning("  ⚠ 数据为空，跳过图表生成")
            return ""
        
        logger.info("📈 生成语言分布图表...")
        
        # 设置图表样式
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cross-Lingual Topic Composition Analysis', fontsize=16, fontweight='bold')
        
        # 1. 主题类型分布饼图
        type_counts = composition_df['Topic_Type'].value_counts()
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']
        ax1.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
                colors=colors[:len(type_counts)], startangle=90)
        ax1.set_title('Topic Type Distribution', fontweight='bold')
        
        # 2. 语言比例堆叠条形图
        topics = composition_df['Topic'].tolist()
        zh_pct = composition_df['Chinese_Percentage'].tolist()
        ru_pct = composition_df['Russian_Percentage'].tolist()
        en_pct = composition_df['English_Percentage'].tolist()
        
        x = range(len(topics))
        ax2.bar(x, zh_pct, label='Chinese', color='#FF6B6B', alpha=0.8)
        ax2.bar(x, ru_pct, bottom=zh_pct, label='Russian', color='#4ECDC4', alpha=0.8)
        
        # 计算英文的底部位置
        en_bottom = [zh_pct[i] + ru_pct[i] for i in range(len(topics))]
        ax2.bar(x, en_pct, bottom=en_bottom, label='English', color='#45B7D1', alpha=0.8)
        
        ax2.set_xlabel('Topic ID')
        ax2.set_ylabel('Language Percentage (%)')
        ax2.set_title('Language Composition by Topic', fontweight='bold')
        ax2.legend()
        ax2.set_xticks(x[::max(1, len(x)//10)])  # 显示部分x轴标签以避免重叠
        ax2.set_xticklabels([f'T{topics[i]}' for i in range(0, len(topics), max(1, len(topics)//10))])
        
        # 3. 主导语言分布
        dominant_langs = composition_df['Dominant_Language'].value_counts()
        lang_colors = {'zh': '#FF6B6B', 'ru': '#4ECDC4', 'en': '#45B7D1', 'mixed': '#96CEB4', 'unknown': '#DDA0DD'}
        colors_for_dominant = [lang_colors.get(lang, '#gray') for lang in dominant_langs.index]
        
        bars = ax3.bar(range(len(dominant_langs)), dominant_langs.values, color=colors_for_dominant, alpha=0.8)
        ax3.set_xlabel('Dominant Language')
        ax3.set_ylabel('Number of Topics')
        ax3.set_title('Dominant Language Distribution', fontweight='bold')
        ax3.set_xticks(range(len(dominant_langs)))
        lang_labels = {'zh': 'Chinese', 'ru': 'Russian', 'en': 'English', 'mixed': 'Mixed', 'unknown': 'Unknown'}
        ax3.set_xticklabels([lang_labels.get(lang, lang) for lang in dominant_langs.index])
        
        # 在柱状图上添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 4. 语言平衡度散点图
        sino_russian_topics = composition_df[
            (composition_df['Chinese_Percentage'] > 10) & 
            (composition_df['Russian_Percentage'] > 10)
        ]
        
        if not sino_russian_topics.empty:
            scatter = ax4.scatter(sino_russian_topics['Chinese_Percentage'], 
                                sino_russian_topics['Russian_Percentage'],
                                c=sino_russian_topics['English_Percentage'], 
                                cmap='viridis', alpha=0.7, s=60)
            
            # 添加对角线表示平衡点
            max_val = max(sino_russian_topics['Chinese_Percentage'].max(), 
                         sino_russian_topics['Russian_Percentage'].max())
            ax4.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Balance Line')
            
            ax4.set_xlabel('Chinese Percentage (%)')
            ax4.set_ylabel('Russian Percentage (%)')
            ax4.set_title('Sino-Russian Topic Balance\n(Color = English %)', fontweight='bold')
            ax4.legend()
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('English Percentage (%)')
        else:
            ax4.text(0.5, 0.5, 'No Sino-Russian\nTopics Found', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Sino-Russian Topic Balance', fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if output_path is None:
            output_path = Path(self.results_paths['output_dir']) / '跨语言分析图'
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为PNG和PDF
        png_path = output_path.with_suffix('.png')
        pdf_path = output_path.with_suffix('.pdf')
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"  ✓ 跨语言分析图表已保存: {png_path}")
        logger.info(f"  ✓ PDF版本已保存: {pdf_path}")
        
        return str(png_path)
    
    def save_language_composition_report(self,
                                       composition_df: pd.DataFrame,
                                       document_languages: List[str]) -> str:
        """
        保存语言构成报告
        
        Args:
            composition_df: 主题语言构成数据
            document_languages: 文档语言列表
        
        Returns:
            保存的文件路径
        """
        output_path = Path(self.results_paths['cross_lingual_file'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if not composition_df.empty:
                # 添加总体统计信息
                total_stats = {
                    'Topic': 'OVERALL',
                    'Total_Documents': len(document_languages),
                    'Chinese_Count': document_languages.count('zh'),
                    'Russian_Count': document_languages.count('ru'),
                    'English_Count': document_languages.count('en'),
                    'Mixed_Count': document_languages.count('mixed'),
                    'Unknown_Count': document_languages.count('unknown'),
                }
                
                # 计算总体百分比
                total = total_stats['Total_Documents']
                total_stats.update({
                    'Chinese_Percentage': (total_stats['Chinese_Count'] / total) * 100,
                    'Russian_Percentage': (total_stats['Russian_Count'] / total) * 100,
                    'English_Percentage': (total_stats['English_Count'] / total) * 100,
                    'Mixed_Percentage': (total_stats['Mixed_Count'] / total) * 100,
                    'Unknown_Percentage': (total_stats['Unknown_Count'] / total) * 100,
                    'Dominant_Language': max(
                        [('zh', total_stats['Chinese_Count']),
                         ('ru', total_stats['Russian_Count']),
                         ('en', total_stats['English_Count'])],
                        key=lambda x: x[1]
                    )[0],
                    'Topic_Type': 'Overall_Statistics'
                })
                
                # 合并数据
                final_df = pd.concat([
                    composition_df,
                    pd.DataFrame([total_stats])
                ], ignore_index=True)
                
                # 保存CSV
                final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                logger.info(f"  ✓ 跨语言构成报告已保存: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"  ✗ 保存跨语言报告失败: {e}")
            return ""
    
    def run_full_cross_lingual_analysis(self,
                                       documents: List[str],
                                       topics: List[int]) -> Dict[str, Any]:
        """
        运行完整的跨语言分析
        
        Args:
            documents: 文档列表
            topics: 主题列表
            
        Returns:
            完整的跨语言分析结果
        """
        logger.info("🚀 开始完整跨语言主题成分分析...")
        
        # 1. 分析文档语言
        document_languages = self.analyze_document_languages(documents)
        
        # 2. 分析主题语言构成
        composition_df = self.analyze_topic_language_composition(topics, document_languages)
        
        # 3. 生成可视化图表
        chart_path = ""
        if not composition_df.empty:
            chart_path = self.generate_language_distribution_chart(composition_df)
        
        # 4. 保存分析报告
        report_path = self.save_language_composition_report(composition_df, document_languages)
        
        # 5. 生成分析摘要
        summary = self._generate_analysis_summary(composition_df, document_languages)
        
        results = {
            'document_languages': document_languages,
            'composition_df': composition_df,
            'chart_path': chart_path,
            'report_path': report_path,
            'summary': summary
        }
        
        logger.info("✅ 跨语言主题成分分析完成")
        return results
    
    def _generate_analysis_summary(self,
                                  composition_df: pd.DataFrame,
                                  document_languages: List[str]) -> Dict[str, Any]:
        """生成分析摘要"""
        if composition_df.empty:
            return {}
        
        # 统计信息
        total_docs = len(document_languages)
        total_topics = len(composition_df)
        
        # 语言分布
        lang_dist = {
            'chinese': document_languages.count('zh'),
            'russian': document_languages.count('ru'),
            'english': document_languages.count('en'),
            'mixed': document_languages.count('mixed'),
            'unknown': document_languages.count('unknown')
        }
        
        # 主题类型分布
        topic_types = composition_df['Topic_Type'].value_counts().to_dict()
        
        # 关键洞察
        insights = []
        
        # 中俄共通议题
        sino_russian_topics = composition_df[
            (composition_df['Chinese_Percentage'] > 20) & 
            (composition_df['Russian_Percentage'] > 20)
        ]
        insights.append(f"发现 {len(sino_russian_topics)} 个中俄共通议题")
        
        # 语言主导议题
        zh_dominant = len(composition_df[composition_df['Dominant_Language'] == 'zh'])
        ru_dominant = len(composition_df[composition_df['Dominant_Language'] == 'ru'])
        en_dominant = len(composition_df[composition_df['Dominant_Language'] == 'en'])
        
        insights.append(f"中文主导议题: {zh_dominant} 个")
        insights.append(f"俄文主导议题: {ru_dominant} 个")
        insights.append(f"英文主导议题: {en_dominant} 个")
        
        return {
            'total_documents': total_docs,
            'total_topics': total_topics,
            'language_distribution': lang_dist,
            'topic_type_distribution': topic_types,
            'key_insights': insights
        }
