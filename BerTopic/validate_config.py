#!/usr/bin/env python3
"""
配置文件验证工具
================
验证config.yaml的完整性和正确性
"""

import yaml
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            self.config_path = Path(__file__).parent / "config.yaml"
        else:
            self.config_path = Path(config_path)
        self.errors = []
        self.warnings = []
        
    def validate(self) -> bool:
        """执行验证"""
        print("🔍 开始验证配置文件...\n")
        
        # 加载配置
        config = self._load_config()
        if config is None:
            return False
        
        # 执行各项检查
        self._check_structure(config)
        self._check_file_paths(config)
        self._check_parameters(config)
        self._check_column_names(config)
        
        # 输出结果
        return self._report_results()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 成功加载配置文件: {self.config_path}\n")
            return config
        except FileNotFoundError:
            self.errors.append(f"配置文件不存在: {self.config_path}")
            return None
        except yaml.YAMLError as e:
            self.errors.append(f"YAML格式错误: {e}")
            return None
    
    def _check_structure(self, config: Dict[str, Any]):
        """检查配置结构"""
        required_sections = [
            'data_paths',
            'results_paths', 
            'data_processing',
            'bertopic_params',
            'visualization',
            'analysis',
            'system'
        ]
        
        for section in required_sections:
            if section not in config:
                self.errors.append(f"缺少必需的配置节: {section}")
        
        print("📋 配置结构检查完成")
    
    def _check_file_paths(self, config: Dict[str, Any]):
        """检查文件路径"""
        if 'data_paths' in config:
            for key, path in config['data_paths'].items():
                file_path = Path(path)
                if not file_path.exists():
                    self.warnings.append(f"数据文件不存在: {path}")
                else:
                    print(f"  ✓ {key}: {path}")
        
        print("📁 文件路径检查完成\n")
    
    def _check_parameters(self, config: Dict[str, Any]):
        """检查参数合理性"""
        if 'bertopic_params' in config:
            params = config['bertopic_params']
            
            # 检查min_topic_size
            if 'min_topic_size' in params:
                size = params['min_topic_size']
                if size < 5:
                    self.warnings.append(f"min_topic_size={size} 太小，可能产生噪声主题")
                elif size > 100:
                    self.warnings.append(f"min_topic_size={size} 太大，可能遗漏重要主题")
            
            # 检查nr_topics
            if 'nr_topics' in params:
                nr = params['nr_topics']
                if nr != "auto" and nr is not None:
                    if not isinstance(nr, int) or nr < 2:
                        self.errors.append(f"nr_topics必须是'auto'或>=2的整数，当前值: {nr}")
            
            # 检查embedding_model
            if 'embedding_model' in params:
                model = params['embedding_model']
                recommended_models = [
                    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                    'sentence-transformers/all-MiniLM-L6-v2',
                    'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
                ]
                if model not in recommended_models:
                    self.warnings.append(f"使用非标准嵌入模型: {model}")
        
        print("🔧 参数检查完成")
    
    def _check_column_names(self, config: Dict[str, Any]):
        """检查列名配置"""
        if 'data_processing' in config:
            proc = config['data_processing']
            
            # 检查文本列
            if 'text_column' in proc:
                text_col = proc['text_column']
                common_names = ['Incident', 'Unit_Text', 'Text', 'Content']
                if text_col not in common_names:
                    self.warnings.append(
                        f"文本列名'{text_col}'不常见，请确认数据中存在此列"
                    )
            
            # 检查元数据列
            if 'metadata_columns' in proc:
                cols = proc['metadata_columns']
                if len(cols) == 0:
                    self.warnings.append("未指定任何元数据列")
                
                # 检查框架列
                frame_cols = [c for c in cols if c.startswith('Frame_')]
                if len(frame_cols) == 0:
                    self.warnings.append("未找到框架列(Frame_*)，将无法生成框架分析")
        
        print("📊 列名检查完成\n")
    
    def _report_results(self) -> bool:
        """输出验证结果"""
        print("="*60)
        print("验证结果")
        print("="*60)
        
        if self.errors:
            print("\n❌ 错误 (必须修复):")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print("\n⚠️  警告 (建议检查):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✨ 完美！配置文件没有任何问题")
        elif not self.errors:
            print("\n✅ 配置文件可以使用，但建议处理警告")
        else:
            print("\n❌ 配置文件存在错误，请修复后再运行")
        
        print("="*60)
        return len(self.errors) == 0


def suggest_fix(config_path: str = None):
    """提供修复建议"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    print("\n💡 配置修复建议:")
    print("-"*40)
    print("1. 确保所有数据文件路径正确")
    print("2. 检查text_column是否与数据中的列名匹配")
    print("3. min_topic_size建议范围: 10-30")
    print("4. 对于中文数据，使用multilingual模型")
    print("-"*40)
    
    # 提供示例配置
    print("\n📝 最小可用配置示例:")
    print("""
data_paths:
  media_data: "data/媒体_最终分析数据库.xlsx"
  social_media_data: "data/社交媒体_最终分析数据库.xlsx"

results_paths:
  output_dir: "results"

data_processing:
  text_column: "Incident"
  metadata_columns: ["Source", "日期"]

bertopic_params:
  language: "multilingual"
  min_topic_size: 15
""")


def main():
    """主函数"""
    print("="*60)
    print("📋 BERTopic配置文件验证器")
    print("="*60 + "\n")
    
    validator = ConfigValidator()
    is_valid = validator.validate()
    
    if not is_valid:
        suggest_fix()
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1)