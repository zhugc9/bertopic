#!/usr/bin/env python3
"""
依赖检查脚本
============
检查BERTopic项目的所有依赖是否正确安装
"""

import sys
import importlib
from typing import Dict, List, Tuple
from pathlib import Path
import yaml


def _ensure_utf8_output():
    """Avoid UnicodeEncodeError when console encoding is GBK."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ensure_utf8_output()

# 核心依赖 (必需)
CORE_DEPS = {
    'bertopic': '主题建模核心',
    'sentence_transformers': '句子嵌入模型',
    'umap': '降维算法',
    'hdbscan': '聚类算法',
    'sklearn': '基础机器学习工具',
    'pandas': '数据处理',
    'numpy': '数值计算',
    'plotly': '交互可视化',
    'matplotlib': '基础绘图',
    'yaml': '配置解析',
    'openpyxl': 'Excel读写支持',
}

# 超参数优化依赖
TUNING_DEPS = {
    'optuna': '超参数优化'
}

# 多语言处理依赖
MULTILINGUAL_DEPS = {
    'spacy': '自然语言处理',
    'langdetect': '语言检测',
    'jieba': '中文分词',
    'pymorphy2': '俄语形态学'
}

# 高级可视化与分析依赖
ADVANCED_DEPS = {
    'seaborn': '统计可视化',
    'networkx': '网络图分析',
    'scipy': '科学计算',
    'tqdm': '进度条显示',
    'requests': '网络请求工具',
}

def check_dependency(module_name: str, description: str) -> Tuple[bool, str]:
    """检查单个依赖"""
    try:
        importlib.import_module(module_name)
        return True, f"✅ {description}"
    except ImportError as e:
        return False, f"❌ {description} - 未安装"
    except Exception as e:
        return False, f"⚠️ {description} - 导入错误: {e}"

def check_dependencies_group(deps: Dict[str, str], group_name: str) -> Tuple[int, int]:
    """检查依赖组"""
    print(f"\n🔍 {group_name}:")
    print("-" * 40)
    
    passed = 0
    total = len(deps)
    
    for module, desc in deps.items():
        success, message = check_dependency(module, desc)
        print(f"  {message}")
        if success:
            passed += 1
    
    return passed, total

def check_python_version() -> bool:
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python版本过低！需要Python 3.8或更高版本")
        print(f"   当前版本: {sys.version}")
        return False
    print(f"✅ Python版本: {sys.version.split()[0]}")
    return True

def validate_config(config_path: Path = None) -> bool:
    """验证配置文件是否包含关键设置"""
    if config_path is None:
        # 从 tools/ 目录向上找到项目根目录
        config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        print("❌ 未找到 config.yaml 配置文件")
        return False

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f) or {}

        # 必填块
        if 'output_settings' not in raw_config:
            print("❌ 缺少 output_settings：请在 config.yaml 填写输出目录和中文文件名")
            return False

        if 'topic' not in raw_config:
            print("❌ 缺少 topic：至少需要设置 min_documents_per_topic、expected_topics、text_language")
            return False

        # 至少一个数据路径
        data_cfg = raw_config.get('data', {}).get('files', {})
        if not any(data_cfg.get(k) for k in ('traditional_media', 'social_media')):
            print("❌ data.files 里没有任何有效路径，请填写最少一个 Excel 文件")
            return False

        print("✅ 配置文件验证通过")
        return True

    except yaml.YAMLError as e:
        print(f"❌ 配置文件格式错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 配置文件验证失败: {e}")
        return False

def check_data_files() -> bool:
    """检查数据文件"""
    try:
        # 从 tools/ 目录向上找到项目根目录
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查配置的数据路径
        data_paths = config.get('data_paths', {})
        media_data = data_paths.get('media_data')
        social_data = data_paths.get('social_media_data')

        # 兼容用户友好配置
        if not media_data and not social_data:
            friendly_paths = config.get('data', {}).get('files', {})
            media_data = friendly_paths.get('traditional_media')
            social_data = friendly_paths.get('social_media')
        
        found_files = 0
        if media_data and media_data != "null" and Path(media_data).exists():
            found_files += 1
            print(f"✅ 找到传统媒体数据: {media_data}")
            
        if social_data and social_data != "null" and Path(social_data).exists():
            found_files += 1
            print(f"✅ 找到社交媒体数据: {social_data}")
        
        if found_files == 0:
            print("❌ 未找到数据文件")
            print("   请在config.yaml中正确配置data.files")
            return False
        
        print(f"✅ 数据文件检查通过: {found_files} 个")
        return True
        
    except Exception as e:
        print(f"❌ 检查数据文件失败: {e}")
        return False

def run_dependency_check() -> bool:
    """执行完整的环境诊断流程"""
    print("=" * 60)
    print("🎓 BERTopic项目环境诊断")
    print("=" * 60)

    # 1. 检查Python版本
    print("\n🐍 Python版本检查:")
    print("-" * 40)
    python_ok = check_python_version()
    if not python_ok:
        print("\n⚠️ Python版本不满足要求，无法继续检查")
        return False

    total_passed = 0
    total_deps = 0

    # 检查核心依赖
    passed, total = check_dependencies_group(CORE_DEPS, "核心依赖 (必需)")
    total_passed += passed
    total_deps += total
    core_ok = passed == total

    # 检查两阶段分析依赖
    passed, total = check_dependencies_group(TUNING_DEPS, "两阶段分析依赖")
    total_passed += passed
    total_deps += total
    tuning_ok = passed == total

    # 检查多语言处理依赖
    passed, total = check_dependencies_group(MULTILINGUAL_DEPS, "多语言处理依赖")
    total_passed += passed
    total_deps += total
    multilingual_ok = passed == total

    # 检查高级功能依赖
    passed, total = check_dependencies_group(ADVANCED_DEPS, "高级功能依赖")
    total_passed += passed
    total_deps += total
    advanced_ok = passed == total

    # 总结报告
    print("\n" + "=" * 60)
    print("📊 检查结果总结")
    print("=" * 60)

    print(f"总体通过率: {total_passed}/{total_deps} ({total_passed/total_deps*100:.1f}%)")

    print("\n🎯 功能可用性:")
    print(f"  基础分析功能: {'✅ 可用' if core_ok else '❌ 不可用'}")
    print(f"  两阶段调参: {'✅ 可用' if tuning_ok else '❌ 不可用'}")
    print(f"  多语言处理: {'✅ 可用' if multilingual_ok else '❌ 不可用'}")
    print(f"  高级可视化: {'✅ 可用' if advanced_ok else '❌ 不可用'}")

    if not core_ok:
        print("\n⚠️ 核心依赖缺失，系统无法正常运行")
        print("💡 请运行: pip install -r requirements.txt")
    elif total_passed == total_deps:
        print("\n所有依赖检查通过，系统就绪")
    else:
        print(f"\n✅ 核心功能可用，{total_deps - total_passed}个可选依赖缺失")
        print("💡 如需完整功能，请运行: pip install -r requirements.txt")

    # 语言模型检查
    print("\n🌍 语言模型检查:")
    models = [
        ('zh_core_web_sm', '中文模型'),
        ('en_core_web_sm', '英文模型'),
        ('ru_core_news_sm', '俄文模型')
    ]

    for model_name, desc in models:
        try:
            import spacy
            spacy.load(model_name)
            print(f"  ✅ {desc}")
        except Exception:
            print(f"  ❌ {desc} - 未安装")

    # 配置文件检查
    print("\n📝 配置文件检查:")
    print("-" * 40)
    config_ok = validate_config()

    # 数据文件检查
    print("\n📊 数据文件检查:")
    print("-" * 40)
    data_ok = check_data_files()

    # 最终总结
    print("\n" + "=" * 60)
    print("📋 完整诊断结果")
    print("=" * 60)
    print(f"  Python版本: {'✅ 通过' if python_ok else '❌ 失败'}")
    print(f"  核心依赖: {'✅ 通过' if core_ok else '❌ 失败'}")
    print(f"  配置文件: {'✅ 通过' if config_ok else '❌ 失败'}")
    print(f"  数据文件: {'✅ 通过' if data_ok else '❌ 失败'}")

    all_ok = core_ok and config_ok and data_ok
    if all_ok:
        print("\n🎉 所有检查通过，系统可以正常运行！")
    else:
        print("\n⚠️ 部分检查未通过，请根据上述提示修复问题")
    
    print("=" * 60)
    return all_ok

def main():
    """CLI入口，保持向后兼容"""
    success = run_dependency_check()
    return sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()