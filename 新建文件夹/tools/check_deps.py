#!/usr/bin/env python3
"""
依赖检查脚本
============
检查BERTopic项目的所有依赖是否正确安装
"""

import sys
import importlib
from typing import Dict, List, Tuple


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

# 两阶段分析依赖
TUNING_DEPS = {
    'optuna': '超参数优化',
    'gensim': '主题一致性评估'
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

def run_dependency_check() -> bool:
    """执行完整的依赖检查流程"""
    print("=" * 60)
    print("🎓 BERTopic项目依赖检查")
    print("=" * 60)

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
        return False

    if total_passed == total_deps:
        print("\n🎉 所有依赖检查通过，系统完全就绪！")
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

    print("\n" + "=" * 60)
    return core_ok


def main():
    """CLI入口，保持向后兼容"""
    success = run_dependency_check()
    return sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
