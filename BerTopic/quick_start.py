#!/usr/bin/env python3
"""
快速启动脚本 - 一键运行BERTopic分析
=====================================
包含环境检查、依赖安装提示、数据验证等功能
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python版本过低！需要Python 3.8或更高版本")
        print(f"   当前版本: {sys.version}")
        return False
    print(f"✅ Python版本: {sys.version.split()[0]}")
    return True


def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'bertopic',
        'pandas',
        'numpy',
        'sentence_transformers',
        'umap',
        'hdbscan',
        'plotly',
        'yaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n📦 请运行以下命令安装依赖:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True


def check_data_files():
    """检查数据文件"""
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    required_files = [
        "媒体_最终分析数据库.xlsx",
        "社交媒体_最终分析数据库.xlsx"
    ]
    
    if not data_dir.exists():
        print("❌ 未找到data文件夹")
        print("   请创建data文件夹并放入数据文件")
        data_dir.mkdir(exist_ok=True)
        return False
    
    missing_files = []
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print("❌ 缺少以下数据文件:")
        for file in missing_files:
            print(f"   - data/{file}")
        print("\n📁 请将上述文件放入data文件夹")
        return False
    
    print("✅ 数据文件就绪")
    return True


def check_config():
    """检查配置文件"""
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.yaml"
    if not config_path.exists():
        print("❌ 未找到config.yaml配置文件")
        return False
    print("✅ 配置文件存在")
    return True


def create_directories():
    """创建必要的目录"""
    script_dir = Path(__file__).parent
    dirs = ["data", "results", "results/trained_model"]
    for dir_path in dirs:
        (script_dir / dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ 目录结构已创建")


def run_analysis():
    """运行主分析程序"""
    print("\n" + "="*60)
    print("🚀 开始运行BERTopic分析...")
    print("="*60 + "\n")
    
    try:
        # 运行主程序
        script_dir = Path(__file__).parent
        main_script = script_dir / "main.py"
        subprocess.run([sys.executable, str(main_script)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 分析过程出错: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ 未找到main.py文件")
        return False


def main():
    """主函数"""
    print("="*60)
    print("🔍 BERTopic议题分析系统 - 环境检查")
    print("="*60)
    
    # 执行检查
    checks = [
        ("Python版本", check_python_version),
        ("依赖包", check_dependencies),
        ("配置文件", check_config),
        ("数据文件", check_data_files),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        if not check_func():
            all_passed = False
    
    if not all_passed:
        print("\n" + "="*60)
        print("⚠️  请先解决上述问题后再运行")
        print("="*60)
        sys.exit(1)
    
    # 创建目录
    create_directories()
    
    print("\n" + "="*60)
    print("✨ 所有检查通过！")
    print("="*60)
    
    # 询问是否继续
    response = input("\n是否开始分析？(y/n): ").strip().lower()
    if response == 'y':
        success = run_analysis()
        if success:
            print("\n" + "="*60)
            print("🎉 分析完成！请查看results文件夹中的结果")
            print("="*60)
    else:
        print("已取消运行")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断运行")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1)