#!/usr/bin/env python3
"""
BERTopic主题分析系统 - 统一入口
===============================
包含环境检查、配置验证、分析执行的完整流程
"""

import os
import sys
import yaml
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


def check_python_version() -> bool:
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python版本过低！需要Python 3.8或更高版本")
        print(f"   当前版本: {sys.version}")
        return False
    print(f"✅ Python版本: {sys.version.split()[0]}")
    return True


def check_dependencies() -> bool:
    """检查依赖包"""
    required_packages = [
        'bertopic', 'pandas', 'numpy', 'sentence_transformers',
        'umap', 'hdbscan', 'plotly', 'yaml', 'streamlit'
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
        print("\n📦 请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True


def validate_config(config_path: Path = None) -> bool:
    """验证配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        print("❌ 未找到config.yaml配置文件")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 基本结构验证
        required_sections = ['data_paths', 'bertopic_params', 'results_paths']
        for section in required_sections:
            if section not in config:
                print(f"❌ 配置缺少必需的[{section}]部分")
                return False
        
        print("✅ 配置文件验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置文件验证失败: {e}")
        return False


def check_data_files() -> bool:
    """检查数据文件"""
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        print("❌ 未找到data文件夹")
        print("   请创建data文件夹并放入数据文件")
        return False
    
    # 检查是否有Excel文件
    excel_files = list(data_dir.glob("*.xlsx"))
    if not excel_files:
        print("❌ data文件夹中没有Excel文件")
        print("   请放入.xlsx格式的数据文件")
        return False
    
    print(f"✅ 找到数据文件: {len(excel_files)} 个")
    return True


def run_analysis():
    """运行主题分析"""
    print("\n" + "="*60)
    print("🚀 开始BERTopic主题分析...")
    print("="*60 + "\n")
    
    try:
        # 导入分析模块
        sys.path.append(str(Path(__file__).parent))
        from topic_analyzer.data_loader import DataLoader
        from topic_analyzer.model import TopicAnalyzer
        
        # 配置日志
        log_file = Path(__file__).parent / 'analysis.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        # 加载配置
        config_path = Path(__file__).parent / "config.yaml"
        
        # 加载数据
        print("📁 加载数据...")
        loader = DataLoader(str(config_path))
        data = loader.load_and_prepare_data()
        documents = data['texts']
        print(f"✅ 已加载 {len(documents)} 条文档")
        
        # 训练模型
        print("🤖 训练主题模型...")
        analyzer = TopicAnalyzer(str(config_path))
        topic_model = analyzer.train_model(documents)
        
        # 生成结果
        print("📊 生成分析结果...")
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # 保存主题信息
        topic_info = topic_model.get_topic_info()
        topic_info.to_csv(results_dir / "topics_summary.csv", index=False, encoding='utf-8-sig')
        
        # 生成可视化
        try:
            viz_html = topic_model.visualize_topics()
            with open(results_dir / "topic_visualization.html", "w", encoding='utf-8') as f:
                f.write(viz_html.to_html())
        except Exception as e:
            print(f"⚠️ 可视化生成失败: {e}")
        
        print(f"\n🎉 分析完成！")
        print(f"📁 结果保存在: {results_dir}")
        print(f"📊 发现主题数: {len(topic_info) - 1}")  # -1排除噪声主题
        
        return True
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        logging.error(f"Analysis failed: {e}")
        return False


def show_menu():
    """显示主菜单"""
    print("\n🚀 BERTopic主题分析系统")
    print("="*40)
    print("1. 运行主题分析")
    print("2. 启动Web界面")
    print("3. 只检查环境")
    print("4. 退出")
    print("="*40)


def main():
    """主函数"""
    print("="*60)
    print("🔍 BERTopic主题分析系统 - 环境检查")
    print("="*60)
    
    # 环境检查
    checks = [
        ("Python版本", check_python_version),
        ("依赖包", check_dependencies),
        ("配置文件", validate_config),
        ("数据文件", check_data_files),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n🔍 检查{check_name}...")
        if not check_func():
            all_passed = False
    
    if not all_passed:
        print("\n" + "="*60)
        print("⚠️ 请先解决上述问题后再运行")
        print("="*60)
        return
    
    print("\n" + "="*60)
    print("✨ 环境检查通过！")
    print("="*60)
    
    # 显示菜单
    while True:
        show_menu()
        try:
            choice = input("\n请选择操作 (1-4): ").strip()
            
            if choice == '1':
                if run_analysis():
                    break
            elif choice == '2':
                print("\n🌐 启动Web界面...")
                print("请运行: python web_ui.py")
                print("或双击: run_web_ui.bat")
                break
            elif choice == '3':
                print("\n✅ 环境检查已完成，可以进行分析")
                break
            elif choice == '4':
                print("\n👋 再见！")
                break
            else:
                print("\n❌ 请输入1-4之间的数字")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断运行")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        sys.exit(1)