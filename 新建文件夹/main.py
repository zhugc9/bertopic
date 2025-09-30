#!/usr/bin/env python3
"""
BERTopic主题分析系统 - 统一入口
===============================
集成超参数优化、多语言预处理、高级可视化的完整分析流程
"""

import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from tools.workflow_manager import WorkflowManager
from tools.check_deps import run_dependency_check
from topic_analyzer.pipeline import AnalysisPipeline

# 为了向后兼容，保留BERTopicWorkflow别名
BERTopicWorkflow = WorkflowManager

def _ensure_utf8_output():
    """在Windows控制台等环境下尝试启用UTF-8输出，避免emoji导致的编码错误"""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ensure_utf8_output()

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
    """调用集中化的依赖检查工具"""
    return run_dependency_check()

def validate_config(config_path: Path = None) -> bool:
    """验证配置文件是否包含关键设置"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

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
        # 从config.yaml读取数据路径
        config_path = Path(__file__).parent / "config.yaml"
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
            print("   请在config.yaml中正确配置data_paths")
            return False
        
        print(f"✅ 找到数据文件: {found_files} 个")
        return True
        
    except Exception as e:
        print(f"❌ 检查数据文件失败: {e}")
        return False

def run_analysis():
    """运行主题分析"""
    print("\n" + "="*60)
    print("🚀 开始BERTopic主题分析...")
    print("="*60 + "\n")
    
    try:
        # 导入分析模块
        sys.path.append(str(Path(__file__).parent))
        from topic_analyzer.pipeline import AnalysisPipeline
        
        # 加载配置 - 直接加载YAML文件
        config_path = Path(__file__).parent / "config.yaml"
        from config_loader import load_runtime_config
        runtime_config = load_runtime_config(config_path)
        pipeline = AnalysisPipeline(config_path, runtime_config)
        
        print("📁 加载数据...")
        result = pipeline.run_analysis()
        print(f"✅ 已加载 {len(result.documents)} 条文档")

        # 生成结果
        print("📊 生成分析结果...")
        pipeline.generate_results(result)
        
        # 检查是否启用SOTA可视化
        sota_config = pipeline.runtime_config.get('visualization', {}).get('sota_charts', {})
        if sota_config.get('enable', True):
            print("🎨 生成SOTA级可视化...")
            sota_charts = pipeline.topic_analyzer.generate_sota_visualizations(
                result.topic_model, result.documents, result.topics, result.metadata_df
            )
            if sota_charts:
                print(f"✅ 生成SOTA图表: {len(sota_charts)} 个")
        
        # 基础统计
        topic_info = result.topic_model.get_topic_info()
        n_topics = len(topic_info) - 1  # -1排除噪声主题
        
        print(f"\n🎉 分析完成！")
        print(f"📁 结果保存在: {Path(pipeline.runtime_config['results_paths']['output_dir'])}")
        print(f"📊 发现主题数: {n_topics}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        logging.error(f"Analysis failed: {e}")
        return False


def show_menu():
    """显示主菜单"""
    print("\n🚀 BERTopic主题分析系统")
    print("="*50)
    print("💡 提示：所有分析参数通过 config.yaml 文件控制")
    print("-"*50)
    print("1. 开始主题分析 (推荐)")
    print("2. 检查系统环境")
    print("3. 退出")
    print("="*50)
    print("📝 使用说明：")
    print("   - 修改 config.yaml 设置分析参数")
    print("   - 选择1开始分析，系统会自动读取您的配置")
    print("="*50)


def main():
    """主函数"""
    log_file = Path(__file__).parent / 'analysis.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
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
                # 基于config.yaml的智能分析
                print("\n🎯 正在读取 config.yaml 配置...")
                if run_user_friendly_analysis():
                    break
            elif choice == '2':
                print("\n✅ 环境检查已完成，可以进行分析")
                break
                
            elif choice == '3':
                print("\n👋 再见！")
                break
                
            else:
                print("\n❌ 请输入1-3之间的数字")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


def run_user_friendly_analysis() -> bool:
    """
    运行用户友好的分析（基于config.yaml用户配置）
    """
    try:
        config_path = Path(__file__).parent / "config.yaml"
        
        from config_loader import load_runtime_config
        runtime_config = load_runtime_config(config_path)
        analysis_mode = runtime_config.get('analysis', {}).get('mode', 'analyze')
        
        print(f"\n🎯 检测到运行模式: {analysis_mode}")
        
        if analysis_mode == 'tune':
            return run_tuning_phase(runtime_config, config_path)
        else:
            return run_analysis_phase(runtime_config, config_path)
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_tuning_phase(runtime_config: Dict[str, Any], config_path: Path) -> bool:
    """
    运行第一阶段：机器调参
    
    Args:
        translator: 配置翻译器
        
    Returns:
        是否成功
    """
    print("\n🔍 第一阶段：机器自动调参 (海选模式)")
    print("=" * 50)
    print("💡 机器将尝试数百种参数组合，为您筛选出Top 5最佳候选")
    print("⏰ 预计耗时：30-60分钟（取决于数据量和试验次数）")
    print("☕ 您可以去喝杯咖啡，机器会不知疲倦地工作...")
    
    try:
        pipeline = AnalysisPipeline(config_path, runtime_config)
        optimization_results = pipeline.run_hyperparameter_search()
        if optimization_results:
            from tools.tuning_manager import save_tuning_results
            save_tuning_results(str(config_path), optimization_results)
            success = True
        else:
            success = False
        
        if success:
            print("\n🎉 第一阶段调参完成！")
            print("📋 机器已为您筛选出Top 5最佳参数组合")
            print("\n🔄 下一步操作：")
            print("1. 查看 results/候选参数选择指南.txt")
            print("2. 在config.yaml中设置 selected_candidate: X (1-5)")
            print("3. 将 mode 改为 'analyze' 并重新运行")
            print("4. 对比不同候选参数的分析结果")
        else:
            print("\n❌ 调参失败")
            
        return success
        
    except Exception as e:
        print(f"\n❌ 调参阶段失败: {e}")
        return False


def run_analysis_phase(runtime_config: Dict[str, Any], config_path: Path) -> bool:
    """
    运行第二阶段：正式分析
    
    Args:
        translator: 配置翻译器
        
    Returns:
        是否成功
    """
    print("\n📊 第二阶段：正式分析 (深度模式)")
    print("=" * 50)
    
    # 检查是否有候选参数
    candidate_config = runtime_config.get('candidate_parameters', {})
    selected_candidate = candidate_config.get('selected_candidate', 1)
    candidate_key = f'candidate_{selected_candidate}'
    candidate = candidate_config.get(candidate_key)
    
    if candidate:
        print(f"🎯 使用候选参数 {selected_candidate}")
        print(f"📋 参数特征: {candidate.get('description', '未知')}")
        print(f"📈 一致性分数: {candidate.get('coherence_score', 'N/A')}")
    else:
        print("⚠️ 未找到候选参数，使用默认配置")
    
    # 应用候选参数覆盖运行时配置
    if candidate:
        bertopic_params = runtime_config.setdefault('bertopic_params', {})
        umap_params = bertopic_params.setdefault('umap_params', {})
        hdbscan_params = bertopic_params.setdefault('hdbscan_params', {})

        umap_keys = ['n_neighbors', 'n_components', 'min_dist', 'metric']
        for key in umap_keys:
            if key in candidate and candidate[key] is not None:
                umap_params[key] = candidate[key]

        hdbscan_keys = ['min_cluster_size', 'min_samples', 'cluster_selection_method']
        for key in hdbscan_keys:
            if key in candidate and candidate[key] is not None:
                hdbscan_params[key] = candidate[key]

        if 'min_topic_size' in candidate and candidate['min_topic_size'] is not None:
            bertopic_params['min_topic_size'] = candidate['min_topic_size']
    
    try:
        # 转换技术配置
        pipeline = AnalysisPipeline(config_path, runtime_config)
        result = pipeline.run_analysis()
        pipeline.generate_results(result)
        success = True
        
        if success:
            print(f"\n🎉 第二阶段分析完成！")
            results_dir = Path(runtime_config['results_paths']['output_dir'])
            print(f"📁 结果保存在: {results_dir}")
            generate_analysis_summary(results_dir, runtime_config, selected_candidate)
            
            # 如果是候选参数，提示对比其他候选
            if f'candidate_{selected_candidate}' in candidate_config:
                print(f"\n💡 建议：")
                print(f"• 已完成候选 {selected_candidate} 的分析")
                print(f"• 可尝试其他候选参数 (1-5) 进行对比")
                print(f"• 最终选择最符合您研究需求的参数组合")
        else:
            print(f"\n❌ 分析失败")
            
        return success
        
    except Exception as e:
        print(f"\n❌ 分析阶段失败: {e}")
        return False


def generate_user_summary(results_dir: Path, mode: str, user_config: dict):
    """
    生成用户友好的结果摘要
    
    Args:
        results_dir: 结果目录
        mode: 分析模式
        user_config: 用户配置
    """
    try:
        summary_file = results_dir / 'analysis_summary_for_user.txt'
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("🎓 BERTopic主题分析结果摘要\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"📊 分析模式: {mode}\n")
            f.write(f"📅 分析时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
            
            # 数据信息
            data_paths = user_config.get('data_paths', {})
            f.write("📁 分析数据:\n")
            if data_paths.get('media_data'):
                f.write(f"  - 传统媒体数据: {data_paths['media_data']}\n")
            if data_paths.get('social_media_data'):
                f.write(f"  - 社交媒体数据: {data_paths['social_media_data']}\n")
            f.write("\n")
            
            # 分析设置
            topic_config = user_config.get('topic_settings', {})
            f.write("⚙️ 分析设置:\n")
            f.write(f"  - 主题数量: {topic_config.get('expected_topics', 'auto')}\n")
            f.write(f"  - 最小主题大小: {topic_config.get('min_documents_per_topic', 15)}\n")
            f.write(f"  - 关键词精度: {topic_config.get('advanced', {}).get('ngram_range', [1, 3])}\n")
            f.write(f"  - 语言模式: {topic_config.get('text_language', 'multilingual')}\n\n")
            
            # 功能启用状态
            features = user_config.get('features', {})
            auto_tuning_cfg = user_config.get('hyperparameter_optimization', {})
            viz_config = user_config.get('visualization', {})
            f.write("🔧 功能启用状态:\n")
            f.write(f"  - 超参数优化: {'✓' if auto_tuning_cfg.get('enable') else '✗'}\n")
            f.write(f"  - 学术级可视化: {'✓' if viz_config.get('sota_charts', {}).get('enable', True) else '✗'}\n")
            f.write(f"  - 时间分析: {'✓' if features.get('time_evolution', {}).get('enable') else '✗'}\n")
            f.write(f"  - 来源分析: {'✓' if features.get('source_comparison', {}).get('enable') else '✗'}\n")
            f.write(f"  - 框架分析: {'✓' if features.get('frame_analysis', {}).get('enable') else '✗'}\n\n")
            
            # 结果文件
            f.write("📄 主要结果文件:\n")
            f.write("  - 主题摘要表.csv: 主题关键词和统计信息\n")
            f.write("  - 文档主题分布表.csv: 文档与主题的对应关系\n")
            f.write("  - 图表清单.txt: 生成的图表清单\n")
            f.write("  - 图表文件夹: 包含所有可视化图表\n\n")
            
            # 使用建议
            f.write("💡 结果使用建议:\n")
            f.write("  1. 查看 主题摘要表.csv 了解发现的主题\n")
            f.write("  2. 查看图表文件夹中的可视化结果\n")
            f.write("  3. 如需调整，修改 config.yaml 中的参数重新运行\n")
            f.write("  4. 论文写作可直接使用生成的高质量图表\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("🎉 分析完成！如有问题请检查配置文件或联系技术支持。\n")
            f.write("=" * 60 + "\n")
        
        print(f"📋 用户摘要已生成: {summary_file}")
        
    except Exception as e:
        print(f"⚠️ 用户摘要生成失败: {e}")


def generate_analysis_summary(results_dir: Path, user_config: dict, candidate_num: int):
    """
    生成分析阶段的结果摘要
    
    Args:
        results_dir: 结果目录
        user_config: 用户配置
        candidate_num: 候选参数编号
    """
    try:
        summary_file = results_dir / f'候选{candidate_num}_分析摘要.txt'
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"🎓 候选参数 {candidate_num} 分析结果摘要\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"📅 分析时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
            f.write(f"🎯 使用候选: 第 {candidate_num} 组参数\n\n")
            
            # 候选参数信息
            candidate_config = user_config.get('candidate_parameters', {})
            if f'candidate_{candidate_num}' in candidate_config:
                candidate = candidate_config[f'candidate_{candidate_num}']
                f.write("📊 参数信息:\n")
                f.write(f"  - 一致性分数: {candidate.get('coherence_score', 'N/A')}\n")
                f.write(f"  - 参数特征: {candidate.get('description', '未知')}\n")
                f.write(f"  - 最小主题大小: {candidate.get('min_topic_size', 'N/A')}\n")
                f.write(f"  - UMAP邻居数: {candidate.get('n_neighbors', 'N/A')}\n")
                f.write(f"  - 聚类大小: {candidate.get('min_cluster_size', 'N/A')}\n\n")
            
            # 数据信息
            data_paths = user_config.get('data_paths', {})
            f.write("📁 分析数据:\n")
            if data_paths.get('media_data'):
                f.write(f"  - 传统媒体数据: {data_paths['media_data']}\n")
            if data_paths.get('social_media_data'):
                f.write(f"  - 社交媒体数据: {data_paths['social_media_data']}\n")
            f.write("\n")
            
            # 结果文件
            f.write("📄 生成的文件:\n")
            f.write("  - 主题关键词表.csv: 发现的主题及关键词\n")
            f.write("  - 文档主题映射表.csv: 每个文档的主题归属\n")
            f.write("  - 图表文件: 论文级可视化图表\n")
            f.write("  - 分析报告: 详细的统计信息\n\n")
            
            f.write("💡 下一步建议:\n")
            f.write("  1. 查看主题关键词表，评估主题质量\n")
            f.write("  2. 检查图表文件，确认可视化效果\n")
            f.write("  3. 如需对比，可选择其他候选参数重新分析\n")
            f.write("  4. 选定最佳参数后，可用于论文写作\n\n")
            
            f.write("=" * 60 + "\n")
            f.write(f"🎉 候选 {candidate_num} 分析完成！\n")
            f.write("=" * 60 + "\n")
        
        print(f"📋 分析摘要已生成: {summary_file}")
        
    except Exception as e:
        print(f"⚠️ 分析摘要生成失败: {e}")


def run_advanced_analysis(mode: str = 'standard') -> bool:
    """
    运行高级分析（向后兼容）
    
    Args:
        mode: 分析模式 (quick/standard/research)
    """
    # 为向后兼容保留此函数，但建议使用用户友好版本
    return run_user_friendly_analysis()


def create_cli_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="BERTopic主题分析系统 - 博士生友好版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎓 使用说明（专为计算传播学博士生设计）:
  
  python main.py                    # 交互式界面（推荐）
  python main.py --run              # 直接运行分析（读取config.yaml）
  python main.py --check            # 检查系统环境
  
💡 所有分析参数都在 config.yaml 文件中设置：
  - 数据文件路径
  - 分析模式选择  
  - 主题数量和精度
  - 可视化设置
  
📝 论文写作流程：
  1. 修改 config.yaml 设置参数
  2. 运行 python main.py --run
  3. 查看 results/ 文件夹获取结果
  4. 使用生成的高质量图表写论文
        """
    )
    
    parser.add_argument(
        '--run', '-r',
        action='store_true',
        help='直接运行分析（读取config.yaml配置）'
    )
    
    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='仅检查系统环境'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )
    
    return parser


if __name__ == "__main__":
    try:
        parser = create_cli_parser()
        args = parser.parse_args()
        
        # 检查参数
        if args.run:
            # 直接运行分析
            print("🎯 直接运行模式：读取 config.yaml 配置...")
            success = run_user_friendly_analysis()
            sys.exit(0 if success else 1)
            
        elif args.check:
            # 仅环境检查
            print("🔍 系统环境检查...")
            checks = [
                ("Python版本", check_python_version),
                ("依赖包", check_dependencies),
                ("配置文件", lambda: validate_config(Path(args.config))),
                ("数据文件", check_data_files),
            ]
            
            all_passed = True
            for check_name, check_func in checks:
                print(f"\n检查{check_name}...")
                if not check_func():
                    all_passed = False
            
            if all_passed:
                print("\n✅ 所有检查通过，系统就绪")
                print("💡 可以运行: python main.py --run 开始分析")
                sys.exit(0)
            else:
                print("\n❌ 检查失败，请解决上述问题")
                sys.exit(1)
        else:
            # 交互式模式
            main()
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断运行")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)