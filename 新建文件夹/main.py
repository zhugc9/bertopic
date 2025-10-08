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
from topic_analyzer.pipeline import AnalysisPipeline

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

def run_analysis():
    """运行主题分析"""
    print("\n" + "="*60)
    print("开始BERTopic主题分析")
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
        
        print(f"\n分析完成")
        print(f"结果保存在: {Path(pipeline.runtime_config['results_paths']['output_dir'])}")
        print(f"发现主题数: {n_topics}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        logging.error(f"Analysis failed: {e}")
        return False


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
    print("\n第一阶段：自动调参")
    print("=" * 50)
    
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
            print("\n第一阶段调参完成")
            print("\n下一步操作：")
            print("1. 查看 results/候选参数选择指南.txt")
            print("2. 在config.yaml中设置 selected_candidate: X (1-5)")
            print("3. 将 mode 改为 'analyze' 并重新运行")
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
        print(f"📈 质量分数: {candidate.get('quality_score', 'N/A')}")
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
        
        # 检查是否启用SOTA可视化
        sota_config = pipeline.runtime_config.get('visualization', {}).get('sota_charts', {})
        if sota_config.get('enable', True):
            print("🎨 生成SOTA级可视化...")
            try:
                sota_charts = pipeline.topic_analyzer.generate_sota_visualizations(
                    result.topic_model, result.documents, result.topics, result.metadata_df
                )
                if sota_charts:
                    print(f"✅ 生成SOTA图表: {len(sota_charts)} 个")
            except Exception as e:
                print(f"⚠ SOTA可视化生成时遇到问题: {e}")
        
        if success:
            print(f"\n第二阶段分析完成")
            results_dir = Path(runtime_config['results_paths']['output_dir'])
            print(f"结果保存在: {results_dir}")
            generate_analysis_summary(results_dir, runtime_config, selected_candidate)
        else:
            print(f"\n❌ 分析失败")
            
        return success
        
    except Exception as e:
        import traceback
        print(f"\n❌ 分析阶段失败: {e}")
        print(f"\n详细错误信息：")
        traceback.print_exc()
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
            f.write("  - 图表文件夹: 包含所有可视化图表\n\n")
            
            # 使用建议
            f.write("💡 结果使用建议:\n")
            f.write("  1. 查看 主题摘要表.csv 了解发现的主题\n")
            f.write("  2. 查看图表文件夹中的可视化结果\n")
            f.write("  3. 如需调整，修改 config.yaml 中的参数重新运行\n")
            f.write("  4. 论文写作可直接使用生成的高质量图表\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("分析完成。如有问题请检查配置文件。\n")
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
        summary_file = results_dir / f'5-候选{candidate_num}_分析摘要.txt'
        
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
                f.write(f"  - 质量分数: {candidate.get('quality_score', 'N/A')}\n")
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
  python tools/check_deps.py        # 检查环境依赖（仅诊断时使用）
  
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
            try:
                input("\n按任意键退出...")
            except EOFError:
                pass
            sys.exit(0 if success else 1)
            
        else:
            # 显示帮助
            parser.print_help()
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断运行")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)