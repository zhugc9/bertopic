#!/usr/bin/env python3
"""
BERTopic主题分析系统 - 统一入口
===============================
集成超参数优化、多语言预处理、高级可视化的完整分析流程
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from tools.workflow_manager import WorkflowManager

# 为了向后兼容，保留BERTopicWorkflow别名
BERTopicWorkflow = WorkflowManager


def main():
    """主菜单界面"""
    print("🎓 BERTopic主题分析系统")
    print("=" * 50)
    print("💡 推荐工作流程：")
    print("1. 修改 config.yaml 设置您的参数")
    print("2. 运行 python main.py --run")
    print("3. 查看 results/ 文件夹获取结果")
    print("=" * 50)
    
    show_menu()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.results_dir / 'bertopic_analysis.log'
        self.results_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> bool:
        """加载数据"""
        self.logger.info("📁 开始加载数据...")
        
        try:
            from topic_analyzer.data_loader import DataLoader
            
            self.data_loader = DataLoader(self.config)
            self.documents, self.metadata_df = self.data_loader.load_and_prepare_data()
            
            self.logger.info(f"✅ 数据加载完成：{len(self.documents)} 个文档")
            
            if self.metadata_df is not None:
                self.logger.info(f"  元数据维度: {self.metadata_df.shape}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 数据加载失败: {e}")
            return False
    
    def run_hyperparameter_optimization(self) -> Optional[Dict]:
        """运行超参数优化"""
        opt_config = self.config.get('hyperparameter_optimization', {})
        
        if not opt_config.get('enable', False):
            self.logger.info("⏭️ 超参数优化已禁用，跳过")
            return None
        
        if not self.analyzer:
            from topic_analyzer.model import TopicAnalyzer
            self.analyzer = TopicAnalyzer(self.config)
        
        self.logger.info("🔍 开始超参数优化...")
        
        try:
            optimization_results = self.analyzer.optimize_hyperparameters(self.documents)
            
            if optimization_results:
                self.logger.info("✅ 超参数优化完成")
                return optimization_results
            else:
                self.logger.warning("⚠️ 超参数优化未返回结果")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 超参数优化失败: {e}")
            return None
    
    def train_topic_model(self, use_optimized_params: bool = False) -> bool:
        """训练主题模型"""
        if not self.analyzer:
            from topic_analyzer.model import TopicAnalyzer
            self.analyzer = TopicAnalyzer(self.config)
        
        self.logger.info("🤖 开始训练主题模型...")
        
        try:
            if use_optimized_params:
                self.topic_model, self.topics = self.analyzer.train_with_optimized_parameters(self.documents)
                self.logger.info("  使用优化参数训练")
            else:
                self.topic_model, self.topics = self.analyzer.train_bertopic_model(self.documents)
                self.logger.info("  使用默认配置训练")
            
            # 模型统计
            topic_info = self.topic_model.get_topic_info()
            n_topics = len(topic_info) - 1  # 排除噪声主题
            
            self.logger.info(f"✅ 模型训练完成")
            self.logger.info(f"  发现主题数: {n_topics}")
            self.logger.info(f"  总文档数: {len(self.documents)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 模型训练失败: {e}")
            return False
    
    def generate_results(self) -> bool:
        """生成分析结果"""
        if not self.topic_model:
            self.logger.error("模型未训练，无法生成结果")
            return False
        
        self.logger.info("📊 生成分析结果...")
        
        try:
            # 1. 基础结果
            self.logger.info("  生成基础结果...")
            self.analyzer.generate_results(
                self.topic_model, 
                self.documents, 
                self.topics, 
                self.metadata_df
            )
            
            # 2. 高级可视化
            viz_config = self.config.get('visualization', {}).get('sota_charts', {})
            if viz_config.get('enable', True):
                self.logger.info("  生成高级可视化...")
                charts = self.analyzer.generate_sota_visualizations(
                    self.topic_model,
                    self.documents, 
                    self.topics,
                    self.metadata_df
                )
                
                if charts:
                    self.logger.info(f"  ✓ 生成图表: {len(charts)} 个")
                else:
                    self.logger.warning("  ⚠ 图表生成失败")
            else:
                self.logger.info("  ⏭️ 高级可视化已禁用")
            
            self.logger.info("✅ 分析结果生成完成")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 结果生成失败: {e}")
            return False
    
    def run_complete_analysis(self, optimize_params: bool = False) -> bool:
        """运行完整分析流程"""
        self.logger.info("🚀 开始完整主题分析流程...")
        
        # 1. 加载数据
        if not self.load_data():
            return False
        
        # 2. 超参数优化（可选）
        optimization_results = None
        if optimize_params:
            optimization_results = self.run_hyperparameter_optimization()
        
        # 3. 训练模型
        use_optimized = optimization_results is not None
        if not self.train_topic_model(use_optimized_params=use_optimized):
            return False
        
        # 4. 生成结果
        if not self.generate_results():
            return False
        
        self.logger.info("🎉 完整分析流程成功完成！")
        return True


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
        
        # 基本结构验证 - 适配新的用户友好配置格式
        required_sections = ['topic_parameters', 'output_settings']
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
    try:
        # 从config.yaml读取数据路径
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查配置的数据路径
        data_paths = config.get('data_paths', {})
        media_data = data_paths.get('media_data')
        social_data = data_paths.get('social_media_data')
        
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
        from topic_analyzer.data_loader import DataLoader
        from topic_analyzer.model import TopicAnalyzer
        from config_manager import ConfigManager
        
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
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
        
        # 加载数据
        print("📁 加载数据...")
        loader = DataLoader(config)
        documents, metadata_df = loader.load_and_prepare_data()
        print(f"✅ 已加载 {len(documents)} 条文档")
        
        # 训练模型
        print("🤖 训练主题模型...")
        analyzer = TopicAnalyzer(config)
        topic_model, topics = analyzer.train_bertopic_model(documents)
        
        # 生成结果
        print("📊 生成分析结果...")
        results_dir = Path(config['results_paths']['output_dir'])
        results_dir.mkdir(exist_ok=True)
        
        # 生成完整结果（包含SOTA可视化）
        analyzer.generate_results(topic_model, documents, topics, metadata_df)
        
        # 检查是否启用SOTA可视化
        sota_config = config.get('visualization', {}).get('sota_charts', {})
        if sota_config.get('enable', True):
            print("🎨 生成SOTA级可视化...")
            sota_charts = analyzer.generate_sota_visualizations(
                topic_model, documents, topics, metadata_df
            )
            if sota_charts:
                print(f"✅ 生成SOTA图表: {len(sota_charts)} 个")
        
        # 基础统计
        topic_info = topic_model.get_topic_info()
        n_topics = len(topic_info) - 1  # -1排除噪声主题
        
        print(f"\n🎉 分析完成！")
        print(f"📁 结果保存在: {results_dir}")
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
    print("2. 启动Web界面")
    print("3. 检查系统环境")
    print("4. 退出")
    print("="*50)
    print("📝 使用说明：")
    print("   - 修改 config.yaml 设置分析参数")
    print("   - 选择1开始分析，系统会自动读取您的配置")
    print("="*50)


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
                # 基于config.yaml的智能分析
                print("\n🎯 正在读取 config.yaml 配置...")
                if run_user_friendly_analysis():
                    break
                    
            elif choice == '2':
                print("\n🌐 启动Web界面...")
                print("请运行: streamlit run web_ui.py --server.port 8502")
                print("或双击: run_web.bat")
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


def run_user_friendly_analysis() -> bool:
    """
    运行用户友好的分析（基于config.yaml用户配置）
    """
    try:
        config_path = Path(__file__).parent / "config.yaml"
        
        # 使用配置翻译器转换用户配置
        from tools.config_translator import ConfigTranslator
        translator = ConfigTranslator(str(config_path))
        analysis_mode = translator.get_analysis_mode()
        
        print(f"\n🎯 检测到运行模式: {analysis_mode}")
        
        if analysis_mode == 'tune':
            # 第一阶段：机器调参
            return run_tuning_phase(translator)
        else:
            # 第二阶段：正式分析
            return run_analysis_phase(translator)
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_tuning_phase(translator) -> bool:
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
        # 转换技术配置
        tech_config = translator.translate_to_technical_config()
        temp_config_path = Path("temp_tuning_config.yaml")
        translator.save_technical_config(str(temp_config_path))
        
        # 运行调参工作流
        workflow = BERTopicWorkflow(str(temp_config_path))
        
        # 只运行超参数优化
        loaded_data = workflow.load_data()
        if loaded_data:
            optimization_results = workflow.run_hyperparameter_optimization()
            if optimization_results:
                # 保存调参结果
                from tools.tuning_manager import save_tuning_results
                save_tuning_results(str(translator.config_path), optimization_results)
                success = True
            else:
                success = False
        else:
            success = False
        
        # 清理临时文件
        if temp_config_path.exists():
            temp_config_path.unlink()
        
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


def run_analysis_phase(translator) -> bool:
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
    candidate_config = translator.user_config.get('candidate_parameters', {})
    selected_candidate = candidate_config.get('selected_candidate', 1)
    
    if f'candidate_{selected_candidate}' in candidate_config:
        print(f"🎯 使用候选参数 {selected_candidate}")
        candidate = candidate_config[f'candidate_{selected_candidate}']
        print(f"📋 参数特征: {candidate.get('description', '未知')}")
        print(f"📈 一致性分数: {candidate.get('coherence_score', 'N/A')}")
    else:
        print("⚠️ 未找到候选参数，使用默认配置")
    
    try:
        # 转换技术配置
        tech_config = translator.translate_to_technical_config()
        temp_config_path = Path("temp_analysis_config.yaml")
        translator.save_technical_config(str(temp_config_path))
        
        # 运行完整分析工作流
        workflow = BERTopicWorkflow(str(temp_config_path))
        success = workflow.run_complete_analysis(optimize_params=False)
        
        # 清理临时文件
        if temp_config_path.exists():
            temp_config_path.unlink()
        
        if success:
            print(f"\n🎉 第二阶段分析完成！")
            print(f"📁 结果保存在: {workflow.results_dir}")
            
            # 生成用户友好的结果摘要
            generate_analysis_summary(workflow.results_dir, translator.user_config, selected_candidate)
            
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
            f.write("⚙️ 分析设置:\n")
            topic_config = user_config.get('topic_modeling', {})
            f.write(f"  - 主题数量: {topic_config.get('target_topics', 'auto')}\n")
            f.write(f"  - 最小主题大小: {topic_config.get('min_topic_size', 15)}\n")
            f.write(f"  - 关键词精度: {topic_config.get('keyword_precision', 'standard')}\n")
            f.write(f"  - 语言模式: {topic_config.get('language_mode', 'multilingual')}\n\n")
            
            # 功能启用状态
            analysis_mode_config = user_config.get('analysis_mode', {})
            f.write("🔧 功能启用状态:\n")
            f.write(f"  - 超参数优化: {'✓' if analysis_mode_config.get('enable_hyperparameter_optimization') else '✗'}\n")
            f.write(f"  - 多语言处理: {'✓' if analysis_mode_config.get('enable_multilingual_processing') else '✗'}\n")
            f.write(f"  - 学术级可视化: {'✓' if analysis_mode_config.get('enable_academic_visualizations') else '✗'}\n")
            f.write(f"  - 时间分析: {'✓' if user_config.get('temporal_analysis', {}).get('enable') else '✗'}\n")
            f.write(f"  - 来源分析: {'✓' if user_config.get('source_analysis', {}).get('enable') else '✗'}\n")
            f.write(f"  - 框架分析: {'✓' if user_config.get('frame_analysis', {}).get('enable') else '✗'}\n\n")
            
            # 结果文件
            f.write("📄 主要结果文件:\n")
            f.write("  - topics_summary.csv: 主题关键词和统计信息\n")
            f.write("  - document_topic_mapping.csv: 文档与主题的对应关系\n")
            f.write("  - charts_summary.txt: 生成的图表清单\n")
            f.write("  - 图表文件夹: 包含所有可视化图表\n\n")
            
            # 使用建议
            f.write("💡 结果使用建议:\n")
            f.write("  1. 查看 topics_summary.csv 了解发现的主题\n")
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