"""
调参结果管理器
==============
管理超参数优化的结果和候选参数
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TuningManager:
    """调参结果管理器"""
    
    def __init__(self, config_path: str):
        """
        初始化调参管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def save_tuning_results(self, optimization_results: Dict[str, Any]) -> str:
        """
        保存调参结果并更新config.yaml
        
        Args:
            optimization_results: 优化结果
            
        Returns:
            候选参数文件路径
        """
        logger.info("💾 保存调参结果...")
        
        try:
            # 1. 保存详细的调参报告
            self._save_detailed_report(optimization_results)
            
            # 2. 生成候选参数文件
            candidates_file = self._generate_candidates_file(optimization_results)
            
            # 3. 更新config.yaml中的候选参数
            self._update_config_candidates(optimization_results)
            
            # 4. 生成用户友好的选择指南
            self._generate_selection_guide(optimization_results)
            
            logger.info("✅ 调参结果保存完成")
            return candidates_file
            
        except Exception as e:
            logger.error(f"❌ 调参结果保存失败: {e}")
            raise
    
    def _save_detailed_report(self, results: Dict[str, Any]):
        """保存详细的调参报告"""
        report_file = self.results_dir / 'hyperparameter_tuning_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("BERTopic超参数调优详细报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"调优完成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
            f.write(f"总试验次数: {results.get('n_trials', 'N/A')}\n")
            f.write(f"最佳一致性分数: {results.get('best_score', 'N/A'):.4f}\n")
            f.write(f"优化用时: {results.get('optimization_time', 'N/A'):.1f}秒\n\n")
            
            # Top 5 参数组合
            top_params = results.get('top_parameters', results.get('top_5_results', []))
            f.write("🏆 Top 5 最佳参数组合:\n")
            f.write("-" * 40 + "\n")

            for i, item in enumerate(top_params[:5], 1):
                params = item.get('params', item) if isinstance(item, dict) else {}
                params = params if isinstance(params, dict) else {}
                raw_score = item.get('score') if isinstance(item, dict) else None
                score = raw_score if raw_score is not None else params.get('score', 0)

                f.write(f"\n候选 {i}:\n")
                f.write(f"  一致性分数: {float(score or 0):.4f}\n")
                f.write(f"  最小主题大小: {params.get('min_topic_size', 'N/A')}\n")
                f.write(f"  UMAP邻居数: {params.get('n_neighbors', 'N/A')}\n")
                f.write(f"  UMAP维度: {params.get('n_components', 'N/A')}\n")
                f.write(f"  聚类大小: {params.get('min_cluster_size', 'N/A')}\n")
                f.write(f"  最小样本数: {params.get('min_samples', 'N/A')}\n")

                # 参数特征描述
                description = self._describe_parameters(params)
                f.write(f"  特征描述: {description}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("下一步操作:\n")
            f.write("1. 查看 candidate_parameters.yaml 文件\n")
            f.write("2. 在config.yaml中设置 selected_candidate: X (1-5)\n")
            f.write("3. 将 mode 改为 'analyze' 并重新运行\n")
            f.write("4. 对比不同候选参数的分析结果\n")
    
    def _describe_parameters(self, params: Dict[str, Any]) -> str:
        """描述参数组合的特征"""
        min_topic_size = params.get('min_topic_size', 15)
        n_neighbors = params.get('n_neighbors', 15)
        min_cluster_size = params.get('min_cluster_size', 15)
        
        # 主题粒度
        if min_topic_size <= 10:
            granularity = "细粒度"
        elif min_topic_size >= 30:
            granularity = "粗粒度"
        else:
            granularity = "中等粒度"
        
        # 聚类特征
        if n_neighbors <= 10:
            clustering = "局部聚类"
        elif n_neighbors >= 30:
            clustering = "全局聚类"
        else:
            clustering = "平衡聚类"
        
        # 稳定性
        if min_cluster_size >= 25:
            stability = "高稳定性"
        elif min_cluster_size <= 10:
            stability = "高灵敏性"
        else:
            stability = "中等稳定性"
        
        return f"{granularity}, {clustering}, {stability}"
    
    def _generate_candidates_file(self, results: Dict[str, Any]) -> str:
        """生成候选参数文件"""
        candidates_file = self.results_dir / 'candidate_parameters.yaml'
        
        candidates_data = {
            'tuning_metadata': {
                'tuning_date': datetime.now().isoformat(),
                'n_trials': results.get('n_trials', 0),
                'best_score': results.get('best_score', 0),
                'optimization_time': results.get('optimization_time', 0)
            },
            'candidates': {}
        }
        
        # 添加Top 5候选
        top_params = results.get('top_parameters', results.get('top_5_results', []))
        for i, item in enumerate(top_params[:5], 1):
            params = item.get('params', item) if isinstance(item, dict) else {}
            params = params if isinstance(params, dict) else {}
            raw_score = item.get('score') if isinstance(item, dict) else None
            score = raw_score if raw_score is not None else params.get('score')

            candidates_data['candidates'][f'candidate_{i}'] = {
                'min_topic_size': params.get('min_topic_size'),
                'n_neighbors': params.get('n_neighbors'),
                'n_components': params.get('n_components'),
                'min_cluster_size': params.get('min_cluster_size'),
                'min_samples': params.get('min_samples'),
                'coherence_score': score,
                'description': self._describe_parameters(params),
                'rank': i
            }
        
        with open(candidates_file, 'w', encoding='utf-8') as f:
            yaml.dump(candidates_data, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        logger.info(f"📋 候选参数文件已生成: {candidates_file}")
        return str(candidates_file)
    
    def _update_config_candidates(self, results: Dict[str, Any]):
        """更新config.yaml中的候选参数"""
        try:
            # 读取当前配置
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 确保candidate_parameters部分存在
            if 'candidate_parameters' not in config:
                config['candidate_parameters'] = {}
            
            # 添加候选参数
            top_params = results.get('top_parameters', results.get('top_5_results', []))
            for i, item in enumerate(top_params[:5], 1):
                params = item.get('params', item) if isinstance(item, dict) else {}
                params = params if isinstance(params, dict) else {}
                raw_score = item.get('score') if isinstance(item, dict) else None
                score = raw_score if raw_score is not None else params.get('score')

                config['candidate_parameters'][f'candidate_{i}'] = {
                    'min_topic_size': params.get('min_topic_size'),
                    'n_neighbors': params.get('n_neighbors'),
                    'n_components': params.get('n_components'),
                    'min_cluster_size': params.get('min_cluster_size'),
                    'min_samples': params.get('min_samples'),
                    'coherence_score': score,
                    'description': self._describe_parameters(params)
                }
            
            # 创建备份
            backup_path = self.config_path.parent / f'config_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
            with open(backup_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            # 保存更新的配置
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            logger.info("✅ config.yaml已更新候选参数")
            logger.info(f"📋 配置备份已保存: {backup_path}")
            
        except Exception as e:
            logger.error(f"❌ 更新config.yaml失败: {e}")
    
    def _generate_selection_guide(self, results: Dict[str, Any]):
        """生成候选参数选择指南"""
        guide_file = self.results_dir / '候选参数选择指南.txt'
        
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write("🎯 调参完成！候选参数选择指南\n")
            f.write("=" * 50 + "\n\n")

            f.write("📊 机器推荐的Top 5参数组合:\n")
            f.write("-" * 30 + "\n")

            top_params = results.get('top_parameters', results.get('top_5_results', []))
            for i, item in enumerate(top_params[:5], 1):
                params = item.get('params', item) if isinstance(item, dict) else {}
                params = params if isinstance(params, dict) else {}
                raw_score = item.get('score') if isinstance(item, dict) else None
                score = raw_score if raw_score is not None else params.get('score', 0)
                description = self._describe_parameters(params)

                f.write(f"\n🏆 候选 {i} (一致性分数: {float(score or 0):.4f})\n")
                f.write(f"   特征: {description}\n")
                f.write(f"   参数: 主题大小{params.get('min_topic_size')}, ")
                f.write(f"邻居数{params.get('n_neighbors')}, ")
                f.write(f"聚类大小{params.get('min_cluster_size')}\n")

                # 适用场景建议
                if i == 1:
                    f.write("   💡 推荐: 数学指标最优，通常是最佳起点\n")
                elif params.get('min_topic_size', 15) <= 10:
                    f.write("   💡 适合: 需要发现细粒度主题的研究\n")
                elif params.get('min_topic_size', 15) >= 30:
                    f.write("   💡 适合: 需要宏观主题概览的研究\n")
                else:
                    f.write("   💡 适合: 平衡型研究需求\n")

            f.write("\n" + "🔄 下一步操作:" + "\n")
            f.write("-" * 20 + "\n")
            f.write("1. 在config.yaml中设置: selected_candidate: 1 (选择候选1)\n")
            f.write("2. 修改运行模式: mode: 'analyze'\n")
            f.write("3. 运行分析: python main.py --run\n")
            f.write("4. 查看results/文件夹中的分析结果\n")
            f.write("5. 重复步骤1-4，测试其他候选参数\n")
            f.write("6. 对比不同候选的主题质量，选择最符合研究需求的\n\n")

            f.write("🎓 专家建议:\n")
            f.write("-" * 15 + "\n")
            f.write("• 数学分数高不等于研究价值高\n")
            f.write("• 请结合您的领域知识判断主题是否有意义\n")
            f.write("• 细粒度主题适合深度分析，粗粒度适合宏观概览\n")
            f.write("• 建议测试前3个候选，对比主题关键词的解释力\n")
            f.write("• 最终选择应服务于您的研究问题和论文论点\n")

        logger.info(f"📋 选择指南已生成: {guide_file}")


def save_tuning_results(config_path: str, optimization_results: Dict[str, Any]) -> str:
    """
    便捷函数：保存调参结果
    
    Args:
        config_path: 配置文件路径
        optimization_results: 优化结果
        
    Returns:
        候选参数文件路径
    """
    manager = TuningManager(config_path)
    return manager.save_tuning_results(optimization_results)
