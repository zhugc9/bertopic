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
            # 1. 生成候选参数文件
            candidates_file = self._generate_candidates_file(optimization_results)
            
            # 2. 更新config.yaml中的候选参数
            self._update_config_candidates(optimization_results)
            
            logger.info("✅ 调参结果保存完成")
            return candidates_file
            
        except Exception as e:
            logger.error(f"❌ 调参结果保存失败: {e}")
            raise
    
    def _generate_candidates_file(self, results: Dict[str, Any]) -> str:
        """生成候选参数文件"""
        candidates_file = self.results_dir / '候选参数.yaml'
        
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
        top_params = results.get('top_parameters') or results.get('top_5_results') or []
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
                'quality_score': score,
                'rank': i
            }
        
        with open(candidates_file, 'w', encoding='utf-8') as f:
            yaml.dump(candidates_data, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        logger.info(f"📋 候选参数文件已生成: {candidates_file}")
        return str(candidates_file)
    
    def _update_config_candidates(self, results: Dict[str, Any]):
        """更新config.yaml中的候选参数（KISS原则：从manual读取全部配置，只更新候选参数）"""
        try:
            # KISS方案：从config_manual.yaml读取完整配置作为基础
            manual_config_path = self.config_path.parent / 'config_manual.yaml'
            if manual_config_path.exists():
                with open(manual_config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info("✓ 从config_manual.yaml读取基础配置")
            else:
                # 如果manual不存在，读取当前config
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.warning("⚠ config_manual.yaml不存在，使用当前config.yaml")
            
            # 确保candidate_parameters部分存在
            if 'candidate_parameters' not in config:
                config['candidate_parameters'] = {}
            
            # 只更新候选参数部分
            top_params = results.get('top_parameters') or results.get('top_5_results') or []
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
                    'quality_score': score,
                    'rank': i
                }
            
            # 创建备份
            backup_dir = self.results_dir / 'config_backups'
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f'config_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
            with open(backup_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            # 保存更新的配置到config.yaml
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            logger.info("✅ config.yaml已更新候选参数（所有用户配置已从manual保留）")
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

            top_params = results.get('top_parameters') or results.get('top_5_results') or []
            for i, item in enumerate(top_params[:5], 1):
                params = item.get('params', item) if isinstance(item, dict) else {}
                params = params if isinstance(params, dict) else {}
                raw_score = item.get('score') if isinstance(item, dict) else None
                score = raw_score if raw_score is not None else params.get('score')

                if isinstance(score, (int, float)):
                    score_str = f"{float(score):.4f}"
                elif score is not None:
                    score_str = str(score)
                else:
                    score_str = "N/A"
                f.write(f"\n🏆 候选 {i} (一致性分数: {score_str})\n")
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
