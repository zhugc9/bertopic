"""统一分析管道"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from config_loader import load_runtime_config
from .data_loader import DataLoader
from .model import TopicAnalyzer


@dataclass
class PipelineResult:
    documents: list
    metadata_df: Any
    topic_model: Any
    topics: list


class AnalysisPipeline:
    def __init__(self, config_path: Path, runtime_config: Dict[str, Any] = None):
        self.config_path = Path(config_path)
        if runtime_config is not None:
            self.runtime_config = runtime_config
        else:
            self.runtime_config = load_runtime_config(self.config_path)
        self.data_loader = DataLoader(self.runtime_config)
        self.topic_analyzer = TopicAnalyzer(self.runtime_config)

    def run_analysis(self) -> PipelineResult:
        documents, metadata_df = self.data_loader.load_and_prepare_data()
        topic_model, topics = self.topic_analyzer.train_bertopic_model(documents)
        return PipelineResult(
            documents=documents,
            metadata_df=metadata_df,
            topic_model=topic_model,
            topics=topics,
        )

    def generate_results(self, result: PipelineResult) -> Dict[str, Path]:
        self.topic_analyzer.generate_results(
            result.topic_model,
            result.documents,
            result.topics,
            result.metadata_df,
        )
        results_dir = Path(self.runtime_config['results_paths']['output_dir'])
        return {
            "results_dir": results_dir,
        }

    def run_hyperparameter_search(self) -> Dict[str, Any]:
        documents, _ = self.data_loader.load_and_prepare_data()
        return self.topic_analyzer.optimize_hyperparameters(documents)

