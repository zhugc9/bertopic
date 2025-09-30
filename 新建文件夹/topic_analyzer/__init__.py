"""
Topic Analyzer Package
======================
BERTopic议题分析模块
"""

from .data_loader import DataLoader
from .model import TopicAnalyzer
from .pipeline import AnalysisPipeline, PipelineResult

__version__ = "1.0.0"
__author__ = "BERTopic Analysis System"

__all__ = [
    "DataLoader",
    "TopicAnalyzer",
    "AnalysisPipeline",
    "PipelineResult",
]