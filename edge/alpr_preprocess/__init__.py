"""Khối Video Sampling + IQA dùng độc lập và tích hợp ALPR."""
from .quality import QualityAssessor, QualityConfig, QualityReport
from .sampling import MotionGate, MotionConfig, TemporalSampler
from .pipeline import FrameSelectionPipeline, PipelineConfig

__all__ = [
    "QualityAssessor", "QualityConfig", "QualityReport",
    "MotionGate", "MotionConfig", "TemporalSampler",
    "FrameSelectionPipeline", "PipelineConfig",
]
