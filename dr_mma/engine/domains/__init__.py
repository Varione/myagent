"""Domain specialized agents."""

from dr_mma.engine.domains.code_dev import CodeDevAgent
from dr_mma.engine.domains.data_analysis import DataAnalysisAgent
from dr_mma.engine.domains.engineering_sim import EngineeringSimAgent
from dr_mma.engine.domains.knowledge_mgmt import KnowledgeMgmtAgent
from dr_mma.engine.domains.paper_writing import PaperWritingAgent

__all__ = [
    "PaperWritingAgent",
    "CodeDevAgent",
    "EngineeringSimAgent",
    "DataAnalysisAgent",
    "KnowledgeMgmtAgent",
]
