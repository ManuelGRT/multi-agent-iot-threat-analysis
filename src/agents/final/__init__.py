# src/agents/final/__init__.py
"""Agentes operativos del sistema multiagente.

Usan herramientas MCP y los modelos ``joblib`` desplegados. Cada agente
registra inicio, fin, confianza y errores en la traza del caso.
"""
from src.agents.final.final_standardizer import FinalStandardizer
from src.agents.final.final_detector import FinalDetector
from src.agents.final.final_classifier import FinalClassifier
from src.agents.final.final_mitigator import FinalMitigator
from src.agents.final.final_judge import FinalJudge
from src.agents.final.auditor import CaseAuditor, CaseAuditReport

__all__ = [
    "FinalStandardizer",
    "FinalDetector",
    "FinalClassifier",
    "FinalMitigator",
    "FinalJudge",
    "CaseAuditor",
    "CaseAuditReport",
]
