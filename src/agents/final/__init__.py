# src/agents/final/__init__.py
"""Agentes finales del sistema (Fase 3 del plan de cierre).

Sustituyen a los agentes por reglas/LLM del orquestador original usando las
tools MCP con los modelos .joblib preparados. Cada agente registra un
``TraceEntry`` con inicio, fin, confianza y errores en la traza del caso.
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
