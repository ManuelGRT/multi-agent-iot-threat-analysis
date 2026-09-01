# src/mcp/__init__.py
"""Capa MCP del sistema multiagente (Fase 2 del plan de cierre).

Cinco servidores MCP construidos con FastMCP (SDK oficial) exponen los
artefactos preparados del TFM como herramientas conectables y auditables:

- ``datasets_server``:    datasets crudos y corpus estandarizado.
- ``inference_server``:   estandarizacion Mistral obligatoria (sin fallback),
                          deteccion y clasificacion con modelos .joblib.
- ``case_memory_server``: memoria de casos con trazas (SQLite).
- ``threat_intel_server``: mapeo familia -> ATT&CK/CAPEC y mitigaciones.
- ``evaluation_server``:  metricas, comparacion con baselines e informes.

Cada servidor puede ejecutarse como proceso MCP real::

    python -m src.mcp.inference_server

o consumirse en modo in-process (mismas tools, sin subproceso) a traves de
``src.mcp.client.MCPToolClient`` para tests deterministas.
"""
