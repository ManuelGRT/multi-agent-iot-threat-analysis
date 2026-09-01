# src/mcp/__init__.py
"""Capa MCP del sistema multiagente final.

Tres servidores MCP construidos con FastMCP (SDK oficial) exponen el runtime
online como herramientas conectables y auditables:

- ``inference_server``:   estandarizacion Mistral obligatoria (sin fallback),
                          deteccion y clasificacion con modelos .joblib.
- ``case_memory_server``: memoria de casos con trazas (SQLite).
- ``threat_intel_server``: mapeo familia -> ATT&CK/CAPEC y mitigaciones.

La preparacion de datasets, la sanitizacion y la evaluacion permanecen fuera
del runtime online, bajo ``src.eval`` y los scripts de validacion.

Cada servidor puede ejecutarse como proceso MCP real::

    python -m src.mcp.inference_server

o consumirse en modo in-process (mismas tools, sin subproceso) a traves de
``src.mcp.client.MCPToolClient`` para tests deterministas.
"""
