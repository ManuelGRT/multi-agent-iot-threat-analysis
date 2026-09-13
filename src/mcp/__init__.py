# src/mcp/__init__.py
"""Capa MCP del sistema multiagente final.

Tres servidores MCP construidos con FastMCP (SDK oficial) exponen el runtime
online como herramientas conectables y auditables:

- ``inference_server``:   estandarizacion Mistral viva (sin fallback),
                          deteccion y clasificacion con modelos .joblib.
- ``case_memory_server``: cache de estandarizacion, memoria de casos y trazas
                          persistentes (dos ficheros SQLite en una sola raiz).
- ``threat_intel_server``: tipo de ataque -> ATT&CK/CAPEC y mitigaciones.

La preparacion de datasets, la sanitizacion y la evaluacion permanecen fuera
del runtime online, bajo ``src.eval`` y los scripts de validacion.

Cada servidor puede ejecutarse como proceso MCP real::

    python -m src.mcp.inference_server

o consumirse en modo in-process (mismas tools, sin subproceso) a traves de
``src.mcp.client.MCPToolClient`` para tests deterministas.
"""
