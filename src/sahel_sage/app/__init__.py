"""The offline advisory console (FastAPI + single-file UI).

Imports here stay lazy: `fastapi` lives in the optional ``app`` extra, so the
rest of the package (data, training, retrieval) must remain importable on a
machine that never installs the server.
"""
