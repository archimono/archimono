"""Application-layer command-orchestration workflows for archimono.

This package holds the workflow logic that adapts parsed CLI arguments into core
operations, so the dependency arrow points strictly inward
(``cli -> workflows -> core``). Each submodule remains importable on its own;
adapters import submodules rather than this package root.

``verify`` is intentionally exempt from this package: ``archimono.verification``
already serves as verify's application layer, so there is no
``workflows/verify.py``.
"""
