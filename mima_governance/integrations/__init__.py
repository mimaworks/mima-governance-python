"""Framework integrations for automatic attestation.

All integrations are imported lazily so that missing optional dependencies
do not break the base package. Each integration raises a clear ImportError
with install instructions when its dependency is absent.
"""

_INTEGRATIONS = {
    "MimaLangChainCallback":  "mima_governance.integrations.langchain_callback",
    "MimaLlamaIndexHandler":  "mima_governance.integrations.llamaindex_handler",
    "MimaAutoGenMiddleware":  "mima_governance.integrations.autogen_middleware",
}


def __getattr__(name: str):
    if name in _INTEGRATIONS:
        module_path = _INTEGRATIONS[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_INTEGRATIONS.keys())
