"""Public Revolution API with lazy imports for lightweight tool-specific runners."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "EoHEngine": (".algorithm", "EoHEngine"),
    "Heuristic": (".algorithm", "Heuristic"),
    "VerilogEvaluator": (".evaluation", "VerilogEvaluator"),
    "VerilogEvaluatorICRTL": (".evaluation", "VerilogEvaluatorICRTL"),
    "SynthesisEvaluator": (".evaluation", "SynthesisEvaluator"),
    "DCPTEvaluator": (".dcpt_evaluation", "DCPTEvaluator"),
    "LLMInterface": (".llm", "LLMInterface"),
    "EoHLogger": (".logging", "EoHLogger"),
    "StreamRedirector": (".utils", "StreamRedirector"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
