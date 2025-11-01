"""Minimal nv-one-logger shim that suppresses telemetry and stubs imports."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

__all__ = ["OneLogger", "OneLoggerConfig", "TrainingTelemetryConfig", "get_logger"]


def _ensure_null_handler(logger: logging.Logger) -> None:
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(logging.NullHandler())
    logger.propagate = False


def get_logger(name: Optional[str] = None, *, level: str | int | None = None) -> logging.Logger:
    logger = logging.getLogger(name or "nv-one-logger")
    if level is not None:
        numeric = getattr(logging, str(level).upper(), level) if isinstance(level, str) else level
        if isinstance(numeric, int):
            logger.setLevel(numeric)
    _ensure_null_handler(logger)
    return logger


@dataclass
class OneLoggerConfig:
    name: str = "nv-one-logger"
    level: str = "CRITICAL"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingTelemetryConfig:
    enabled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class _LoggerAdapter:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def debug(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - intentionally silent
        return

    info = debug
    warning = debug
    error = debug
    exception = debug


class OneLogger:
    """Drop-in replacement that swallows all telemetry calls."""

    def __init__(self, config: OneLoggerConfig | None = None) -> None:  # pragma: no cover - simple wiring
        self.config = config or OneLoggerConfig()
        self._logger = get_logger(self.config.name, level=self.config.level)

    def get_logger(
        self,
        name: Optional[str] = None,
        *,
        tags: Optional[Mapping[str, Any]] = None,
    ) -> _LoggerAdapter:
        logger = get_logger(name or self.config.name)
        if tags:
            logger = logging.LoggerAdapter(logger, extra={"tags": dict(tags)})  # type: ignore[arg-type]
        return _LoggerAdapter(logger)  # type: ignore[arg-type]

    def log_metrics(self, metrics: Mapping[str, Any], *, step: Optional[int] = None) -> None:  # pragma: no cover
        return

    def log_table(
        self,
        *,
        columns: Iterable[str],
        data: Iterable[Mapping[str, Any]],
        step: Optional[int] = None,
    ) -> None:  # pragma: no cover
        return

    def update_config(self, updates: MutableMapping[str, Any]) -> None:  # pragma: no cover
        for key, value in updates.items():
            setattr(self.config, key, value)


class TrainingTelemetryProvider:
    """Context manager consumed by NeMo's telemetry hooks."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - noop
        self.config = TrainingTelemetryConfig()

    def __enter__(self) -> "TrainingTelemetryProvider":  # pragma: no cover - noop
        return self

    def __exit__(self, *_exc: Any) -> bool:  # pragma: no cover - noop
        return False


def _noop(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - intentional noop
    return None


_TRAINING_CALLBACKS: Dict[str, Any] = {
    "on_app_start": _noop,
    "on_app_end": _noop,
    "on_run_start": _noop,
    "on_run_end": _noop,
    "on_stage_start": _noop,
    "on_stage_end": _noop,
}


def _make_provider(*_args: Any, **_kwargs: Any) -> TrainingTelemetryProvider:  # pragma: no cover - noop
    return TrainingTelemetryProvider()


_KNOWN_MODULE_SYMBOLS: Dict[str, Dict[str, Any]] = {
    "nv_one_logger.api": {
        "OneLogger": OneLogger,
        "OneLoggerConfig": OneLoggerConfig,
        "get_logger": get_logger,
    },
    "nv_one_logger.api.config": {
        "OneLoggerConfig": OneLoggerConfig,
    },
    "nv_one_logger.api.logger": {
        "OneLogger": OneLogger,
        "get_logger": get_logger,
    },
    "nv_one_logger.training_telemetry": {},
    "nv_one_logger.training_telemetry.api": {
        "TrainingTelemetryConfig": TrainingTelemetryConfig,
        "TrainingTelemetryProvider": TrainingTelemetryProvider,
        "get_training_telemetry_provider": _make_provider,
        "log_app_metrics": _noop,
        **_TRAINING_CALLBACKS,
    },
    "nv_one_logger.training_telemetry.api.config": {
        "TrainingTelemetryConfig": TrainingTelemetryConfig,
    },
    "nv_one_logger.training_telemetry.api.metrics": {
        "log_app_metrics": _noop,
    },
    "nv_one_logger.training_telemetry.api.callbacks": _TRAINING_CALLBACKS,
    "nv_one_logger.training_telemetry.api.training_telemetry_provider": {
        "TrainingTelemetryProvider": TrainingTelemetryProvider,
        "get_training_telemetry_provider": _make_provider,
    },
}


class _NoOpObject:
    """Object that absorbs all interactions for unknown symbols."""

    def __call__(self, *args: Any, **kwargs: Any) -> "_NoOpObject":  # pragma: no cover - trivial
        return self

    def __getattr__(self, _name: str) -> "_NoOpObject":  # pragma: no cover - trivial
        return self

    def __iter__(self):  # pragma: no cover - trivial
        return iter(())

    def __enter__(self) -> "_NoOpObject":  # pragma: no cover - trivial
        return self

    def __exit__(self, *_exc: Any) -> bool:  # pragma: no cover - trivial
        return False


def _module_getattr(module_name: str, name: str):  # pragma: no cover - trivial
    mapping = _KNOWN_MODULE_SYMBOLS.get(module_name)
    if mapping and name in mapping:
        return mapping[name]

    if name and name[0].isupper():
        return type(name, (_NoOpObject,), {})

    return _noop


class _NvOneLoggerStubLoader(importlib.abc.Loader):
    def create_module(self, spec: importlib.machinery.ModuleSpec):
        module = types.ModuleType(spec.name)
        module.__file__ = __file__
        module.__package__ = spec.name
        module.__path__ = []  # type: ignore[attr-defined]
        mapping = _KNOWN_MODULE_SYMBOLS.get(spec.name)
        if mapping:
            module.__dict__.update(mapping)
            module.__all__ = sorted(mapping)  # type: ignore[attr-defined]
        else:
            module.__all__ = []  # type: ignore[attr-defined]
        module.__getattr__ = lambda attr, name=spec.name: _module_getattr(name, attr)  # type: ignore[attr-defined]
        return module

    def exec_module(self, module: types.ModuleType) -> None:  # pragma: no cover - nothing to execute
        return


class _NvOneLoggerStubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):
        if not fullname.startswith("nv_one_logger"):
            return None

        existing = importlib.machinery.PathFinder.find_spec(fullname, path)
        if existing is not None:
            return None

        spec = importlib.machinery.ModuleSpec(fullname, _NvOneLoggerStubLoader(), is_package=True)
        spec.submodule_search_locations = []  # type: ignore[assignment]
        return spec


if not any(isinstance(finder, _NvOneLoggerStubFinder) for finder in sys.meta_path):
    sys.meta_path.append(_NvOneLoggerStubFinder())

