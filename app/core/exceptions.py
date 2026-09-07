from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base application exception with structured HTTP metadata."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "app_error",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class DependencyInitializationError(AppError):
    def __init__(
        self,
        message: str = "Failed to initialize application dependency.",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code="dependency_initialization_failed",
            status_code=500,
            details=details,
        )


class ResourceNotReadyError(AppError):
    def __init__(
        self,
        message: str = "Requested application resource is not ready.",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code="resource_not_ready",
            status_code=503,
            details=details,
        )


class NotFoundError(AppError):
    def __init__(
        self,
        message: str = "Requested resource was not found.",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code="resource_not_found",
            status_code=404,
            details=details,
        )
