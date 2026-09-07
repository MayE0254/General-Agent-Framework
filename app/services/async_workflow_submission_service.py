from __future__ import annotations

from typing import TYPE_CHECKING

from celery.result import AsyncResult

from app.core import NotFoundError
from app.models import AsyncWorkflowSubmissionRecord
from app.repositories import AsyncWorkflowSubmissionRepository
from app.tasks.celery_app import celery_app

if TYPE_CHECKING:
    from app.schemas import WorkflowRunRequest


_PENDING_STATES = {"pending", "received", "started", "retry"}


class AsyncWorkflowSubmissionService:
    """Persist and refresh async workflow submissions from Celery state."""

    def __init__(self, repository: AsyncWorkflowSubmissionRepository) -> None:
        self._repository = repository

    def get_by_request_id(
        self, request_id: str
    ) -> AsyncWorkflowSubmissionRecord | None:
        return self._repository.get_by_request_id(request_id)

    def create_submission(
        self,
        request: WorkflowRunRequest,
        *,
        task_id: str,
        task_name: str,
        queue_name: str,
    ) -> AsyncWorkflowSubmissionRecord:
        record = AsyncWorkflowSubmissionRecord(
            task_id=task_id,
            request_id=request.request_id,
            session_id=request.session_id,
            user_id=request.user_id,
            initial_agent_id=request.initial_agent_id,
            task_name=task_name,
            queue_name=queue_name,
            status="pending",
            message="workflow submitted to async queue",
            metadata_payload=request.model_dump(mode="json"),
        )
        saved = self._repository.add(record)
        self._repository.commit()
        return saved

    def get_submission(self, task_id: str) -> AsyncWorkflowSubmissionRecord:
        record = self._repository.get_by_task_id(task_id)
        if record is None:
            raise NotFoundError(
                "Async workflow submission was not found.",
                details={"task_id": task_id},
            )
        return record

    def refresh_submission_status(
        self,
        task_id: str,
        *,
        raise_if_missing: bool = True,
    ) -> AsyncWorkflowSubmissionRecord | None:
        record = self._repository.get_by_task_id(task_id)
        if record is None:
            if raise_if_missing:
                raise NotFoundError(
                    "Async workflow submission was not found.",
                    details={"task_id": task_id},
                )
            return None

        async_result = AsyncResult(task_id, app=celery_app)
        status = async_result.status.lower()
        self._apply_async_result(record, status=status, async_result=async_result)
        self._repository.commit()
        return record

    def list_submissions(
        self,
        *,
        status: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
        refresh_statuses: bool = True,
    ) -> dict[str, object]:
        items, total = self._repository.list_submissions(
            status=status,
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )
        if refresh_statuses:
            dirty = False
            for item in items:
                if item.status in _PENDING_STATES:
                    async_result = AsyncResult(item.task_id, app=celery_app)
                    next_status = async_result.status.lower()
                    dirty = self._apply_async_result(
                        item,
                        status=next_status,
                        async_result=async_result,
                    ) or dirty
            if dirty:
                self._repository.commit()
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {
                "status": status,
                "request_id": request_id,
                "session_id": session_id,
                "user_id": user_id,
            },
        }

    def close(self) -> None:
        self._repository.close()

    def _apply_async_result(
        self,
        record: AsyncWorkflowSubmissionRecord,
        *,
        status: str,
        async_result: AsyncResult,
    ) -> bool:
        changed = False
        message = _status_message(status)
        if record.status != status:
            record.status = status
            changed = True
        if record.message != message:
            record.message = message
            changed = True

        if status == "success" and isinstance(async_result.result, dict):
            if record.result_payload != async_result.result:
                record.result_payload = async_result.result
                changed = True
            if record.traceback is not None:
                record.traceback = None
                changed = True
        elif status == "failure":
            traceback = async_result.traceback or ""
            if record.traceback != traceback:
                record.traceback = traceback
                changed = True
        return changed


def _status_message(status: str) -> str:
    messages = {
        "pending": "workflow is pending in async queue",
        "received": "workflow has been received by worker",
        "started": "workflow execution started by worker",
        "retry": "workflow execution is being retried",
        "success": "workflow completed successfully",
        "failure": "workflow execution failed",
    }
    return messages.get(status, f"workflow status is {status}")
