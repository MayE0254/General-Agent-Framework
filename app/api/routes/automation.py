from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.dependencies import AuthenticatedUser, get_current_user
from app.business import business_automation_agents

router = APIRouter(tags=["automation"])


class AutomationAgentResponse(BaseModel):
    id: str
    name: str
    description: str
    schedule: str
    owner_role: str
    business: str
    enabled: bool


@router.get("/automation/agents", response_model=list[AutomationAgentResponse])
def list_automation_agents(
    _: AuthenticatedUser = Depends(get_current_user),
) -> list[AutomationAgentResponse]:
    """List the scheduled automation agents (beat-driven, non-personal)."""
    return [
        AutomationAgentResponse(
            id=item["id"],
            name=item["name"],
            description=item["description"],
            schedule=item["schedule"],
            owner_role=item["owner_role"],
            business=item["business"],
            enabled=item["enabled"],
        )
        for item in business_automation_agents()
    ]
