from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_agent_registry
from app.agents.registry import AgentRegistry
from app.schemas import AgentMetadataResponse

router = APIRouter(tags=["agents"])


@router.get("/agents", response_model=list[AgentMetadataResponse])
def list_agents(
    registry: AgentRegistry = Depends(get_agent_registry),
) -> list[AgentMetadataResponse]:
    return [
        AgentMetadataResponse(
            agent_id=item.agent_id,
            agent_name=item.agent_name,
            agent_role=item.agent_role,
            agent_kind=item.agent_kind,
            description=item.description,
            allowed_tools=item.allowed_tools,
            tags=item.tags,
        )
        for item in registry.list_metadata()
    ]
