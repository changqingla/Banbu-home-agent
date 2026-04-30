from __future__ import annotations

from dataclasses import dataclass

from .base import ReactiveTool
from .device_list import ListRelevantDevicesTool
from .device_state import GetDeviceStateTool
from .execute_plan import ExecutePlanTool


@dataclass(frozen=True)
class ReactiveToolRegistry:
    tools: tuple[ReactiveTool, ...]

    @classmethod
    def default(cls) -> "ReactiveToolRegistry":
        return cls(
            tools=(
                GetDeviceStateTool(),
                ListRelevantDevicesTool(),
                ExecutePlanTool(),
            )
        )

    def get(self, name: str) -> ReactiveTool | None:
        for tool in self.tools:
            if tool.spec.name == name:
                return tool
        return None

    def prompt_specs(self) -> list[dict]:
        specs: list[dict] = []
        for tool in self.tools:
            spec = tool.spec
            specs.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                    "safety": list(spec.safety),
                }
            )
        return specs
