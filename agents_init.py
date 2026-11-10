from typing import Type
from .agent import Agent, Playback
from .templates.my_agent import MyAgent
AVAILABLE_AGENTS: dict[str, Type[Agent]] = {"myagent": MyAgent}
