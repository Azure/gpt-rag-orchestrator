from typing import Optional

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies
from plugins.realtime.realtime_types import RTActionRequest

class RealTimeVoiceStrategy(BaseAgentStrategy):
    """
    An optimized Real-Time Voice Interaction strategy
    """
    def __init__(self):
        super().__init__()
        self.strategy_type = AgentStrategies.REALTIME_VOICE

        self.model: Optional[str]= None
        self.temperature: Optional[float] = 0.7
        self.max_response_output_tokens: Optional[int] = 1024
        self.tools: list = [] 
        self.conversation_id: Optional[str] = None

    async def initiate_agent_flow(self, user_message: str):
        # For real-time voice, the flow is handled elsewhere
        self.default_session_config: dict = {
            "instructions": await self._read_prompt("default"),
            # "model": self.model,
            "temperature": self.temperature,
            "max_response_output_tokens": self.max_response_output_tokens,
            "tool_choice":"auto" if len(self.tools) > 0 else "none",
            "tools": self.tools
        }
        return True
    
    async def _get_realtime_session_config(self, name: str) -> dict:
        """
        Retrieve the session configuration for real-time voice interactions.
        """
        # For simplicity, return the default session config
        prompt = await self._read_prompt(name)
        self.default_session_config["instructions"] = prompt
        return self.default_session_config if prompt else {}
    
    async def handle_realtime_voice_action(self, action: RTActionRequest):
        """
        Handle real-time voice actions.
        """
        # Implement real-time voice action handling logic here
        # action type will same name as function to call
        result = await getattr(self, f"handle_{action.type}_action")(**action.payload)
        return  result
    
    async def handle_get_prompt_action(self,**kwargs) -> str:
        """
        Return the system prompt for real-time voice interactions.
        """
        self._default_prompt  = await self._read_prompt(kwargs.get("name"))
        return self._default_prompt
    
    async def handle_get_session_config_action(self, **kwargs) -> dict:
        """
        Return the session configuration for real-time voice interactions.
        """
        if kwargs.get("name") =="default":
            return self.default_session_config
        return await self._get_realtime_session_config(kwargs.get("name"))
