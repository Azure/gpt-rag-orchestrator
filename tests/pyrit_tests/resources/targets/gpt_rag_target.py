# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import json
import logging
from typing import Optional

from pyrit.common import net_utility
from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestResponse
from pyrit.models import construct_response_from_request
from pyrit.prompt_target import PromptTarget,PromptChatTarget
from pyrit.common import default_values
from pyrit.models import PromptRequestPiece


logger = logging.getLogger(__name__)


class GptRagTarget(PromptChatTarget):
    API_KEY_ENVIRONMENT_VARIABLE: str = "GPT_RAG_ORCH_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "GPT_RAG_ORCH_ENDPOINT"

    def __init__(
        self,
        *,
        endpoint: str = None,
        api_key: str = None,
        memory: MemoryInterface = None,
    ):
       
        super().__init__(memory=memory)
        self._conversation_id = None
        self._endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )
        self._api_key = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self._validate_request(prompt_request=prompt_request)
        request: PromptRequestPiece = prompt_request.request_pieces[0]

        prompt_req_res_entries = self._memory.get_conversation(conversation_id=request.conversation_id)
        prompt_req_res_entries.append(prompt_request)

        logger.info(f"Sending the following prompt to the prompt target: {prompt_request}")
        
        response = await self._complete_text_async(request.converted_value,request.conversation_id)

        response_entry = construct_response_from_request(request=request, response_text_pieces=[response])

        return response_entry

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")
    
    async def _complete_text_async(self, text: str,conversation_id :str) -> str:
        payload: dict[str, object] = {
            "question": text,
            "conversation_id": conversation_id
        }

        resp = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=self._endpoint, method="GET", request_body=payload, post_type="json",headers={"x-functions-key": self._api_key}
        )

        if not resp.text:
            raise ValueError("The chat returned an empty response.")

        answer = json.loads(resp.text)["answer"]
        logger.info(f'Received the following response from the prompt target "{answer}"')
        return answer