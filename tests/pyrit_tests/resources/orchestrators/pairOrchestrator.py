# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging
import uuid
from collections import defaultdict
from typing import Optional

from tqdm.auto import tqdm

from pyrit.common.path import DATASETS_PATH
from pyrit.exceptions import pyrit_json_retry, InvalidJsonException
from pyrit.memory import MemoryInterface
from pyrit.models import PromptTemplate, PromptRequestResponse, PromptRequestPiece, Score
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import PromptNormalizer, NormalizerRequest, NormalizerRequestPiece
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class PAIROrchestrator(Orchestrator):
    """
    This orchestrator implements the Prompt Automatic Iterative Refinement (PAIR) algorithm

    This algorithm was published and described in the paper: Chao, Patrick, et al. Jailbreaking Black Box Large
    Language Models in Twenty Queries. arXiv:2310.08419, arXiv, 13 Oct. 2023. arXiv.org, http://arxiv.org/abs/2310.08419

    The authors published a reference implementation in the following repository: https://github.com/patrickrchao/\
    JailbreakingLLMs/blob/main/system_prompts.py

    """

    def __init__(
        self,
        *,
        memory: Optional[MemoryInterface] = None,
        memory_labels: Optional[dict[str, str]] = None,
        verbose: bool = False,
        prompt_target: PromptChatTarget,
        desired_target_response_prefix: str,
        red_teaming_chat: PromptChatTarget,
        conversation_objective: str,
        number_of_conversation_streams: int = 20,
        max_conversation_depth: int = 3,
        stop_on_first_success: bool = True,
        scorer: Scorer,
        scorer_sensitivity: float = 1.0,
        prompt_converters: Optional[list[PromptConverter]] = None,
        single_turn_jailbreak_only: bool = True,
    ) -> None:
        """
        Initializes the PAIR orchestrator to run the PAIR algorithm against two targets.

        Args:
            memory: The memory interface to use. If None, a new memory interface will be created.
            memory_labels (dict[str, str], optional): A free-form dictionary for tagging prompts with custom labels.
            These labels can be used to track all prompts sent as part of an operation, score prompts based on
            the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category category.
            Users can define any key-value pairs according to their needs. Defaults to None.
            verbose: Whether to print debug information. Defaults to False.
            prompt_target: The target model to jailbreak.
            desired_target_response_prefix: An example of a desired response from the target. This is used to compare
                against the target's actual response to determine if the target has been compromised. E.g., "Sure, here
                is code for a keylogger."
            red_teaming_chat: The attacker model to use to jailbreak the target.
            conversation_objective: The objective for the attacker to achieve, as a string. E.g. "Generate a phishing
                email" or "Generate code for a keylogger", etc. Defaults to "Sure, I'd be happy to assist with that".
            max_conversation_depth: The maximum number of turns in a conversation. Defaults to 3 based on the PAIR's
                paper.
            number_of_conversation_streams: The number of independent conversation streams to run for PAIR algorithm.
                defaults to 20 based on the PAIR's paper.
            stop_on_first_success: Whether to stop the algorithm after the first successful attack. Defaults to True.
                If False, the algorithm will continue to run until all conversation streams are exhausted possibly
                finding multiple successful jailbreaks.
            scorer (Score, optional): The scorer to use for scoring the target's response.
            scorer_sensitivity (float, optional): The sensitivity of the scorer.
                This value indicates the scorer's tolerance for scores below the maximum score value.
                Note that the sensitivity refers to the normalized score (value-min)/(max-min).
                For example, for a scoring scale 1-10, the scores map as follows:
                1 -> 0.0, 2 -> 0.11, 3 -> 0.22, ..., 10 -> 1.0.
                A sensitivity of 0.7 would consider a score of 7 or above as a successful jailbreak.
                Defaults to 1.0 which is equivalent to 10/10 in the scale of the PAIR paper.
            prompt_converters (list[PromptConverter], optional): List of prompt converters. These are stacked in
                the order they are provided. The default PAIR implementation does not use any converters.
        """
        super().__init__(
            memory=memory, memory_labels=memory_labels, verbose=verbose, prompt_converters=prompt_converters
        )

        self.successful_jailbreaks: list[PromptRequestResponse] = []
        self._prompt_target = prompt_target
        self._adversarial_target = red_teaming_chat
        self._max_conversation_depth = max_conversation_depth
        self._stop_on_first_success = stop_on_first_success
        self._conversation_objective = conversation_objective
        self._number_of_conversation_streams = number_of_conversation_streams
        self._last_attacker_conversation_id = ""
        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._single_turn_jailbreak_only = single_turn_jailbreak_only
        self._scorer_sensitivity = scorer_sensitivity
        self._scorer = scorer
        self._desired_target_response_prefix = desired_target_response_prefix

        # Load the prompt templates for the attacker
        self._attacker_prompt_template = PromptTemplate.from_yaml_file(
            file=DATASETS_PATH / "orchestrators" / "pair" / "attacker_system_prompt.yaml"
        )

        self._validate()

    def _validate(self) -> None:
        if self._scorer_sensitivity < 0 or self._scorer_sensitivity > 1:
            raise ValueError("scorer_sensitivity must be between 0 and 1")

        if self._scorer.scorer_type != "float_scale":
            raise ValueError("PAIR orchestrator requires a scorer with a 'float_scale' scorer_type")

    @pyrit_json_retry
    async def _get_attacker_response_and_store(
        self, *, target_response: str, start_new_conversation: bool = False
    ) -> str:
        """
        Generates an attacker response based on the target response.

        Args:
            target_response: The response from the target.
            start_new_conversation: Whether to start a new conversation with the attacker. Defaults to False.

        Returns:
            The attacker response.
        """
        if start_new_conversation:
            self._last_attacker_conversation_id = str(uuid.uuid4())
            attacker_system_prompt = self._attacker_prompt_template.apply_custom_metaprompt_parameters(
                goal=self._conversation_objective, target_str=self._desired_target_response_prefix
            )
            self._adversarial_target.set_system_prompt(
                system_prompt=attacker_system_prompt, conversation_id=self._last_attacker_conversation_id
            )
        # Send a new request to the attacker
        attacker_response = await self._prompt_normalizer.send_prompt_async(
            normalizer_request=NormalizerRequest(
                request_pieces=[
                    NormalizerRequestPiece(
                        request_converters=self._prompt_converters,
                        prompt_value=target_response,
                        prompt_data_type="text",
                    )
                ]
            ),
            target=self._adversarial_target,
            conversation_id=self._last_attacker_conversation_id,
            labels=self._global_memory_labels,
            orchestrator_identifier=self.get_identifier(),
        )
        return self._parse_attacker_response(response=attacker_response)

    async def _get_target_response_and_store(
        self, *, text: str, conversation_id: Optional[str] = None
    ) -> PromptRequestResponse:
        """Interact with the target in a zero-shot fashion.

        A new UUID will be generated for each interaction with the target. This zero-shot interation is how the PAIR
        algorithm is specified
        Args:
            text: prompt to send to target

        Returns:
            response from target.
        """
        curr_conversation_id = conversation_id or str(uuid.uuid4())
        target_response = await self._prompt_normalizer.send_prompt_async(
            normalizer_request=NormalizerRequest(
                request_pieces=[
                    NormalizerRequestPiece(
                        request_converters=self._prompt_converters, prompt_value=text, prompt_data_type="text"
                    )
                ]
            ),
            target=self._prompt_target,
            conversation_id=curr_conversation_id,
            labels=self._global_memory_labels,
            orchestrator_identifier=self.get_identifier(),
        )
        return target_response

    def _parse_attacker_response(self, *, response: PromptRequestResponse) -> str:
        try:
            json_response = json.loads(response.request_pieces[0].converted_value)
            attacker_improvement_rationale = json_response["improvement"]  # noqa
            attacker_suggested_prompt = json_response["prompt"]
        except (json.JSONDecodeError, KeyError):
            # This forces the @pyrit_json_retry decorator to retry the function
            raise InvalidJsonException(message="Error parsing attacker response. Retrying...")
        return attacker_suggested_prompt

    def _should_stop(self, *, prompt_responses: list) -> bool:
        """
        Determines if the algorithm should stop based on the _stop_on_first_success flag and existing jailbreaks.
        """
        if prompt_responses and self._stop_on_first_success:
            return True
        return False

    async def _process_conversation_stream(self, *, index: int) -> list[PromptRequestResponse]:
        """
        Processes a single conversation stream in the PAIR algorithm.
        """
        jailbreaks: list[PromptRequestResponse] = []
        prompt_to_improve_upon = "Hello world!"
        t = tqdm(range(self._max_conversation_depth), disable=not self._verbose, leave=False, desc="Conversation depth")
        conversation_id = str(uuid.uuid4())
        for depth in t:
            try:
                attacker_prompt_suggestion = await self._get_attacker_response_and_store(
                    target_response=prompt_to_improve_upon, start_new_conversation=(depth == 0)
                )
            except InvalidJsonException:
                logger.warn("Invalid JSON response from attacker despite retries.")
                break
            if self._single_turn_jailbreak_only:
                # Create a new conversation ID for each turn
                conversation_id = str(uuid.uuid4())
            target_response = await self._get_target_response_and_store(
                text=attacker_prompt_suggestion,
                conversation_id=conversation_id,
            )
            # This is called JUDGE in the paper
            score = (
                await self._scorer.score_async(
                    request_response=target_response.request_pieces[0], task=self._conversation_objective
                )
            )[0]
            if self._is_jailbreak_successful(score=score):
                jailbreaks.append(target_response)
                break
            prompt_to_improve_upon = target_response.request_pieces[0].converted_value
        return jailbreaks

    def _is_jailbreak_successful(self, *, score: Score) -> bool:
        """
        Determines if a jailbreak attempt was successful based on the scorer's response (the paper calls is "judge").
        """
        score_value = float(score.score_value)
        return float(score_value) >= self._scorer_sensitivity

    async def run(self) -> list[PromptRequestResponse]:
        """
        Runs the PAIR algorithm against the target and attacker.
        """

        prompt_response_pair_for_jailbreaks: list[PromptRequestResponse] = []

        t = tqdm(range(self._number_of_conversation_streams), disable=not self._verbose, desc="Stream")
        for conversation_stream_index in t:
            if self._should_stop(prompt_responses=prompt_response_pair_for_jailbreaks):
                break
            new_jailbreaks = await self._process_conversation_stream(index=conversation_stream_index)
            prompt_response_pair_for_jailbreaks += new_jailbreaks

        self.successful_jailbreaks.extend(prompt_response_pair_for_jailbreaks)
        return prompt_response_pair_for_jailbreaks

    def _group_by_conversation_id(self, *, pieces: list[PromptRequestPiece]) -> dict[str, list[PromptRequestPiece]]:
        grouped_pieces = defaultdict(list)
        for piece in pieces:
            grouped_pieces[piece.conversation_id].append(piece)
        return grouped_pieces

    def print(self, *, normalized_score_threshold: float | None = None) -> None:
        """Prints the conversations that have been processed by the orchestrator.

        This method retrieves all the prompt pieces from the memory, filters them based on the orchestrator class and
        optionally based on the orchestrator instance and successful jailbreaks. The filtered pieces are then grouped by
        conversation ID and printed.

        Args:
            normalized_score_threshold: The score threshold to filter the successful jailbreaks. If None, it defaults
                to the orchestrator's scorer sensitivity. Defaults to None. This must be a value between 0 and 1.
        """
        score_threshold = normalized_score_threshold or self._scorer_sensitivity
        if score_threshold < 0.0 or score_threshold > 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        scores = self._memory.get_scores_by_orchestrator_id(orchestrator_id=self.get_identifier()["id"])
        scores_above_threshold = [s for s in scores if float(s.score_value) >= score_threshold]
        filtered_pieces = self._memory.get_prompt_request_pieces_by_id(
            prompt_ids=[str(s.prompt_request_response_id) for s in scores_above_threshold]
        )
        if not filtered_pieces:
            print("No conversations with scores above the score threshold found.")
        grouped_pieces = self._group_by_conversation_id(pieces=list(filtered_pieces))
        # Prints conversation
        for idx, (conversation_id, pieces) in enumerate(grouped_pieces.items(), start=1):
            score_for_piece = list(filter(lambda s: s.prompt_request_response_id == pieces[0].id, scores))[0]
            print(f"Conversation ID: {conversation_id} (Conversation {idx}) (Score: {score_for_piece})")
            for piece in pieces:
                normalized_text = piece.converted_value.replace("\n", " ")
                print(f"    {piece.role}: {normalized_text}")
