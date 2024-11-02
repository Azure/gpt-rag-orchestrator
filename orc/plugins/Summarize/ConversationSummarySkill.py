from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from semantic_kernel.kernel import Kernel
    from semantic_kernel.orchestration.sk_context import SKContext


class ConversationSummarySkill:
    """
    Semantic skill that enables conversations summarization.
    """

    from semantic_kernel.skill_definition import sk_function

    # The max tokens to process in a single semantic function call.
    _max_tokens = 1024
    _temperature = 0.1
    _top_p = 0.5
    _presence_penalty = 0.0
    _frequency_penalty = 0.0

    _summarize_conversation_prompt_template = (
        "[SUMMARIZATION RULES]\n"
        + "DONT WASTE WORDS\n"
        + "USE SHORT, CLEAR, COMPLETE SENTENCES.\n"
        + "DO NOT USE BULLET POINTS OR DASHES.\n"
        + "USE ACTIVE VOICE.\n"
        + "MAXIMIZE DETAIL, MEANING\n"
        + "FOCUS ON THE CONTENT\n\n"
        + "[BANNED PHRASES]\n"
        + "This article\n"
        + "This document\n"
        + "This page\n"
        + "This material\n"
        + "[END LIST]\n\n"
        + "Summarize:\n"
        + "Hello how are you?\n"
        + "+++++\n"
        + "Hello\n\n"
        + "Summarize this\n"
        + "{{$input}}\n"
        + "+++++\n"
    )

    def __init__(
        self, 
        kernel: "Kernel", 
        temperature: None,
        top_p: None,
        presence_penalty: None,
        frequency_penalty: None,
        max_tokens: None,
    ):
        _max_tokens = int(max_tokens) if max_tokens != None else ConversationSummarySkill._max_tokens
        _temperature = float(temperature) if temperature != None else ConversationSummarySkill._temperature
        _top_p = float(top_p) if top_p != None else ConversationSummarySkill._top_p
        _presence_penalty = float(presence_penalty) if presence_penalty != None else ConversationSummarySkill._presence_penalty
        _frequency_penalty = float(frequency_penalty) if frequency_penalty != None else ConversationSummarySkill._frequency_penalty

        logging.info(
            "[summary_plug] " +
            " _max_tokens: " + str(_max_tokens) + 
            " _temperature: " + str(_temperature) + 
            " _top_p: " + str(_top_p) + 
            " _presence_penalty: " + str(_presence_penalty) + 
            " _frequency_penalty: " + str(_frequency_penalty)
        )

        self._summarizeConversationFunction = kernel.create_semantic_function(
            ConversationSummarySkill._summarize_conversation_prompt_template,
            skill_name=ConversationSummarySkill.__name__,
            description=(
                "Given a section of a conversation transcript, summarize the part of"
                " the conversation."
            ),
            max_tokens=_max_tokens,
            temperature=_temperature,
            top_p=_top_p,
            presence_penalty=_presence_penalty,
            frequency_penalty=_frequency_penalty,
        )

    @sk_function(
        description="Given a long conversation transcript, summarize the conversation.",
        name="SummarizeConversation",
        input_description="A long conversation transcript.",
    )
    async def summarize_conversation_async(
        self, input: str, context: "SKContext"
    ) -> "SKContext":
        """
        Given a long conversation transcript, summarize the conversation.

        :param input: A long conversation transcript.
        :param context: The SKContext for function execution.
        :return: SKContext with the summarized conversation result.
        """
        from semantic_kernel.text import text_chunker
        from semantic_kernel.text.function_extension import (
            aggregate_chunked_results_async,
        )

        lines = text_chunker._split_text_lines(
            input, ConversationSummarySkill._max_tokens, True
        )
        paragraphs = text_chunker._split_text_paragraph(
            lines, ConversationSummarySkill._max_tokens
        )

        return await aggregate_chunked_results_async(
            self._summarizeConversationFunction, paragraphs, context
        )
