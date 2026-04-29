"""
RobustAdvancedMemAgent extracted from test_advanced_robust.py for reuse.
"""

import logging

from llm_text_parsers import (
    parse_keywords_response,
    parse_plain_text_answer,
    parse_relevant_parts,
)
from llm_controller import RobustLLMController
from memory_layer_robust import RobustAgenticMemorySystem

logger = logging.getLogger("amem_robust")


class RobustAdvancedMemAgent:
    """Agent using the robust memory system with plain-text LLM calls."""

    def __init__(
        self,
        model,
        backend,
        retrieve_k,
        temperature_c5,
        sglang_host="http://localhost",
        sglang_port=30000,
        attention_model: str = "/data/hyc/models/Qwen2.5-14B-Instruct",
        attention_max_length: int = 4096,
    ):
        self.memory_system = RobustAgenticMemorySystem(
            model_name="all-MiniLM-L6-v2",
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retriever_llm = RobustLLMController(
            backend=backend,
            model=model,
            api_key=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.attention_model = attention_model
        self.attention_max_length = attention_max_length

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(
            content,
            k=k,
            use_attention=True,
            attention_model_name=self.attention_model,
            attention_max_length=self.attention_max_length,
        )

    def retrieve_memory_no_attention(self, content, k=10):
        return self.memory_system.find_related_memories_raw(
            content,
            k=k,
            use_attention=False,
            attention_model_name=self.attention_model,
            attention_max_length=self.attention_max_length,
        )

    @staticmethod
    def _category5_options(question: str, answer: str) -> list:
        """Stable options order for fair comparisons."""
        import hashlib

        h = hashlib.md5(question.encode("utf-8")).digest()[0]
        if h % 2 == 0:
            return ["Not mentioned in the conversation", answer]
        return [answer, "Not mentioned in the conversation"]

    def build_user_prompt(self, question: str, category: int, answer: str, context: str) -> tuple:
        """Build user_prompt and temperature given a context."""
        assert category in [1, 2, 3, 4, 5]
        if category == 5:
            answer_tmp = self._category5_options(question, answer)
            user_prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:"""
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
            temperature = 0.7
        elif category == 3:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7
        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7
        return user_prompt, temperature

    def retrieve_memory_llm(self, memories_text, query):
        """Select relevant parts of conversation memories — plain text, no JSON schema."""
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

Conversation memories:
{memories_text}

Question: {query}

Return only the relevant parts of the conversation that would help answer this specific question.
If no parts are relevant, return the input unchanged."""

        response = self.retriever_llm.llm.get_completion(prompt)
        return parse_relevant_parts(response)

    def generate_query_llm(self, question):
        """Generate query keywords — plain text, no JSON schema."""
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""

        response = self.retriever_llm.llm.get_completion(prompt)
        result = parse_keywords_response(response)
        logger.debug("generate_query_llm response: %s", result)
        return result

    def answer_question(self, question: str, category: int, answer: str) -> tuple:
        """Generate answer for a question — plain text, no JSON schema."""
        keywords = self.generate_query_llm(question)
        raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)
        user_prompt, temperature = self.build_user_prompt(question, category, answer, raw_context)

        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning("answer_question failed: %s — returning empty", e)
            response = ""
        return response, user_prompt, raw_context

    def answer_question_with_context(
        self, question: str, category: int, answer: str, raw_context: str
    ) -> tuple:
        """Answer with provided context (used for attention/no-attention comparisons)."""
        user_prompt, temperature = self.build_user_prompt(question, category, answer, raw_context)
        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning("answer_question_with_context failed: %s — returning empty", e)
            response = ""
        return parse_plain_text_answer(response), user_prompt, raw_context

