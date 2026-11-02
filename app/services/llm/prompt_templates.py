"""
Prompt templates for LLM interactions.

Provides reusable prompt templates for different use cases.
"""

from typing import Any


class PromptTemplates:
    """
    Collection of prompt templates for various LLM use cases.

    All templates are designed to be customizable and composable.
    """

    # System prompts
    CHAT_SYSTEM_PROMPT = """You are a helpful, friendly, and knowledgeable AI assistant. Your goal is to provide accurate, useful, and safe information to users.

Guidelines:
- Be clear and concise in your responses
- If you're not sure about something, admit it
- Prioritize safety and ethics
- Be conversational but professional
- Ask clarifying questions when needed"""

    RAG_SYSTEM_PROMPT = """You are a helpful AI assistant with access to relevant context from a knowledge base. Your goal is to answer user questions using the provided context while being clear about what information comes from the context versus your general knowledge.

Guidelines:
- Use the provided context to answer questions when relevant
- Cite sources when referencing specific information from context
- If the context doesn't contain relevant information, rely on your general knowledge
- Be explicit about when you're using information from context vs. general knowledge
- If the context contradicts your knowledge, mention this discrepancy"""

    # Template methods
    @staticmethod
    def get_chat_system_prompt() -> str:
        """
        Get the standard chat system prompt.

        Returns:
            str: Chat system prompt
        """
        return PromptTemplates.CHAT_SYSTEM_PROMPT

    @staticmethod
    def get_rag_system_prompt() -> str:
        """
        Get the RAG system prompt.

        Returns:
            str: RAG system prompt
        """
        return PromptTemplates.RAG_SYSTEM_PROMPT

    @staticmethod
    def get_cs_system_prompt(language: str = "zh") -> str:
        """
        Get the e-commerce customer-service persona for the dialogue
        generators (tool / RAG / direct paths).

        Wording stays aligned with the agent path's SYSTEM_PROMPT so both
        halves of the product speak with one voice. The identity rule
        (no. 5) anchors self-description: a provider-fallback turn once
        self-identified as the underlying LLM (observed live 2026-09-30)
        — the assistant's identity must be stable no matter which model
        serves the turn.
        """
        handoff_phrase = "转人工" if language == "zh" else "human agent"
        return (
            "你是电商平台的智能客服助手。\n"
            "回答规则：\n"
            "1. 用与用户当前消息相同的语言回答（无法判断时用英语），"
            "友好、专业、简洁。\n"
            "2. 优先依据对话中的参考资料和工具结果回答；资料中没有的内容如实"
            "说明，不要编造订单号、金额或时效。\n"
            "3. 不对优惠、赔偿、时效做无依据的承诺。\n"
            f"4. 无法解决的问题，建议用户回复「{handoff_phrase}」发起人工协助请求。"
            "请求只代表可创建工单，不保证有客服在线、已经接通、必定接手或响应时间。\n"
            "5. 当用户问「你是谁」或你的身份时，回答自己是电商平台的智能客服"
            "助手；不要提及或自称背后的模型或开发公司。技术实现类提问"
            "（如是否用到知识图谱、用什么模型）也以客服助手身份简要回应："
            "用户明确提到的能力（如知识图谱）在产品层面如实确认并说明用途，"
            "不否认也不夸大；但不展开技术栈、模型选型等内部实现细节，"
            "可请对方参考项目介绍页，随后把话题带回购物服务。\n"
            "6. 用户消息里嵌着售后诉求（如「我要退货」）时，主动为其办理，"
            "不要只回答字面问题。"
        )

    @staticmethod
    def get_general_knowledge_prompt(language: str = "zh") -> str:
        """
        Persona for the general-knowledge lane (general_knowledge intent).

        The user asked something with no relation to the shop (health,
        science, daily life); the product answers from model knowledge
        instead of refusing. The persona's job is to keep that answer
        unmistakably separate from shop facts: no store policies, no
        invented order/pricing detail, conservative on sensitive topics.
        The deterministic disclaimer prefix (i18n) frames the answer;
        this persona governs the body under it.
        """
        if language == "en":
            return (
                "You are a customer-service assistant answering a general "
                "knowledge question that is unrelated to this store.\n"
                "Rules:\n"
                "1. Reply in the same language as the user's message "
                "(English when unclear). Be friendly, concise, practical.\n"
                "2. Answer from general knowledge only. Do NOT state or "
                "imply anything about this store's policies, prices, "
                "delivery times, stock, or services — for those, tell the "
                "user to ask the shop directly.\n"
                "3. Health, legal, and financial topics: give widely "
                "accepted general guidance only. Never diagnose, never "
                "prescribe medication or dosages, never give individual "
                "legal/financial advice. For serious, urgent, or "
                "persistent symptoms, advise seeing a qualified "
                "professional.\n"
                "4. Keep it short: a few sentences or a short list. No "
                "disclaimers of your own — the fixed prefix already "
                "labels this answer as general reference."
            )
        return (
            "你是电商平台的智能客服助手，正在回答一个与本店无关的通识问题。\n"
            "回答规则：\n"
            "1. 用与用户当前消息相同的语言回答（无法判断时用英语），"
            "友好、简洁、实用。\n"
            "2. 只依据通用常识回答。不得提及或暗示本店的政策、价格、"
            "配送时效、库存或服务；涉及店铺的问题请用户直接咨询本店。\n"
            "3. 涉及健康、法律、财务话题：只给广为接受的通用建议。"
            "不做诊断、不推荐药物或剂量、不提供个人化法律/投资建议；"
            "症状严重、紧急或持续时，提示尽快就医或咨询专业人士。\n"
            "4. 篇幅简短：几句话或一个短列表。不要自行添加免责声明——"
            "固定的前缀已标明这是通用参考信息。"
        )

    @staticmethod
    def format_chat_prompt(system_prompt: str, **kwargs: Any) -> str:
        """
        Format a chat prompt with variables.

        Args:
            system_prompt: System prompt template with placeholders
            **kwargs: Variables to substitute in template

        Returns:
            str: Formatted prompt
        """
        return system_prompt.format(**kwargs)

    @staticmethod
    def format_rag_prompt(
        query: str,
        retrieved_docs: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        """
        Format a RAG prompt with query and retrieved documents.

        Args:
            query: User's query
            retrieved_docs: List of retrieved documents with 'content' and 'source' keys
            system_prompt: Optional custom system prompt

        Returns:
            str: Formatted RAG prompt
        """
        if system_prompt is None:
            system_prompt = PromptTemplates.RAG_SYSTEM_PROMPT

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "")
            source = doc.get("source", "Unknown")
            context_parts.append(f"[Document {i} - {source}]\n{content}\n")

        context = "\n".join(context_parts) if context_parts else "No relevant documents found."

        # Format the prompt
        prompt = f"""{system_prompt}

Relevant Context:
{context}

User Query: {query}

Answer:"""

        return prompt

    @staticmethod
    def get_summarization_prompt(text: str, max_length: int | None = None) -> str:
        """
        Get a summarization prompt.

        Args:
            text: Text to summarize
            max_length: Optional maximum length for summary

        Returns:
            str: Summarization prompt
        """
        if max_length:
            instruction = f"Summarize the following text in {max_length} characters or less."
        else:
            instruction = "Summarize the following text concisely."

        return f"""{instruction}

Text:
{text}

Summary:"""

    @staticmethod
    def get_intent_detection_prompt(
        query: str,
        intents: list[str],
    ) -> str:
        """
        Get an intent detection prompt.

        Args:
            query: User query to classify
            intents: List of possible intents

        Returns:
            str: Intent detection prompt
        """
        intents_str = ", ".join(intents)

        return f"""Classify the user's query into one of the following intents: {intents_str}

User Query: {query}

Respond with only the intent name, nothing else.

Intent:"""

    @staticmethod
    def format_template(template: str, **kwargs: Any) -> str:
        """
        Format a custom template with variables.

        Args:
            template: Template string with placeholders
            **kwargs: Variables to substitute

        Returns:
            str: Formatted template
        """
        return template.format(**kwargs)

    @staticmethod
    def get_memory_aware_prompt(
        conversation_summary: str,
        current_message: str,
    ) -> str:
        """
        Get a memory-aware prompt with conversation context.

        Args:
            conversation_summary: Summary of previous conversation
            current_message: Current user message

        Returns:
            str: Memory-aware prompt
        """
        return f"""{PromptTemplates.CHAT_SYSTEM_PROMPT}

Conversation Summary:
{conversation_summary}

Current Message: {current_message}

Response:"""

    @staticmethod
    def get_code_generation_prompt(
        requirements: str,
        language: str = "Python",
    ) -> str:
        """
        Get a code generation prompt.

        Args:
            requirements: Description of code requirements
            language: Programming language

        Returns:
            str: Code generation prompt
        """
        return f"""You are an expert programmer. Write {language} code to fulfill the following requirements:

{requirements}

Provide only the code with brief comments. Include error handling and best practices.

Code:"""

    @staticmethod
    def get_few_shot_prompt(
        examples: list[dict[str, str]],
        test_input: str,
        task_description: str | None = None,
    ) -> str:
        """
        Get a few-shot learning prompt.

        Args:
            examples: List of example dictionaries with 'input' and 'output'
            test_input: Input to process
            task_description: Optional description of the task

        Returns:
            str: Few-shot prompt
        """
        examples_text = ""
        for i, example in enumerate(examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Input: {example['input']}\n"
            examples_text += f"Output: {example['output']}\n\n"

        if task_description:
            prompt = f"""{task_description}

{examples_text}Input: {test_input}
Output:"""
        else:
            prompt = f"""{examples_text}Input: {test_input}
Output:"""

        return prompt

    @staticmethod
    def get_chain_of_thought_prompt(problem: str) -> str:
        """
        Get a chain-of-thought reasoning prompt.

        Args:
            problem: Problem statement to solve

        Returns:
            str: Chain-of-thought prompt
        """
        return f"""Think step by step to solve the following problem. Show your reasoning process.

Problem: {problem}

Reasoning:"""

    @staticmethod
    def get_multi_turn_prompt(
        turns: list[tuple[str, str]],
        current_user_message: str,
    ) -> str:
        """
        Get a multi-turn conversation prompt.

        Args:
            turns: List of (user_message, assistant_response) tuples
            current_user_message: Current user message

        Returns:
            str: Multi-turn conversation prompt
        """
        conversation = ""
        for i, (user_msg, assistant_msg) in enumerate(turns, 1):
            conversation += f"Turn {i}:\n"
            conversation += f"User: {user_msg}\n"
            conversation += f"Assistant: {assistant_msg}\n\n"

        conversation += f"User: {current_user_message}\n"
        conversation += "Assistant:"

        return f"""{PromptTemplates.CHAT_SYSTEM_PROMPT}

{conversation}"""
