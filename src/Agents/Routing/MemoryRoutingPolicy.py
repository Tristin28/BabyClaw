from dataclasses import dataclass
import re


@dataclass(frozen=True)
class MemoryRoutingDecision:
    """
        Deterministic memory scope for the current task.

        The router decides the workflow route. This helper decides how much
        memory that route is allowed to see. No LLM call is used here because
        memory retrieval changes the planning context and should stay under
        infrastructure control.
    """
    use_short_term: bool
    use_long_term: bool
    allowed_memory_types: tuple[str, ...]
    long_term_mode: str
    long_term_k: int
    short_term_k: int
    reason: str


class MemoryRoutingPolicy:
    USER_MEMORY_KEYWORDS = {
        "my name",
        "remember",
        "what do you know",
        "what do you remember",
        "my preference",
        "i prefer",
        "my project",
        "about me",
    }

    FOLLOWUP_PATTERNS = [
        #Using Regex patterns as a sort of heuristic method to catch follow-up questions that rely on recent conversation
        r"\bcontinue\b",
        r"\banother\b",
        r"\bsame\b",
        r"\bprevious\b",
        r"\bthat\b",
        r"\bthis\b",
        r"\bit\b",
    ]

    @classmethod
    def decide(cls, user_task: str, route: dict) -> MemoryRoutingDecision:
        task_type = route.get("task_type", "direct_response")
        task = (user_task or "").lower()

        if task_type == "memory_question":
            return MemoryRoutingDecision(
                use_short_term=False,
                use_long_term=True,
                allowed_memory_types=("user_fact", "user_preference"),
                long_term_mode="full",
                long_term_k=8,
                short_term_k=0,
                reason="Memory question: retrieve durable user facts/preferences strongly."
            )

        if task_type == "contextual_followup":
            return MemoryRoutingDecision(
                use_short_term=True,
                use_long_term=True,
                allowed_memory_types=("user_fact", "user_preference"),
                long_term_mode="full",
                long_term_k=5,
                short_term_k=5,
                reason="Contextual follow-up: use recent chat plus relevant durable memory."
            )

        if task_type in {"workspace_read", "workspace_summarise", "workspace_mutation"}:
            return MemoryRoutingDecision(
                use_short_term=False,
                use_long_term=True,
                allowed_memory_types=("user_fact", "user_preference"),
                long_term_mode="relevant_only",
                long_term_k=3,
                short_term_k=0,
                reason="Workspace task: only task-relevant durable memory may be used."
            )

        if cls.direct_response_needs_memory(task):
            return MemoryRoutingDecision(
                use_short_term=False,
                use_long_term=True,
                allowed_memory_types=("user_fact", "user_preference"),
                long_term_mode="relevant_only",
                long_term_k=3,
                short_term_k=0,
                reason="Direct response contains explicit memory-related signals."
            )

        return MemoryRoutingDecision(
            use_short_term=False,
            use_long_term=False,
            allowed_memory_types=(),
            long_term_mode="none",
            long_term_k=0,
            short_term_k=0,
            reason="Standalone direct response: memory is not needed."
        )

    @classmethod
    def direct_response_needs_memory(cls, task: str) -> bool:
        if any(keyword in task for keyword in cls.USER_MEMORY_KEYWORDS):
            return True

        return any(re.search(pattern, task) for pattern in cls.FOLLOWUP_PATTERNS)
