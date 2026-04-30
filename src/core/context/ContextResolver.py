'''
    ContextResolver deterministically maps vague references in the user task
    ("it", "this", "the previous answer", "the file", ...) to concrete session
    state captured in ActiveContext.

    The goal is to keep the planner free from the burden of guessing what
    pronouns refer to. PlanCompiler still owns literal-content extraction, but
    contextual reference words are now stripped from that path so they cannot
    be mistaken for the literal text the user wants written.

    No LLM is used here. If a reference cannot be resolved with high confidence,
    the resolver returns should_ask_clarification=True instead of guessing.
'''

import re

from pathlib import PurePosixPath

from src.action_constants import CONTEXTUAL_REFERENCE_PRONOUNS, CONTEXTUAL_REFERENCE_PHRASES
from src.tools.utils import WorkspaceConfig 
from src.core.context.ActiveContext import ActiveContext


# Verbs that indicate the reference is the *content* the user wants written or saved somewhere. Example: "save it in a text file" -> "it" is content.
SAVE_VERBS = {
    "save",
    "store",
    "put",
    "append",
    "persist",
    "dump",
    "log",
}

# Verbs that indicate the reference is a *file* the user wants to act on, Example: "edit it" -> "it" is the file.
FILE_VERBS = {
    "edit",
    "show",
    "open",
    "read",
    "view",
    "display",
    "delete",
    "remove",
    "summarise",
    "summarize",
    "rename",
    "move",
    "copy",
}

MAX_INLINE_CONTENT_CHARS = 4000


class ContextResolver:
    def __init__(self, active_context: ActiveContext, workspace_config: WorkspaceConfig =None):
        self.active_context = active_context
        self.workspace_config = workspace_config

    def resolve(self, user_task: str) -> dict:
        '''
            Inspect user_task and active context, return a structured resolution.

            The returned dict matches the shape described in the design notes:
            has_references, resolved_references, unresolved_references,
            planner_context, should_ask_clarification, clarification_question.
        '''
        empty = self.empty_resolution()

        if not isinstance(user_task, str) or user_task.strip() == "":
            return empty

        normalised = self.normalise(user_task)
        intent = self.classify_intent(normalised)

        candidates = self.find_reference_candidates(normalised)
        if not candidates:
            return empty

        resolved_references = []
        unresolved_references = []

        for candidate in candidates:
            resolution = self.resolve_candidate(candidate=candidate, intent=intent)

            if resolution is None:
                continue

            if resolution.get("resolved"):
                resolved_references.append(resolution)
            else:
                unresolved_references.append(resolution)

        if not resolved_references and not unresolved_references:
            return empty

        planner_context = self.build_planner_context(resolved_references)

        should_ask = bool(unresolved_references) and not resolved_references
        clarification_question = None
        if should_ask:
            clarification_question = self.build_clarification_question(unresolved_references)

        return {
            "has_references": True,
            "resolved_references": resolved_references,
            "unresolved_references": unresolved_references,
            "planner_context": planner_context,
            "should_ask_clarification": should_ask,
            "clarification_question": clarification_question,
        }

    def empty_resolution(self) -> dict:
        return {
            "has_references": False,
            "resolved_references": [],
            "unresolved_references": [],
            "planner_context": {},
            "should_ask_clarification": False,
            "clarification_question": None,
        }

    def normalise(self, user_task: str) -> str:
        return " ".join(user_task.strip().lower().split())

    def classify_intent(self, normalised_task: str) -> str:
        '''
            Returns one of: "save_content", "edit_file", "show_file", "ambiguous".

            Distinguishing intent is what makes "save it" mean a content
            reference and "edit it" mean a file reference.
        '''
        words = re.findall(r"[a-z]+", normalised_task)
        if not words:
            return "ambiguous"

        # The "in/into/inside <ref> write X" pattern is reserved for
        # PlanCompiler's literal-content extraction; we deliberately do not
        # treat the reference as content here.
        if re.search(r"\b(?:inside|into|in)\s+(?:it|this|that|them|those|the\s+file)\s+write\b", normalised_task):
            return "ambiguous"

        if "write" in words:
            # "write it in a file" / "write it to a file" -> save content
            if re.search(r"\bwrite\s+(?:it|this|that|them|those|the\s+previous\s+answer|the\s+previous\s+response|the\s+result|the\s+output|the\s+generated\s+content)\b", normalised_task):
                return "save_content"

        for verb in SAVE_VERBS:
            if re.search(rf"\b{verb}\b", normalised_task):
                return "save_content"

        for verb in FILE_VERBS:
            if re.search(rf"\b{verb}\b", normalised_task):
                if verb in {"show", "open", "read", "view", "display", "summarise", "summarize"}:
                    return "show_file"
                return "edit_file"

        return "ambiguous"

    def find_reference_candidates(self, normalised_task: str) -> list[dict]:
        '''
            Returns the list of contextual reference phrases found in the task,
            with their span. Multi-word phrases are matched first so they win
            over the bare pronoun "the file" instead of "file".
        '''
        candidates = []
        consumed_spans: list[tuple[int, int]] = []

        for phrase in sorted(CONTEXTUAL_REFERENCE_PHRASES, key=len, reverse=True):
            for match in re.finditer(rf"\b{re.escape(phrase)}\b", normalised_task):
                if self.span_overlaps(match.span(), consumed_spans):
                    continue
                consumed_spans.append(match.span())
                candidates.append({"phrase": phrase, "span": match.span(), "kind": "phrase"})

        # "the file" is a softer reference than the longer phrases, so keep it
        # separate and only match if not already consumed.
        for match in re.finditer(r"\bthe\s+file\b", normalised_task):
            if self.span_overlaps(match.span(), consumed_spans):
                continue
            consumed_spans.append(match.span())
            candidates.append({"phrase": "the file", "span": match.span(), "kind": "phrase"})

        for pronoun in CONTEXTUAL_REFERENCE_PRONOUNS:
            for match in re.finditer(rf"\b{pronoun}\b", normalised_task):
                if self.span_overlaps(match.span(), consumed_spans):
                    continue
                consumed_spans.append(match.span())
                candidates.append({"phrase": pronoun, "span": match.span(), "kind": "pronoun"})

        candidates.sort(key=lambda candidate: candidate["span"][0])
        return candidates

    def span_overlaps(self, span: tuple[int, int], consumed: list[tuple[int, int]]) -> bool:
        start, end = span
        for c_start, c_end in consumed:
            if start < c_end and c_start < end:
                return True
        return False

    def resolve_candidate(self, candidate: dict, intent: str) -> dict | None:
        phrase = candidate["phrase"]

        if phrase in {"the previous answer", "previous answer", "the previous response", "previous response"}:
            return self.resolve_to_previous_assistant_response(phrase=phrase)

        if phrase in {"the result", "the output", "the generated content"}:
            return self.resolve_to_previous_generated_or_tool_result(phrase=phrase)

        if phrase in {"the previous file", "the last file"}:
            return self.resolve_to_previous_file(phrase=phrase)

        if phrase == "the file":
            return self.resolve_file_reference(phrase=phrase, intent=intent)

        # Pronoun: "it", "this", "that", "them", "those". Resolution depends
        # entirely on intent. We refuse to resolve when intent is ambiguous.
        if intent == "save_content":
            return self.resolve_to_previous_assistant_response(phrase=phrase)

        if intent in {"edit_file", "show_file"}:
            return self.resolve_file_reference(phrase=phrase, intent=intent)

        return None

    def resolve_to_previous_assistant_response(self, phrase: str) -> dict:
        content = (
            self.active_context.last_assistant_response
            or self.active_context.last_generated_content
        )

        if content:
            source_type = (
                "previous_assistant_response"
                if self.active_context.last_assistant_response
                else "last_generated_content"
            )

            return {
                "phrase": phrase,
                "resolved": True,
                "source_type": source_type,
                "content": self.truncate_content(content),
                "path": None,
                "confidence": 1.0,
                "reason": f"Resolved '{phrase}' to the {source_type.replace('_', ' ')}.",
            }

        return self.unresolved(phrase=phrase, reason="No previous assistant response is available to resolve this reference.")

    def resolve_to_previous_generated_or_tool_result(self, phrase: str) -> dict:
        content = (
            self.active_context.last_generated_content
            or self.active_context.last_tool_result
            or self.active_context.last_assistant_response
        )

        if content:
            if content == self.active_context.last_generated_content:
                source_type = "last_generated_content"
            elif content == self.active_context.last_tool_result:
                source_type = "last_tool_result"
            else:
                source_type = "previous_assistant_response"

            return {
                "phrase": phrase,
                "resolved": True,
                "source_type": source_type,
                "content": self.truncate_content(content),
                "path": None,
                "confidence": 1.0,
                "reason": f"Resolved '{phrase}' to the {source_type.replace('_', ' ')}.",
            }

        return self.unresolved(phrase=phrase, reason="No previous result or output is available to resolve this reference.")

    def resolve_to_previous_file(self, phrase: str) -> dict:
        path = (
            self.active_context.last_modified_file
            or self.active_context.last_created_file
            or self.active_context.last_viewed_file
            or self.active_context.active_file
        )

        if path and self.is_safe_workspace_path(path):
            return {
                "phrase": phrase,
                "resolved": True,
                "source_type": self.classify_path_source(path),
                "content": None,
                "path": path,
                "confidence": 1.0,
                "reason": f"Resolved '{phrase}' to the most recent file in session state.",
            }

        if path and not self.is_safe_workspace_path(path):
            return self.unresolved(phrase=phrase, reason="The previous file path is outside the workspace; refusing to resolve.")

        return self.unresolved(phrase=phrase, reason="No previous file is recorded in session state.")

    def resolve_file_reference(self, phrase: str, intent: str) -> dict:
        active_file = self.active_context.active_file

        if active_file and self.is_safe_workspace_path(active_file):
            return {
                "phrase": phrase,
                "resolved": True,
                "source_type": "active_file",
                "content": None,
                "path": active_file,
                "confidence": 1.0,
                "reason": f"Resolved '{phrase}' to the active file.",
            }

        if active_file and not self.is_safe_workspace_path(active_file):
            return self.unresolved(phrase=phrase, reason="The active file path is outside the workspace; refusing to resolve.")

        return self.unresolved(phrase=phrase, reason="No active file is recorded in session state.")

    def classify_path_source(self, path: str) -> str:
        if path == self.active_context.last_modified_file:
            return "last_modified_file"
        if path == self.active_context.last_created_file:
            return "last_created_file"
        if path == self.active_context.last_viewed_file:
            return "last_viewed_file"
        return "active_file"

    def is_safe_workspace_path(self, path: str) -> bool:
        if not isinstance(path, str) or path.strip() == "":
            return False

        cleaned = path.strip().replace("\\", "/")

        if cleaned.startswith("/"):
            return False

        if cleaned.startswith("../") or "/../" in cleaned or cleaned == "..":
            return False

        if PurePosixPath(cleaned).is_absolute():
            return False

        if self.workspace_config is None:
            return True

        try:
            self.workspace_config.resolve_workspace_path(cleaned)
        except Exception:
            return False

        return True

    def truncate_content(self, content: str) -> str:
        if not isinstance(content, str):
            return ""

        if len(content) <= MAX_INLINE_CONTENT_CHARS:
            return content

        return content[:MAX_INLINE_CONTENT_CHARS] + "\n... [truncated]"

    def unresolved(self, phrase: str, reason: str) -> dict:
        return {
            "phrase": phrase,
            "resolved": False,
            "source_type": None,
            "content": None,
            "path": None,
            "confidence": 0.0,
            "reason": reason,
        }

    def build_planner_context(self, resolved_references: list[dict]) -> dict:
        '''
            Flatten the resolved references into a small structured context object for the planner. Only the most relevant content/path is
            promoted to the top-level keys so the planner does not have to scan a list to find them.
        '''
        planner_context: dict = {}

        for ref in resolved_references:
            source_type = ref.get("source_type")
            content = ref.get("content")
            path = ref.get("path")

            if content and "resolved_content" not in planner_context:
                planner_context["resolved_content"] = content
                planner_context["content_source"] = source_type

            if path and "resolved_file_path" not in planner_context:
                planner_context["resolved_file_path"] = path
                planner_context["file_source"] = source_type

        if self.active_context.active_file:
            planner_context.setdefault("active_file", self.active_context.active_file)

        if self.active_context.last_generated_content:
            planner_context.setdefault(
                "last_generated_content",
                self.truncate_content(self.active_context.last_generated_content),
            )

        return planner_context

    def build_clarification_question(self, unresolved_references: list[dict]) -> str:
        if not unresolved_references:
            return None

        first = unresolved_references[0]
        phrase = first.get("phrase", "the reference")
        return (
            f"What does '{phrase}' refer to? "
            f"There is no recent assistant response or active file in session state."
        )
