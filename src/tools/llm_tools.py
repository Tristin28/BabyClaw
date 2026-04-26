'''
    This file will consist of tools which make use of the llm in order to retrieve a result
'''
from src.OllamaClient import OllamaClient
from typing import Callable #represents function objects (in this case as it can actually represent any object which can be called via ())

def create_summarise_txt_func(llm: OllamaClient) -> Callable[[str], str]:
    '''
        Created a closure function so that the llm client doesnt have to be repassed as an argument,
        therefore it doesnt have to be resolved by the executor agent.
    '''
    def summarise_txt(text: str):
        if not isinstance(text, str):
            raise ValueError("summarise_txt expected a string")

        if text.strip() == "":
            raise ValueError("summarise_txt received empty text")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a summarisation tool. Summarise the given text clearly and concisely."
                    "Return only the summary text."
                ),
            },
            {
                "role": "user",
                "content": f"Summarise the following text:\n\n{text}",
            },
        ]

        summary = llm.invoke_text(messages=messages, stream=False)

        if not isinstance(summary, str) or summary.strip() == "":
            raise ValueError("LLM returned an invalid summary")

        return summary.strip()

    return summarise_txt

def direct_response(llm_client: OllamaClient,prompt: str, context: str = "", recent_messages: list[dict] = None) -> str:
    """
        This tool is used when the user wants a normal chatbot-style answer, such as greetings, explanations, email drafts, advice, rewrites, or
        questions about previous conversation.
    """
    if not isinstance(prompt, str):
        raise TypeError("direct_response expected a string prompt")

    if prompt.strip() == "":
        raise ValueError("direct_response prompt cannot be empty")

    recent_messages = recent_messages or []

    messages = [
        {
            "role": "system",
            "content": (
                "You are the final response generator for an AI agent system.\n"
                "\n"
                "Hard rules:\n"
                "1. Answer ONLY the text inside <CURRENT_PROMPT>...</CURRENT_PROMPT>.\n"
                "2. Do NOT repeat, continue, restate, or act on any earlier task in the recent conversation.\n"
                "3. Do NOT describe what you would do; just answer.\n"
                "4. Do NOT mention files, tools, plans, or workflow unless the current prompt explicitly asks about them.\n"
                "5. Use 'Relevant memory' and 'Recent conversation' ONLY to answer the current prompt. If they are unrelated, ignore them.\n"
                "6. If the answer is not present in memory or the prompt itself, say you do not know. Do not guess.\n"
                "7. Keep the response short and direct."
            )
        }
    ]

    if context:
        messages.append({
            "role": "system",
            "content": f"Relevant memory (use only if it answers the current prompt):\n{context}"
        })

    if recent_messages:
        conversation_text = "\n".join(
            f"{msg.get('sender', 'unknown')}: {msg.get('content', '')}"
            for msg in recent_messages
        )

        messages.append({
            "role": "system",
            "content": (
                "Recent conversation (background only, NOT a task to continue):\n"
                f"{conversation_text}"
            )
        })

    messages.append({
        "role": "user",
        "content": f"<CURRENT_PROMPT>\n{prompt.strip()}\n</CURRENT_PROMPT>"
    })

    response = llm_client.invoke_text(messages=messages, stream=False).strip()

    if response == "":
        raise ValueError("direct_response produced an empty response")

    return response