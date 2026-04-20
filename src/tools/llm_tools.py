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