from ollama import chat
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLM_LOG_PATH = PROJECT_ROOT / "Memory" / "llm_calls.jsonl"

'''
    This class is used in order to wrap the respective ollama API calls to the respective LLM
    Meaning each LLM based agent will make use of this instance in order to communicate to the LLM through ollama's methods
'''
class OllamaClient:
    def __init__(self, model: str = "qwen2.5:3b", log_path: str | Path = DEFAULT_LLM_LOG_PATH, enable_logging: bool = True):
        self.model = model
        self.log_path = Path(log_path) if log_path else None
        self.enable_logging = enable_logging

    def make_json_safe(self, value):
        if isinstance(value, dict):
            return {
                str(key): self.make_json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                self.make_json_safe(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return [
                self.make_json_safe(item)
                for item in value
            ]

        if isinstance(value, set):
            return [
                self.make_json_safe(item)
                for item in sorted(value, key=str)
            ]

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, Path):
            return str(value)

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        return str(value)

    def write_llm_call_log(self, entry: dict) -> None:
        """
            Best-effort append-only LLM call logging.
            Logging must never affect the workflow result.
        """
        if not self.enable_logging or self.log_path is None:
            return

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            safe_entry = self.make_json_safe(entry)

            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(safe_entry, ensure_ascii=False) + "\n")
        except Exception:
            return

    def build_log_entry(self, call_type: str, messages: list[dict], stream: bool, schema: dict = None) -> dict:
        options = {
            "temperature": 0,
            "top_p": 0.1
        }

        entry = {
            "call_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_type": call_type,
            "model": self.model,
            "stream": stream,
            "options": options,
            "messages": messages,
        }

        if schema is not None:
            entry["schema"] = schema

        return entry

    def invoke_text(self, messages: list[dict], stream) -> str:
        '''
            Each ask method will let the respective agent send its respective role based prompt (where it will be a list of dictionaries) to the LLM running on 
            the machine and then it receives a response back from the LLM in text and not in JSON format
        '''
        options = {
            "temperature": 0,
            "top_p": 0.1
        }
        log_entry = self.build_log_entry(call_type="text", messages=messages, stream=stream)
        
        response = chat(
            model=self.model,
            messages=messages,
            stream=stream,
            options=options
        )

        raw_response = response.message.content
        log_entry["raw_response"] = raw_response
        self.write_llm_call_log(log_entry)

        return raw_response
    
    def invoke_json(self,messages: list[dict], stream, schema: dict) -> dict:
        '''
            Creating a new method which is the same as invoke_text but returns response in JSON structured format defined by the respective agent but still text 
            however loads method will turn it into a dict object so planner agent can make use of it, so that no encoding of 
            what tools need to be executed by the executor is needed
        '''
        options = {
            "temperature": 0, #A paramater which controls how strongly the model prefers the most probable next token over the other possible tokens
            "top_p": 0.1 #So that the model doesnt have too many words to choose from i.e. giving it a smaller subset of words
        }
        log_entry = self.build_log_entry(call_type="json", messages=messages, stream=stream, schema=schema)

        response = chat(
            model=self.model,
            messages=messages,
            stream=stream,
            format=schema,
            options=options
        )

        raw_response = response.message.content
        log_entry["raw_response"] = raw_response

        try:
            parsed_response = json.loads(raw_response)
        except Exception as e:
            log_entry["parse_error"] = str(e)
            self.write_llm_call_log(log_entry)
            raise

        log_entry["parsed_response"] = parsed_response
        self.write_llm_call_log(log_entry)

        return parsed_response
