from ollama import chat
import json

'''
    This class is used in order to wrap the respective ollama API calls to the respective LLM
    Meaning each LLM based agent will make use of this instance in order to communicate to the LLM through ollama's methods
'''
class OllamaClient:
    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model

    def ask_text(self, messages: str) -> str:
        '''
            Each ask method will let the respective agent send its respective role based prompt (where it will be a list of messages) to the LLM running on 
            the machine and then it receives a response back from the LLM in text and not in JSON format
        '''
        
        response = chat(
            model=self.model,
            messages=messages,
        )

        return response.message.content
    
    def ask_json(self,messages: str,schema: dict) -> dict:
        '''
            Same thing as ask_text but it will return in JSON structured format but still text however loads method will turn it into dict
            This is because planner agent can make use of it, so that no encoding of what tools need to be executed by the executor is needed
        '''

        response = chat(
            model=self.model,
            messages=messages,
            format=schema,
        )
        return json.loads(response.message.content)