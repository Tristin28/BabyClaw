from ollama import chat
import json

'''
    This class is used in order to wrap the respective ollama API calls to the respective LLM
    Meaning each LLM based agent will make use of this instance in order to communicate to the LLM through ollama's methods
'''
class OllamaClient:
    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model

    def invoke_text(self, messages: list[dict], stream, options: dict) -> str:
        '''
            Each ask method will let the respective agent send its respective role based prompt (where it will be a list of dictionaries) to the LLM running on 
            the machine and then it receives a response back from the LLM in text and not in JSON format
        '''
        
        response = chat(
            model=self.model,
            messages=messages,
            stream=stream,
            options = {
                "temperature": 0.1 
            }
        )

        return response.message.content
    
    def invoke_json(self,messages: list[dict], stream, schema: dict, options: dict) -> dict:
        '''
            Creating a new method which is the same as invoke_text but returns response in JSON structured format defined by the respective agent but still text 
            however loads method will turn it into a dict object so planner agent can make use of it, so that no encoding of 
            what tools need to be executed by the executor is needed
        '''

        response = chat(
            model=self.model,
            messages=messages,
            stream=stream,
            format=schema,
            options = {
                "temperature": 0.1 #A paramater which controls how strongly the model prefers the most probable next token over the other possible tokens
            }
        )
        return json.loads(response.message.content)