from abc import ABC
from src.message import Message

class Agent(ABC):
    def __init__(self,name):
        self.name = name

    def get_message(self,conversation_id, step_index, receiver, target_agent, message_type, status, visibility, response):
        return Message(conversation_id=conversation_id, step_index=step_index, sender=self.name, receiver=receiver, target_agent=target_agent,
                       message_type=message_type, status=status, response=response, visibility=visibility
                       )
