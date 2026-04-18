'''
    DTO are used because this just a transfer of values (fields) however since these fields are not related to the agents' state
    they are not set as fields but as an external object which will only be used through behaviour to instantiate it and then could send it 
'''
from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass
class Message:
    def __init__(self, conversation_id: int, step_index: int, sender: str ,receiver: str, target_agent: str, message_type: str, status: str, response: dict, 
                 visibility: str ,timestamp: str 
                 ):
        self.conversation_id = conversation_id
        self.step_index = step_index #Identifies which part of the system is being performed
        self.sender = sender
        self.receiver = receiver
        self.target_agent = target_agent
        self.message_type = message_type
        self.status = status
        self.response = response
        self.visibility = visibility #Identifies whether message being sent is internally via agents or externally to the user
        self.timestamp = datetime.now(timezone.utc).isoformat()
