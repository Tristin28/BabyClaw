'''
    DTO are used because this just a transfer of values (fields) however since these fields are not related to the agents' state
    they are not set as fields but as an external object which will only be used through behaviour from the agents to instantiate it and then could send it
    to the coordinator in order to handle the respective fields that it needs to use to send it to the respective agent, and also send the entire object to the
    relational db so that it logs every step
'''
from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass
class Message:
    def __init__(self, conversation_id: int, step_index: int, sender: str ,receiver: str, target_agent: str, message_type: str, status: str, response: dict, 
                 visibility: str
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
    
    def __str__(self):
        '''
            Method is used as the toString in order for when the object is called to be printed or in a string concatenation it always uses this conversion into str
        '''
        return (
            f"Message(conversation_id={self.conversation_id}, "
            f"step_index={self.step_index}, sender={self.sender}, "
            f"receiver={self.receiver}, target_agent={self.target_agent}, "
            f"message_type={self.message_type}, status={self.status}, "
            f"response={self.response}, visibility={self.visibility}, "
            f"timestamp={self.timestamp})"
        )
