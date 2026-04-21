from src.Agents.BaseAgent import Agent
from src.message import Message
from src.Memory.MessageRepository import MessageRepository

class MemoryAgent(Agent):
    def __init__(self, db_manager):
        super().__init__("memory")
        self.repo = MessageRepository(db_manager) #using composition because it is just an object which is just being used internally by memory agent

    def get_recent_messages(self, conversation_id: int, k: int)-> list[dict]:
        '''
            Method will use the MessageRepositry instance's behaviour in order to query some relational db in order to recieve the respective k last messages
        '''
        k_recent_messages = self.repo.get_recent_messages(conversation_id=conversation_id,k=k)
        return k_recent_messages

    def get_relevant_memory(self, conversation_id: int, task: str) -> str:
        '''
            Method will query a vector db in order to return content which is relative to the respective task that is being passed
        '''
        pass

    def store_message(self, message: Message):
        '''
            Method which lets the coordinator store the respective responses from each agent in order to keep it transparent (which will be kept in the relational db)
            This is done by forwarding the message from the memoryagent object to the message repositry instance in order for it to handle the internal working
        '''
        self.repo.store_message(message=message)

    def store_memory(self):
        '''
            Have to decide whether to use some llm in order to determine what type of memory will be stored inside the vector db and how it will be stored
            That is how the chunks are split so that specific chunks are embeded together in order to have it have meaning and not be pointless
        '''
        pass