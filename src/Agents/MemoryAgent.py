from src.Agents.BaseAgent import Agent
from src.message import Message
from src.Memory.MessageRepository import MessageRepository
from src.Memory.VectorRepository import VectorRepository
from src.Memory.sql_database import DatabaseManager
from src.OllamaClient import OllamaClient

class MemoryAgent(Agent):
    SCHEMA = {
            "type": "object",
            "properties": {
                "should_store": {
                    "type": "boolean",
                    "description": "Whether useful long-term memory should be stored."
                },
                "memories": {
                    "type": "array",
                    "description": "List of memory entries to store.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "memory_type": {
                                "type": "string",
                                "enum": ["summary", "fact", "preference", "decision"], #llm can only pick one of these as memory_types
                                "description": "Type of memory."
                            },
                            "content": {
                                "type": "string",
                                "description": "Short meaningful memory chunk to store."
                            }
                        },
                        "required": ["memory_type", "content"]
                    }
                }
            },
            "required": ["should_store", "memories"]
        }

    def __init__(self, db_manager: DatabaseManager , llm_client: OllamaClient):
        super().__init__("memory")
        self.llm_client = llm_client
        self.sql_repo = MessageRepository(db_manager) #using composition because it is just an object which is just being used internally by memory agent
        self.vector_repo = VectorRepository()

    def get_recent_messages(self, conversation_id: int, k: int)-> list[dict]:
        '''
            Method will use the MessageRepositry instance's behaviour in order to query some relational db in order to recieve the respective k last messages
        '''
        return self.sql_repo.get_recent_messages(conversation_id=conversation_id,k=k)

    def get_relevant_memory(self, task: str, k: int) -> str:
        '''
            Method will query a vector db in order to return content which is relative to the respective task that is being passed
        '''
        results =  self.vector_repo.retrieve_relevant_memory(task=task, k=k)

        #Applying threshold filtering because even though 10 documents would be retrieved it wouldnt necessarly mean they are actually related to the task sent
        k_relative_chunks = results["documents"][0] #returning the k most relevant chunks to the query sent 
        dists = results["distances"][0] #respective distances from query to those chunks

        filtered = [chunk for chunk, dist in zip(k_relative_chunks, dists) if dist < 0.6]
        return "\n".join(filtered)

    def store_message(self, message: Message) -> Message:
        '''
            Method which lets the coordinator store the respective responses from each agent in order to keep it transparent (which will be kept in the relational db)
            This is done by forwarding the message from the memoryagent object to the message repositry instance in order for it to handle the internal working
        '''
        try:
            self.sql_repo.store_message(message=message)
            status="completed"
            response={
                "stored": True,
                "logged_message_type": message.message_type
            }
        except Exception as e:
            status="failed"
            response={
                "stored": False,
                "error": str(e)
            }
        
        return Message(conversation_id=message.conversation_id, step_index=message.step_index, sender="memory", receiver="coordinator", 
                       target_agent=None, message_type="sql_store", status=status,response=response, visibility="internal") 

    def build_metadata(self, conversation_id: int, memory_type: str, source: str, timestamp: str) -> dict:
        return {
            "conversation_id": conversation_id,
            "memory_type": memory_type,
            "source": source,
            "timestamp": timestamp
        }
    
    def build_messages(self, user_task: str, planner_response: dict, executor_response: dict, reviewer_response: dict) -> list[dict]:
        return [
                {
                    "role": "system",
                    "content": """
                        You are a memory selection agent.

                        Your job is to decide whether useful long-term semantic memory should be stored from a completed and accepted workflow.

                        Store only:
                        - user preferences
                        - important facts
                        - key decisions
                        - concise useful summaries of valuable results

                        Do NOT store:
                        - temporary execution logs
                        - repeated tool steps
                        - low-value intermediate outputs
                        - noise or routine workflow details

                        Rules:
                        1. Return only valid JSON matching the provided schema.
                        2. If nothing useful should be remembered, return should_store=false and memories=[].
                        3. If memory should be stored, each memory entry must contain exactly one short meaningful idea.
                        4. Do not combine multiple unrelated ideas into one memory chunk.
                        5. Keep each memory content short, clear, and reusable for future semantic retrieval.
                                        """
                },
                {
                    "role": "user",
                    "content": f"""
                                User task:
                                {user_task}

                                Planner response:
                                {planner_response}

                                Executor response:
                                {executor_response}

                                Reviewer response:
                                {reviewer_response}
                            """
                }
        ]

    def store_long_term_memory(self, user_task: str, planner_msg: Message, executor_msg: Message, reviewer_msg: Message) -> Message:
        '''
            Using an llm in order to determine what type of memory will be stored inside the vector db and how it will be stored
            That is how the chunks are split so that specific chunks are embeded together in order to have it have meaning and not be pointless
        '''

        accepted = reviewer_msg.response.get("accepted", False)
        if not accepted:
            return Message(conversation_id=reviewer_msg.conversation_id, step_index=reviewer_msg.step_index, sender="memory", receiver="coordinator",
                           target_agent=None, message_type="memory_store", status="completed", response={"stored": False, "reason": "Reviewer did not accept workflow"},
                           visibility="internal")

        messages = self.build_messages(user_task=user_task, planner_response=planner_msg.response, executor_response=executor_msg.response, 
                                       reviewer_response=reviewer_msg.response)
        
        try:
            memory_result = self.llm_client.invoke_json(messages=messages, stream=False, schema=self.SCHEMA)

            if not memory_result["should_store"]:
                response={"stored": False}
                
            for memory in memory_result["memories"]:
                metadata = self.build_metadata(conversation_id=reviewer_msg.conversation_id, memory_type=memory["memory_type"], source="reviewer_accepted",
                                            timestamp=reviewer_msg.timestamp)

                self.vector_repo.store_memory(text=memory["content"], metadata=metadata)

            response={"stored": True}
            status = "completed"

        except Exception as e:
            status="failed",
            response={
                "stored": False,
                "error": str(e)
            }

        return Message(conversation_id=reviewer_msg.conversation_id, step_index=reviewer_msg.step_index, sender="memory", receiver="coordinator", target_agent=None, 
                       message_type="memory_store", status=status, response=response, visibility="internal")