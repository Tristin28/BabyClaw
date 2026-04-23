from src.Agents.BaseAgent import Agent
from src.message import Message
from src.Memory.MessageRepository import MessageRepository
from src.Memory.VectorRepository import VectorRepository
from src.Memory.sql_database import DatabaseManager
from src.OllamaClient import OllamaClient
from datetime import datetime, timezone

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
                                "enum": ["user_fact", "user_preference", "task_lesson"], #llm can only pick one of these as memory_types
                                "description": "Type of memory."
                            },
                            "topic":{
                                 "type": "string",
                                 "description": "A short reusable label describing what this memory is about (e.g. current_project, tools_used, explanation_style, learning_goal). Use lowercase snake_case and avoid full sentences."
                            },
                            "content": {
                                "type": "string",
                                "description": "Short meaningful reusable memory chunk to store."
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Confidence score between 0 and 1 representing how reliable this memory is."
                            }
                        },
                        "required": ["memory_type", "topic", "content", "confidence"]
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

    def get_relevant_memory(self, task: str, k: int) -> str:
        '''
            Method will query a vector db in order to return content which is relative to the respective task that is being passed
        '''
        results =  self.vector_repo.retrieve_relevant_memory(task=task, k=k)

        #Applying threshold filtering because even though 10 documents would be retrieved it wouldnt necessarly mean they are actually related to the task sent
        retrieved_memories = results["documents"][0] #returning the k most relevant chunks to the query sent, where documents is a key in the returned dict that stores [[strings]]
        distances = results["distances"][0] #respective distances from query to those chunks, where the distances key is stores [[integers]]
        meta_data = results["metadatas"][0] #stores [[dict]]

        candidates = [(fact, meta) for fact, dist, meta in zip(retrieved_memories, distances,meta_data) if dist < 0.6]
        
        #retrieving the respective facts which do not conflict by applying the respective method
        filtered_candidates = self.resolve_conflicts(candidates=candidates)

        return "\n".join(fact for fact, _ in filtered_candidates)
    
    def resolve_conflicts(self, candidates):
        best_by_topic = {} #dictionary which stores the most relevant candidates based on their respective memory_type
        for fact, meta in candidates:
            memory_topic = meta.get("topic")
            timestamp = meta.get("timestamp","")
            confidence = meta.get("confidence",0.0)

            if memory_topic not in best_by_topic:
                best_by_topic[memory_topic]= (fact,meta)
                continue

            _, old_meta = best_by_topic[memory_topic]

            old_timestamp = old_meta.get("timestamp","")
            old_confidence = old_meta.get("confidence",0.0)

            if timestamp > old_timestamp:
                best_by_topic[memory_topic] = (fact, meta)
            elif timestamp == old_timestamp and confidence > old_confidence:
                best_by_topic[memory_topic] = (fact, meta)
            
        return list(best_by_topic.values()) #returning a list of tuples, where a tuple is (the fact, and its metadata)

    def build_metadata(self, conversation_id: int, topic: str, memory_type: str, source: str, confidence: float, timestamp: str) -> dict:
        return {
            "conversation_id": conversation_id,
            "memory_type": memory_type,
            "topic": topic,
            "source": source,
            "confidence": confidence,
            "timestamp": timestamp
        }
    
    def build_messages(self, user_task: str, episode_summary: str) -> list[dict]:
        return [
                {
                    "role": "system",
                    "content": """
                                You are a memory selection agent.

                                Your job is to decide whether useful long-term semantic memory should be stored from a completed workflow that was accepted by the reviewer.

                                Store only these kinds of memories:

                                1. user_fact
                                - Stable facts about the user that may help in future tasks.
                                - Example: the user's project, tools they use, recurring goals, or stable background context.

                                2. user_preference
                                - Stable preferences about how the user likes responses, explanations, or workflows.
                                - Example: prefers short conceptual explanations first, prefers step-by-step guidance.

                                3. task_lesson
                                - A reusable lesson learned from this successful task that may help future similar tasks.
                                - Store this only if it is generalisable and useful beyond this single run.

                                Do NOT store:
                                - system architecture facts
                                - coordinator, planner, executor, reviewer, or memory agent role descriptions
                                - fixed workflow rules already defined by the system
                                - temporary execution logs
                                - repeated tool calls or step-by-step tool traces
                                - low-value intermediate outputs
                                - raw summaries of the conversation
                                - one-off details that are unlikely to help in future tasks
                                - anything that is already obvious from the system design

                                Storage rules:
                                1. Return only valid JSON matching the provided schema.
                                2. If nothing useful should be remembered, return should_store=false and memories=[].
                                3. Each memory entry must contain exactly one short meaningful reusable idea.
                                4. Do not combine multiple unrelated ideas into one memory entry.
                                5. Keep memory content short, clear, and useful for future retrieval.
                                6. Prefer stable and reusable memories over temporary details.
                                7. A task_lesson must describe a lesson that can help with future similar tasks, not just describe what happened here.
                                8. Do not store system facts just because they appeared in the planner, executor, or reviewer outputs.
                                """
                },
                {
                    "role": "user",
                    "content": f"""
                                User task:
                                {user_task}

                                Episode summary:
                                {episode_summary}
                                """        
                }
        ]
    

    def validate_llm_response(self, response: dict):
        if not response.get("should_store"):
            return []

        valid_memories = []

        ALLOWED_MEMORY_TYPES = {"user_fact", "user_preference", "task_lesson"}

        for memory in response.get("memories", []):
            memory_type = memory.get("memory_type")
            topic = memory.get("topic")
            content = memory.get("content")
            confidence = memory.get("confidence")

            if memory_type not in ALLOWED_MEMORY_TYPES:
                continue

            if not isinstance(topic, str) or topic.strip() == "":
                continue

            if not isinstance(content, str) or content.strip() == "":
                continue

            if not isinstance(confidence, (int, float)):
                continue

            if not (0.0 <= confidence <= 1.0):
                continue

            valid_memories.append({
                "memory_type": memory_type,
                "topic": topic,
                "content": content,
                "confidence": confidence
            })

        return valid_memories


    def store_long_term_memory(self, user_task: str, episode_summary: str, conversation_id: int, step_index:int) -> Message:
        '''
            Using an llm in order to determine what type of memory will be stored inside the vector db and how it will be stored
            That is how the chunks are split so that specific chunks are embeded together in order to have it have meaning and not be pointless
        '''
        messages = self.build_messages(user_task=user_task, episode_summary=episode_summary)
        
        try:
            memory_result = self.llm_client.invoke_json(messages=messages, stream=False, schema=self.SCHEMA)

            if not memory_result["should_store"]:
                response={"stored": False}
                return Message(conversation_id=conversation_id, step_index=step_index, sender="memory", receiver="coordinator", target_agent=None, 
                       message_type="memory_store", status="completed", response=response, visibility="internal")
            
            
            valid_memories = self.validate_llm_response(response=memory_result)
            if valid_memories:
                for memory in valid_memories:
                    metadata = self.build_metadata(conversation_id=conversation_id, memory_type=memory["memory_type"], topic = memory["topic"], 
                                                    source="reviewer_accepted", confidence=memory["confidence"],timestamp=datetime.now(timezone.utc).isoformat())

                    self.vector_repo.store_memory(text=memory["content"], metadata=metadata)

                response={"stored": True}
                status = "completed"
            else:
                response={"stored": False}
                status = "completed"

        except Exception as e:
            status="failed"
            response={
                "stored": False,
                "error": str(e)
            }

        return Message(conversation_id=conversation_id, step_index=step_index, sender="memory", receiver="coordinator", target_agent=None, 
                       message_type="memory_store", status=status, response=response, visibility="internal")