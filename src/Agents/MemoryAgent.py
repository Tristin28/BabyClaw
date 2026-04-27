from src.Agents.BaseAgent import Agent
from src.message import Message
from src.Memory.MessageRepository import MessageRepository
from src.Memory.VectorRepository import VectorRepository
from src.Memory.sql_database import DatabaseManager
from src.OllamaClient import OllamaClient
from datetime import datetime, timezone

class MemoryAgent(Agent):
    #Squared L2 distance ceiling used when filtering vector-store hits.
    #Chroma's default embedding (all-MiniLM-L6-v2) returns squared L2 distance,
    #so a value of approximate 1.2 corresponds to roughly cosine similarity 0.4 which still
    #captures paraphrased matches like "what is my name?" vs a stored fact.
    RELEVANCE_THRESHOLD = 1.2

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
                                "enum": ["user_fact", "user_preference"], #llm can only pick one of these as memory_types
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
            between the entire components (agents + user) of the system
        '''
        return self.sql_repo.get_recent_messages(conversation_id=conversation_id,k=k)

    def get_pinned_facts_text(self) -> str:
        '''
            Always-on context: pulls every stored user_fact and user_preference
            and dedupes by topic (newest wins). Lets the planner/responder see
            the user's name/friend/etc. on tasks where similarity search would
            never surface them.
        '''
        sections = []

        for memory_type in ("user_fact", "user_preference"):
            results = self.vector_repo.get_facts_by_type(memory_type=memory_type, max_items=20)
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []

            by_topic = {}
            for fact, meta in zip(docs, metas):
                topic = (meta or {}).get("topic", "")
                ts = (meta or {}).get("timestamp", "")
                if topic not in by_topic or ts > by_topic[topic][1]:
                    by_topic[topic] = (fact, ts)

            if by_topic:
                lines = "\n".join(f"- {fact}" for fact, _ in by_topic.values())
                sections.append(f"{memory_type}:\n{lines}")

        return "\n\n".join(sections)
    
    def get_recent_conversation_messages(self,conversation_id: int, k: int = 5):
        '''
            Method which will filter out the respective k recent messages so that only the external visible messages are retrieved for context for the llm's to 
            reason about.
        '''
        messages = self.get_recent_messages(conversation_id=conversation_id, k=50)

        clean_messages = []

        for msg in messages:
            if msg.get("visibility") != "external":
                continue

            if msg.get("sender") not in {"user", "assistant"}:
                continue

            if msg.get("message_type") not in {"user_message", "assistant_message"}:
                continue

            content = msg.get("content", "")

            if content:
                clean_messages.append({
                    "sender": msg["sender"],
                    "content": content
                })

        return clean_messages[-k:]
    
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
        
        #Only returning to indicate whether it actually saved the information or not
        return Message(conversation_id=message.conversation_id, step_index=message.step_index, sender="memory", receiver="coordinator", 
                       target_agent=None, message_type="sql_store", status=status,response=response, visibility="internal") 

    def get_relevant_memory(self, task: str, k: int) -> str:
        '''
            Method will query a vector db in order to return content which is relative to the respective task that is being passed
        '''
        results = self.vector_repo.retrieve_relevant_memory(task=task, k=k)

        #Guarding against an empty collection or a query that returned no rows.
        documents_outer = results.get("documents") or []

        relevance_lines = []
        if documents_outer and documents_outer[0]:
            #Applying threshold filtering because even though k documents would be retrieved it wouldnt necessarly mean they are actually related to the task sent
            retrieved_memories = results["documents"][0] #returning the k most relevant chunks to the query sent
            distances = results["distances"][0] #respective distances from query to those chunks
            meta_data = results["metadatas"][0] #stores [[dict]]

            candidates = [(fact, meta) for fact, dist, meta in zip(retrieved_memories, distances, meta_data) if dist <= MemoryAgent.RELEVANCE_THRESHOLD]

            #retrieving the respective facts which do not conflict by applying the respective method
            filtered_candidates = self.resolve_conflicts(candidates=candidates)
            relevance_lines = [fact for fact, _ in filtered_candidates]

        pinned = self.get_pinned_facts_text()

        sections = []
        if pinned:
            sections.append(f"Known about the user (always include):\n{pinned}")
        if relevance_lines:
            sections.append("Task-relevant memory:\n" + "\n".join(relevance_lines))

        return "\n\n".join(sections)
    
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
    
    def build_messages(self, user_task: str) -> list[dict]:
        return [
                {
                    "role": "system",
                    "content": """
                                You are a memory selection agent.

                                Your job is to decide whether the user's latest message contains a useful long-term memory.

                                Only store information that the user literally stated in the user task.

                                CRITICAL RULES FOR USER FACTS:
                                - A noun on its own is USELESS. Always store the relationship or role with the name.
                                -   Example: WRONG: content="Jake", topic=friend_name
                                             RIGHT: content="Jake is the user's friend", topic=friend_jake
                                             WRONG: content="John", topic=professor
                                             RIGHT: content="John is the user's professor", topic=professor_john

                                - If the user says "my friend <NAME>", store: "<NAME> is the user's friend".
                                - If the user says "my professor <NAME>", store: "<NAME> is the user's professor".
                                - If you cannot identify the relationship/role from the user's text, do NOT store the name.

                                The content field must be a complete sentence that makes sense on its own when retrieved months later.

                                Allowed memory types:

                                1. user_fact
                                - Stable facts explicitly stated by the user.
                                - Example: "my name is Tristin" -> user_fact, topic=user_name

                                2. user_preference
                                - Stable preferences explicitly stated by the user.
                                - Example: "I prefer short explanations" -> user_preference, topic=explanation_style

                                Do NOT store:
                                - task lessons
                                - facts from planner output
                                - facts from executor output
                                - facts from reviewer output
                                - facts from assistant replies
                                - inferred preferences
                                - guesses
                                - temporary task details
                                - file names created during a task
                                - tool usage details
                                - anything not clearly stated by the user

                                Storage rules:
                                1. Return only valid JSON matching the schema.
                                2. If the user did not explicitly state a stable fact/preference, return should_store=false and memories=[].
                                3. Each memory must be short and reusable.
                                4. The memory content must be grounded in the exact user task.
                                """
                },
                {
                    "role": "user",
                    "content": f"""
                                User task:
                                {user_task}
                                """        
                }
        ]
    
    def is_grounded_in_user_task(self, memory: dict, user_task: str) -> bool:
        """
        Checks that the memory content is meaningfully grounded in what the user literally said.
        This prevents storing names/facts invented by the LLM.
        """

        content = memory.get("content", "").lower()
        task = user_task.lower()

        # Remove very common words that do not prove grounding.
        stop_words = {
            "the", "a", "an", "is", "are", "am", "i", "me", "my", "you",
            "user", "users", "to", "of", "and", "or", "in", "on", "for",
            "that", "this", "it", "with", "as", "be", "was"
        }

        task_words = {
            word.strip(".,!?;:'\"()[]{}").lower()
            for word in task.split()
            if len(word.strip(".,!?;:'\"()[]{}")) > 2
        }

        content_words = {
            word.strip(".,!?;:'\"()[]{}").lower()
            for word in content.split()
            if len(word.strip(".,!?;:'\"()[]{}")) > 2
        }

        task_words = task_words - stop_words
        content_words = content_words - stop_words

        overlap = task_words.intersection(content_words)

        return len(overlap) >= 1
    
    def validate_llm_response(self, response: dict, user_task: str):
        if not response.get("should_store"):
            return []

        valid_memories = []

        ALLOWED_MEMORY_TYPES = {"user_fact", "user_preference"}

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
        messages = self.build_messages(user_task=user_task)
        
        try:
            memory_result = self.llm_client.invoke_json(messages=messages, stream=False, schema=self.SCHEMA)

            if not memory_result["should_store"]:
                response={"stored": False}
                return Message(conversation_id=conversation_id, step_index=step_index, sender="memory", receiver="coordinator", target_agent=None, 
                       message_type="memory_store", status="completed", response=response, visibility="internal")
            
            
            valid_memories = self.validate_llm_response(response=memory_result, user_task=user_task)
            if valid_memories:
                for memory in valid_memories:

                    if self.should_reject_memory(memory):
                        continue

                    existing = self.get_relevant_memory(task=memory["content"], k=3)
                    if memory["content"].lower() in existing.lower():
                        #filtering out any infromation which already existss in the database
                        continue

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

        #Only returning message to indicate whether storing relative content succeeded or not
        return Message(conversation_id=conversation_id, step_index=step_index, sender="memory", receiver="coordinator", target_agent=None, 
                       message_type="memory_store", status=status, response=response, visibility="internal")
    
    def should_reject_memory(self, memory: dict) -> bool:
        '''
            Filtering the respective llm generated summary deterministicly based on specific keywords so that it doesnt store any garbage from 
            the it from the episode (task) that the system handles
        '''
        content = memory.get("content", "").lower()
        topic = memory.get("topic", "").lower()
        memory_type = memory.get("memory_type", "")

        rejected_topics = {
            "file_system_structure",
            "tool_usage",
            "workspace_state",
            "current_file",
            "temporary_file",
            "age_preference"
        }

        rejected_phrases = [
            "file named",
            "located in their current directory",
            "used to read",
            "prefers using tools",
            "current directory",
            "hello.txt"
        ]

        if topic in rejected_topics:
            return True

        if any(phrase in content for phrase in rejected_phrases):
            return True

        if memory_type == "user_preference":
            preference_markers = [
                "prefers",
                "likes",
                "wants",
                "from now on",
                "always",
                "going forward"
            ]

            if not any(marker in content for marker in preference_markers):
                return True

        return False