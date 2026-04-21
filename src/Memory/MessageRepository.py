from src.Memory.database import DatabaseManager
import json 
from src.message import Message

'''
    This class is going to handle what queries to tell the dbms to execute, while then the memory agent's job will just be when to store and what to retrieve
'''
class MessageRepository():
    def __init__(self,db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def store_message(self, message: Message):
        with self.db_manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    conversation_id,
                    step_index,
                    sender,
                    receiver,
                    target_agent,
                    message_type,
                    status,
                    response,
                    visibility,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.conversation_id,
                    message.step_index,
                    message.sender,
                    message.receiver,
                    message.target_agent,
                    message.message_type,
                    message.status,
                    json.dumps(message.response),
                    message.visibility,
                    message.timestamp,
                )
            )
            conn.commit()

    def get_recent_messages(self, conversation_id: int, k: int) -> list[dict]:
        with self.db_manager.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT sender, response, timestamp
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (conversation_id, k)
            ).fetchall()

        rows = list(reversed(rows))

        return [{"sender": row["sender"], 
                 "content": json.loads(row["response"]) #converting back into a python object (i.e. dict in this case)
                 }
                for row in rows
                ]