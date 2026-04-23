from src.Memory.sql_database import DatabaseManager
import json 
from src.message import Message

'''
    This class is going to handle what queries to tell the dbms to execute , while then the memory agent's job will just be when to store and what to retrieve
    That is this repository decides how to store/retrieve the Message object and memory agent will tell it when to do it
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
                    json.dumps(message.response), #converting into text because of how SQLite stores data (i.e. doesnt store python obj)
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
                 "content": row["response"] #Leaving it as text because this will be passed inside the prompt planner
                 }
                for row in rows
                ]