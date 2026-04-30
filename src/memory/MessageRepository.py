from src.memory.sql_database import DatabaseManager
import json 
from src.core.message import Message
from datetime import datetime
from pathlib import Path


'''
    This class is going to handle what queries to tell the dbms to execute , while then the memory agent's job will just be when to store and what to retrieve
    That is this repository decides how to store/retrieve the Message object and memory agent will tell it when to do it
'''
class MessageRepository():
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def make_json_safe(self, value):
        """
            Converts Python objects that JSON cannot store into JSON-safe values.

            This is important because execution_state may contain sets, and SQLite stores
            the message response as JSON text using json.dumps().
        """

        if isinstance(value, dict):
            return {
                str(key): self.make_json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                self.make_json_safe(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return [
                self.make_json_safe(item)
                for item in value
            ]

        if isinstance(value, set):
            return [
                self.make_json_safe(item)
                for item in sorted(value, key=str)
            ]

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, (bytes, bytearray)):
            return f"<{len(value)} bytes>"

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        #Fallback for any unexpected object type.
        return str(value)
    
    def store_message(self, message: Message):
        safe_response = self.make_json_safe(message.response)

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
                    json.dumps(safe_response), #converting into text because of how SQLite stores data (i.e. doesnt store python obj)
                    message.visibility,
                    message.timestamp,
                )
            )
            conn.commit()

    def get_recent_messages(self, conversation_id: int, k: int) -> list[dict]:
        with self.db_manager.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT sender, message_type, response, visibility, timestamp
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (conversation_id, k)
            ).fetchall()

        rows = list(reversed(rows))

        messages = []

        for row in rows:
            try:
                response = json.loads(row["response"])
            except Exception:
                response = {}

            content = ""

            if isinstance(response, dict):
                content = response.get("content", "")

            messages.append({
                "sender": row["sender"],
                "message_type": row["message_type"],
                "response": response,
                "content": content,
                "visibility": row["visibility"],
                "timestamp": row["timestamp"]
            })

        return messages