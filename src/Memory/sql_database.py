'''
    This class is only used in order to handle the creation of the connection to the dbms from the MemoryManager 
'''
import sqlite3

class DatabaseManager():
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_connection(self) -> sqlite3.Connection: 
        '''
            Creates the database file or loads up into memory, and retrieves a connection with the dbms (i.e. object will be communicating with dbms)
            Where this method will be called each time the usage of the db is needed (called every time instead of having it open until program's life time because if it were like that
            it would waste resource space, and slows down the program which is pointless since db queries are not done often)
        '''
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row #Letting the connection object to access the columns with their respective names and not have to remember their index
        return conn
    
    def init_db(self):
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    step_index INTEGER NOT NULL,
                    sender TEXT NOT NULL,
                    receiver TEXT NOT NULL,
                    target_agent TEXT,
                    message_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response TEXT NOT NULL,
                    visibility TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.commit()