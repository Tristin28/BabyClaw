'''
    vector db will live throughout the VectorRepository's lifetime 
    (which will be lifetime of the program as this repo will be composed inside memory agent where this agent lives throughout entire lifetime of program), 
    hence there is no need for some db manager in this case

    Note:
        Not building an embedding function (model) as i am using the automatic one which is done internally in chroma_db
'''
import chromadb
from pathlib import Path
import uuid 

class VectorRepository():
    CHROMA_PATH = Path(__file__).resolve().parents[2] / "Memory" / "chroma_db"
    def __init__(self):
        self.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.CHROMA_PATH) #Since only one vector db will be used composition style is used
        self.collection = self.client.get_or_create_collection(name="memory") 

    def store_memory(self, text: str, metadata: dict):
        '''
            Metadata - explains where the memory came from, 
            text will be a composed summary from an llm (where it will basically tell us what to store) and it is what is changed into a numeric vector, 
            and memory_id is set so that every vector has a unique_id
        '''
        memory_id = str(uuid.uuid4()) #Storing a unique id each time
        self.collection.add(ids=[memory_id], documents=[text], metadatas=[metadata])

    def retrieve_relevant_memory(self, task: str, k: int) -> dict:
        return self.collection.query(query_texts=[task], n_results=k)

    def get_all_memories(self) -> dict:
        '''
            Debug/helper method used to visually confirm that memories are actually being stored inside the vector database.
        '''
        return self.collection.get()