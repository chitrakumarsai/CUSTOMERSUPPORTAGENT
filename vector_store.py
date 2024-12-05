from chromadb.api.types import Embeddings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import os
from dotenv import load_dotenv
from chromadb import PersistentClient, EmbeddingFunction, Embeddings
from typing import List
import json

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
MODEL_NAME = "text-embedding-3-small"
DB_PATH = "./.chroma_db"
FAQ_FILE_PATH = './FAQ.json'
INVENTORY_FILE_PATH = "./inventory.json"

embed_model = OpenAIEmbedding(embed_batch_size=10)
Settings.embed_model = embed_model

class Product:
    def __init__(self, name: str, id: str, description: str, price: float, quantity: int):
        self.name = name
        self.id = id
        self.description = description
        self.price = price
        self.quantity = quantity
        
class QuestionAnswerPair:
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer


class CustomEmbeddingClass(EmbeddingFunction):
    def __init__(self, model_name):
        self.embedding_model = OpenAIEmbedding(embed_batch_size=10, model=MODEL_NAME)

    def __call__(self, input_text:List[str]) -> Embeddings:
        return [self.embedding_model.get_text_embedding(text) for text in input_text]


# db = PersistentClient(DB_PATH)
# custom_embedding_funtion = CustomEmbeddingClass(MODEL_NAME)
# collections = db.get_or_create_collection(name = 'FAQ', embedding_function=custom_embedding_funtion)




# def query_faqs(query):
#     return collections.query(query_texts=[query],n_results=3)

class ShopVectorStore:
    def __init__(self):
        db = PersistentClient(DB_PATH)
        custom_embedding_funtion = CustomEmbeddingClass(MODEL_NAME)

        self.faq_collections = db.get_or_create_collection(name = 'FAQ', embedding_function=custom_embedding_funtion)

        self.inventory_collections = db.get_or_create_collection(name = 'Inventory', embedding_function=custom_embedding_funtion)

        if self.faq_collections.count() == 0:
            self._load_faq_collections(FAQ_FILE_PATH)

        if self.inventory_collections.count() == 0:
            self._load_inventory_collections(INVENTORY_FILE_PATH)
    
    def _load_faq_collections(self, faq_file_path: str):
        with open(faq_file_path, 'r') as f:
            faqs = json.load(f)

        self.faq_collections.add(
            documents=[faq['question'] for faq in faqs] + [faq['answer'] for faq in faqs],
            ids=[str(i) for i in range(0, len(faqs)*2)],
            metadatas=faqs + faqs
        )
    def _load_inventory_collections(self, inventory_file_path: str):
        with open(inventory_file_path, 'r') as f:
            inventories = json.load(f)

        self.inventory_collections.add(
            documents=[inventory['description'] for inventory in inventories],
            ids=[str(i) for i in range(0, len(inventories))],
            metadatas=inventories
        )

    def query_faqs(self, query: str):
        return self.faq_collections.query(query_texts=[query], n_results=3)

    def query_inventories(self, query: str):
        return self.inventory_collections.query(query_texts=[query], n_results=3)


