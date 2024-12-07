from langchain_core.tools import tool
from typing import List, Dict
from vector_store import ShopVectorStore

store = ShopVectorStore()

@tool
def query_knowledge_base(query: str) -> List[Dict[str, str]]:
    """
    Look up information in a knowledge base to help with answering customer questions and getting information on business processes.

    Args:
        query (str): Question to ask the knowledge base.
    Return:
        List[Dict[str, str]]: Potentially questions answer pairs from the knowledge base.
    """
    return store.query_faqs(query=query)

store.query_faqs


@tool
def search_for_product_recommendations(description: str):
    """
    Look up information in a knowledge base to help with product recommendation for customers. For example:

    "High-Strength Nylon 6,6 Resin for automotive"
    "Industrial Nylon Fiber for textiles and industrial fabrics."
    "Nylon 6,6 Yarn - High Tenacity for heavy-duty sewing and rope production."

    Args:
        query (str): Description of product features
    Return:
        List[Dict[str, str]]: Potentially relavent product features.
    """
    return store.query_inventories(query=description)
