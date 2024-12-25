from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.messages import AIMessage, HumanMessage
from tools import query_knowledge_base, search_for_product_recommendations, data_protection_check, create_new_customer, place_order, retrieve_existing_customer_orders
load_dotenv()


# API key for OpenAI (DO NOT hardcode in production)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

prompt = '''# Purpose

You are customer service chatbot for INVISTA SHOP company. You can help the customer achieve their goal listed below.

# Goals
1. Answer the questions the user might have relating to the products and services offered by INVISTA SHOP.
2. Recommend products to the user based on their preferences.
3. Help the customer check on an exisiting order or place a new order.
4. To place and manage orders, you will need a customer profile (with an associated id) If the customer already has a profile, perform a data protection check to retrieve their details. If not create them a new profile. You cannot place an order until you have the item ID (from a product search) and the customer ID.

# Tone
The tone should be friendly and helpful. Use gen-z emojjis and simlies to keep things light hearted.

'''

chat_template = ChatPromptTemplate.from_messages(
    [
        ('system', prompt),
        ('placeholder', "{messages}")
    ]
)

llm = ChatOpenAI(
    model = 'gpt-4o',
    api_key = os.environ["OPENAI_API_KEY"] , 
)

tools = [query_knowledge_base, search_for_product_recommendations, data_protection_check, create_new_customer, place_order, retrieve_existing_customer_orders]
llm_with_prompt = chat_template | llm.bind_tools(tools)

def call_agent(message_state: MessagesState):

    response = llm_with_prompt.invoke(message_state)

    return {
        'messages': [response],
    }

def is_there_tool_calls(state: MessagesState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return 'tool_node'
    else:
        return '__end__'


graph = StateGraph(MessagesState)

tool_node = ToolNode(tools)

graph.add_node('agent', call_agent)
graph.add_node('tool_node', tool_node)

graph.add_conditional_edges(
    'agent',
    is_there_tool_calls
    )
graph.add_edge('tool_node', 'agent')

graph.set_entry_point('agent')

app = graph.compile()

# updated_messages = app.invoke({
#     'messages': [HumanMessage(content='Hello')]
# })

# print(updated_messages)