from langchain_core.tools import tool
from typing import List, Dict
from vector_store import ShopVectorStore
import json

vector_store = ShopVectorStore()

customers_database = [
    {"name": "John Doe", "postcode": "SW1A 1AA", "dob": "1990-01-01", "customer_id": "CUST001", "first_line_address": "123 Main St", "phone_number": "07712345678", "email": "john.doe@example.com"},
    {"name": "Jane Smith", "postcode": "E1 6AN", "dob": "1985-05-15", "customer_id": "CUST002", "first_line_address": "456 High St", "phone_number": "07723456789", "email": "jane.smith@example.com"},
]

order_database = [
    { "order_id" : "ORD001", "customer_id" : "CUST001", "status" : "processing", "items" : ["Industrial Nylon Fiber"], "quantity": [2]},
    { "order_id" : "ORD002", "customer_id" : "CUST002", "status" : "shipped", "items" : ["Nylon Composite Sheets", "Nylon Braided Cord"], "quantity": [1, 3]}
    
]

with open('inventory.json', 'r') as f:
    inventory_database = json.load(f)

data_protection_checks = []

@tool
def data_protection_check(name: str, postcode: str, year_of_birth: int, month_of_birth: int, day_of_birth: int) -> str:
    """
    Perform a data protection check against a customer to retrieve customer details.

    Args:
        name (str): Customer first and last name
        postcode (str): Customer registered address
        day_of_birth (int): The day the customer was born
        month_of_birth (int): The month the customer was born
        year_of_birth (int): The year the customer was born
        

    Returns:
        Dict: Customer details (name, postcode, dob, customer_id, first_line_address, email)
    """
    data_protection_checks.append(
        {
            'name': name,
            'postcode': postcode,
            'year_of_birth': year_of_birth,
            'month_of_birth': month_of_birth,
            'day_of_birth': day_of_birth
        }
    )
    for customer in customers_database:
        if (customer['name'].lower() == name.lower() and
            customer['postcode'].lower() == postcode.lower() and
            int(customer['dob'][0:4]) == year_of_birth and
            int(customer["dob"][5:7]) == month_of_birth and
            int(customer["dob"][8:10]) == day_of_birth):
            return f"DPA check passed - Retrieved customer details:\n{customer}"

    return "DPA check failed, no customer with these details found"

@tool
def create_new_customer(first_name: str, surname: str, year_of_birth: int, month_of_birth: int, day_of_birth: int, postcode: str, first_line_of_address: str, phone_number: str, email: str) -> str:
    """
    Creates a customer profile, so that they can place orders.

    Args:
        first_name (str): Customers first name
        surname (str): Customers surname
        day_of_birth (int): Day customer was born
        month_of_birth (int): Month customer was born
        year_of_birth (int): Year customer was born
        postcode (str): Customer's postcode
        first_line_address (str): Customer's first line of address
        phone_number (str): Customer's phone number
        email (str): Customer's email address

    Returns:
        str: Confirmation that the profile has been created or any issues with the inputs
    """
    if len(phone_number) != 11:
        return "Phone number must be 11 digits"
    customer_id = len(customers_database) + 1
    customers_database.append({
        'name': first_name + ' ' + surname,
        'dob': f'{year_of_birth}-{month_of_birth:02}-{day_of_birth:02}',
        'postcode': postcode,
        'first_line_address': first_line_of_address,
        'phone_number': phone_number,
        'email': email,
        'customer_id': f'CUST{customer_id}'
    })
    return f"Customer registered, with customer_id {f'CUST{customer_id}'}"
    

@tool
def query_knowledge_base(query: str) -> List[Dict[str, str]]:
    """
    Looks up information in a knowledge base to help with answering customer questions and getting information on business processes.

    Args:
        query (str): Question to ask the knowledge base

    Return:
        List[Dict[str, str]]: Potentially relevant question and answer pairs from the knowledge base
    """
    return vector_store.query_faqs(query=query)



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
    return vector_store.query_inventories(query=description)


@tool
def retrieve_existing_customer_orders(customer_id: str):
    """
    Retrieves the orders associated with a customer, including the status, items and ids.

    Args:
        customer_id (str): Customer unique id associated with the orders

    Returns:
        List[Dict]: All the orders associated with the customer_id passed in
    """
    customer_orders = [order for order in order_database if order['customer_id'] == customer_id]
    if not customer_orders:
        return f"No orders found for this customer: {customer_id}"
    return customer_orders


@tool

def place_order(items: Dict[str, int], customer_id: str):
    """
    Places an order for a customer with the items and quantities they want.

    Args:
        items (Dict[str, int]): The items the customer wants to order and the quantity
        customer_id (str): The customer unique id associated with the order

    Returns:
        str: Confirmation that the order has been placed or any issues with the inputs or it has not been placed due to an issue

    Example arguments:
    items = { 'N007' : 2, 'N008' : 1 }
    customer_id = 'CUST001'
    """

    # Check that the item ids are valid
    # Check that the quantities of items are valid
    availability_messages = []
    valid_item_ids = [
        item['id'] for item in inventory_database
    ]
    for item_id, quatity in items.items():
        if item_id not in valid_item_ids:
            availability_messages.append(f"Item {item_id} not found in the inventory")
        else:
            inventory_item = [item for item in inventory_database if item['id'] == item_id][0]
            if quatity > inventory_item['quantity']:
                availability_messages.append(f"There is inssufficient quantity in the inventory for this item {inventory_item['name']} \n Available quantity: {inventory_item['quantity']}\n Requested quantity: {quatity}") 

    if availability_messages:
        return "Order cannot be placed due to the following issues: \n" + "\n".join(availability_messages)
    
    # Place the order (in pretend database)

    order_database.append({
        'order_id': f'ORD{len(order_database) + 1}',
        'customer_id': customer_id,
        'status': 'Waiting for processing',
        'items': list(items.keys()),
        'quantity': list(items.values())
        })

    # Update the inventory
    for item_id, quatity in items.items():
        inventory_item = [item for item in inventory_database if item['id'] == item_id][0]
        inventory_item['quantity'] -= quatity
    
    return f"Order with id {order_database[-1]['order_id']} has been placed successfully"