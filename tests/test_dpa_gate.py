"""The data-protection gate: orders are only read or placed for customers verified in this conversation.

Run from the project root (tools.py reads inventory.json from the working directory):
    uv run --no-project --with pytest --with langchain-core --with langgraph pytest tests -q
"""

import importlib
import sys
import types
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

PROJECT = Path(__file__).resolve().parent.parent
JOHN = {"name": "John Doe", "postcode": "SW1A 1AA", "year_of_birth": 1990, "month_of_birth": 1,
        "day_of_birth": 1}
ITEM = {"N001": 1}  # first inventory item; quantity is checked against inventory.json


@pytest.fixture
def tools(monkeypatch):
    """A fresh tools module per test, with the vector store (OpenAI + Chroma) stubbed out."""
    fake = types.ModuleType("vector_store")
    fake.ShopVectorStore = type("ShopVectorStore", (), {
        "query_faqs": lambda self, query: [], "query_inventories": lambda self, query: []})
    monkeypatch.setitem(sys.modules, "vector_store", fake)
    monkeypatch.chdir(PROJECT)
    monkeypatch.syspath_prepend(str(PROJECT))
    sys.modules.pop("tools", None)
    return importlib.import_module("tools")


def order(tools, customer_id):
    return tools.place_order.invoke({"items": ITEM, "customer_id": customer_id})


def test_orders_are_refused_when_no_conversation_is_bound(tools):
    assert "data protection check" in order(tools, "CUST001")
    assert "data protection check" in tools.retrieve_existing_customer_orders.invoke(
        {"customer_id": "CUST001"})


def test_orders_are_refused_for_a_customer_not_verified_in_this_conversation(tools):
    orders_before = len(tools.order_database)
    with tools.customer_session(set()):
        result = order(tools, "CUST001")

    assert "data protection check" in result
    assert len(tools.order_database) == orders_before


def test_passing_the_check_unlocks_only_that_customer(tools):
    with tools.customer_session(set()):
        assert "passed" in tools.data_protection_check.invoke(JOHN)
        assert "placed successfully" in order(tools, "CUST001")
        assert "ORD001" in str(tools.retrieve_existing_customer_orders.invoke({"customer_id": "CUST001"}))
        assert "data protection check" in order(tools, "CUST002")


def test_a_failed_check_verifies_nobody(tools):
    with tools.customer_session(set()):
        assert "failed" in tools.data_protection_check.invoke({**JOHN, "postcode": "WRONG"})
        assert "data protection check" in order(tools, "CUST001")


def test_creating_a_profile_verifies_the_new_customer(tools):
    profile = {"first_name": "Ada", "surname": "Lovelace", "year_of_birth": 1990,
               "month_of_birth": 12, "day_of_birth": 10, "postcode": "N1 1AA",
               "first_line_of_address": "1 Analytical St", "phone_number": "07700900123",
               "email": "ada@example.com"}
    with tools.customer_session(set()):
        created = tools.create_new_customer.invoke(profile)
        new_id = created.rsplit(" ", 1)[-1]

        assert "placed successfully" in order(tools, new_id)


def test_conversations_do_not_share_verification(tools):
    alice_session, bob_session = set(), set()
    with tools.customer_session(alice_session):
        tools.data_protection_check.invoke(JOHN)
    with tools.customer_session(bob_session):
        assert "data protection check" in order(tools, "CUST001")


def test_gate_holds_when_tools_run_inside_a_langgraph_graph(tools):
    # Same shape as chatbot.py: a compiled StateGraph running a ToolNode, which may execute tools
    # on worker threads; the conversation's verified set must still reach them.
    from langgraph.graph import MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    graph = StateGraph(MessagesState)
    graph.add_node("tool_node", ToolNode([tools.data_protection_check, tools.place_order]))
    graph.set_entry_point("tool_node")
    graph.set_finish_point("tool_node")
    app = graph.compile()
    calls = AIMessage(content="", tool_calls=[
        {"name": "place_order", "args": {"items": ITEM, "customer_id": "CUST001"}, "id": "1"}])
    verify = AIMessage(content="", tool_calls=[
        {"name": "data_protection_check", "args": JOHN, "id": "2"}])

    with tools.customer_session(set()):
        refused = app.invoke({"messages": [calls]})["messages"][-1].content
        app.invoke({"messages": [verify]})
        placed = app.invoke({"messages": [calls]})["messages"][-1].content

    assert "data protection check" in refused
    assert "placed successfully" in placed
