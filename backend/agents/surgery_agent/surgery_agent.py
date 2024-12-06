from langgraph.graph.state import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from .nodes.anatomy_node import anatomy_node
from .nodes.complication_management_node import complication_management_node
from .nodes.expert_surgeon_node import expert_surgeon_node
from .nodes.infection_prevention_node import infection_prevention_node
from .nodes.response_generation_node import response_generation_node
from .nodes.rewrite_node import rewrite
from .nodes.insights_accumulator_node import insight_accumulator_node

from .routing_functions.grade_document import grade_document

from .tools.query_database import query_database

from state import State

def surgical_agent(state):
    builder = StateGraph(State)

    # Define retriever tool node
    tools = [query_database]
    tool_node = ToolNode(tools)

    # Add nodes
    builder.add_node("Anatomy Specialist", anatomy_node)
    builder.add_node("Infection Prevention Specialist", infection_prevention_node)
    builder.add_node("Complication Manager", complication_management_node)
    builder.add_node("Insight Accumulator", insight_accumulator_node)
    builder.add_node("Expert Surgeon", expert_surgeon_node)
    builder.add_node('Answer Generation', response_generation_node)
    builder.add_node("Retriever", tool_node)
    builder.add_node("Rewrite", rewrite)

    # Define edges
    builder.add_edge(START, "Anatomy Specialist")
    builder.add_edge('Anatomy Specialist', 'Infection Prevention Specialist')
    builder.add_edge("Infection Prevention Specialist", "Complication Manager")
    builder.add_edge("Complication Manager", "Expert Surgeon")
    builder.add_edge('Expert Surgeon', 'Retriever')
    builder.add_conditional_edges("Retriever",
                                grade_document,
                                ['Insight Accumulator', 'Rewrite']
                                )
    builder.add_edge('Rewrite', 'Expert Surgeon')
    builder.add_edge('Insight Accumulator', 'Answer Generation')
    builder.add_edge('Answer Generation', END)
    agent = builder.compile()
    result = agent.invoke(state)
    return result['answer']
    