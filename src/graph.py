from src.nodes import chatbot_node, response_node
from src.state import ChatState
from langgraph.graph import StateGraph, START, END

#DEFINE GRAPH
graph = StateGraph(ChatState)

#DEFINE NODES
graph.add_node('chatbot_node',chatbot_node)
graph.add_node('response_node',response_node)

#DEFINE CONNECTION
graph.add_edge(START,'chatbot_node')
graph.add_edge('chatbot_node','response_node')
graph.add_edge('response_node',END)

#COMPILE THE GRAPH
def build_graph(checkpointer):
    return graph.compile(checkpointer=checkpointer)