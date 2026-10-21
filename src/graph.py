from src.nodes import chatbot_node, response_node
from src.state import ChatState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from src.tools import tools

#DEFINE TOOLNODE
tool_node=ToolNode(tools)

#DEFINE GRAPH
graph = StateGraph(ChatState)

#DEFINE ROUTHING FUNCTION
def route_after_chatbot(state:ChatState):
    last_message=state['messages'][-1]

    if last_message.tool_calls:
        return "tool_node"

    return "response_node"

#DEFINE NODES
graph.add_node('chatbot_node',chatbot_node)
graph.add_node('tool_node',tool_node)
graph.add_node('response_node',response_node)

#DEFINE CONNECTION
graph.add_edge(START,'chatbot_node')

#DECIDE TO USE TOOL
graph.add_conditional_edges(
    "chatbot_node",
    route_after_chatbot,
    {
        "tool_node":"tool_node",
        "response_node":"response_node"
    }
)

#After executing a tool, return to the AI.
graph.add_edge("tool_node","chatbot_node")

#Finish after preparing the final answer.
graph.add_edge('response_node',END)

#COMPILE THE GRAPH
def build_graph(checkpointer):
    return graph.compile(checkpointer=checkpointer)