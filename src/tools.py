from langchain.tools import tool
import requests
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
from src.rag.retrieval import RetrievalService


_retrieval_service = None

load_dotenv()

#GET SEARCH TOOL ==============================================================================
search_tool=TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)

#GET RAG TOOL ================================================================================

# =========================================================
# RAG KNOWLEDGE-BASE TOOL
# =========================================================

# Create the service only when the tool is first called.
# This avoids loading the embedding model during module import.
_retrieval_service = None


@tool
def search_knowledge_base(question: str) -> str:
    """Search uploaded PDF documents for information relevant to a question.

    Use this tool when the answer may be found in the user's uploaded
    documents. Return the retrieved excerpts with their source and page.
    """
    global _retrieval_service

    if not question.strip():
        return "Please provide a question to search for."

    try:
        if _retrieval_service is None:
            _retrieval_service = RetrievalService()

        documents = _retrieval_service.retrieve(
            question=question,
            top_k=5,
            namespace="default",
        )

        if not documents:
            return "No relevant information was found in the knowledge base."

        results = []

        for number, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "Unknown")
            page = document.metadata.get("page")

            # PyPDFLoader page numbers start at zero.
            page_label = (
                page + 1 if isinstance(page, int) else "Unknown"
            )

            results.append(
                f"Result {number}\n"
                f"Source: {source}\n"
                f"Page: {page_label}\n"
                f"Content:\n{document.page_content}"
            )

        return "\n\n---\n\n".join(results)

    except Exception:
        import logging

        logging.getLogger(__name__).exception(
            "Knowledge-base search failed."
        )
        return "Knowledge-base search failed. Check the application logs."
    

#GET WEATHER TOOL ============================================================================
@tool
def get_weather(city: str) -> str:
    
    """
    Retrieve the current weather information for a specified city.

    Provides the current weather conditions, temperature, feels-like
    temperature, humidity, and wind speed using the OpenWeather API.

    Args:
        city (str): The name of the city to retrieve weather information for.

    Returns:
        str: A formatted summary of the current weather, or an error
        message if the API key is missing, the city is not found,
        or the weather data cannot be retrieved.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "Error: OpenWeather API key is not configured."

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()
        data = response.json()

        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        wind_speed = data["wind"]["speed"]

        return (
            f"Current weather in {data['name']}:\n"
            f"Conditions: {description}\n"
            f"Temperature: {temperature}°C\n"
            f"Feels like: {feels_like}°C\n"
            f"Humidity: {humidity}%\n"
            f"Wind speed: {wind_speed} m/s"
        )

    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return f"City '{city}' was not found."
        if response.status_code == 401:
            return "Weather API key is invalid or not activated."
        return f"Weather API error: {e}"

    except requests.exceptions.RequestException as e:
        return f"Unable to retrieve weather: {e}"


#CALCULATOR TOOL =========================================================================
import ast
import math
import operator

from langchain_core.tools import tool


# =========================================================
# 1. ALLOWED MATHEMATICAL OPERATORS
# =========================================================

# Python provides functions for mathematical operations.
# For example, operator.add(10, 5) returns 15.
#
# AST identifies which operator appears in the expression.
# This dictionary maps each allowed AST operator to its
# corresponding Python function.
#
# We only execute operators explicitly included here.

OPERATORS = {
    ast.Add: operator.add,          # 10 + 5
    ast.Sub: operator.sub,          # 10 - 5
    ast.Mult: operator.mul,         # 10 * 5
    ast.Div: operator.truediv,      # 10 / 5
    ast.FloorDiv: operator.floordiv,# 10 // 3
    ast.Mod: operator.mod,          # 10 % 3
    ast.Pow: operator.pow,          # 2 ** 3
    ast.UAdd: operator.pos,         # +5
    ast.USub: operator.neg,         # -5
}


# =========================================================
# 2. ALLOWED MATHEMATICAL FUNCTIONS
# =========================================================

# These are the only mathematical functions users can call.
#
# The dictionary key is the name users type.
# The dictionary value is the actual Python function.
#
# Example:
#     sqrt(16) -> math.sqrt(16) -> 4.0
#
# We deliberately do not allow arbitrary Python functions.

MATH_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "log": math.log,
}


# =========================================================
# 3. SAFETY LIMITS
# =========================================================

# Maximum number of characters in the user's expression.
MAX_EXPRESSION_LENGTH = 200

# Maximum number of AST nodes in one expression.
MAX_AST_NODES = 100

# Maximum recursive nesting depth.
MAX_DEPTH = 20

# Maximum permitted absolute numeric value.
MAX_MAGNITUDE = 1e100

# Maximum absolute exponent.
MAX_EXPONENT = 100


# =========================================================
# 4. VALIDATE NUMERIC VALUES
# =========================================================

def validate_number(value):
    """
    Check that a value is a permitted, finite number.

    Rejects booleans, unsupported types, non-finite floats,
    and excessively large results.
    """

    # bool is a subclass of int in Python.
    # Using type(...) instead of isinstance(...) rejects True
    # and False as numeric inputs.
    if type(value) not in (int, float):
        raise ValueError("Only real numbers are supported.")

    # Limit the size of integers before further processing.
    if isinstance(value, int) and abs(value) > 10**100:
        raise ValueError("Integer result is too large.")

    # Reject NaN, positive infinity, and negative infinity.
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Result must be finite.")

    # Reject numbers beyond our configured magnitude limit.
    if abs(value) > MAX_MAGNITUDE:
        raise ValueError("Number exceeds the permitted limit.")

    # Return the original value if all checks pass.
    return value


# =========================================================
# 5. EVALUATE AST NODES
# =========================================================

def calculate(node, depth=0):
    """
    Evaluate one AST node recursively.

    A node represents a piece of the expression:
    a number, unary operation, function call, or binary operation.

    The function evaluates only explicitly approved node types.
    """

    # SAFETY CHECK:
    # Each recursive call increases depth.
    # Reject expressions that are nested too deeply.
    if depth > MAX_DEPTH:
        raise ValueError("Expression is too deeply nested.")

    # -----------------------------------------------------
    # CASE 1: THE NODE IS A NUMBER
    # -----------------------------------------------------
    # Examples: 10, 3.5
    #
    # ast.Constant represents literal values in the AST.
    if isinstance(node, ast.Constant):

        # Accept only integers and floating-point numbers.
        if type(node.value) not in (int, float):
            raise ValueError("Only numeric constants are allowed.")

        # Validate the number and return it.
        return validate_number(node.value)

    # -----------------------------------------------------
    # CASE 2: THE NODE IS A UNARY OPERATION
    # -----------------------------------------------------
    # A unary operation applies to one value.
    #
    # Examples:
    #     -5 -> unary minus
    #     +5 -> unary plus
    #
    # node.op identifies the sign.
    # node.operand identifies the value receiving the sign.
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:

        # Recursively evaluate the operand.
        # For -5, this returns the number 5.
        operand = calculate(node.operand, depth + 1)

        # Find the approved sign function and apply it.
        # For -5, this calls operator.neg(5).
        result = OPERATORS[type(node.op)](operand)

        # Validate and return the result.
        return validate_number(result)

    # -----------------------------------------------------
    # CASE 3: THE NODE IS A FUNCTION CALL
    # -----------------------------------------------------
    # Examples: sqrt(16), sin(0), log(100)
    #
    # ast.Call identifies a function call.
    # ast.Name ensures the function is referenced by a simple
    # name, such as sqrt, rather than an attribute expression.
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):

        # Look up the requested function in our allowlist.
        # If it doesn't exist, .get() returns None.
        function = MATH_FUNCTIONS.get(node.func.id)

        # Reject functions that are not explicitly approved.
        if function is None:
            raise ValueError("Unsupported mathematical function.")

        # We don't allow keyword arguments.
        # We also limit the number of arguments to two.
        if node.keywords or len(node.args) > 2:
            raise ValueError("Invalid function arguments.")

        # Evaluate each argument recursively.
        # For sqrt(16), args becomes [16].
        args = [
            calculate(arg, depth + 1)
            for arg in node.args
        ]

        # Execute only the approved function.
        # function(*args) passes each list item as an argument.
        # For sqrt(16), this is equivalent to math.sqrt(16).
        result = function(*args)

        # Validate and return the result.
        return validate_number(result)

    # -----------------------------------------------------
    # CASE 4: THE NODE IS A BINARY OPERATION
    # -----------------------------------------------------
    # A binary operation combines two values.
    #
    # Examples: 10 + 5, 10 * 5, 10 / 2
    #
    # node.left  -> left-hand side
    # node.op    -> operator
    # node.right -> right-hand side
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:

        # Calculate the left-hand side first.
        left = calculate(node.left, depth + 1)

        # Calculate the right-hand side.
        right = calculate(node.right, depth + 1)

        # Prevent excessively large exponent calculations.
        # Example: reject 2 ** 100000.
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ValueError("Exponent exceeds the permitted limit.")

        # Look up and execute the approved operator.
        # For 10 * 5:
        #     type(node.op) is ast.Mult
        #     OPERATORS[ast.Mult] is operator.mul
        #     operator.mul(10, 5) returns 50
        result = OPERATORS[type(node.op)](left, right)

        # Validate and return the result.
        return validate_number(result)

    # -----------------------------------------------------
    # CASE 5: UNSUPPORTED EXPRESSION
    # -----------------------------------------------------
    # If no allowed case matches, reject the expression.
    #
    # Examples:
    #     x + 5
    #     [1, 2, 3]
    #     __import__('os')
    raise ValueError("Unsupported mathematical expression.")


# =========================================================
# 6. PARSE AND VALIDATE THE USER'S INPUT
# =========================================================

def evaluate_expression(expression: str):
    """
    Accept a mathematical expression as text,
    parse it into an AST, and evaluate the AST safely.
    """

    # Ensure the input is a string.
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")

    # Remove unnecessary spaces from the beginning and end.
    expression = expression.strip()

    # Reject empty input.
    if not expression:
        raise ValueError("Expression cannot be empty.")

    # Limit input length to reduce abuse and excessive parsing.
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression is too long.")

    # Convert the text into an AST.
    # mode="eval" means one expression, not a full Python program.
    tree = ast.parse(expression, mode="eval")

    # Count all nodes in the parsed expression.
    # Reject expressions with too many nodes.
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("Expression is too complex.")

    # tree.body is the root node of the expression.
    # Pass it to calculate() to evaluate it.
    return calculate(tree.body)


# =========================================================
# 7. EXPOSE THE CALCULATOR AS A LANGCHAIN TOOL
# =========================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression using approved operators
    and functions.

    Supports +, -, *, /, //, %, **, unary signs, sqrt, sin, cos,
    and log.

    Examples:
        10 + 5
        (10 + 5) * 2
        -5 + 10
        sqrt(16)
        log(100)

    Returns the result as text or a readable error message.
    """

    try:
        # Run the expression through our validation and evaluator.
        result = evaluate_expression(expression)

        # LangChain tools commonly return text.
        return str(result)

    except (
        ValueError,
        SyntaxError,
        ZeroDivisionError,
        OverflowError,
        TypeError,
        RecursionError,
    ):
        # Avoid exposing internal exception details to the user.
        return (
            "Error: Invalid expression or calculation "
            "outside the permitted limits."
        )

tools = [get_weather,calculator,search_tool,search_knowledge_base]