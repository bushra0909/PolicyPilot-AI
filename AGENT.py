
import json
import re
import requests

from RAG import retrieve_answer


# 1. Define the tools the AI agent can use.
# Llama chooses a tool; Python performs the actual operation.

def add(a, b):
    """Add two numbers."""
    return a + b


def calculate(a, b):
    """Multiply two numbers."""
    return a * b


def search_company_policy(question):
    """Answer a policy question and return supporting evidence."""
    return retrieve_answer(question)


# 2. Create a reusable connection to the local Ollama API.
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

session = requests.Session()


# 3. Ask Llama to select one tool using a short JSON response.
def choose_tool(question):
    prompt = f"""
You are a tool-selection agent.
Choose exactly one tool for the user's request.

Tools:
1. add: adds two numbers.
2. calculate: multiplies two numbers.
3. search_company_policy: answers company policy questions.

Return ONLY a JSON object. Do not explain your choice.

Examples:
{{"tool":"add","a":4,"b":5}}
{{"tool":"calculate","a":4,"b":5}}
{{"tool":"search_company_policy","question":"How many paid leave days are allowed?"}}

User request: {question}

JSON:
"""

# Send the prompt to Ollama and wait for its response.
    response = session.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "10m",
            "options": {
                "temperature": 0,  # Makes responses more consistent.
                "num_predict": 80  # Limits the response to about 80 tokens.
            }
        },
        timeout=(5, 120)  # 5 seconds to connect, 120 seconds to receive.
    )

# Check whether Ollama's API request failed.
    response.raise_for_status()

# Get the text returned by Ollama.
    raw = response.json()["response"]

# Convert the AI's JSON text into a Python dictionary.
    decision = json.loads(raw)

 # Check that the AI returned a dictionary.
    if not isinstance(decision, dict):
        raise ValueError("The agent did not return a JSON object.")

# Send the AI's decision back to the calling code.
    return decision


# 4. Check the AI's decision and run the selected tool.
def execute_tool(decision):

    # Get the name of the tool selected by the AI.
    tool = decision.get("tool")

    # If the AI selected addition or calculation:
    if tool in ("add", "calculate"):

        # Get the two numbers provided by the AI.
        a = decision.get("a")
        b = decision.get("b")

        # Check that both values are numbers, not missing,
        # text, or True/False.
        if (
            isinstance(a, bool) or isinstance(b, bool)
            or not isinstance(a, (int, float))
            or not isinstance(b, (int, float))
        ):
            raise ValueError("The selected tool needs two numbers.")

        # Run the addition function if the tool is "add".
        if tool == "add":
            return add(a, b), ""

        # Otherwise, run the calculation function.
        return calculate(a, b), ""

    # If the AI selected the company policy search tool:
    if tool == "search_company_policy":

        # Get the question that needs to be answered.
        policy_question = decision.get("question")

        # Check that the question is valid and not empty.
        if not isinstance(policy_question, str) or not policy_question.strip():
            raise ValueError("The policy question is missing.")

        # Search the policy document and return its result.
        return search_company_policy(policy_question), None

    # If the AI selected a tool we don't recognize, show an error.
    raise ValueError("The agent selected an unknown tool.")


# 5. Run the assistant and display the final answer.
def main():

    # Ask the user to enter a question and remove extra spaces.
    question = input("Ask a question: ").strip()

    # If the user entered nothing, show a message and stop.
    if not question:
        print("Please enter a question.")
        return

    # Try to process the question without crashing the program.
    try:

        # Ask the AI to choose which tool should be used.
        decision = choose_tool(question)

        # Check the decision and execute the selected tool.
        result = execute_tool(decision)

        # Policy tools return (answer, evidence).
        if decision["tool"] == "search_company_policy":

            # The policy function returns (answer, evidence).
            answer, evidence = result[0]

            # Display the answer.
            print("\nANSWER")
            print(answer)

            # Display evidence only if some was returned.
            if evidence:
                print("\nSUPPORTING EVIDENCE")
                print(evidence)

        else:
            # Maths tools return (number, empty string).
            answer, _ = result

            # Display the answer.
            print("\nANSWER")
            print(answer)

    # Show a message if Ollama takes too long to respond.
    except requests.Timeout:
        print("\nThe local AI model took too long to respond. Please try again.")

    # Show a message if Python cannot connect to Ollama.
    except requests.RequestException:
        print("\nCould not connect to the local AI service. Check that Ollama is running.")

    # Show a message if the decision or data is invalid.
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        print("\nThe agent could not process this request correctly. Please try again.")


# Run main() only when this file is executed directly.
if __name__ == "__main__":
    main()
