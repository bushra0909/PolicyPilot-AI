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

    # A short output limit reduces unnecessary model generation.
    response = session.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "10m",
            "options": {
                "temperature": 0,
                "num_predict": 80
            }
        },
        timeout=(5, 120)
    )

    # Raise an error if Ollama returns an unsuccessful HTTP status.
    response.raise_for_status()

    # Extract the model's JSON response.
    raw = response.json()["response"]
    decision = json.loads(raw)

    if not isinstance(decision, dict):
        raise ValueError("The agent did not return a JSON object.")

    return decision


# 4. Validate and execute the selected tool.
def execute_tool(decision):
    tool = decision.get("tool")

    if tool in ("add", "calculate"):
        # Reject missing or non-numeric arguments.
        a = decision.get("a")
        b = decision.get("b")

        if (
            isinstance(a, bool) or isinstance(b, bool)
            or not isinstance(a, (int, float))
            or not isinstance(b, (int, float))
        ):
            raise ValueError("The selected tool needs two numbers.")

        if tool == "add":
            return add(a, b), ""

        return calculate(a, b), ""

    if tool == "search_company_policy":
        policy_question = decision.get("question")

        if not isinstance(policy_question, str) or not policy_question.strip():
            raise ValueError("The policy question is missing.")

        return search_company_policy(policy_question), None

    raise ValueError("The agent selected an unknown tool.")


# 5. Run the assistant and display a clean, user-facing result.
def main():
    question = input("Ask a question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    try:
        decision = choose_tool(question)
        result = execute_tool(decision)

        # Policy tools return (answer, evidence); maths tools return a number.
        if decision["tool"] == "search_company_policy":
            answer, evidence = result[0]

            print("\nANSWER")
            print(answer)

            if evidence:
                print("\nSUPPORTING EVIDENCE")
                print(evidence)
        else:
            answer, _ = result

            print("\nANSWER")
            print(answer)

    except requests.Timeout:
        print("\nThe local AI model took too long to respond. Please try again.")
    except requests.RequestException:
        print("\nCould not connect to the local AI service. Check that Ollama is running.")
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        print("\nThe agent could not process this request correctly. Please try again.")


# Run only when AG.py is executed directly.
if __name__ == "__main__":
    main()
