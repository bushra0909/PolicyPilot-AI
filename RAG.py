import requests
import chromadb


# 1. Create a ChromaDB client and collection
client = chromadb.Client()
collection = client.create_collection(name="company_policy")


# 2. Read the company policy document
with open("company_policy.txt", "r", encoding="utf-8") as file:
    document = file.read()


# 3. Split the document into chunks
chunks = [
    section.strip()
    for section in document.split("\n## ")
    if section.strip()
]


# 4. Create embeddings for the policy chunks
response = requests.post(
    "http://localhost:11434/api/embed",
    json={
        "model": "nomic-embed-text",
        "input": chunks
    },
    timeout=120
)
response.raise_for_status()
embeddings = response.json()["embeddings"]


# 5. Store the chunks and embeddings in ChromaDB
collection.add(
    documents=chunks,
    embeddings=embeddings,
    # create a unique ID for each chunk
    ids=[f"chunk_{i}" for i in range(len(chunks))] )


# 6. Retrieve policy evidence and generate an answer
def retrieve_answer(question):

    # Convert the user's question into an embedding.
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": question
        },
        timeout=120
    )
    response.raise_for_status()
    question_embedding = response.json()["embeddings"][0]

    # Retrieve the two closest policy chunks.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=2,
        include=["documents", "distances"]
    )

    closest_distance = results["distances"][0][0]

    # Reject distant matches using a provisional cutoff.
    if closest_distance > 0.6:
        return (
            "I don't know. The policy document doesn't provide "
            "relevant information.",
            ""
        )

    # Combine the retrieved evidence.
    relevant_chunks = results["documents"][0]
    relevant_info = "\n".join(relevant_chunks)

    # Ask Llama to answer using only the retrieved evidence.
    prompt = f"""
You are a company policy assistant.

Answer the question using only the policy information below.
Do not invent rules or add unsupported details.
If the information does not answer the question, say:
"I don't know. The policy document doesn't provide that information."

Give a concise answer first.
Do not repeat the entire policy section.

Policy information:
{relevant_info}

Question:
{question}

Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "keep_alive": "10m"
            },
            timeout=180
        )
        response.raise_for_status()
        answer = response.json()["response"].strip()

    except (requests.RequestException, KeyError, ValueError):
        return (
            "I couldn't generate an answer right now. Please try again.",
            relevant_info
        )

    # Return the answer and its supporting evidence.
    return answer, relevant_info


# 7. Run the assistant when this file is executed directly.
if __name__ == "__main__":
    question = input("Ask a company policy question: ")
    answer, evidence = retrieve_answer(question)

    print("\nANSWER")
    print(answer)

    if evidence:
        print("\nSUPPORTING EVIDENCE")
        print(evidence)
