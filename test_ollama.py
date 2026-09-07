from ollama import Client

client = Client(
    host="http://127.0.0.1:11434"
)

try:
    response = client.embed(
        model="nomic-embed-text:latest",
        input="What is the process for requesting production access?"
    )

    embedding = response["embeddings"][0]

    print("\nSUCCESS - Ollama embedding is working!")
    print("Embedding dimension:", len(embedding))
    print("First 10 values:")
    print(embedding[:10])

except Exception as error:

    print("\nERROR - Ollama embedding failed:")
    print(error)