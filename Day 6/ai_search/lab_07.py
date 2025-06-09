import os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.openai import OpenAIClient

# === Environment Variables ===
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_SERVICE_ENDPOINT", "https://your-openai-endpoint.openai.azure.com")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-openai-api-key")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://your-search-service.search.windows.net")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY", "your-search-api-key")
AZURE_INDEX_NAME = os.getenv("AZURE_INDEX_NAME", "your-index-name")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "your-deployment-name")

# === Retrieve Documents Using Azure Cognitive Search ===
def retrieve_documents(query):
    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_INDEX_NAME, credential=credential)
    results = search_client.search(query)
    documents = [doc for doc in results]
    return documents

# === Generate Answer Using Azure OpenAI Service ===
def generate_answer(query, documents):
    # Combine document content into a context string
    context = "\n".join(doc.get("content", "") for doc in documents)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    openai_client = OpenAIClient(endpoint=AZURE_OPENAI_ENDPOINT, credential=AzureKeyCredential(AZURE_OPENAI_API_KEY))
    response = openai_client.get_completions(
        deployment_id=AZURE_DEPLOYMENT_NAME,
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    answer = response.choices[0].text.strip() if response.choices else ""
    return answer

# === RAG Pipeline: Retrieve Documents and Generate Answer ===
def rag_pipeline(query):
    documents = retrieve_documents(query)
    if not documents:
        return "No documents found."
    answer = generate_answer(query, documents)
    return answer

if __name__ == "__main__":
    query = input("Enter your query: ")
    result = rag_pipeline(query)
    print("Answer:", result)