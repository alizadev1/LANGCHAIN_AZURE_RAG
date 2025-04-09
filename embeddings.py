import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    model=os.getenv("azure_openai_api_model")
)

text = """
LangChain is a framework for developing applications powered by language models. It helps with chaining different components like LLMs, retrievers, and vector stores to build RAG (retrieval augmented generation) pipelines.
This script shows how to split text, embed it with OpenAI, and store it in Azure Cognitive Search.
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
chunks = text_splitter.split_text(text)

print(f"Split into {len(chunks)} chunks.")

test_vector = embedding_model.embed_query(chunks[0])
print("\nVector length:", len(test_vector))

page_id = "0029983029493489348"

client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

documents = []
for i, chunk in enumerate(chunks):
    vector = embedding_model.embed_query(chunk)
    documents.append({
        "id": f"chunk_{i}",
        "content": chunk,
        "vector": vector,
        "metadata": {
            "page_id": page_id
        }
    })

result = client.upload_documents(documents)
for res in result:
    print(f"📄 Document ID: {res.key} | Status: {res.status_code} | Error: {res.error_message}")
print("this is page id",page_id)