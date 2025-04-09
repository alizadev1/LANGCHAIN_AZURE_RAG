import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

load_dotenv()

azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

vectorstore = AzureSearch(
    azure_search_endpoint=azure_search_endpoint,
    azure_search_key=azure_search_key,
    index_name=index_name,
    embedding_function=embedding_model
)

query = "What is LangChain used for?"
results = vectorstore.similarity_search(query, k=2)

print("\n🔍 Top matches:\n")
for i, doc in enumerate(results, start=1):
    print(f"{i}. Content: {doc.page_content}")
    print(f"   Metadata: {doc.metadata}")
