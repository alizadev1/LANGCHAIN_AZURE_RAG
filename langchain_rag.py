import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_community.retrievers import AzureAISearchRetriever

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
)


text = """
LangChain is a framework for developing applications powered by language models. It helps with chaining different components like LLMs, retrievers, and vector stores to build RAG (retrieval augmented generation) pipelines.
This script shows how to split text, embed it with OpenAI, and store it in Azure Cognitive Search..
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
chunks = text_splitter.split_text(text)

print(f"Split into {len(chunks)} chunks.")
test_vector = embedding_model.embed_query(chunks[0])

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

print("This is page id", page_id)
print("Using deployment:", os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"))

vectorstore = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_KEY,
    index_name=AZURE_SEARCH_INDEX_NAME,
    embedding_function=embedding_model,
    vector_field_name="vector",
    text_key="content",
    embedding_dim=1536,
    search_type="vector",
    vector_search_config="default"
)

service_name = os.getenv("AZURE_SEARCH_ENDPOINT").split("//")[1].split(".")[0]
index_name=AZURE_SEARCH_INDEX_NAME

retriever = AzureAISearchRetriever(
    service_name=service_name,
    index_name=index_name,
    content_key="content",
    top_k=5,
    api_key=AZURE_SEARCH_KEY, 
    filter=f"metadata/page_id eq '{page_id}'",
)

# search_results = client.search(
#     search_text="LangChain",
#     filter=f"metadata/page_id eq '{page_id}'",
#     select="id,metadata,version,content",
#     top=5
# )

# # Print the retrieved results
# for result in search_results:
#     print(f"ID: {result['id']}, Metadata: {result['metadata']}, Version: {result['version']}, Content: {result['content']}")


chat_llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

qa = RetrievalQA.from_chain_type(
    llm=chat_llm,
    retriever=retriever,
    return_source_documents=True
)

query = input("\nAsk your question: ")
result = qa.invoke(query)

print("Answer:", result["result"])
