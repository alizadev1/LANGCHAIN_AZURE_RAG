import os
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv()

# Get credentials from .env
endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
key = os.getenv("AZURE_SEARCH_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Initialize client
index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Fetch the index definition
index = index_client.get_index(index_name)

# Print all fields
print(f"📄 Fields for index: {index_name}")
vector_field_name = None
for field in index.fields:
    print(f"- {field.name}: {field.type}")
    # Check if the field type matches vector type (Collection(Edm.Single))
    if field.type == "Collection(Edm.Single)":
        vector_field_name = field.name

if vector_field_name:
    print(f"\nThe vector field is named: {vector_field_name}")
else:
    print("\nNo vector field found.")
