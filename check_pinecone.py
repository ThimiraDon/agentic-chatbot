
from dotenv import load_dotenv
from pinecone import Pinecone
import os

load_dotenv(".env", override=True)

client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.environ["PINECONE_INDEX_NAME"]

details = client.describe_index(index_name)

print("Index name:", index_name)
print("Index host:", details.host)
print("Index dimension:", getattr(details, "dimension", None))
print("Index metric:", getattr(details, "metric", None))
print("Index status:", details.status)
print("Index schema:", getattr(details, "schema", None))
