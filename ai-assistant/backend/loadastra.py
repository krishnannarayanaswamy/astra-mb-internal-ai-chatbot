from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
from pathlib import Path
from constants import ASTRA_COLLECTION_NAME
import json
from langchain_experimental.text_splitter import SemanticChunker

import os

token=os.environ['ASTRA_DB_APPLICATION_TOKEN']
api_endpoint=os.environ['ASTRA_DB_API_ENDPOINT']
keyspace=os.environ['ASTRA_DB_KEYSPACE']
openai_api_key=os.environ["OPENAI_API_KEY"]

def get_embeddings_model() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)

vstore = AstraDBVectorStore(
    embedding=get_embeddings_model(),
    collection_name=ASTRA_COLLECTION_NAME,
    api_endpoint=api_endpoint,
    token=token,
    namespace=keyspace,
)

text_splitter = SemanticChunker(
    get_embeddings_model(), breakpoint_threshold_type="percentile"
)

with open("../data/THẨM-ĐỊNH-TÍN-DỤNG.txt") as f:
    digital_onboarding = f.read()

docs = text_splitter.create_documents([digital_onboarding])
#print(docs[0].page_content)

print(len(docs))

inserted_ids = vstore.add_documents(docs)
print(f"\nInserted {len(inserted_ids)} documents.")
