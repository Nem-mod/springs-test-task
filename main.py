import os
import chromadb
import logging
import sys
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core import PromptHelper, ServiceContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool


OPENAI_API_KEY = "<YOUR_API_KEY>"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Defining ChatGPT model
OPENAI_MODEL = "gpt-3.5-turbo-16k"

# ChatGPT completion setup
OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.1,  # respond to accuracy of llm (from 0.1 up to 2)
    "max_tokens": 1000,  # max amount of tokens that llm is uses
    "top_p": 1,  # top value of temperature
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "request_timeout": 60.0,
}

# Base prompt
LLM_BASE_PROMPT = ""

# Create instance of llm
llm = OpenAI(model=OPENAI_MODEL, temperature=0)
# Define embeddings model
embed_model = OpenAIEmbedding()

# Set up Node parser
node_parser = SimpleNodeParser.from_defaults(
    chunk_size=1024,
    chunk_overlap=20
)

prompt_helper = PromptHelper(
    context_window=4096,
    num_output=256,
    chunk_overlap_ratio=0.1,
    chunk_size_limit=None
)

# Logging
llama_debug = LlamaDebugHandler()
callback_manager = CallbackManager([llama_debug])

service_context = {
    "llm": llm,
    "embed_model": embed_model,
    "node_parser": node_parser,
    "prompt_helper": prompt_helper,
    "system_prompt": LLM_BASE_PROMPT,
    "callback_manager": callback_manager
}


db2 = chromadb.PersistentClient(path="./chroma_db")

chroma_stripe_collection = db2.get_or_create_collection("stripe_embeddings")
chroma_coffeB_collection = db2.get_or_create_collection("coffeB_embeddings")

# load documents
stripe_documents = SimpleDirectoryReader("./data/stripe/").load_data()
coffeB_documents = SimpleDirectoryReader("./data/coffeB/").load_data()

# set up ChromaVectorStore and load in data
stripe_vector_store = ChromaVectorStore(chroma_collection=chroma_stripe_collection)
stripe_storage_context = StorageContext.from_defaults(vector_store=stripe_vector_store)
stripe_index = VectorStoreIndex.from_documents(
    stripe_documents, **service_context, storage_context=stripe_storage_context
)

coffeB_vector_store = ChromaVectorStore(chroma_collection=chroma_coffeB_collection)
coffeB_storage_context = StorageContext.from_defaults(vector_store=coffeB_vector_store)
coffeB_index = VectorStoreIndex.from_documents(
    coffeB_documents, **service_context, storage_context=coffeB_storage_context
)


# initialize tools
list_tool = QueryEngineTool.from_defaults(
    query_engine=stripe_index.as_query_engine(),
    description="Useful for retrieving information about Stripe.",
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=coffeB_index.as_query_engine(),
    description="Manual for coffeB coffe machine. Retrieves full guidance how to use coffeB machine.",
)

# initialize router query engine (single selection, pydantic)
query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)




logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

while True:
    response = query_engine.query(input())
    print(f"----------------------------------\n\n"
          f"Answer: {response}"
          f"----------------------------------\n\n")