# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, missing-module-docstring, missing-function-docstring

import json
import os
import yaml
import argparse
import getpass

from llama_index.core.node_parser import SentenceSplitter

# from llama_index.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI

# from llama_index.llms import OpenAI
from llama_index.core import (
    StorageContext,
    ServiceContext,
)
from llama_index.readers.string_iterable import StringIterableReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores.deeplake import DeepLakeVectorStore

from global_variable import (
    PILLS_JSON_FILE_CLEANED,
    PILLS_JSON_FILE_CLEANED_2,
    VECTOR_STORE_PATH_DESCRIPTION,
)

parser = argparse.ArgumentParser()
parser.add_argument("--credentials", action="store_true")
args = parser.parse_args()

if args.credentials:
    os.environ["ACTIVELOOP_TOKEN"] = input("Copy and paste your ActiveLoop token: ")
    os.environ["OPENAI_API_KEY"] = input("Copy and paste your OpenAI API key: ")

else:
    from dotenv import load_dotenv

    load_dotenv()  # take environment variables from .env.
    os.environ["ACTIVELOOP_TOKEN"] = os.getenv("ACTIVELOOP_TOKEN")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

print("credentials entered")


def create_storage_and_service_contexts(
    vector_store_path: str,
):
    vector_store = load_vector_store(
        vector_store_path, overwrite=True, token=os.environ["ACTIVELOOP_TOKEN"]
    )
    loader = StringIterableReader()
    chunks = create_chunks(get_pills_info())
    documents = loader.load_data(texts=chunks)

    node_parser = SentenceSplitter.from_defaults(separator="\n")
    nodes = node_parser.get_nodes_from_documents(documents)

    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of docs: {len(documents)}")

    # To ensure same id's per run, we manually set them.
    for idx, node in enumerate(nodes):
        node.id_ = f"node_{idx}"

    llm = OpenAI(model="gpt-4")
    # text_splitter = SentenceSplitter(separator="\n", chunk_size=1024, chunk_overlap=20)

    service_context = ServiceContext.from_defaults(llm=llm)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        service_context=service_context,
    )
    return service_context, storage_context, nodes, llm, index


def create_chunks(pills_info: dict):
    chunks = [
        pills_info[el]["description"] if pills_info[el]["description"] else "."
        for el in pills_info
    ]
    return chunks


def load_vector_store(
    vector_store_path: str,
    overwrite: bool = False,
    token: str = None,
    runtime: dict = {"tensor_db": True},
):

    vector_store = DeepLakeVectorStore(
        dataset_path=vector_store_path,
        overwrite=overwrite,
        runtime=runtime,
        read_only=True,
        token=token,
    )

    return vector_store


def get_pills_info():
    with open(PILLS_JSON_FILE_CLEANED, "r", encoding="utf-8") as file:
        pills_info = json.load(file)
    return pills_info


def get_pills_info2():
    with open(PILLS_JSON_FILE_CLEANED_2, "r", encoding="utf-8") as file:
        pills_info = json.load(file)
    return pills_info


def get_index_and_nodes_from_activeloop(vector_store_path: str):
    vector_store = load_vector_store(
        vector_store_path=vector_store_path, token=os.environ["ACTIVELOOP_TOKEN"]
    )
    chunks = []
    for el in vector_store.client:
        chunks.append(str(el.text.data()["value"]))

    loader = StringIterableReader()
    documents = loader.load_data(texts=chunks)
    node_parser = SentenceSplitter.from_defaults(separator="\n")
    nodes = node_parser.get_nodes_from_documents(documents)

    # To ensure same id's per run, we manually set them.
    for idx, node in enumerate(nodes):
        node.id_ = f"node_{idx}"

    llm = OpenAI(model="gpt-4")

    service_context = ServiceContext.from_defaults(llm=llm)
    index = VectorStoreIndex(nodes=nodes)
    return index, nodes, service_context


def get_index_and_nodes_after_visual_similarity(filenames: list):
    """
    Takes the filenames of the similar images and after having the id of the similar images return the index and nodes (based on description similarity)
    """
    vector_store = load_vector_store(
        vector_store_path=VECTOR_STORE_PATH_DESCRIPTION,
        token=os.environ["ACTIVELOOP_TOKEN"],
    )

    conditions = " or ".join(f"filename == '{name}'" for name in filenames)
    tql_query = f"select * where {conditions}"

    # filtered_elements = vector_store.vectorstore.search(query=tql_query)
    filtered_elements = vector_store._vectorstore.search(query=tql_query)
    chunks = []
    for el in filtered_elements["text"]:
        chunks.append(el)

    loader = StringIterableReader()
    documents = loader.load_data(texts=chunks)
    node_parser = SentenceSplitter.from_defaults(separator="\n")
    nodes = node_parser.get_nodes_from_documents(documents)

    # To ensure same id's per run, we manually set them.
    for idx, node in enumerate(nodes):
        node.id_ = f"node_{idx}"

    llm = OpenAI(model="gpt-4")

    service_context = ServiceContext.from_defaults(llm=llm)
    index = VectorStoreIndex(nodes=nodes)
    return index, nodes, service_context, filtered_elements


def keep_best_k_unique_nodes(reranked_nodes_bm25, reranked_nodes_vector, k=4):
    """
    Keeps the best k unique nodes from the two lists of nodes
    """
    all_nodes = []
    node_ids = set()
    for n in reranked_nodes_bm25 + reranked_nodes_vector:
        if n.node.node_id not in node_ids:
            all_nodes.append(n)
            node_ids.add(n.node.node_id)
    if len(all_nodes) > k:
        return all_nodes[:k]
    return all_nodes
