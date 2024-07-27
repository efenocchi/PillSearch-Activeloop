import nest_asyncio

nest_asyncio.apply()


from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.llama_pack import download_llama_pack


# load in some sample data
reader = SimpleDirectoryReader(input_files=["paul_graham_essay.txt"])
documents = reader.load_data()

# parse nodes
node_parser = SimpleNodeParser.from_defaults(separator="\n")
nodes = node_parser.get_nodes_from_documents(documents)


HybridFusionRetrieverPack = download_llama_pack(
    "HybridFusionRetrieverPack",
    "./hybrid_fusion_pack",
    # leave the below commented out (was for testing purposes)
    # llama_hub_url="https://raw.githubusercontent.com/run-llama/llama-hub/jerry/add_llama_packs/llama_hub",
)

hybrid_fusion_pack = HybridFusionRetrieverPack(
    nodes, chunk_size=256, vector_similarity_top_k=2, bm25_similarity_top_k=2
)

# this will run the full pack
response = hybrid_fusion_pack.run("What did the author do during his time in YC?")

print(str(response))

modules = hybrid_fusion_pack.get_modules()
