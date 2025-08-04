from modules.vector_store import get_embedder

embedder = get_embedder()
print(embedder.embed_query("This is a test"))
