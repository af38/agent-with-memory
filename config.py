config = {
    "vector_store" : {
        "provider": "qdrant",
        "config" : {
            "collection_name": "qdrant_v2",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 384
            },
    },
    "llm": {
        "provider": "groq",
        "config":{
            "model": "groq/compound",
            "api_key": os.getenv("GROQ_API_KEY"),
            "temperature": 0.1
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dims": 384
        }
    }
}