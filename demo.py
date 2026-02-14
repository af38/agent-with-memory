from mem0 import Memory
from groq import Groq

from dotenv import load_dotenv
import os

load_dotenv()

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
    "embedder": {                                 # ADDED: Embedding configuration
        "provider": "huggingface",                # Local embeddings via Hugging Face
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",  # Lightweight & fast
            "embedding_dims": 384
        }
    }
}

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize Mem0 with Groq
memory = Memory.from_config(config)



def chat_with_memories(message: str, user_id: str = "default_user") -> str:

    relevant_memories = memory.search(query=message, user_id=user_id, limit=3)
    memories_str = "\n".join(
        f"- {entry['memory']}" for entry in relevant_memories["results"]
    )

    print(memories_str)

    # Generate Assistant response
    system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    response = groq_client.chat.completions.create(
        model="groq/compound",
        messages=messages
    )

    assistant_response = response.choices[0].message.content
    # Create new memories from the conversation
    messages.append({"role": "assistant", "content": assistant_response})
    # This is where the magic happens
    memory.add(messages, user_id=user_id, metadata={"source": "demo"})

    return assistant_response

def main():
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        print(f"AI: {chat_with_memories(user_input)}")


if __name__ == "__main__":
    main()
