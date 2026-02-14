from mem0 import Memory
from groq import Groq

from dotenv import load_dotenv
import os
import config

load_dotenv()


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

    memory.add(messages, user_id=user_id)

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
