import requests

OLLAMA_HOST = 'http://localhost:11434'  # Replace with your Mac's IP

def chat_with_ollama(prompt):
    response = requests.post(
        f'{OLLAMA_HOST}/api/generate',
        json={
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False
        }
    )
    data = response.json()
    return data['response']

print(chat_with_ollama("What is the capital of France?"))
