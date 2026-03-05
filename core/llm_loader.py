import requests
from transformers import data


class LLM:
    """
    Handles interaction with local Ollama LLM.
    """

    def __init__(self, model_name="mistral"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str):
        response = requests.post(
            self.url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]
    

    def rewrite_query(self, query: str, history_text: str):
        rewrite_prompt = f"""
    You are a query rewriting assistant.

    Rewrite the user's question into a clear, standalone question.
    Use the conversation history if needed.

    Conversation History:
    {history_text}

    User Question:
    {query}

    Rewritten Question:
    """

        response = requests.post(
            self.url,
            json={
                "model": self.model_name,
                "prompt": rewrite_prompt,
                "stream": False
            }
        )

        data = response.json()
        print("DEBUG:", data)
        return data.get("response", "").strip()