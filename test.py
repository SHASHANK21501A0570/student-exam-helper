import ollama
response=ollama.chat(model="mistral",messages=[{"role":"user","content":"What is the capital of France?"}])
print(response)