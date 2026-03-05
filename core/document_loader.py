import os
from core.pdf_loader import load_pdf

def load_documents(folder_path="data"):
    documents = []


    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)

            print(f"Loading {file}...")

            text = load_pdf(path)

            documents.append({
                "text": text,
                "source": file
            })

    return documents

