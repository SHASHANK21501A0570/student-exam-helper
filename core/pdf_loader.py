import os
from pypdf import PdfReader
def load_pdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        reader=PdfReader(file_path)
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {e}")
    
    extracted_text=[]
    for page_number, pages in enumerate(reader.pages):
        try:
            text=pages.extract_text()
            if text:
                extracted_text.append(text)
        except Exception as e:
            print(f"Error extracting text from page {page_number}: {e}")

    full_text="\n".join(extracted_text)
    full_text=full_text.strip()
    return full_text
