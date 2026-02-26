from core.pdf_loader import load_pdf

pdf_path = "data/data1.pdf"  # Replace with your actual file name

text = load_pdf(pdf_path)

print("\n--- PDF Loaded Successfully ---\n")
print(f"Total Characters Extracted: {len(text)}\n")
print("Preview:\n")
print(text[:500])