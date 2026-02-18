
import pypdf
import sys

try:
    reader = pypdf.PdfReader("c:/Users/Sanju/OneDrive/Desktop/project/w1999522_Sanjula_PPRS.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print(text)
except Exception as e:
    print(f"Error reading PDF: {e}")
