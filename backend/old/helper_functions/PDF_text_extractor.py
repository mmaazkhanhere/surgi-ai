from PyPDF2 import PdfReader
from io import BytesIO

def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file using PyPDF2.
    """
    pdf_reader = PdfReader(BytesIO(file.read()))
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() or ""
    return text