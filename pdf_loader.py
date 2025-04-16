import fitz 
def extract_from_pdf(file):
    # 'file' can be a file path or a file-like object from Streamlit
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text