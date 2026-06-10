import os
import fitz
from langchain_core.documents import Document

def parse_pdf_with_links(file_path: str) -> list[Document]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: PDF not found at '{file_path}'")

    doc = fitz.open(file_path)
    documents = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        links = page.get_links()
        link_urls = [link.get("uri") for link in links if "uri" in link]

        metadata = {
            "source": os.path.basename(file_path),
            "page": page_num + 1,
            "hyperlinks": link_urls
        }

        documents.append(Document(page_content=text, metadata=metadata))

    doc.close()
    return documents
