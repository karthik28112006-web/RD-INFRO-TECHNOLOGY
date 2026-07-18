from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Custom business knowledge data for your chatbot
KNOWLEDGE_BASE = [
    "Hello! Welcome to our automated customer support channel.",
    "Our refund policy allows complete returns within 30 days of item purchase.",
    "Standard shipping takes 3 to 5 business days across the country.",
    "You can track your live orders using your customer portal dashboard."
]

def load_local_retriever():
    docs = [Document(page_content=text) for text in KNOWLEDGE_BASE]
    # Creates a local fallback vector store
    db = Chroma.from_documents(docs, embedding=None)
    return db.as_retriever(search_kwargs={"k": 1})