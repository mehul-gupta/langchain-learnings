import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_DIR = Path(os.getenv("DOCS_DIR", "documents"))
NAMESPACE = "knowledge-base"


def build_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-002")
    return PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=embeddings,
        namespace=NAMESPACE,
    )


def load_and_chunk_document(file_path: Path):
    relative_path = str(file_path.relative_to(DOCS_DIR))

    loader = Docx2txtLoader(str(file_path))
    docs = loader.load()

    for doc in docs:
        doc.metadata.update({"source": relative_path, "filename": file_path.name})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=250, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_number"] = i

    return chunks


def main():
    print(f"Starting ingestion into namespace '{NAMESPACE}'")

    files = sorted(DOCS_DIR.rglob("*.docx"))

    if not files:
        print(f"No DOCX files found under '{DOCS_DIR}'")
        return

    print(f"Found {len(files)} DOCX files")
    vector_store = build_vector_store()
    total_chunks = 0

    for file_path in files:
        chunks = load_and_chunk_document(file_path)
        total_chunks += len(chunks)

        relative_path = str(file_path.relative_to(DOCS_DIR))

        print(f"\nProcessing: {relative_path}")
        print("  Deleting existing chunks...")
        vector_store.delete(filter={"source": {"$eq": relative_path}})

        print(f"  Uploading {len(chunks)} chunks...")
        vector_store.add_documents(chunks)
        print("  Upload complete")

    print("\nIngestion completed")
    print(f"Files processed : {len(files)}")
    print(f"Chunks uploaded : {total_chunks}")


if __name__ == "__main__":
    main()
