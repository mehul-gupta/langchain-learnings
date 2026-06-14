import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
NAMESPACE = "knowledge-base"


def build_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-002")
    vector_store = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=embeddings,
        namespace=NAMESPACE,
    )
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def build_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def answer_question(question, retriever, llm):
    docs = retriever.invoke(question)

    if not docs:
        return "No relevant documents found.", []

    context = "\n\n".join(doc.page_content for doc in docs)

    response = llm.invoke(
        [
            SystemMessage(content=f"""
You are an assistant answering questions from a knowledge base.

Use only the provided context to answer.

If the answer is not present in the context, respond:
"I could not find that information in the knowledge base."

Do not make up facts.

Context:
{context}
"""),
            HumanMessage(content=question),
        ]
    )

    return response.content, docs


def print_sources(docs):
    if not docs:
        return

    print("\nSources:")
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        chunk = doc.metadata.get("chunk_number", "unknown")
        citation = f"{source} (chunk {chunk})"

        if citation not in seen:
            print(f"- {citation}")
            seen.add(citation)


def main():
    print("Knowledge Base Ready")
    print("Type 'exit' to quit")

    retriever = build_retriever()
    llm = build_llm()

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "exit":
            print("Goodbye")
            break
        if not question:
            continue

        print("Searching knowledge base...")
        answer, docs = answer_question(question, retriever, llm)
        print("\nAnswer:")
        print(answer)
        print_sources(docs)


if __name__ == "__main__":
    main()
