import argparse
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def main():
    parser = argparse.ArgumentParser(description="Index a pdf file to ask questions against.")
    parser.add_argument("--pdf", type=str, required=True, help="Pass a pdf file to be indexed.")
    args = parser.parse_args()

    documents = UnstructuredPDFLoader(args.pdf).load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), return_source_documents=True)

    chat_history = []
    query = "What is this document about?"
    print(qa({"question": query, "chat_history": chat_history})["answer"])
    
    try:
        while True:
            query = input("Query: ")
            print(qa({"question": query, "chat_history": chat_history})["answer"])
    except KeyboardInterrupt:
        print("Goodbye!")


if __name__ == "__main__":
    main()
