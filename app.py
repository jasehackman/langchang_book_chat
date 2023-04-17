import argparse
import os

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def main():
    parser = argparse.ArgumentParser(description="Index a pdf file to ask questions against.")
    parser.add_argument("--path", type=str, required=False, help="Pass a directory where the pdfs are stored.", default="docs/")
    args = parser.parse_args()

    files = get_files(args.path)
    
    embeddings_api = OpenAIEmbeddings()
    llm = OpenAI(temperature=1)
    
    tools = []
    for file in files:
        document_in_chunks, file_name = load_document(file)
        
        # Create an index in memory on the document and a retriever
        vectorstore = index_document(document_in_chunks, file_name, embeddings_api)

        retriever = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

        # Ask GPT to provide a description of the document
        description = get_document_description(llm, vectorstore)
        # Create a tool object and add it to the list of tools
        tool = Tool(name=file_name, func=retriever, description=description)
        tools.append(tool)

    # Initialize the agent
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    chat_history = []

    try:
        while True:
            query = input("Query: ")
            print(agent.run({"input": query, "chat_history": chat_history}))
    except KeyboardInterrupt:
        print("Goodbye!")



def get_files(directory_path):
    return [os.path.join(directory_path, file) for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]

def load_document(file_name):
    documents = UnstructuredPDFLoader(file_name).load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    document_in_chunks = text_splitter.split_documents(documents)

    # strip the extension from the file name
    name_without_ext = os.path.splitext(os.path.basename(file_name))[0].replace(" ", "_").lower()

    return document_in_chunks, name_without_ext

def index_document(documents, file_name, embeddings_api):
    print(f"Indexing {file_name}...")
    return Chroma.from_documents(documents, embeddings_api, collection_name=file_name)

def get_document_description(llm, vectorstore):
    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
    query = "What is this document about? Summarize it for me, and provide bullets of key points."
    return qa({"question": query, "chat_history": []})["answer"]


if __name__ == "__main__":
    main()
