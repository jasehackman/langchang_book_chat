import os
import argparse
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import AgentType
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI


def main():
    parser = argparse.ArgumentParser(description="Index a pdf file to ask questions against.")
    parser.add_argument("--path", type=str, required=False, help="Pass a directory where the pdfs are stored.", default="docs/")
    args = parser.parse_args()

    files = get_files(args.path)
    
    embeddings = OpenAIEmbeddings()
    llm = OpenAI(temperature=0)
    chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)
    vectorstores=[]
    retrievers=[]
    tools = []
    for file in files:
        # Load the pdf file and split it into chunks
        documents = UnstructuredPDFLoader(file).load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        documents = text_splitter.split_documents(documents)

        # strip the extension from the file name
        name_without_ext = os.path.splitext(os.path.basename(file))[0].replace(" ", "_").lower()
        
        # Create a vectorstore and a retriever
        print(f"Indexing {name_without_ext}...")
        vectorstore = Chroma.from_documents(documents, embeddings, collection_name=name_without_ext)
        vectorstores.append(vectorstore)
        retriever = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
        retrievers.append(retriever)

        # Ask GPT to provide a description of the document
        chat_history = []
        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
        query = "What is this document about?"
        description = qa({"question": query, "chat_history": chat_history})["answer"]

        # Create a tool object and add it to the list of tools
        tool = Tool(name=file, func=retriever, description=description)
        tools.append(tool)

    # Initialize the agent
    agent = initialize_agent(tools, chat_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    chat_history = []

    try:
        while True:
            query = input("Query: ")
            print(agent.run({"input": query, "chat_history": chat_history}))
    except KeyboardInterrupt:
        print("Goodbye!")



def get_files(directory_path):
    return [os.path.join(directory_path, file) for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]


if __name__ == "__main__":
    main()
