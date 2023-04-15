import argparse
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import UnstructuredPDFLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script accepts arguments.")
    parser.add_argument("--pdf", type=str, required=True, help="Pass a pdf file to be indexed.")

    args = parser.parse_args()
    loader = UnstructuredPDFLoader(args.pdf)
    index = VectorstoreIndexCreator().from_loaders([loader])
    query = "What is this document about?"
    print(index.query(query))

    print("Hello! I can answer questions about the document you gave me.")
    try:

        while True:
            query = input("Query: ")
            print(index.query(query))
    except KeyboardInterrupt:
        print("Goodbye!")