import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

DATA_PATH = "./data"
DB_PATH = "./db"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

for file in os.listdir(DATA_PATH):

    if file.endswith(".txt"):

        print(f"\n Processing: {file}")

        file_path = os.path.join(DATA_PATH, file)

        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        chunks = splitter.split_documents(docs)
        print(f"Chunks: {len(chunks)}")

        # remove .txt → match DB names
        folder_name = file.replace(".txt", "")

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=os.path.join(DB_PATH, folder_name)
        )

        db.persist()

        print(f"DB created at: ./db/{folder_name}")

print("\nlALL DATABASES CREATED SUCCESSFULLY")