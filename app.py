from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are a Hospital AI Assistant.

Answer ONLY using the context below.
If the answer is not in the context, say: "I don't have that information."

<context>
{context}
</context>

Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)

# Department -> DB mapping
DB_MAP = {
    "Doctors": "01_doctors",
    "Appointments": "02_appointments_scheduling",
    "Billing": "03_billing_pricing",
    "Insurance": "04_insurance",
    "Reception": "05_reception_general_info",
    "Medical": "06_medical_knowledge_base",
    "Emergency": "07_emergency_handling",
    "Lab": "08_lab_diagnostics",
    "FAQ": "09_faq_dataset",
    "Policies": "10_policies_rules"
}


@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    # Keeps "Doctors" pre-selected on first load (matches the UI design),
    # and remembers the user's last pick + question after they submit.
    selected_department = "Doctors"
    question_text = ""

    if request.method == "POST":
        selected_department = request.form.get("department", "Doctors")
        question_text = request.form.get("question", "")

        try:
            # Correct mapping
            db_folder = DB_MAP.get(selected_department)

            if not db_folder:
                answer = "Invalid department selected."
                return render_template(
                    "index.html",
                    answer=answer,
                    selected_department=selected_department,
                    question_text=question_text
                )

            db_path = f"./db/{db_folder}"

            # Load vector DB
            vector_db = Chroma(
                persist_directory=db_path,
                embedding_function=embedding_model
            )

            # Retrieve documents
            docs = vector_db.similarity_search(question_text, k=3)

            print("DEBUG DOCS:", docs)

            if not docs:
                answer = "No relevant information found."
            else:
                result = document_chain.invoke({
                    "input": question_text,
                    "context": docs  # pass documents directly
                })

                if isinstance(result, dict):
                    answer = result.get("output_text", str(result))
                else:
                    answer = str(result)

        except Exception as e:
            answer = f"Error: {str(e)}"

    return render_template(
        "index.html",
        answer=answer,
        selected_department=selected_department,
        question_text=question_text
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)