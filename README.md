# 🏥 Hospital AI Assistant (RAG)

An AI-powered Hospital Assistant built using Retrieval-Augmented Generation (RAG) to answer hospital-related queries using a structured medical knowledge base.

---

## ⚙️ Tech Stack

- **Embeddings Model:** HuggingFace `all-MiniLM-L6-v2`
- **LLM:** Groq `llama-3.1-8b-instant`
- **Vector Database:** ChromaDB
- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS, JavaScript
- **Frameworks/Libraries:** LangChain, HuggingFace, Groq API

---

## 📚 Knowledge Base / Dataset

The system uses a structured hospital dataset divided into multiple domains for better retrieval in RAG:

- **01_doctors** → Doctor profiles, specialization, availability  
- **02_appointments_scheduling** → Booking rules, scheduling system  
- **03_billing_pricing** → Consultation fees, treatment costs, billing details  
- **04_insurance** → Insurance policies, claim process, coverage  
- **05_reception_general_info** → Hospital timings, front desk info, general details  
- **06_medical_knowledge_base** → Diseases, symptoms, treatments  
- **07_emergency_handling** → Emergency instructions, first aid guidance  
- **08_lab_diagnostics** → Lab tests, reports, diagnostic procedures  
- **09_faq_dataset** → Frequently asked questions  
- **10_policies_rules** → Hospital rules, patient guidelines, policies  

---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/chaithanyagollapalli08-lab/Hospital_AI_Assistant_RAG.git
cd Hospital_AI_Assistant_RAG
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

Activate:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

### 5. Run Application
```bash
python app.py
```

Open in browser:
```
http://127.0.0.1:5000
```

---

## 🧠 How it Works

- User enters a query in the chat interface  
- Query is converted into embeddings using `all-MiniLM-L6-v2`  
- Vector database retrieves relevant chunks from hospital datasets  
- Retrieved context is passed to Groq LLM (`llama-3.1-8b-instant`)  
- LLM generates a context-aware response  
- Response is displayed in the UI  

---

## 🚀 Future Improvements

- Voice-based assistant for patients   
- Multilingual support 
- Appointment booking automation   
- Patient history tracking system  
- Streaming chat responses (real-time UI)   
- Upload medical reports (PDF/image analysis)  
- Improved hybrid search (BM25 + vector search)   

---

## 👨‍💻 Author

**Chaithanya Gollapalli**  
 Email:chaithanyagollapalli08@gmail.com
