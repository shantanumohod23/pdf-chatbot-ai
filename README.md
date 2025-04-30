# **🤖 PDF Chatbot AI — A Generative AI Application**

A **Generative AI-powered chatbot** built with **Streamlit** that allows users to upload PDF documents and get **detailed, human-like answers** to their questions using **state-of-the-art language models**.

---

## **🚀 Features**

✅ Upload and analyze PDF files 📄\
✅ Extract and segment text using **PyPDF2** + **NLTK** ✂️\
✅ Convert text into semantic embeddings using **all-mpnet-base-v2** 🔍\
✅ Retrieve the most relevant context based on user questions 🧠\
✅ Generate **context-aware answers** using **FLAN-T5-Large** (Generative AI) ✨\
✅ Clean and interactive UI with **Streamlit** 💬

---

## **🛠️ Installation**

1⃣ **Clone the repository**
```bash
git clone https://github.com/shantanumohod23/pdf-chatbot-ai.git
cd pdf-chatbot-ai
```

2⃣ **Create and activate a virtual environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## **▶️ How to Run**

Once installed, start the Streamlit app:
```bash
streamlit run app.py
```

Upload a PDF file, ask your question, and let the AI generate an insightful answer! 🤖📚

---

## **📁 Folder Structure**
```
📂 pdf-chatbot-ai
 ├── app.py            # Main Streamlit app
 ├── requirements.txt  # Required Python libraries
 ├── config.toml       # Streamlit config
 ├── README.md         # Project documentation
 ├── LICENSE           # License file
```

---

## **🧠 Technologies Used**
- **Python** 🐍
- **Streamlit** – UI for interaction
- **PyPDF2** – PDF text extraction
- **NLTK** – Sentence segmentation
- **Sentence Transformers** – `all-mpnet-base-v2` for semantic similarity
- **Hugging Face Transformers** – `google/flan-t5-large` for generative responses

---

## **🌱 Future Improvements**
✅ Add support for scanned PDFs via OCR\
✅ Expand to handle multiple documents\
✅ Let users select different AI models\
✅ Deploy online via Hugging Face Spaces or Streamlit Cloud

---

## **🙋‍♂️ About the Creator**
📌 **Shantanu Mohod** – Passionate about AI, Data Science, and Full-Stack Development  
🔗 GitHub: [shantanumohod23](https://github.com/shantanumohod23)  
🔗 LinkedIn: [Shantanu Mohod](https://www.linkedin.com/in/shantanumohod)

---
