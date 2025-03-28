import streamlit as st
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')  # Use 'punkt' for sentence tokenization
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

st.title("Improved PDF Chatbot with Detailed Answers")

# Function to chunk text using sentence segmentation with overlap
def chunk_text(text, chunk_size=5, overlap=2):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    i = 0
    while i < len(sentences):
        # Create a chunk from a group of sentences
        chunk = " ".join(sentences[i: i + chunk_size])
        chunks.append(chunk)
        # Move index forward by chunk_size - overlap
        i += (chunk_size - overlap)
    return chunks

# File uploader widget
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf_file:
    # Extract text from PDF
    reader = PdfReader(pdf_file)
    pdf_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pdf_text += page_text + "\n"
    
    if not pdf_text.strip():
        st.error("No text could be extracted from this PDF.")
    else:
        # Optionally, display the extracted text
        if st.checkbox("Show extracted text"):
            st.write(pdf_text)
        
        # Advanced chunking: split text into overlapping chunks
        chunks = chunk_text(pdf_text, chunk_size=5, overlap=2)
        st.write(f"Extracted {len(chunks)} text chunks.")
        
        if len(chunks) == 0:
            st.error("No chunks available after splitting the text.")
        else:
            # Load a stronger embedding model for better semantic capture
            st.info("Loading embedding model...")
            embedding_model = SentenceTransformer('all-mpnet-base-v2')
            chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
            
            # Load a generative model for more detailed answers
            st.info("Loading generative model for detailed answers...")
            gen_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")
            
            # Input for user question
            question = st.text_input("Ask a question about the PDF:")
            
            if st.button("Get Detailed Answer") and question:
                # Embed the user question
                question_embedding = embedding_model.encode(question, convert_to_tensor=True)
                
                # Compute cosine similarity between the question and each chunk
                cos_scores = util.cos_sim(question_embedding, chunk_embeddings)
                
                # Determine the number of top chunks to retrieve (up to 5)
                top_k = min(5, cos_scores.shape[1])
                top_results = torch.topk(cos_scores, k=top_k, dim=1)
                top_indices = top_results.indices[0].tolist()
                
                # Combine the top retrieved chunks into one context string
                combined_context = " ".join([chunks[i] for i in top_indices])
                
                # Construct the prompt for the generative model
                prompt = (
                    f"Answer the following question in detail based on the context below:\n\n"
                    f"Context: {combined_context}\n\n"
                    f"Question: {question}\n\n"
                    f"Detailed Answer:"
                )
                
                detailed_result = gen_pipeline(prompt, max_length=300)
                
                st.write("### Detailed Answer:")
                st.write(detailed_result[0]['generated_text'])
                
                # Optional: show the combined context for debugging
                if st.checkbox("Show combined context"):
                    st.write(combined_context)
