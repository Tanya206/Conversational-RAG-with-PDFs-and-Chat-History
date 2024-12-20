# RAG Q&A Conversational Chatbot with PDF and Chat History

## Overview
This project implements a Retrieval-Augmented Generation (RAG) conversational chatbot capable of answering questions based on uploaded PDF documents. It supports maintaining chat history and enhances context-awareness for improved interactions.

The system uses:
- **LangChain** for chaining and managing prompts.
- **Chroma** for storing and retrieving document embeddings.
- **Streamlit** for building a user-friendly web interface.
- **Groq API** as the primary LLM backend.

## Features
- Upload multiple PDF documents to extract content.
- Query the chatbot for answers based on the PDF content.
- Maintain session-based chat history for enhanced interaction.
- Utilize embeddings for semantic search.

## Prerequisites
1. Python 3.8 or later.
2. A valid Groq API key.
3. Access to HuggingFace embeddings.
4. Installed dependencies (see below).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/rag-pdf-chatbot.git
   cd rag-pdf-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add the following lines:
     ```env
     HF_TOKEN=your_huggingface_token
     GROQ_API_KEY=your_groq_api_key
     ```

## Usage
1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open the Streamlit app in your browser (typically at `http://localhost:8501`).

3. Enter your Groq API key when prompted.

4. Upload one or more PDF files.

5. Input questions in the text box to query the chatbot based on the uploaded content.

6. View responses along with chat history.

## File Structure
```
root
├── app.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
├── chroma_data/         # Directory for storing embeddings
└── temp.pdf             # Temporary storage for uploaded PDFs
```

## Key Components
### 1. PDF Loading
- Uses **PyPDFLoader** to extract content from uploaded PDF files.

### 2. Document Splitting and Embeddings
- Splits documents into smaller chunks using **RecursiveCharacterTextSplitter**.
- Generates embeddings using **HuggingFaceEmbeddings** with the `all-MiniLM-L6-v2` model.

### 3. Chroma Vectorstore
- Stores embeddings in a persistent directory (`./chroma_data`).
- Retrieves relevant document chunks for answering queries.

### 4. LLM Integration
- Queries are contextualized and answered using the Groq API with the `gemma2-9b-it` model.

### 5. Conversational Chat
- Maintains session-specific chat history using **ChatMessageHistory**.

## Error Handling
- Ensure the API keys are valid and correctly configured in `.env`.
- Confirm that dependencies are installed and match the required versions in `requirements.txt`.
- For debugging, check the Streamlit logs in the terminal.

## Dependencies
- `streamlit`
- `langchain`
- `langchain_chroma`
- `langchain_huggingface`
- `langchain_groq`
- `chromadb`
- `PyPDFLoader`
- `dotenv`

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Future Enhancements
- Add support for more document types (e.g., Word, Excel).
- Improve UI/UX with better session management.
- Expand LLM support with additional APIs.
- Implement more advanced error handling and logging.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- [LangChain Documentation](https://langchain.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Groq API Documentation](https://groq.com/docs)

