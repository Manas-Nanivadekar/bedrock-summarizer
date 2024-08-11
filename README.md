# Chat with PDF using AWS Bedrock

This repository contains a Streamlit application that allows users to interact with PDF documents by asking questions. The application uses AWS Bedrock for language model services and LangChain for document processing and question-answering. The core functionality revolves around ingesting PDF documents, vectorizing them, and using a language model to generate answers to user queries.

## Features

- **PDF Ingestion**: Load PDF documents from a directory and split them into manageable chunks for processing.
- **Vectorization**: Use AWS Bedrock embeddings to vectorize the document chunks and store them in a FAISS index for efficient retrieval.
- **Question Answering**: Leverage a language model from AWS Bedrock to generate detailed answers to user questions based on the most relevant document chunks.
- **Streamlit Interface**: Simple web interface for interacting with the system, including features to update or create vector stores and ask questions.

## Installation

To get started with this project, follow the instructions below.

### Prerequisites

- Python 3.8+
- AWS credentials configured for access to Bedrock services
- Necessary Python packages

### Clone the Repository

```bash
git clone https://github.com/Manas-Nanivadekar/bedrock-summarizer.git
cd chat-with-pdf-bedrock
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Directory Structure

Ensure that your project directory looks something like this:

```
chat-with-pdf-bedrock/
│
├── data/                  # Directory containing PDF files
├── faiss_index/           # Directory where FAISS index is stored
├── app.py                 # Main application file
├── README.md              # This README file
└── requirements.txt       # Python dependencies
```

### Configuration

Make sure you have your AWS credentials set up correctly in your environment. The code uses the AWS SDK (`boto3`) to interact with Bedrock services.

### Running the Application

To run the Streamlit application, simply execute:

```bash
streamlit run app.py
```

This will start the application, and you can interact with it via your web browser.

### Updating or Creating Vector Store

- Navigate to the sidebar in the Streamlit app.
- Click the "Vectors Update" button to process the PDF files in the `data` directory and create/update the FAISS index.

### Asking Questions

- Enter your question in the text input box on the main page.
- Click the "Claude Output" button to get the response generated by the AWS Bedrock language model.
