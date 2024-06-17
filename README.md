Introducing a Chatbot for Resume Screening

This innovative solution is more efficient, user-friendly, and effective than traditional keyword-based methods.

How it Works

The chatbot uses a hybrid retrieval method to retrieve relevant resumes based on job descriptions. It employs two techniques:

Adaptive Retrieval: The chatbot uses RAG/RAG Fusion to search for similar resumes and narrow down the pool of applicants to the most relevant profiles.
Keyword-based Retrieval: When provided with applicant IDs, the chatbot retrieves additional information about specified candidates.
Why Resume Screening Matters

Despite the increasing number of applicants, there are limited tools that can effectively and reliably assist the screening process. Existing methods often rely on keyword-based approaches, which struggle to accurately handle the complexity of natural language in human-written documents. This project aims to address this gap by integrating LLM-based methods into the recruitment domain.

Why RAG/RAG Fusion Matters

RAG-like frameworks, such as RAG Fusion, enhance the reliability of chatbots by providing an external knowledge base for LLM agents. This increases the relevance and accuracy of generated answers, which is crucial in data-intensive environments like recruitment.

Demo and Setup

You can access the demo chatbot interface here: Streamlit. The demo uses a synthetic resume dataset and a Kaggle job description dataset.

System Description

The chatbot structure is designed to be suitable for real-world use cases:

Chat History Access: The LLM is fed with the entire conversation and the latest retrieved documents for every message, allowing it to perform follow-up tasks.
Query Classification: The LLM extracts necessary information to decide whether to toggle the retrieval process on/off.
Small-to-Big Retrieval: The retrieval process is performed using text chunks for efficiency.
Tech Stacks

The project uses:

Langchain, OpenAI, and HuggingFace for RAG pipeline and chatbot construction.
Faiss for vector indexing and similarity retrieval.
Streamlit for user interface development.
Installation and Setup

To set up the project locally:

Clone the project: git clone https://github.com/katariyaVivek/RS-RAG.git
Install dependencies: pip install requirements.txt
Run the Streamlit demo: streamlit run demo/interface.py
Contributions

The demo chatbot is still a work in progress, and any suggestions, feedback, or contributions are highly appreciated! Please share them at the project's issue tracker.

Acknowledgement

Inspired by RAG Fusion.

remove POC part
Here is the rewritten text without the "Proof of Concept" part:

Introducing a Chatbot for Resume Screening

As part of my thesis, I developed a chatbot that assists hiring managers in the resume screening process. This innovative solution is more efficient, user-friendly, and effective than traditional keyword-based methods.

How it Works

The chatbot uses a hybrid retrieval method to retrieve relevant resumes based on job descriptions. It employs two techniques:

Adaptive Retrieval: The chatbot uses RAG/RAG Fusion to search for similar resumes and narrow down the pool of applicants to the most relevant profiles.
Keyword-based Retrieval: When provided with applicant IDs, the chatbot retrieves additional information about specified candidates.
Why Resume Screening Matters

Despite the increasing number of applicants, there are limited tools that can effectively and reliably assist the screening process. Existing methods often rely on keyword-based approaches, which struggle to accurately handle the complexity of natural language in human-written documents. This project aims to address this gap by integrating LLM-based methods into the recruitment domain.

Why RAG/RAG Fusion Matters

RAG-like frameworks, such as RAG Fusion, enhance the reliability of chatbots by providing an external knowledge base for LLM agents. This increases the relevance and accuracy of generated answers, which is crucial in data-intensive environments like recruitment.

Demo and Setup

You can access the demo chatbot interface here: Streamlit. The demo uses a synthetic resume dataset and a Kaggle job description dataset.

System Description

The chatbot structure is designed to be suitable for real-world use cases:

Chat History Access: The LLM is fed with the entire conversation and the latest retrieved documents for every message, allowing it to perform follow-up tasks.
Query Classification: The LLM extracts necessary information to decide whether to toggle the retrieval process on/off.
Small-to-Big Retrieval: The retrieval process is performed using text chunks for efficiency.
Tech Stacks

The project uses:

Langchain, OpenAI, and HuggingFace for RAG pipeline and chatbot construction.
Faiss for vector indexing and similarity retrieval.
Streamlit for user interface development.
Installation and Setup

Contributions

The demo chatbot is still a work in progress, and any suggestions, feedback, or contributions are highly appreciated! Please share them at the project's issue tracker.

Acknowledgement

Inspired by RAG Fusion.
