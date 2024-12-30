# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import pipeline
# import torch
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# # from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
# from langchain_community.llms import HuggingFacePipeline
# from langchain.chains import RetrievalQA

# # Define the checkpoint
# checkpoint = "LaMini-T5-738M"

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# # Load the model directly onto the CPU or GPU
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Load the model
# base_model = AutoModelForSeq2SeqLM.from_pretrained(
#     checkpoint, 
#     torch_dtype=torch.float32
# )

# # Move the model to the desired device
# base_model.to(device)

# @st.cache_resource
# def llm_pipeline():
#     pipe = pipeline(
#         'text2text-generation',
#         model=base_model,
#         tokenizer=tokenizer,
#         max_length=256,
#         do_sample=True,
#         temperature=0.3,
#         top_p=0.95
#     )
#     local_llm = HuggingFacePipeline(pipeline=pipe)
#     return local_llm

# @st.cache_resource
# def qa_llm():
#     llm = llm_pipeline()
#     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#     db = Chroma(persist_directory="db", embedding_function=embeddings)
#     retriever = db.as_retriever()
#     qa = RetrievalQA.from_chain_type(
#         llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
#     return qa

# def process_answer(instruction):
#     qa = qa_llm()
#     generated_text = qa(instruction)
#     answer = generated_text['result']
#     return answer, generated_text

# def main():
#     st.title("Search Your PDF 🐦📄")
#     with st.expander("About the App"):
#         st.markdown(
#             """
#             This is a Generative AI powered Question and Answering app that responds to questions about your PDF File.
#             """
#         )
#     question = st.text_area("Enter your Question")
#     if st.button("Ask"):
#         st.info("Your Question: " + question)

#         st.info("Your Answer")
#         answer, metadata = process_answer(question)
#         st.write(answer)
#         st.write(metadata)

# if __name__ == '__main__':
#     main()



import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Define the checkpoint
checkpoint = "LaMini-T5-738M"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load the model directly onto the CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, 
    torch_dtype=torch.float32
)

# Move the model to the desired device
base_model.to(device)

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=512,
        do_sample=True,
        temperature=0.5,
        top_p=0.9
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def process_answer(instruction):
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text

def main():
    st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")
    
    st.title("🔍 PDF Q&A Assistant 📘")
    st.markdown(
        """
        Welcome to the PDF Q&A Assistant! This application leverages Generative AI to answer your questions based on the content of your PDF file.
        """
    )
    
    with st.expander("📖 About This App", expanded=True):
        st.markdown(
            """
            This tool allows you to interactively ask questions about your PDF documents. 
            Simply type your question below, and the AI will provide you with an answer based on the extracted information.
            """
        )
    
    st.subheader("💬 Ask Your Question:")
    question = st.text_area("Type your question here:", height=150)
    
    if st.button("Submit Question", key="submit_button"):
        if question.strip() == "":
            st.warning("Please enter a question before submitting.")
        else:
            st.info("You asked: " + question)
            with st.spinner("Generating your answer..."):
                answer, metadata = process_answer(question)
                st.success("Here's your answer:")
                st.write(answer)
                st.markdown("### Additional Context:")
                st.write(metadata)

    st.sidebar.header("📝 Instructions")
    st.sidebar.markdown(
        """
        1. Enter your question in the text area.
        2. Click the 'Submit Question' button to receive an answer.
        3. Review the answer and any additional context provided.
        """
    )

if __name__ == '__main__':
    main()