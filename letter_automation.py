import os
### THIS CAN BE A SECURITY RISK, UNCOMMENT IF YOU DO NOT WANT TO MANUALLY insert NAME, EMAIL, PHONE NUMBER
# name = input("Enter your name: ")
name = "name"

# email = input("Enter your email address: ")
email = "email"

# phone = input("Enter your phone number with (XXX)-XXX-XXXX format: ")
phone = "phone"

company_name = input("Enter the company name you are applying for: ")
url =  input("Enter the url for job description:  ")
os.environ["USER_AGENT"] = f"coverletter_code/1.0 (Contact:[lukasa])"

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document as LangChainDoc
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyMuPDFLoader
from docx import Document
from docx.shared import Pt




# Use your own ChatGPT API Key
with open("/Users/lukasamare/Desktop/random_project/openaiapikey", "r") as file:
    apikey = file.read().strip()

def load_resume(resume_filepath):
    loader = PyMuPDFLoader(resume_filepath)
    resume  = loader.load()
    return resume
# Put your resume file path make sure it is a pdf
resume = load_resume("/Users/lukasamare/Desktop/JOB:internship/updated_resumes.pdf")[0].page_content




def get_documents_from_web(url):
    loader =  WebBaseLoader(url)
    docs = loader.load()
    if docs and docs[0].page_content.strip():
        splitter =  RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 50)
        split_docs =  splitter.split_documents(docs)
        return split_docs
    else:
        description = docs[0].metadata.get("description","")
        if not description.strip():
            raise ValueError("No valid content found in page_content or description.")
        return [LangChainDoc(page_content=description, metadata={"source": url})]

docs  = get_documents_from_web(url)


def create_db(docs):
    embedding =  OpenAIEmbeddings(api_key=apikey)
    vector_store = FAISS.from_documents(docs,embedding=embedding)
    return vector_store

vectorStore = create_db(docs)
    
def create_chain(vectorStore, name, email, phone):
    llm1 = ChatOpenAI(api_key=apikey,
    temperature = 0,
    model = "gpt-3.5-turbo",
    )
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system",f'You are a data science candidate in the middle of his/her Masters program named {name} with some level of experience. Using the resume make a cover letter based off of the context given. Make sure to include some statistics from my resume to impress. This is formal but I do not need [Company Address],[City, State, Zip], or [Date] to start the letter. Use {email} and {phone} at the end.'),
        ("human", '{input}'),
        ("user", '{context}')

    ])
    
    chain = create_stuff_documents_chain(llm=llm1, prompt=prompt)

    retriever = vectorStore.as_retriever(search_kwargs = {'k':10})
    retrieval_chain = create_retrieval_chain(retriever, chain)
    return retrieval_chain
    

chain = create_chain(vectorStore, name, email, phone)
response =  chain.invoke({"input" :resume })

# Define the filename to save the response
file_name_txt = company_name+"_coverletter.txt"
file_name_docx = company_name+"_coverletter.docx"
answer = response.get('answer', '')
print(answer)

# Save the response to a text file
with open(file_name_txt, "w") as file:
    file.write(answer)

def save_to_docx(content, filename):
    doc = Document()
    paragraph = doc.add_paragraph(content)
    run = paragraph.runs[0]
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)

    doc.save(filename)
save_to_docx(answer, file_name_docx)


print(f"Response saved to {file_name_txt}")
print(f"Response saved to {file_name_docx}")