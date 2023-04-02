from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import tiktoken
import os
import streamlit as st

company_information = """* Optimize your Electric Vehicle Operations: Reduce energy costs, avoid late departures, and ensure reliable charging operations
* Optimize Energy Usage: Avoid high operation costs
"""

example_email = ""

map_prompt = """Below is a section of a website about {prospect}
What is some interesting information and news about electric fleets about {prospect}. If the information is not about {prospect}, exclude it from your summary.
{text}
INTRIGUING INFORMATION:"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "prospect"])

combine_prompt = """
Your goal is to write a personalized outbound email from {sales_rep}, a sales rep at {company} to {prospect}.

A good email is personalized and combines information about the two companies on how they can help each other. Be sure to use value selling: A sales methodology that focuses on how your product or service will provider value to the customer. 

INFORMATION ABOUT {company}:
{company_information}

INFORMATION ABOUT {prospect}:
{text}

INCLUDE THE FOLLOWING PIECE IN YOUR RESPONSE:
- Start the email with the sentence: "We love that {prospect}..." then insert something specific about the electric fleet information provided.
- The sentence: "We can help you do XYZ by ABC" Replace XYZ with what {prospect} does and ABC with what {company} does
- End your email with a call-to-action such as asking them to set up time to talk more

EXAMPLE EMAIL:
{example_email}

YOUR RESPONSE:

"""

combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["sales_rep", "company", "prospect", "text", "company_information", "example_email"])

def get_company_page(link1, link2, link3):
    loader = UnstructuredURLLoader(urls=[link1, link2, link3])
    return loader.load()

def split_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 0
    )
    return text_splitter.split_documents(data)


def chain_prompts(docs, prospect):
    llm = OpenAI(temperature=.7, openai_api_key=os.environ['OPENAI_API_KEY'])
    chain = load_summarize_chain(llm,
                             chain_type="map_reduce",
                             map_prompt=map_prompt_template,
                             combine_prompt= combine_prompt_template,
                             verbose=False
                            )
    return chain({"input_documents": docs,
                "company" : "AmpControl",
                "company_information": company_information,
                "sales_rep": "Jo",
                "prospect" : prospect,
                "example_email" : example_email,
               })

st.set_page_config(page_title="AmpControl Sales Mail Generatorl", page_icon=":robot:")
st.header("AmpControl Sales Mail Generator")

company_name = st.text_input(label="", placeholder="Prospect Name")

col1, col2, col3 = st.columns(3)

with col1:
   company_url_1 = st.text_input(label= "", placeholder="Prospect URL 1")

with col2:
    company_url_2 = st.text_input(label="", placeholder="Prospect URL 2")

with col3:
    company_url_3 = st.text_input(label="", placeholder="Prospect URL 3")

with st.expander("Advanced Settings"):
    company_information = st.text_area(value=company_information, label="AmpControl Value Proposition", placeholder="Enter the value proposition of AmpControl")
    example_email = st.text_area(value="", label="Example Email", placeholder="Enter an example email")

if st.button('Generate email', type = 'primary'):
    if company_url_1 == "" or company_url_2 == "" or company_url_3 == "" or company_name == "":
        st.session_state.company_url_1_label = "🚨"
        st.error('Please fill out the required fields', icon="🚨")
    else:
        with st.spinner('Loading...'):
            data = get_company_page(link1=company_url_1, link2=company_url_2, link3=company_url_3)
            docs = split_documents(data)
            output = chain_prompts(docs, company_name)
            mail = output['output_text']
            st.write(mail)

    