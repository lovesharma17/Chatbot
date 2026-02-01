import validators , streamlit as st
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader , UnstructuredURLLoader
import os 
from dotenv import load_dotenv 

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

## Streamlit App

st.set_page_config(page_title="Summarize Text from YT or Website" , page_icon="🕊")
st.title("🕊 Summarize Text from YT or Website")
st.subheader("Summarize URL")

## Get the Groq API key and URL field
api_key = st.sidebar.text_input(label="Enter your GROQ API key" , type="password")
URL = st.text_input("URL" , label_visibility="collapsed")
llm = ChatGroq(model_name = "llama-3.1-8b-instant")

prompt_template = """
Provide a summary of the following content in 300 words:
content = {text}
"""
prompt = PromptTemplate(template=prompt_template , input_variables=["text"])
if st.button("Summarize the content from YT or Website"):
    ## Validate all the inputs
    if not api_key.strip() or not URL.strip():
        st.error("Please provide the information")
    elif not validators.url(URL):
        st.error("Please provide a valid Url.")
    else:
        try:
            with st.spinner("waiting..."):
                ## Loading the website or yt video data
                if "youtube.com" in URL:
                    loader = YoutubeLoader.from_youtube_url(URL,add_video_info=False)
                else:
                    loader = UnstructuredURLLoader(urls=[URL] , ssl_verify = False)
                docs = loader.load()

                ## Chain for summarization
                chain = load_summarize_chain(llm , chain_type="stuff" , prompt = prompt)
                output_summary = chain.run(docs)
                st.success("Summary generated!")
                st.write(output_summary)
        except Exception as e :
            st.exception(e)
            if hasattr(e, "response") and e.response is not None:
              st.code(e.response.text)
              