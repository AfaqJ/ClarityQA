from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

def get_llm_chain():
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a helpful assistant. Use the following context to answer the user's question.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
    )
    
    llm = Ollama(model="mistral")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain
