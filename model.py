from langchain import PromptTemplate 
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import ctransformers
from langchain.chains import RetrievalQA #need different types of chain like conversations retrieval or import retrieval in this case
import chainlit as cl 

DB_FAISS_PATH = "vectorstores/db_faiss" #path

custom_prompt_template = '''You're being given some information to a user's question.
If are not familiar with a question/don't know, don't make up an answer. 

Context: {}
Question: {question}

Only returns the helpful answer below and nothing else. 
Useful answer: 
'''

def set_custom_prompt(): 
    '''
    Prompt template for Question/Answer retrieval for each vector stores
    '''

    prompt = PromptTemplate(template=custom_prompt_template, 
                            input_variables=[
                                'context', 
                                'question'
                            ])
    
    return prompt

def load_llm(): 
    llm = ctransformers(
        model = 'llama-2-7b-chat.ggmlv3.q8_0.bin', 
        model_type = 'llama', 
        max_new_tokens = 512, 
        temperature = 0.0
    )
    return llm 

def retrieval_qa_chain(llm, prompt, db): 
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff', 
        retriever = db.as_retriever(search_kwargs={'k': 2}),
        return_source_document = True, 
        chain_type_kwargs = {'prompt': prompt}
    )
    
    return qa_chain

def qa_bot(): 
    embeddings = HuggingFaceBgeEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                          model_kwargs = {'device': 'cpu'})
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True) 
    llm = load_llm() 
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db) 

    return qa

def final_result(query): 
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

## chainlit function ##
@cl.on_chat_start 
async def start(): 
    chain = qa_bot()
    msg = cl.Message(content='Starting the bot...') 
    await msg.send
    msg.content = 'Hi, Welcome to Medical Bot. How may I be of assistance?'
    await msg.update() 
    cl.user_session.set('chain', chain)

@cl.on_message
async def main(message): 
    chain = cl.cl.user_session.set('chain')
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=['FINAL', 'ANSWER']
    )
    cb.answer_reaced = True
    allow_dangerous_deserialization = True,
    res = await chain.acall(message, callback=[cb])
    answer = res['result']
    sources = res['source_documents']

    if sources: 
        answer += f'\nSources:' + str(sources) 
    else: 
        answer += f'\nNo Sources found.'

    await cl.Message(content=answer).send()