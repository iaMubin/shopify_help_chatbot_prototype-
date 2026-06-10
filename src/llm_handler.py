import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def retrieve_and_rerank(query: str, vectorstore, top_k_retrieve: int = 10, top_k_rerank: int = 3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k_retrieve})
    retrieved_docs = retriever.invoke(query)
    
    unique_docs = []
    seen_content = set()
    for doc in retrieved_docs:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)
            
    cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
    pairs = [[query, doc.page_content] for doc in unique_docs]
    scores = cross_encoder.predict(pairs)
    
    scored_docs = list(zip(unique_docs, scores))
    sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    
    return sorted_docs[:top_k_rerank]

def filter_relevant_links(links: list) -> list:
    ignore_keywords = [
        'login', 'privacy', 'terms', 'admin.shopify.com', 
        'user/login', 'search', 'contact', 'twitter', 
        'facebook', 'youtube', 'instagram', 'legal'
    ]
    return [link for link in links if not any(kw in link.lower() for kw in ignore_keywords)]

def get_llm_chain():
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an elite, highly professional Shopify customer success manager. Your goal is to provide exceptional, accurate, and easy-to-understand support.\n\n"
            "Context:\n{context}\n\n"
            "Available Reference Links:\n{links}\n\n"
            "Instructions:\n"
            "1. Tone: Warm, professional, empathetic, and authoritative. Speak like a premium brand representative.\n"
            "2. Structure: Use short paragraphs. Use bullet points for steps, lists, or multiple conditions to make it highly readable.\n"
            "3. Hyperlinks: Seamlessly embed 1 or 2 most relevant links naturally within your text using Markdown (e.g., [Shopify return rules](url)). Never list raw URLs.\n"
            "4. Accuracy: Base your answer STRICTLY on the provided context. If the context lacks the answer, politely inform the user that you cannot provide that information right now."
        )),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}")
    ])
    
    return prompt | llm | StrOutputParser()

def generate_rag_response_lcel(query: str, sorted_docs: list, chat_history: ChatMessageHistory, rag_chain) -> str:
    context_chunks = []
    source_links = set()
    
    for doc, score in sorted_docs:
        context_chunks.append(doc.page_content)
        if "hyperlinks" in doc.metadata:
            cleaned_links = filter_relevant_links(doc.metadata["hyperlinks"])
            for link in cleaned_links:
                if "help.shopify.com" in link and len(link) > 35:
                    source_links.add(link)
                
    context_text = "\n\n".join(context_chunks)
    links_text = "\n".join(source_links) if source_links else "No reference links available."
    
    response = rag_chain.invoke({
        "context": context_text,
        "links": links_text,
        "messages": chat_history.messages,
        "input": query
    })
    
    chat_history.add_user_message(query)
    chat_history.add_ai_message(response)
    
    return response