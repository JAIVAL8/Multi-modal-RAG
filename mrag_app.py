# rag_app.py

"""
Multimodal RAG application with a Gradio interface.
It uses a graph-based approach (LangGraph) for adaptive retrieval,
generation, and self-correction based on the provided notebook.
"""

import base64,os
import logging
from typing import List, Literal, Tuple
from pprint import pprint

import gradio as gr
from faster_whisper import WhisperModel
from langchain import hub
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.types import Command

import config
from utils import setup_environment, read_all_images, encode_image_base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pydantic Models for Structured Output ---
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question, choose to route it to web search or a vectorstore containing router manual information."
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination presence in a generated answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    """Binary score to assess if an answer addresses the question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

# --- LangGraph State ---
class GraphState(TypedDict):
    """Represents the state of our graph."""
    question: str
    imgquery: str
    generation: str
    documents: List[Document]
    imgrefrences: List[str]

# --- Helper function for formatting docs ---
def format_docs_for_llm(docs: List[Document]) -> Tuple[str, List[str]]:
    """Formats retrieved documents into a string context and extracts image references."""
    context_parts, image_refs = [], []
    for doc in docs:
        metadata = doc.metadata
        if metadata.get('type') == "text":
            context_parts.append(f"[Page {metadata.get('page')}] {doc.page_content}")
        elif metadata.get('type') == "image":
            image_refs.append(metadata.get('image_id'))
            context_parts.append(f"[Page {metadata.get('page')}] IMAGE INFO: {doc.page_content}")
    return "\n\n".join(context_parts), image_refs

# --- Graph Node Functions ---
def retrieve(state: GraphState, retriever):
    """Retrieves documents from the vector store."""
    logging.info("--- NODE: RETRIEVE ---")
    question = state["question"]
    retrieved_docs = retriever.invoke(question)
    context_text, image_refs = format_docs_for_llm(retrieved_docs)
    return {"documents": context_text, "imgrefrences": image_refs, "question": question}

def generate(state: GraphState, rag_chain):
    """Generates an answer using the RAG chain."""
    logging.info("--- NODE: GENERATE ---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def transform_query(state: GraphState, rewriter):
    """Transforms the query to a better version."""
    logging.info("--- NODE: TRANSFORM QUERY ---")
    question = state["question"]
    better_question = rewriter.invoke({"question": question})
    return {"question": better_question}

def web_search(state: GraphState, web_search_tool):
    """Performs a web search."""
    logging.info("--- NODE: WEB SEARCH ---")
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs["results"]])
    web_results = Document(page_content=web_results)
    
    current_docs = state.get("documents", [])
    if isinstance(current_docs, str): # Ensure it's a list
        current_docs = [Document(page_content=current_docs)]
    current_docs.append(web_results)
    
    return {"documents": current_docs, "question": question}

# --- Graph Edge Functions ---
def route_question(state: GraphState, router, img_summarizer):
    """Routes the question to the appropriate tool (vectorstore or web search)."""
    logging.info("--- EDGE: ROUTE QUESTION ---")
    question = state["question"]
    image_b64 = state["imgquery"]

    if image_b64:
        logging.info("Image detected, summarizing to refine query...")
        question = img_summarizer.invoke({"question": question, "image_b64_url": image_b64})
        logging.info(f"Refined query: {question}")
    
    source = router.invoke({"question": question})
    logging.info(f"Routing decision: {source.datasource}")
    if source.datasource == "web_search":
        return Command(update={"question": question}, goto="web_search")
    elif source.datasource == "vectorstore":
        return Command(update={"question": question}, goto="retrieve")    
    # if source.datasource == "web_search":
    #     return "web_search"
    # elif source.datasource == "vectorstore":
    #     return "vectorstore"

def grade_generation(state: GraphState, hallucination_grader, answer_grader):
    """Determines whether the generation is grounded and answers the question."""
    logging.info("--- EDGE: GRADE GENERATION ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score
    
    if grade == "no":
        logging.warning("Generation is not grounded in documents, re-trying.")
        return "not supported"
    
    logging.info("Generation is grounded in documents.")
    score = answer_grader.invoke({"question": question, "generation": generation})
    grade = score.binary_score
    
    if grade == "yes":
        logging.info("Generation addresses the question.")
        return "useful"
    else:
        logging.warning("Generation does not address the question, transforming query.")
        return "not useful"

# --- Main Application Class ---
class RAGApplication:
    def __init__(self):
        setup_environment()
        os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT
        
        self.all_images = read_all_images(config.OUTPUT_IMAGE_DIR)
        
        self.model = ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMP)
        self.model_high = ChatOpenAI(model=config.HIGH_PERF_LLM_MODEL, temperature=config.LLM_TEMP)
        self.embedding_model = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
        
        self._setup_components()
        self._compile_graph()

    def _setup_components(self):
        # Tools and Retrievers
        self.web_search_tool = TavilySearch(max_results=5)
        self._setup_retrievers()

        # Chains and Graders
        self.rag_chain = hub.pull("rlm/rag-prompt") | self.model | StrOutputParser()
        
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. Only return the rewritten question."),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")
        ])
        self.question_rewriter = rewrite_prompt | self.model | StrOutputParser()
        
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
        ])
        self.hallucination_grader = hallucination_prompt | self.model_high.with_structured_output(GradeHallucinations)
        
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a grader assessing whether an answer addresses / resolves a question. Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
        ])
        self.answer_grader = answer_prompt | self.model_high.with_structured_output(GradeAnswer)
        
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at routing a user question to a vectorstore or web search. The vectorstore contains documents related to a Wifi router manual. Use the vectorstore for questions on Wifi Router topics. Otherwise, use web-search."),
            ("human", "{question}")
        ])
        self.question_router = router_prompt | self.model.with_structured_output(RouteQuery)
        
        img_summary_prompt = ChatPromptTemplate.from_messages([
           ("system","""You are an assistant that converts images into search queries.
            Your task:
            - Look at the image and extract all meaningful details (text, numbers, objects, entities, or statistics).
            - Rewrite these details as a single, clear, concise query that could be used for web search or knowledge retrieval.
            - The query should capture the user's intent and the core content of the image.
            - Do not explain the image. Only return the final query as plain text.
            """), 
            ("human", [{"type": "text", "text": "{question}"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_b64_url}"}}])
        ])
        self.img_summarizer = img_summary_prompt | self.model | StrOutputParser()

    def _setup_retrievers(self):
        vectorstore_text = FAISS.load_local(config.FAISS_INDEX_PATH, self.embedding_model, allow_dangerous_deserialization=True)
        vectorstore_img = FAISS.load_local(config.FAISS_IMG_INDEX_PATH, self.embedding_model, allow_dangerous_deserialization=True)
        text_retriever = vectorstore_text.as_retriever(search_kwargs={"k": config.TEXT_RETRIEVER_K})
        image_retriever = vectorstore_img.as_retriever(search_kwargs={"k": config.IMAGE_RETRIEVER_K})
        ensemble_retriever = EnsembleRetriever(retrievers=[text_retriever, image_retriever], weights=config.ENSEMBLE_WEIGHTS)
        reranker = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL, model_kwargs={"device": "cuda"})
        compressor = CrossEncoderReranker(model=reranker, top_n=config.RERANKER_TOP_N)
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)

    def _compile_graph(self):
        workflow = StateGraph(GraphState)
        
        workflow.add_node("retrieve", lambda state: retrieve(state, self.compression_retriever))
        workflow.add_node("generate", lambda state: generate(state, self.rag_chain))
        workflow.add_node("transform_query", lambda state: transform_query(state, self.question_rewriter))
        workflow.add_node("web_search", lambda state: web_search(state, self.web_search_tool))
        workflow.add_node("route_question",lambda state: route_question(state, self.question_router, self.img_summarizer))

        # workflow.set_conditional_entry_point(
        #     lambda state: route_question(state, self.question_router, self.img_summarizer),
        #     {"web_search": "web_search", "vectorstore": "retrieve"}
        # )
        workflow.set_entry_point("route_question")

        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("web_search", "generate")
        workflow.add_conditional_edges(
            "generate",
            lambda state: grade_generation(state, self.hallucination_grader, self.answer_grader),
            {"not supported": "generate", "useful": END, "not useful": "transform_query"}
        )
        workflow.add_edge("transform_query", "retrieve")

        self.rag_graph = workflow.compile()
        logging.info("RAG graph compiled successfully.")
        
    def run_query(self, query_text: str, image_path: str = ""):
        image_b64 = encode_image_base64(image_path) if image_path else ""
        
        if not query_text.strip() and not image_b64:
            return "Please provide a question or an image.", []

        if not query_text.strip() and image_b64:
            query_text = "Describe the attached image in the context of router troubleshooting."

        logging.info(f"Executing graph with query: '{query_text}' | Image provided: {bool(image_path)}")
        inputs = {"question": query_text, "imgquery": image_b64, "documents": [], "imgrefrences": []}
        response = self.rag_graph.invoke(inputs)

        return self._format_response_for_ui(response)

    def _format_response_for_ui(self, response: dict):
        generation = response.get("generation", "No answer could be generated.")
        img_refs = response.get("imgrefrences", [])
        images = [self.all_images[ref] for ref in img_refs if ref in self.all_images]
        
        md_response = f"### Answer\n{generation}\n\n"
        if response.get("documents"):
            # Ensure documents are in a readable format for UI
            docs_content = response['documents']
            if isinstance(docs_content, list):
                 docs_content = "\n\n---\n\n".join([doc.page_content for doc in docs_content])
            md_response += f"### Supporting Context\n```\n{docs_content}\n```"
            
        return md_response, images

# --- Gradio UI ---
def build_ui(rag_app: RAGApplication):
    stt_model = WhisperModel(config.STT_MODEL_SIZE, device="cuda", compute_type="float32")
    
    def transcribe_audio(audio_file):
        if not audio_file: return ""
        segments, _ = stt_model.transcribe(audio_file, beam_size=5)
        return " ".join([seg.text for seg in segments]).strip()

    with gr.Blocks(title="Multimodal RAG Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Multimodal RAG Assistant\nAsk questions about your router using text, audio, or an image!")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="filepath", label="Upload Image (Optional)")
                audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Or Record Audio Query")
            with gr.Column(scale=2):
                text_in = gr.Textbox(label="Your Question", lines=4)
                run_btn = gr.Button("🔍 Ask", variant="primary")
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=2):
                answer_out = gr.Markdown(label="💡 Answer & Context")
            with gr.Column(scale=1):
                img_out = gr.Gallery(label="📷 Retrieved Images", show_label=True, columns=1, object_fit="contain", height="auto")
        
        audio_in.change(fn=transcribe_audio, inputs=audio_in, outputs=text_in)
        run_btn.click(
            fn=rag_app.run_query,
            inputs=[text_in, image_in],
            outputs=[answer_out, img_out]
        )
    
    demo.launch()

if __name__ == "__main__":
    app = RAGApplication()
    build_ui(app)