# ============================================================================
# FILE: core/tools.py
# PURPOSE: Initialize and manage all tools (web search, RAG)
# ============================================================================

from typing import Dict
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
import logging

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Centralized tool management.
    WHY? Separation of concern - all tool logic in one place.
    """
    
    def __init__(self):
        self.web_search_tool = self._init_web_search()
        self.rag_tool = self._init_rag()
    
    def _init_web_search(self) -> Tool:
        """Initialize Google Serper for internet search."""
        serper = GoogleSerperAPIWrapper()
        return Tool(
            name="web_search",
            func=serper.run,
            description="Search the internet for current information about Azure services"
        )
    
    def _init_rag(self) -> Tool:
        """Initialize RAG search for vector database."""
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = Chroma(
            collection_name="AzureDocs",
            persist_directory="./chroma_db_AzureDocs",
            embedding_function=embeddings,
        )
        
        def rag_search(query: str, k: int = 5) -> str:
            """Search AWS documentation vector database."""
            try:
                docs = vector_store.similarity_search(query, k=k)
                if not docs:
                    return "No relevant documentation found."
                
                results = []
                for i, doc in enumerate(docs, 1):
                    content = doc.page_content.strip()[:2000]  # Limit length
                    results.append(f"[Document {i}]:\n{content}\n")
                
                return "\n---\n".join(results)
            except Exception as e:
                logging.error(f"RAG error: {str(e)}")
                return f"Error: {str(e)}"
        
        return Tool(
            name="RAG_search",
            func=rag_search,
            description="Search AWS documentation for accurate architectural guidance"
        )
    
    def get_all_tools(self) -> Dict[str, Tool]:
        """Get all tools as a dictionary."""
        return {
            self.web_search_tool.name: self.web_search_tool,
            self.rag_tool.name: self.rag_tool,
        }


class LLMManager:
    """Manage LLM instances."""
    
    def __init__(self):
        self.mini_llm = ChatOpenAI(model="gpt-4o-mini")
        self.reasoning_llm = ChatOpenAI(model="gpt-4o")
    
    def get_mini_llm(self):
        """Get fast, cheap LLM for quick tasks."""
        return self.mini_llm
    
    def get_reasoning_llm(self):
        """Get powerful LLM for complex reasoning."""
        return self.reasoning_llm
    
    def get_mini_with_tools(self, tools: list) -> object:
        """Bind tools to mini LLM for tool calling."""
        return self.mini_llm.bind_tools(tools)
    
    def get_reasoning_structured(self, schema):
        """Get reasoning LLM with structured output."""
        return self.reasoning_llm.with_structured_output(schema)