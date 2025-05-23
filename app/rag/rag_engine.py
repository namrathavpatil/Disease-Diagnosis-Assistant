import logging
from typing import List, Dict, Any
from together import Together
from app.config import settings
from app.core.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

class RAGEngine:
    """Retrieval-Augmented Generation engine for medical Q&A."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.together_client = Together(api_key=settings.together_api_key)
        logger.info("RAG Engine initialized")
    
    def retrieve_context(self, query: str, max_nodes: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context from the knowledge graph."""
        # Get nodes that are most relevant to the query
        relevant_nodes = self.knowledge_graph.search_nodes(query, limit=max_nodes)
        
        # Format the context
        context = []
        for node in relevant_nodes:
            node_data = {
                'id': node['id'],
                'type': node['type'],
                'name': node['name'],
                'metadata': node['metadata']
            }
            context.append(node_data)
        
        return context
    
    def format_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Format the prompt for the LLM with retrieved context."""
        prompt = f"""You are a medical assistant specialized in infectious diseases, particularly cholera. 
Your responses should be accurate, evidence-based, and aligned with current medical guidelines.

Use the following context to answer the question. If you cannot find the answer in the context, you may provide general medical knowledge about cholera, but clearly indicate that this information is not from the provided context.

Important guidelines:
1. Always prioritize information from the provided context
2. If context is insufficient, provide general medical knowledge but clearly state this
3. Be precise about transmission routes, symptoms, and treatments
4. Include relevant medical terminology
5. Cite sources when possible
6. If information seems incorrect or outdated, note this

Context:
"""
        for item in context:
            prompt += f"\n{item['type'].upper()}: {item['name']}\n"
            if 'metadata' in item and item['metadata']:
                for key, value in item['metadata'].items():
                    prompt += f"{key}: {value}\n"
        
        prompt += f"\nQuestion: {query}\nAnswer:"
        return prompt
    
    def generate_answer(self, query: str, max_context_nodes: int = 5) -> Dict[str, Any]:
        """Generate an answer using RAG."""
        try:
            # Retrieve relevant context
            context = self.retrieve_context(query, max_context_nodes)
            if not context:
                return {
                    'answer': "I couldn't find any relevant information in the knowledge base to answer your question.",
                    'context': []
                }
            
            # Format the prompt
            prompt = self.format_prompt(query, context)
            
            # Generate response using Together AI
            response = self.together_client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[
                    {"role": "system", "content": """You are a medical assistant specialized in infectious diseases, particularly cholera. 
Your responses should be accurate, evidence-based, and aligned with current medical guidelines.
Always prioritize information from the provided context, but if it's insufficient, you may provide general medical knowledge while clearly indicating this.
Be precise about transmission routes, symptoms, and treatments, and include relevant medical terminology."""},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return {
                'answer': response.choices[0].message.content,
                'context': context
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                'answer': "I encountered an error while processing your question. Please try again.",
                'context': []
            }
    
    def answer_question(self, query: str) -> Dict[str, Any]:
        """Main method to answer a question using RAG."""
        logger.info(f"Processing question: {query}")
        result = self.generate_answer(query)
        logger.info("Answer generated successfully")
        return result 