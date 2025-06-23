import os
import json
import logging
from typing import List, Dict, Any
from openai import OpenAI

logger = logging.getLogger(__name__)

class LLMEntityExtractor:
    """
    Uses LLM (via Together AI) to extract medical entities from text.
    """
    def __init__(self, api_key: str = None, model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        # Use Together AI with a serverless model (Mixtral-8x7B)
        self.client = OpenAI(
            api_key=api_key or "b5f3b531715a443f991a0538ef36636bfc857f32c4719e04b81b101b6c8c3b51",
            base_url="https://api.together.xyz/v1"
        )
        self.model_name = model_name
        
        # Prompt template for entity extraction
        self.prompt_template = """
You are a medical entity extraction expert. Extract all disease names, drug names, and symptoms from the following text.

Return ONLY a valid JSON array of objects with 'text' and 'type' fields. The 'type' should be one of: 'disease', 'drug', 'symptom'.

Example output format:
[
    {"text": "diabetes", "type": "disease"},
    {"text": "metformin", "type": "drug"},
    {"text": "high blood sugar", "type": "symptom"}
]

Text to analyze: {text}

JSON response:"""

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using LLM and return as a list of dicts.
        """
        try:
            # Prepare the prompt
            prompt = self.prompt_template.format(text=text)
            
            # Call the LLM
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a medical entity extraction expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Extract the response content
            result = response.choices[0].message.content.strip()
            
            # Try to parse the JSON response
            try:
                # Clean up the response - remove any markdown formatting
                if result.startswith("```json"):
                    result = result[7:]
                if result.endswith("```"):
                    result = result[:-3]
                
                entities = json.loads(result.strip())
                
                # Validate and filter entities
                valid_entities = []
                for entity in entities:
                    if isinstance(entity, dict) and 'text' in entity and 'type' in entity:
                        if entity['type'] in ['disease', 'drug', 'symptom']:
                            valid_entities.append({
                                'text': entity['text'].strip(),
                                'type': entity['type'],
                                'confidence': 0.9  # High confidence for LLM extraction
                            })
                
                logger.info(f"Extracted {len(valid_entities)} entities using LLM")
                return valid_entities
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.error(f"Raw response: {result}")
                return []
                
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return []
    
    def extract_entities_with_relationships(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and infer relationships using LLM.
        """
        try:
            # Enhanced prompt for entity and relationship extraction
            relationship_prompt = f"""
You are a medical knowledge extraction expert. Analyze the following text and extract:
1. All disease names, drug names, and symptoms
2. Relationships between these entities

Return a JSON object with 'entities' and 'relationships' arrays.

Example format:
{{
    "entities": [
        {{"text": "diabetes", "type": "disease"}},
        {{"text": "metformin", "type": "drug"}},
        {{"text": "high blood sugar", "type": "symptom"}}
    ],
    "relationships": [
        {{"source": "diabetes", "target": "high blood sugar", "type": "has_symptom"}},
        {{"source": "metformin", "target": "diabetes", "type": "treats"}}
    ]
}}

Text to analyze: {text}

JSON response:"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a medical knowledge extraction expert. Always respond with valid JSON only."},
                    {"role": "user", "content": relationship_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean up the response
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            
            data = json.loads(result.strip())
            
            # Validate and return
            entities = data.get('entities', [])
            relationships = data.get('relationships', [])
            
            # Validate entities
            valid_entities = []
            for entity in entities:
                if isinstance(entity, dict) and 'text' in entity and 'type' in entity:
                    if entity['type'] in ['disease', 'drug', 'symptom']:
                        valid_entities.append({
                            'text': entity['text'].strip(),
                            'type': entity['type'],
                            'confidence': 0.9
                        })
            
            # Validate relationships
            valid_relationships = []
            for rel in relationships:
                if isinstance(rel, dict) and all(k in rel for k in ['source', 'target', 'type']):
                    # Find the entity types for source and target
                    source_type = None
                    target_type = None
                    
                    for entity in valid_entities:
                        if entity['text'].lower() == rel['source'].lower():
                            source_type = entity['type']
                        if entity['text'].lower() == rel['target'].lower():
                            target_type = entity['type']
                    
                    # Only add relationship if we can determine both types
                    if source_type and target_type:
                        valid_relationships.append({
                            'source': rel['source'].strip(),
                            'target': rel['target'].strip(),
                            'type': rel['type'],
                            'source_type': source_type,
                            'target_type': target_type,
                            'confidence': 0.8
                        })
            
            return {
                'entities': valid_entities,
                'relationships': valid_relationships
            }
            
        except Exception as e:
            logger.error(f"Error extracting entities and relationships: {e}")
            return {'entities': [], 'relationships': []}

    def extract_graph_from_prompt(self, prompt: str) -> dict:
        """
        Given a prompt that asks for a knowledge graph in JSON format, call the LLM and parse the result.
        Returns a dict with 'nodes' and 'edges'.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a medical knowledge graph builder. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            result = response.choices[0].message.content.strip()
            # Clean up the response
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            data = json.loads(result.strip())
            # Validate structure
            if not isinstance(data, dict) or "nodes" not in data or "edges" not in data:
                logger.error(f"LLM did not return a valid graph JSON: {data}")
                return {}
            return data
        except Exception as e:
            logger.error(f"Error extracting graph from LLM: {e}")
            return {} 