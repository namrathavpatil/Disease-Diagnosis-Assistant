import re
import logging
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
import spacy
from app.data.orphanet_collector import OrphanetDisease
from app.data.fda_collector import FDACollector
from app.data.pubmed_collector import PubMedArticle

logger = logging.getLogger(__name__)

@dataclass
class ExtractedEntity:
    """Represents an extracted entity from text."""
    text: str
    type: str  # 'disease', 'drug', 'symptom', 'treatment', etc.
    confidence: float
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] = None

class MedicalEntityExtractor:
    """Extracts medical entities from text and builds knowledge graph relationships."""
    
    def __init__(self):
        """Initialize the entity extractor."""
        try:
            # Load English language model for NLP
            import scispacy
            from scispacy.abbreviation import AbbreviationDetector
            from scispacy.linking import EntityLinker
            nlp = spacy.load("en_core_sci_sm")
            nlp.add_pipe("abbreviation_detector")
            nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
            logger.info("Loaded scispaCy model for medical entity extraction")
        except ImportError:
            nlp = spacy.load("en_core_web_sm")
            logger.warning("scispaCy not installed, using default spaCy model")
        
        # Medical entity patterns - improved
        self.disease_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:disease|syndrome|disorder|condition)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:cancer|carcinoma|tumor|neoplasm)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:diabetes|hypertension|arthritis|asthma)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:heart disease|lung disease|kidney disease)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:genetic disorder|autoimmune disease)\b'
        ]
        
        self.drug_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b(?=\s+(?:tablet|capsule|injection|cream|ointment))',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b(?=\s+(?:mg|mcg|g|ml))',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b(?=\s+(?:prescribed|administered|taken))',
            r'\b(?:Aspirin|Ibuprofen|Metformin|Insulin|Penicillin|Amoxicillin|Lisinopril|Atorvastatin)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b(?=\s+(?:treats|relieves|reduces|prevents))'
        ]
        
        self.symptom_patterns = [
            r'\b(?:pain|fever|headache|nausea|vomiting|diarrhea|constipation|fatigue|weakness|dizziness)\b',
            r'\b(?:cough|sneeze|runny nose|sore throat|rash|itching|swelling|inflammation)\b',
            r'\b(?:high blood sugar|low blood sugar|high blood pressure|chest pain|shortness of breath)\b',
            r'\b(?:vision problems|hearing loss|memory loss|confusion|depression|anxiety)\b'
        ]
        
        # Initialize collectors
        self.fda_collector = FDACollector()
        self.orphanet_collector = None  # Will be set if API key available
        
    def set_orphanet_collector(self, collector):
        """Set the Orphanet collector if available."""
        self.orphanet_collector = collector
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract medical entities from text."""
        entities = []
        
        # Use spaCy for basic entity recognition
        doc = self.nlp(text)
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['DISEASE', 'CONDITION', 'MEDICAL_CONDITION']:
                entities.append(ExtractedEntity(
                    text=ent.text,
                    type='disease',
                    confidence=0.8,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char
                ))
            elif ent.label_ in ['DRUG', 'MEDICATION']:
                entities.append(ExtractedEntity(
                    text=ent.text,
                    type='drug',
                    confidence=0.8,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char
                ))
        
        # Use regex patterns for additional extraction
        entities.extend(self._extract_with_patterns(text))
        
        # Remove duplicates and overlapping entities
        entities = self._remove_overlapping_entities(entities)
        
        return entities
    
    def _extract_with_patterns(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using regex patterns."""
        entities = []
        
        # Extract diseases
        for pattern in self.disease_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity_text = match.group().strip()
                # Clean up the extracted text
                if entity_text.lower().startswith('it is used for '):
                    entity_text = entity_text[15:]  # Remove "It is used for "
                elif entity_text.lower().startswith('it is commonly used for '):
                    entity_text = entity_text[24:]  # Remove "It is commonly used for "
                elif entity_text.lower().startswith('metformin is used to treat '):
                    entity_text = entity_text[26:]  # Remove "Metformin is used to treat "
                
                entities.append(ExtractedEntity(
                    text=entity_text,
                    type='disease',
                    confidence=0.7,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        # Extract drugs
        for pattern in self.drug_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity_text = match.group().strip()
                # Clean up the extracted text
                if entity_text.lower().startswith('it is used for '):
                    entity_text = entity_text[15:]  # Remove "It is used for "
                elif entity_text.lower().startswith('it is commonly used for '):
                    entity_text = entity_text[24:]  # Remove "It is commonly used for "
                
                entities.append(ExtractedEntity(
                    text=entity_text,
                    type='drug',
                    confidence=0.6,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        # Extract symptoms
        for pattern in self.symptom_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    type='symptom',
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        return entities
    
    def _remove_overlapping_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove overlapping entities, keeping the one with higher confidence."""
        if not entities:
            return entities
        
        # Sort by confidence (descending) and start position
        entities.sort(key=lambda x: (-x.confidence, x.start_pos))
        
        filtered_entities = []
        for entity in entities:
            # Check if this entity overlaps with any already accepted entity
            overlaps = False
            for accepted in filtered_entities:
                if (entity.start_pos < accepted.end_pos and 
                    entity.end_pos > accepted.start_pos):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def build_relationships(self, entities: List[ExtractedEntity], text: str) -> List[Dict[str, Any]]:
        """Build relationships between extracted entities based on context."""
        relationships = []
        
        # Group entities by type
        diseases = [e for e in entities if e.type == 'disease']
        drugs = [e for e in entities if e.type == 'drug']
        symptoms = [e for e in entities if e.type == 'symptom']
        
        # Disease-Symptom relationships
        for disease in diseases:
            for symptom in symptoms:
                # Check if they appear in the same sentence or nearby
                if self._are_entities_related(disease, symptom, text):
                    relationships.append({
                        'source': disease.text,
                        'target': symptom.text,
                        'type': 'has_symptom',
                        'source_type': 'disease',
                        'target_type': 'symptom',
                        'confidence': min(disease.confidence, symptom.confidence)
                    })
        
        # Drug-Disease relationships (treatment)
        for drug in drugs:
            for disease in diseases:
                if self._are_entities_related(drug, disease, text):
                    relationships.append({
                        'source': drug.text,
                        'target': disease.text,
                        'type': 'treats',
                        'source_type': 'drug',
                        'target_type': 'disease',
                        'confidence': min(drug.confidence, disease.confidence)
                    })
        
        # Drug-Symptom relationships (relieves)
        for drug in drugs:
            for symptom in symptoms:
                if self._are_entities_related(drug, symptom, text):
                    relationships.append({
                        'source': drug.text,
                        'target': symptom.text,
                        'type': 'relieves',
                        'source_type': 'drug',
                        'target_type': 'symptom',
                        'confidence': min(drug.confidence, symptom.confidence)
                    })
        
        # Disease-Disease relationships (causes)
        for disease1 in diseases:
            for disease2 in diseases:
                if disease1 != disease2 and self._are_entities_related(disease1, disease2, text):
                    # Check if one causes the other
                    if self._check_causal_relationship(disease1, disease2, text):
                        relationships.append({
                            'source': disease1.text,
                            'target': disease2.text,
                            'type': 'causes',
                            'source_type': 'disease',
                            'target_type': 'disease',
                            'confidence': min(disease1.confidence, disease2.confidence)
                        })
        
        return relationships
    
    def _are_entities_related(self, entity1: ExtractedEntity, entity2: ExtractedEntity, text: str) -> bool:
        """Check if two entities are related based on proximity and context."""
        # Simple proximity check - within 100 characters
        distance = abs(entity1.start_pos - entity2.start_pos)
        if distance > 100:
            return False
        
        # Check if they're in the same sentence
        sentence_start = max(0, min(entity1.start_pos, entity2.start_pos) - 50)
        sentence_end = min(len(text), max(entity1.end_pos, entity2.end_pos) + 50)
        sentence = text[sentence_start:sentence_end]
        
        # Look for relationship indicators
        relationship_indicators = [
            'treats', 'causes', 'symptoms', 'relieves', 'associated with',
            'leads to', 'results in', 'manages', 'controls', 'prevents'
        ]
        
        return any(indicator in sentence.lower() for indicator in relationship_indicators)
    
    def _check_causal_relationship(self, entity1: ExtractedEntity, entity2: ExtractedEntity, text: str) -> bool:
        """Check if one entity causes another based on context."""
        # Look for causal indicators between the entities
        causal_indicators = [
            'causes', 'leads to', 'results in', 'triggers', 'induces',
            'brings about', 'gives rise to', 'produces', 'creates'
        ]
        
        # Get the text between the two entities
        start = min(entity1.start_pos, entity2.start_pos)
        end = max(entity1.end_pos, entity2.end_pos)
        context = text[start:end].lower()
        
        return any(indicator in context for indicator in causal_indicators)
    
    def enrich_entities(self, entities: List[ExtractedEntity]) -> List[Dict[str, Any]]:
        """Enrich extracted entities with additional information from APIs."""
        enriched_entities = []
        
        for entity in entities:
            enriched_entity = {
                'text': entity.text,
                'type': entity.type,
                'confidence': entity.confidence,
                'metadata': entity.metadata or {}
            }
            
            # Enrich diseases with Orphanet data
            if entity.type == 'disease' and self.orphanet_collector:
                try:
                    diseases = self.orphanet_collector.search_by_name(entity.text)
                    if diseases:
                        disease = diseases[0]  # Take the first match
                        enriched_entity['metadata'].update({
                            'orpha_id': disease.disease_id,
                            'prevalence': disease.prevalence,
                            'inheritance': disease.inheritance,
                            'phenotypes': disease.phenotypes[:5]  # Limit to first 5
                        })
                except Exception as e:
                    logger.warning(f"Failed to enrich disease {entity.text}: {e}")
            
            # Enrich drugs with FDA data
            elif entity.type == 'drug':
                try:
                    drug_labels = self.fda_collector.search_drug_labels(
                        f"openfda.brand_name:{entity.text}",
                        limit=1
                    )
                    if not drug_labels.empty:
                        label = drug_labels.iloc[0]
                        enriched_entity['metadata'].update({
                            'generic_name': label.get('openfda.generic_name', ''),
                            'manufacturer': label.get('openfda.manufacturer_name', ''),
                            'indications': label.get('indications_and_usage', '')[:200]  # Truncate
                        })
                except Exception as e:
                    logger.warning(f"Failed to enrich drug {entity.text}: {e}")
            
            enriched_entities.append(enriched_entity)
        
        return enriched_entities 