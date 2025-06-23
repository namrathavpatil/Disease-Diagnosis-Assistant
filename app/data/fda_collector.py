import requests
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

class FDACollector:
    """
    Collector for U.S. FDA data using the OpenFDA API.
    Provides access to drug labels, adverse events, device recalls, and food data.
    """
    BASE_URL = "https://api.fda.gov"

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache/fda")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def search_drug_labels(self, search_query: str, limit: int = 10) -> pd.DataFrame:
        """
        Search for drug labels by various criteria.
        Args:
            search_query: Search query (e.g., 'active_ingredient:ibuprofen')
            limit: Maximum number of results to return
        Returns:
            DataFrame containing drug label information
        """
        url = f"{self.BASE_URL}/drug/label.json"
        params = {
            "search": search_query,
            "limit": limit
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for result in data.get("results", []):
            results.append({
                "id": result.get("id"),
                "openfda": result.get("openfda", {}),
                "description": result.get("description", []),
                "indications_and_usage": result.get("indications_and_usage", []),
                "dosage_and_administration": result.get("dosage_and_administration", []),
                "warnings": result.get("warnings", []),
                "adverse_reactions": result.get("adverse_reactions", []),
                "drug_interactions": result.get("drug_interactions", []),
                "pregnancy": result.get("pregnancy", []),
                "nursing_mothers": result.get("nursing_mothers", []),
                "pediatric_use": result.get("pediatric_use", []),
                "geriatric_use": result.get("geriatric_use", [])
            })
        
        return pd.DataFrame(results)

    def get_adverse_events(self, search_query: str = None, limit: int = 10) -> pd.DataFrame:
        """
        Get adverse event reports.
        Args:
            search_query: Search query for filtering adverse events
            limit: Maximum number of results to return
        Returns:
            DataFrame containing adverse event information
        """
        url = f"{self.BASE_URL}/drug/event.json"
        params = {"limit": limit}
        
        if search_query:
            params["search"] = search_query
            
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for result in data.get("results", []):
            results.append({
                "safetyreportid": result.get("safetyreportid"),
                "serious": result.get("serious"),
                "seriousnessdeath": result.get("seriousnessdeath"),
                "seriousnessdisabling": result.get("seriousnessdisabling"),
                "seriousnesshospitalization": result.get("seriousnesshospitalization"),
                "seriousnesslifethreatening": result.get("seriousnesslifethreatening"),
                "receivedate": result.get("receivedate"),
                "transmissiondate": result.get("transmissiondate"),
                "patient": result.get("patient", {}),
                "drugs": result.get("drug", []),
                "reactions": result.get("patient", {}).get("reaction", [])
            })
        
        return pd.DataFrame(results)

    def get_device_recalls(self, search_query: str = None, limit: int = 10) -> pd.DataFrame:
        """
        Get device recall information.
        Args:
            search_query: Search query for filtering device recalls
            limit: Maximum number of results to return
        Returns:
            DataFrame containing device recall information
        """
        url = f"{self.BASE_URL}/device/recall.json"
        params = {"limit": limit}
        
        if search_query:
            params["search"] = search_query
            
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for result in data.get("results", []):
            results.append({
                "recall_number": result.get("recall_number"),
                "recall_date": result.get("recall_date"),
                "recall_status": result.get("recall_status"),
                "product_description": result.get("product_description"),
                "reason_for_recall": result.get("reason_for_recall"),
                "classification": result.get("classification"),
                "product_quantity": result.get("product_quantity"),
                "distribution_pattern": result.get("distribution_pattern"),
                "firm_fei_number": result.get("firm_fei_number"),
                "firm_name": result.get("firm_name")
            })
        
        return pd.DataFrame(results)

    def get_food_enforcement(self, search_query: str = None, limit: int = 10) -> pd.DataFrame:
        """
        Get food enforcement reports.
        Args:
            search_query: Search query for filtering food enforcement data
            limit: Maximum number of results to return
        Returns:
            DataFrame containing food enforcement information
        """
        url = f"{self.BASE_URL}/food/enforcement.json"
        params = {"limit": limit}
        
        if search_query:
            params["search"] = search_query
            
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for result in data.get("results", []):
            results.append({
                "recall_number": result.get("recall_number"),
                "recall_initiation_date": result.get("recall_initiation_date"),
                "status": result.get("status"),
                "product_description": result.get("product_description"),
                "reason_for_recall": result.get("reason_for_recall"),
                "classification": result.get("classification"),
                "product_quantity": result.get("product_quantity"),
                "distribution_pattern": result.get("distribution_pattern"),
                "firm_name": result.get("firm_name"),
                "city": result.get("city"),
                "state": result.get("state"),
                "country": result.get("country")
            })
        
        return pd.DataFrame(results)

    def get_drug_info_by_ndc(self, ndc: str) -> Dict:
        """
        Get drug information by NDC (National Drug Code).
        Args:
            ndc: National Drug Code
        Returns:
            Dictionary containing drug information
        """
        url = f"{self.BASE_URL}/drug/ndc.json"
        params = {"search": f"product_ndc:{ndc}"}
        
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("results"):
            return data["results"][0]
        return {}

    def search_drugs_by_ingredient(self, ingredient: str, limit: int = 10) -> pd.DataFrame:
        """
        Search for drugs by active ingredient.
        Args:
            ingredient: Active ingredient name
            limit: Maximum number of results to return
        Returns:
            DataFrame containing drug information
        """
        return self.search_drug_labels(f"active_ingredient:{ingredient}", limit)

    def get_drug_interactions(self, drug_name: str) -> pd.DataFrame:
        """
        Get drug interaction information for a specific drug.
        Args:
            drug_name: Name of the drug
        Returns:
            DataFrame containing drug interaction information
        """
        labels_df = self.search_drug_labels(f"openfda.brand_name:{drug_name}")
        
        interactions = []
        for _, row in labels_df.iterrows():
            drug_interactions = row.get("drug_interactions", [])
            if drug_interactions:
                for interaction in drug_interactions:
                    interactions.append({
                        "drug_name": drug_name,
                        "interaction_text": interaction
                    })
        
        return pd.DataFrame(interactions)

    def build_drug_knowledge_graph(self, drug_names: List[str]) -> Dict:
        """
        Build a knowledge graph from drug information.
        Args:
            drug_names: List of drug names
        Returns:
            Dictionary containing nodes and edges of the knowledge graph
        """
        nodes = []
        edges = []
        
        for drug_name in drug_names:
            # Get drug label information
            labels_df = self.search_drug_labels(f"openfda.brand_name:{drug_name}")
            
            if not labels_df.empty:
                drug_info = labels_df.iloc[0]
                
                # Add drug node
                nodes.append({
                    "id": drug_name,
                    "type": "drug",
                    "name": drug_name,
                    "description": drug_info.get("description", [""])[0] if drug_info.get("description") else ""
                })
                
                # Add active ingredients
                openfda = drug_info.get("openfda", {})
                active_ingredients = openfda.get("substance_name", [])
                
                for ingredient in active_ingredients:
                    nodes.append({
                        "id": ingredient,
                        "type": "ingredient",
                        "name": ingredient
                    })
                    
                    edges.append({
                        "source": drug_name,
                        "target": ingredient,
                        "type": "contains_ingredient"
                    })
                
                # Add adverse reactions
                adverse_reactions = drug_info.get("adverse_reactions", [])
                for reaction in adverse_reactions[:5]:  # Limit to first 5 reactions
                    reaction_id = f"{drug_name}_reaction_{len(nodes)}"
                    nodes.append({
                        "id": reaction_id,
                        "type": "adverse_reaction",
                        "name": reaction[:100] + "..." if len(reaction) > 100 else reaction
                    })
                    
                    edges.append({
                        "source": drug_name,
                        "target": reaction_id,
                        "type": "causes_reaction"
                    })
        
        return {"nodes": nodes, "edges": edges}

if __name__ == "__main__":
    # Example usage
    collector = FDACollector()
    
    # Search for drug labels
    labels_df = collector.search_drug_labels("active_ingredient:ibuprofen")
    print(f"Found {len(labels_df)} drug labels for ibuprofen")
    
    # Get adverse events
    events_df = collector.get_adverse_events("serious:1")
    print(f"Found {len(events_df)} serious adverse events")
    
    # Get device recalls
    recalls_df = collector.get_device_recalls()
    print(f"Found {len(recalls_df)} device recalls")
    
    # Get food enforcement data
    food_df = collector.get_food_enforcement()
    print(f"Found {len(food_df)} food enforcement records")
    
    # Build knowledge graph
    kg = collector.build_drug_knowledge_graph(["Advil", "Tylenol"])
    print(f"Knowledge graph contains:")
    print(f"- {len(kg['nodes'])} nodes")
    print(f"- {len(kg['edges'])} edges") 