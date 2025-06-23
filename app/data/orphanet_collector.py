import requests
import logging
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class OrphanetDisease:
    """Class to represent an Orphanet disease entry with comprehensive metadata."""

    def __init__(
        self,
        disease_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        prevalence: Optional[str] = None,
        inheritance: Optional[List[str]] = None,
        age_of_onset: Optional[List[str]] = None,
        icd10_codes: Optional[List[str]] = None,
        omim_ids: Optional[List[str]] = None,
        medical_specialties: Optional[List[str]] = None,
        phenotypes: Optional[List[Dict[str, Any]]] = None
    ):
        self.disease_id = disease_id
        self.name = name
        self.metadata = metadata or {}
        self.prevalence = prevalence
        self.inheritance = inheritance or []
        self.age_of_onset = age_of_onset or []
        self.icd10_codes = icd10_codes or []
        self.omim_ids = omim_ids or []
        self.medical_specialties = medical_specialties or []
        self.phenotypes = phenotypes or []

    @classmethod
    def from_api(cls, disease_id: str, api_key: Optional[str] = None) -> 'OrphanetDisease':
        """
        Create a disease instance from Orphanet API data using the correct endpoint and response structure.
        
        Args:
            disease_id: Orphanet disease ID (OrphaCode, e.g., '558')
            api_key: Optional API key (if not provided, will look for ORPHANET_API_KEY env var)
            
        Returns:
            OrphanetDisease instance
            
        Raises:
            ValueError: If disease data cannot be found
            requests.RequestException: If API request fails
        """
        api_key = api_key or os.getenv('ORPHANET_API_KEY')
        if not api_key:
            raise ValueError("Orphanet API key is required. Set ORPHANET_API_KEY environment variable or provide api_key parameter.")
        
        base_url = "https://api.orphadata.com"
        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Accept": "application/json"
        }
        
        # Fetch basic disease data using the correct endpoint
        response = requests.get(
            f"{base_url}/rd-cross-referencing/orphacodes/{disease_id}?lang=en",
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        
        # Log the full response for debugging
        logger.info(f"Orphanet API Response for disease {disease_id}: {json.dumps(data, indent=2)}")
        
        # The response structure is: { "data": { ... } }
        disease_data = data.get("data")
        if not disease_data or not isinstance(disease_data, dict):
            raise ValueError(f"No disease data found for ID {disease_id}")
        
        # Log the disease data for debugging
        logger.info(f"Disease data extracted: {json.dumps(disease_data, indent=2)}")
        
        # Extract name from the correct path in the response
        # Prefer 'Preferred term' if available
        preferred_term = None
        if disease_data.get('results') and 'Preferred term' in disease_data['results']:
            preferred_term = disease_data['results']['Preferred term']
        name = preferred_term or disease_data.get("name", {}).get("text", "Unknown Disease")
        orpha_code = disease_data.get("orphacode", disease_id)
        prevalence = disease_data.get("prevalence")
        inheritance = disease_data.get("inheritance", [])
        age_of_onset = disease_data.get("ageOfOnset", [])
        icd10_codes = disease_data.get("icd10Codes", [])
        omim_ids = disease_data.get("omimIds", [])
        
        logger.info(f"Extracted disease name: {name}")
        
        # Fetch phenotypes
        phenotypes = []
        try:
            pheno_response = requests.get(
                f"{base_url}/rd-phenotypes/orphacodes/{disease_id}?lang=en",
                headers=headers
            )
            pheno_response.raise_for_status()
            pheno_data = pheno_response.json()
            if pheno_data and "data" in pheno_data and "phenotypes" in pheno_data["data"]:
                phenotypes = pheno_data["data"]["phenotypes"]
                logger.info(f"Found {len(phenotypes)} phenotypes")
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch phenotypes: {e}")
        
        # Fetch medical specialties
        specialties = []
        try:
            spec_response = requests.get(
                f"{base_url}/rd-medical-specialties/orphacodes/{disease_id}?lang=en",
                headers=headers
            )
            spec_response.raise_for_status()
            spec_data = spec_response.json()
            if spec_data and "data" in spec_data and "medicalSpecialties" in spec_data["data"]:
                specialties = [s.get("name") for s in spec_data["data"]["medicalSpecialties"]]
                logger.info(f"Found {len(specialties)} medical specialties")
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch medical specialties: {e}")
        
        # Create metadata dictionary
        metadata = {
            "source": "Orphanet API",
            "raw_data": data,
            "last_updated": datetime.now().isoformat()
        }
        
        return cls(
            disease_id=orpha_code,
            name=name,
            prevalence=prevalence,
            inheritance=inheritance,
            age_of_onset=age_of_onset,
            icd10_codes=icd10_codes,
            omim_ids=omim_ids,
            phenotypes=phenotypes,
            medical_specialties=specialties,
            metadata=metadata
        )

    @classmethod
    def search_by_name(cls, name: str, api_key: Optional[str] = None) -> List['OrphanetDisease']:
        """Search diseases by name. If no results, try to get the OrphaCode from the /names/{name} endpoint and fetch by code."""
        if not api_key:
            api_key = os.getenv('ORPHANET_API_KEY')

        base_url = "https://api.orphadata.com"
        headers = {
            "Accept": "application/json",
            "Ocp-Apim-Subscription-Key": api_key
        }

        try:
            response = requests.get(
                f"{base_url}/rd-cross-referencing/orphacodes/names/{name}?lang=en",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            # Try to extract ORPHAcode from the response
            results = data.get("data", {}).get("results", {})
            orpha_code = results.get("ORPHAcode")
            if orpha_code:
                # Fetch full disease info by code
                disease = cls.from_api(str(orpha_code), api_key)
                return [disease]
            # Fallback to old logic if 'diseases' list exists
            diseases = []
            for disease_data in data.get("diseases", []):
                disease = cls.from_api(disease_data["orphacode"], api_key)
                diseases.append(disease)
            return diseases
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search Orphanet diseases by name {name}: {str(e)}")
            return []

    @classmethod
    def search_by_icd11(cls, icd11_code: str, api_key: Optional[str] = None) -> List['OrphanetDisease']:
        """Search diseases by ICD-11 code."""
        if not api_key:
            api_key = os.getenv('ORPHANET_API_KEY')

        base_url = "https://api.orphadata.com"
        headers = {
            "Accept": "application/json",
            "Ocp-Apim-Subscription-Key": api_key
        }

        try:
            response = requests.get(
                f"{base_url}/rd-cross-referencing/icd-11s/{icd11_code}?lang=en",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            diseases = []
            for disease_data in data.get("diseases", []):
                disease = cls.from_api(disease_data["orphacode"], api_key)
                diseases.append(disease)
            
            return diseases
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search Orphanet diseases by ICD-11 code {icd11_code}: {str(e)}")
            raise

    @classmethod
    def search_by_omim(cls, omim_code: str, api_key: Optional[str] = None) -> List['OrphanetDisease']:
        """Search diseases by OMIM code."""
        if not api_key:
            api_key = os.getenv('ORPHANET_API_KEY')

        base_url = "https://api.orphadata.com"
        headers = {
            "Accept": "application/json",
            "Ocp-Apim-Subscription-Key": api_key
        }

        try:
            response = requests.get(
                f"{base_url}/rd-cross-referencing/omim-codes/{omim_code}?lang=en",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            diseases = []
            for disease_data in data.get("diseases", []):
                disease = cls.from_api(disease_data["orphacode"], api_key)
                diseases.append(disease)
            
            return diseases
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search Orphanet diseases by OMIM code {omim_code}: {str(e)}")
            raise

    @staticmethod
    def _fetch_phenotypes(disease_id: str, api_key: str) -> List[Dict[str, Any]]:
        """Fetch phenotypes for a disease."""
        base_url = "https://api.orphadata.com"
        headers = {
            "Accept": "application/json",
            "Ocp-Apim-Subscription-Key": api_key
        }

        try:
            response = requests.get(
                f"{base_url}/rd-phenotypes/orphacodes/{disease_id}?lang=en",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            return data.get("phenotypes", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch phenotypes for disease {disease_id}: {str(e)}")
            return []

    @staticmethod
    def _fetch_medical_specialties(disease_id: str, api_key: str) -> List[str]:
        """Fetch medical specialties for a disease."""
        base_url = "https://api.orphadata.com"
        headers = {
            "Accept": "application/json",
            "Ocp-Apim-Subscription-Key": api_key
        }

        try:
            response = requests.get(
                f"{base_url}/rd-medical-specialties/orphacodes/{disease_id}?lang=en",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            return [specialty["name"] for specialty in data.get("medicalSpecialties", [])]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch medical specialties for disease {disease_id}: {str(e)}")
            return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert the disease instance to a dictionary representation."""
        return {
            "disease_id": self.disease_id,
            "name": self.name,
            "metadata": self.metadata,
            "prevalence": self.prevalence,
            "inheritance": self.inheritance,
            "age_of_onset": self.age_of_onset,
            "icd10_codes": self.icd10_codes,
            "omim_ids": self.omim_ids,
            "medical_specialties": self.medical_specialties,
            "phenotypes": self.phenotypes
        }

    def __str__(self) -> str:
        """String representation of the disease."""
        return f"OrphanetDisease(id={self.disease_id}, name={self.name})"

    def __repr__(self) -> str:
        """Detailed string representation of the disease."""
        return (
            f"OrphanetDisease("
            f"disease_id='{self.disease_id}', "
            f"name='{self.name}', "
            f"prevalence='{self.prevalence}', "
            f"inheritance={self.inheritance}, "
            f"age_of_onset={self.age_of_onset}, "
            f"icd10_codes={self.icd10_codes}, "
            f"omim_ids={self.omim_ids}, "
            f"medical_specialties={self.medical_specialties}"
            f")"
        ) 