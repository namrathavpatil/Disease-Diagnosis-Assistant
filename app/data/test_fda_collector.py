import unittest
import pandas as pd
from app.data.fda_collector import FDACollector

class TestFDACollector(unittest.TestCase):
    def setUp(self):
        self.collector = FDACollector()

    def test_search_drug_labels(self):
        result = self.collector.search_drug_labels("active_ingredient:ibuprofen", limit=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_get_adverse_events(self):
        result = self.collector.get_adverse_events(limit=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_get_device_recalls(self):
        result = self.collector.get_device_recalls(limit=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_get_food_enforcement(self):
        result = self.collector.get_food_enforcement(limit=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_search_drugs_by_ingredient(self):
        result = self.collector.search_drugs_by_ingredient("acetaminophen", limit=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_get_drug_interactions(self):
        result = self.collector.get_drug_interactions("Advil")
        self.assertIsInstance(result, pd.DataFrame)

    def test_build_drug_knowledge_graph(self):
        result = self.collector.build_drug_knowledge_graph(["Advil"])
        self.assertIsInstance(result, dict)
        self.assertIn("nodes", result)
        self.assertIn("edges", result)
        self.assertGreater(len(result["nodes"]), 0)

if __name__ == "__main__":
    unittest.main() 