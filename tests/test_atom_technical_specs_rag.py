"""
Automated tests for Kelvin6K ATOM Technical Specifications Extraction & RAG Pipeline.

Verifies:
1. Source PNG images preserved in docs/knowledge/robot_parts/technical_specs/source_images/
2. Structured JSON specification file integrity and three configuration schemas
3. RAG index ingestion of atom_technical_specifications.md
4. Query routing classification to Intent.KNOWLEDGE_QUERY for ATOM questions
5. Direct RAG retriever search responses matching exact source specifications
6. Live / mock FastAPI backend query endpoint response correctness
"""

import json
import os
import unittest
import urllib.request

from agent.query_router import Intent, classify
from agent.rag.retriever import get_store, index_summary, search_documentation


class TestAtomTechnicalSpecsRAG(unittest.TestCase):
    """Test suite for ATOM technical specifications extraction and retrieval."""

    @classmethod
    def setUpClass(cls):
        cls.store = get_store(refresh=True)

    def test_source_evidence_images_exist(self):
        """Source screenshots must be preserved and non-empty."""
        img_dir = os.path.join(
            "docs", "knowledge", "robot_parts", "technical_specs", "source_images"
        )
        img1 = os.path.join(img_dir, "atom_robot_details_table.png")
        img2 = os.path.join(img_dir, "atom_safety_connectivity_table_zoom.png")

        self.assertTrue(os.path.isfile(img1), f"Missing {img1}")
        self.assertTrue(os.path.isfile(img2), f"Missing {img2}")
        self.assertGreater(os.path.getsize(img1), 10000)
        self.assertGreater(os.path.getsize(img2), 10000)

    def test_structured_json_specification(self):
        """JSON specifications file must contain the three distinct configurations."""
        json_path = os.path.join(
            "docs", "knowledge", "robot_parts", "technical_specs", "atom_technical_specifications.json"
        )
        self.assertTrue(os.path.isfile(json_path), f"Missing {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["document_metadata"]["title"], "ATOM Technical Specifications")
        configs = data["document_metadata"]["configurations"]
        self.assertEqual(len(configs), 3)

        c1 = configs[0]
        self.assertEqual(c1["name"], "ATOM – Robot Only")
        c2 = configs[1]
        self.assertEqual(c2["name"], "ATOM – Robot + Rover Base")
        c3 = configs[2]
        self.assertEqual(c3["name"], "ATOM – Robot + Linear Axis")

        summaries = data["configuration_summaries"]
        self.assertEqual(summaries["ATOM – Robot Only"]["printer_weight"], "550kgs")
        self.assertEqual(summaries["ATOM – Robot Only"]["axes"], 3)
        self.assertEqual(summaries["ATOM – Robot Only"]["safety_fence"], "Optional")

        self.assertEqual(summaries["ATOM – Robot + Rover Base"]["printer_weight"], "700kgs")
        self.assertEqual(summaries["ATOM – Robot + Rover Base"]["axes"], 3)
        self.assertEqual(summaries["ATOM – Robot + Rover Base"]["safety_fence"], "yes")

        self.assertEqual(summaries["ATOM – Robot + Linear Axis"]["printer_weight"], "900kgs")
        self.assertEqual(summaries["ATOM – Robot + Linear Axis"]["axes"], 4)
        self.assertEqual(summaries["ATOM – Robot + Linear Axis"]["safety_fence"], "yes")

    def test_rag_index_contains_atom_specs(self):
        """RAG index must have atom_technical_specifications.md chunks."""
        summary = index_summary()
        self.assertIn("atom_technical_specifications.md", summary["documents"])
        self.assertGreaterEqual(summary["documents"]["atom_technical_specifications.md"], 15)

    def test_query_classification_routing(self):
        """Test queries must classify as Intent.KNOWLEDGE_QUERY."""
        q1 = "How many axes does the ATOM Linear Axis configuration have?"
        q2 = "What is the listed printer weight for the Rover Base configuration?"
        q3 = "What safety-fence information is listed for Robot Only?"
        q4 = "Compare the three configurations' equipment dimensions."

        intent1, _ = classify(q1)
        intent2, _ = classify(q2)
        intent3, _ = classify(q3)
        intent4, _ = classify(q4)

        self.assertEqual(intent1, Intent.KNOWLEDGE_QUERY)
        self.assertEqual(intent2, Intent.KNOWLEDGE_QUERY)
        self.assertEqual(intent3, Intent.KNOWLEDGE_QUERY)
        self.assertEqual(intent4, Intent.KNOWLEDGE_QUERY)

    def test_rag_retrieval_question1_axes(self):
        """Question 1: ATOM Linear Axis axes count retrieval."""
        res = search_documentation("How many axes does the ATOM Linear Axis configuration have?")
        self.assertTrue(res["found"])
        top_hit = res["results"][0]
        self.assertIn("atom_technical_specifications.md", top_hit["source"])
        self.assertIn("4 axes", top_hit["text"])
        self.assertGreaterEqual(top_hit["score"], 0.4)

    def test_rag_retrieval_question2_printer_weight(self):
        """Question 2: Rover Base printer weight retrieval."""
        res = search_documentation("What is the listed printer weight for the Rover Base configuration?")
        self.assertTrue(res["found"])
        top_hit = res["results"][0]
        self.assertIn("atom_technical_specifications.md", top_hit["source"])
        self.assertIn("700kgs", top_hit["text"])
        self.assertGreaterEqual(top_hit["score"], 0.4)

    def test_rag_retrieval_question3_safety_fence(self):
        """Question 3: Safety-fence information for Robot Only."""
        res = search_documentation("What safety-fence information is listed for Robot Only?")
        self.assertTrue(res["found"])
        top_hit = res["results"][0]
        self.assertIn("atom_technical_specifications.md", top_hit["source"])
        self.assertIn("Optional", top_hit["text"])
        self.assertGreaterEqual(top_hit["score"], 0.4)

    def test_rag_retrieval_question4_equipment_dimensions(self):
        """Question 4: Equipment dimensions comparison across all 3 configurations."""
        res = search_documentation("Compare the three configurations' equipment dimensions.")
        self.assertTrue(res["found"])
        top_hit = res["results"][0]
        self.assertIn("atom_technical_specifications.md", top_hit["source"])
        self.assertIn("3 × 0.7 × 3", top_hit["text"])
        self.assertIn("3 × 1.2 × 3.5", top_hit["text"])
        self.assertIn("3 × 0.7 × 3.5", top_hit["text"])
        self.assertGreaterEqual(top_hit["score"], 0.4)

    def test_api_server_query_endpoint(self):
        """Verify API POST /api/assistant/query answers technical specification questions."""
        payload = json.dumps({
            "question": "How many axes does the ATOM Linear Axis configuration have?"
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/assistant/query",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["intent"], "KNOWLEDGE_QUERY")
                self.assertTrue(any("atom_technical_specifications.md" in src for src in data["knowledge_sources"]))
                self.assertIn("4 axes", data["answer"])
        except urllib.error.URLError as e:
            self.skipTest(f"FastAPI backend is not running at 127.0.0.1:8000: {e}")


if __name__ == "__main__":
    unittest.main()
