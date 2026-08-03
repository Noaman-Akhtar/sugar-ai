import unittest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from app.ai import RAGAgent

class TestRAGAgentRelevance(unittest.TestCase):
    def setUp(self):
        # Create a mock provider
        self.mock_provider = MagicMock()
        self.mock_provider.get_model_name.return_value = "mock-model"
        self.mock_provider.get_eos_token.return_value = None
        
        # Initialize RAGAgent with mock provider
        self.agent = RAGAgent(provider=self.mock_provider)
        
        # Mock retriever and its underlying vector store
        self.mock_retriever = MagicMock()
        self.mock_vector_store = MagicMock()
        self.mock_retriever.vector_store = self.mock_vector_store
        self.mock_retriever.search_kwargs = {"k": 3}
        self.agent.retriever = self.mock_retriever

    def test_get_relevant_document_above_threshold(self):
        # Set up mock search results returning a score above threshold (0.5)
        doc = Document(page_content="Sugar Labs makes Sugar-AI.", metadata={"source": "test.txt"})
        self.mock_vector_store.similarity_search_with_relevance_scores.return_value = [
            (doc, 0.8)
        ]
        
        result_doc, score = self.agent.get_relevant_document("What is Sugar-AI?", threshold=0.5)
        
        self.assertIsNotNone(result_doc)
        self.assertEqual(result_doc.page_content, "Sugar Labs makes Sugar-AI.")
        self.assertEqual(score, 0.8)
        self.mock_vector_store.similarity_search_with_relevance_scores.assert_called_once_with(
            "What is Sugar-AI?",
            k=3
        )

    def test_get_relevant_document_below_threshold(self):
        # Set up mock search results returning a score below threshold
        doc = Document(page_content="Irrelevant info.", metadata={"source": "test.txt"})
        self.mock_vector_store.similarity_search_with_relevance_scores.return_value = [
            (doc, 0.3)
        ]
        
        result_doc, score = self.agent.get_relevant_document("What is Sugar-AI?", threshold=0.5)
        
        self.assertIsNone(result_doc)
        self.assertEqual(score, 0.0)

    def test_get_relevant_document_empty_results(self):
        # Set up mock search returning empty results
        self.mock_vector_store.similarity_search_with_relevance_scores.return_value = []
        
        result_doc, score = self.agent.get_relevant_document("What is Sugar-AI?", threshold=0.5)
        
        self.assertIsNone(result_doc)
        self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main()
