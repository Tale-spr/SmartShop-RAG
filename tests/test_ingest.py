import unittest
from unittest.mock import patch

from smartshop_rag.rag import ingest


class IngestTestCase(unittest.TestCase):
    def test_main_returns_zero_when_no_failures(self):
        with patch("smartshop_rag.rag.ingest.get_knowledge_source_files", return_value=["a.txt", "b.pdf"]):
            with patch("smartshop_rag.rag.ingest.VectorStoreService") as mock_service_cls:
                mock_service_cls.return_value.load_document.return_value = {
                    "scanned": 2,
                    "loaded": 1,
                    "skipped": 1,
                    "failed": 0,
                }

                result = ingest.main([])

        self.assertEqual(result, 0)
        mock_service_cls.return_value.load_document.assert_called_once_with(reset=False)

    def test_main_returns_one_when_failures_exist(self):
        with patch("smartshop_rag.rag.ingest.get_knowledge_source_files", return_value=["a.txt"]):
            with patch("smartshop_rag.rag.ingest.VectorStoreService") as mock_service_cls:
                mock_service_cls.return_value.load_document.return_value = {
                    "scanned": 1,
                    "loaded": 0,
                    "skipped": 0,
                    "failed": 1,
                }

                result = ingest.main([])

        self.assertEqual(result, 1)

    def test_main_passes_reset_flag(self):
        with patch("smartshop_rag.rag.ingest.get_knowledge_source_files", return_value=["a.txt"]):
            with patch("smartshop_rag.rag.ingest.VectorStoreService") as mock_service_cls:
                mock_service_cls.return_value.load_document.return_value = {
                    "scanned": 1,
                    "loaded": 1,
                    "skipped": 0,
                    "failed": 0,
                }

                result = ingest.main(["--reset"])

        self.assertEqual(result, 0)
        mock_service_cls.return_value.load_document.assert_called_once_with(reset=True)


if __name__ == "__main__":
    unittest.main()

