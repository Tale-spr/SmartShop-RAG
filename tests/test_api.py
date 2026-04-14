import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from smartshop_rag.api.main import create_app
from smartshop_rag.services.chat_service import ChatServiceError


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    @patch("smartshop_rag.api.main.get_dependency_issues", return_value=[])
    def test_health_returns_healthy(self, mock_get_dependency_issues):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    @patch("smartshop_rag.api.main.get_dependency_issues", return_value=["未配置 DASHSCOPE_API_KEY"])
    def test_health_returns_unhealthy(self, mock_get_dependency_issues):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unhealthy")
        self.assertIn("missing_dependencies", response.json())

    @patch("smartshop_rag.api.main.run_chat")
    @patch("smartshop_rag.api.main.get_dependency_issues", return_value=[])
    def test_chat_returns_answer(self, mock_get_dependency_issues, mock_run_chat):
        mock_run_chat.return_value = {
            "user_id": "1001",
            "session_id": "session_001",
            "answer": "这款商品支持七天无理由，具体以页面规则为准。",
            "status_events": [
                {
                    "event_type": "stage.rag",
                    "title": "正在检索知识库",
                    "detail": "正在召回电商客服相关知识并整理参考证据",
                    "created_at": "2025-01-01T10:00:00",
                    "level": "info",
                }
            ],
            "session_summary": "摘要",
        }

        response = self.client.post("/chat", json={"user_id": "1001", "message": "你好"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "这款商品支持七天无理由，具体以页面规则为准。")

    @patch("smartshop_rag.api.main.get_dependency_issues", return_value=["本地向量库不存在"])
    def test_chat_returns_dependency_error_when_not_ready(self, mock_get_dependency_issues):
        response = self.client.post("/chat", json={"user_id": "1001", "message": "你好"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "dependency_not_ready")

    @patch("smartshop_rag.api.main.get_dependency_issues", return_value=[])
    @patch("smartshop_rag.api.main.run_chat", side_effect=ChatServiceError("指定会话不存在", code="session_not_found", status_code=404))
    def test_chat_returns_service_error(self, mock_run_chat, mock_get_dependency_issues):
        response = self.client.post("/chat", json={"user_id": "1001", "message": "你好"})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "session_not_found")

    def test_chat_returns_422_when_missing_required_fields(self):
        response = self.client.post("/chat", json={"user_id": "1001"})

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()

