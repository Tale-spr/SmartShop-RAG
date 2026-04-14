import shutil
import unittest
import uuid
from pathlib import Path

from smartshop_rag.services.chat_service import run_chat


class DummyAgent:
    def __init__(self, answer: str):
        self.answer = answer
        self.seen_context = None

    def execute(self, query: str, runtime_context: dict):
        self.seen_context = runtime_context
        return self.answer


class ChatServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path("tests") / ".tmp" / f"chat_service_{uuid.uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def test_run_chat_creates_session_and_returns_answer(self):
        from smartshop_rag.services import session_service

        original = session_service.get_session_store_dir
        session_service.get_session_store_dir = lambda base_dir=None: self.base_dir if base_dir is None else original(base_dir)
        try:
            agent = DummyAgent("这是客服回答")
            result = run_chat(user_id="demo_user", message="商品支持保修吗？", agent=agent)
        finally:
            session_service.get_session_store_dir = original

        self.assertEqual(result["answer"], "这是客服回答")
        self.assertEqual(result["user_id"], "demo_user")
        self.assertIn("session_id", result)
        self.assertEqual(result["session_data"]["messages"][-1]["content"], "这是客服回答")
        self.assertIsNotNone(agent.seen_context)
        self.assertEqual(agent.seen_context["user_id"], "demo_user")


if __name__ == "__main__":
    unittest.main()
