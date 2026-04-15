import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from smartshop_rag.model.responses_chat import QwenResponsesChatModel


class ResponsesChatModelTestCase(unittest.TestCase):
    def test_serialize_messages_maps_roles(self):
        messages = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
        ]
        serialized = QwenResponsesChatModel._serialize_messages(messages)
        self.assertEqual(
            serialized,
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        )

    def test_extract_output_text_prefers_output_text(self):
        data = {"output_text": "answer", "output": []}
        self.assertEqual(QwenResponsesChatModel._extract_output_text(data), "answer")

    def test_extract_output_text_falls_back_to_message_content(self):
        data = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "first"}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "second"}],
                },
            ]
        }
        self.assertEqual(QwenResponsesChatModel._extract_output_text(data), "first\nsecond")

    @patch("smartshop_rag.model.responses_chat.requests.post")
    def test_generate_calls_responses_endpoint(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"id": "resp_1", "output_text": "done", "usage": {"total_tokens": 12}}
        mock_post.return_value = response

        model = QwenResponsesChatModel(model="qwen3.5-plus", api_key="sk-test")
        result = model.generate([[HumanMessage(content="你好")]])

        self.assertEqual(result.generations[0][0].message.content, "done")
        self.assertEqual(mock_post.call_args.args[0], "https://dashscope.aliyuncs.com/compatible-mode/v1/responses")
        self.assertEqual(mock_post.call_args.kwargs["json"]["model"], "qwen3.5-plus")
        self.assertEqual(
            mock_post.call_args.kwargs["json"]["input"],
            [{"role": "user", "content": "你好"}],
        )


if __name__ == "__main__":
    unittest.main()
