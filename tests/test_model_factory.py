import unittest
from unittest.mock import patch

from smartshop_rag.model.factory import (
    create_chat_model,
    create_embedding_model,
    get_chat_model_name,
    get_embedding_model_name,
    is_responses_api_model,
)
from smartshop_rag.model.responses_chat import QwenResponsesChatModel


class ModelFactoryTestCase(unittest.TestCase):
    def test_get_chat_model_name_returns_configured_role(self):
        self.assertEqual(get_chat_model_name('primary_chat'), 'qwen3.6-plus')
        self.assertEqual(get_chat_model_name('smalltalk_chat'), 'qwen3.5-flash')
        self.assertEqual(get_chat_model_name('rag_chat'), 'qwen3.6-plus')
        self.assertEqual(get_chat_model_name('rewrite_chat'), 'qwen3.6-plus')
        self.assertEqual(get_chat_model_name('rerank_chat'), 'qwen3.6-plus')
        self.assertEqual(get_chat_model_name('eval_chat'), 'qwen3.6-plus')

    def test_get_chat_model_name_raises_for_unknown_role(self):
        with self.assertRaises(ValueError):
            get_chat_model_name('unknown_role')

    def test_get_embedding_model_name_returns_configured_embedding(self):
        self.assertEqual(get_embedding_model_name(), 'text-embedding-v4')

    def test_create_chat_model_uses_responses_backend_for_qwen36_plus(self):
        model = create_chat_model(role='rag_chat')
        self.assertIsInstance(model, QwenResponsesChatModel)
        self.assertEqual(model.model, 'qwen3.6-plus')
        self.assertFalse(model.enable_thinking)

    def test_create_chat_model_uses_responses_backend_for_smalltalk_flash(self):
        model = create_chat_model(role='smalltalk_chat')
        self.assertIsInstance(model, QwenResponsesChatModel)
        self.assertEqual(model.model, 'qwen3.5-flash')
        self.assertFalse(model.enable_thinking)

    def test_create_chat_model_keeps_primary_chat_default_thinking_behavior(self):
        model = create_chat_model(role='primary_chat')
        self.assertIsInstance(model, QwenResponsesChatModel)
        self.assertEqual(model.model, 'qwen3.6-plus')
        self.assertIsNone(model.enable_thinking)

    def test_create_chat_model_uses_chat_tongyi_for_legacy_models(self):
        with patch('smartshop_rag.model.factory.ChatTongyi') as mock_chat:
            create_chat_model(model_name='qwen-turbo', role='rag_chat')
        self.assertEqual(mock_chat.call_args.kwargs['model'], 'qwen-turbo')

    def test_is_responses_api_model_matches_qwen35_and_qwen36_plus(self):
        self.assertTrue(is_responses_api_model('qwen3.5-flash'))
        self.assertTrue(is_responses_api_model('qwen3.5-plus'))
        self.assertTrue(is_responses_api_model('qwen3.5-plus-2026-02-15'))
        self.assertTrue(is_responses_api_model('qwen3.6-plus'))
        self.assertTrue(is_responses_api_model('qwen3.6-plus-2026-04-02'))
        self.assertFalse(is_responses_api_model('qwen-turbo'))
        self.assertFalse(is_responses_api_model('qwen-plus'))

    def test_create_embedding_model_uses_config(self):
        with patch('smartshop_rag.model.factory.DashScopeEmbeddings') as mock_embedding:
            create_embedding_model()
        self.assertEqual(mock_embedding.call_args.kwargs['model'], 'text-embedding-v4')


if __name__ == '__main__':
    unittest.main()
