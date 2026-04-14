import unittest
from unittest.mock import patch

from smartshop_rag.model.factory import create_chat_model, create_embedding_model, get_chat_model_name, get_embedding_model_name


class ModelFactoryTestCase(unittest.TestCase):
    def test_get_chat_model_name_returns_configured_role(self):
        self.assertEqual(get_chat_model_name('primary_chat'), 'qwen-plus')
        self.assertEqual(get_chat_model_name('rag_chat'), 'qwen-plus')
        self.assertEqual(get_chat_model_name('rewrite_chat'), 'qwen-flash')
        self.assertEqual(get_chat_model_name('rerank_chat'), 'qwen-flash')

    def test_get_chat_model_name_raises_for_unknown_role(self):
        with self.assertRaises(ValueError):
            get_chat_model_name('unknown_role')

    def test_get_embedding_model_name_returns_configured_embedding(self):
        self.assertEqual(get_embedding_model_name(), 'text-embedding-v4')

    def test_create_chat_model_uses_role_based_config(self):
        with patch('smartshop_rag.model.factory.ChatTongyi') as mock_chat:
            create_chat_model(role='rag_chat')
        self.assertEqual(mock_chat.call_args.kwargs['model'], 'qwen-plus')

    def test_create_embedding_model_uses_config(self):
        with patch('smartshop_rag.model.factory.DashScopeEmbeddings') as mock_embedding:
            create_embedding_model()
        self.assertEqual(mock_embedding.call_args.kwargs['model'], 'text-embedding-v4')


if __name__ == '__main__':
    unittest.main()
