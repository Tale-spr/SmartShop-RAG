import unittest
from unittest.mock import Mock

from smartshop_rag.agent.tools.agent_tools import create_agent_tools


class AgentToolFactoryTestCase(unittest.TestCase):
    def test_create_agent_tools_builds_rag_tool_from_service(self):
        rag_service = Mock()
        rag_service.rag_summarize.return_value = "ok"

        tools = create_agent_tools(rag_service)
        rag_tool = tools[0]
        result = rag_tool.invoke({"query": "测试问题"})

        rag_service.rag_summarize.assert_called_once_with("测试问题")
        self.assertEqual(result, "ok")

    def test_create_agent_tools_contains_expected_tool_count(self):
        tools = create_agent_tools(Mock())

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "rag_summarize")


if __name__ == "__main__":
    unittest.main()

