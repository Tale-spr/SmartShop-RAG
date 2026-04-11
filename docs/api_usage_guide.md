# API 使用说明

当前项目保留两个接口：

- `GET /health`
- `POST /chat`

## `GET /health`

用于检查运行时依赖是否就绪，包括：

- `DASHSCOPE_API_KEY`
- 本地向量库是否存在
- 主提示词与 RAG 提示词是否存在

示例响应：

```json
{
  "status": "healthy",
  "vector_store_ready": true,
  "dependencies_ready": true,
  "missing_dependencies": null
}
```

## `POST /chat`

用于发起一轮电商客服问答。

请求体：

```json
{
  "user_id": "demo_user",
  "message": "这款商品支持七天无理由吗？",
  "session_id": null
}
```

响应体：

```json
{
  "user_id": "demo_user",
  "session_id": "20260411_120000_000001",
  "answer": "根据当前检索到的资料，……",
  "status_events": [
    {
      "event_type": "stage.rag",
      "title": "正在检索知识库",
      "detail": "正在召回电商客服相关知识并整理参考证据",
      "created_at": "2026-04-11T12:00:00",
      "level": "info"
    }
  ],
  "session_summary": "最近用户关注: 这款商品支持七天无理由吗？"
}
```

说明：

- `user_id` 只用于隔离会话空间
- `session_id` 为空时会自动创建新会话
- 当前接口只返回最终回答与可见状态事件，不返回完整检索 trace
