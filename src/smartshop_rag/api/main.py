from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from smartshop_rag.api.dependencies import build_dependency_error_response, get_dependency_issues, initialize_api_environment
from smartshop_rag.api.schemas import ChatRequest, ChatResponse, ErrorResponse, HealthResponse
from smartshop_rag.services.chat_service import ChatServiceError, run_chat
from smartshop_rag.utils.logger_handler import logger


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    initialize_api_environment()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="SmartShop-RAG API",
        version="0.1.0",
        description="电商客服混合检索 RAG API 服务",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health_check() -> HealthResponse:
        issues = get_dependency_issues()
        return HealthResponse(
            status="healthy" if not issues else "unhealthy",
            vector_store_ready=not any("向量库" in issue for issue in issues),
            dependencies_ready=not issues,
            missing_dependencies=issues or None,
        )

    @app.post(
        "/chat",
        response_model=ChatResponse,
        responses={503: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    def chat(request: ChatRequest):
        issues = get_dependency_issues()
        if issues:
            return build_dependency_error_response()

        try:
            result = run_chat(
                user_id=request.user_id,
                message=request.message,
                session_id=request.session_id,
            )
        except ChatServiceError as exc:
            return _build_service_error_response(exc)

        return ChatResponse(
            user_id=result["user_id"],
            session_id=result["session_id"],
            answer=result["answer"],
            status_events=result["status_events"],
            session_summary=result["session_summary"],
        )

    return app


def _build_service_error_response(exc: ChatServiceError) -> JSONResponse:
    logger.warning(f"[API]请求处理失败: {exc.code} | {str(exc)}")
    payload = ErrorResponse(code=exc.code, message=str(exc))
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


app = create_app()
