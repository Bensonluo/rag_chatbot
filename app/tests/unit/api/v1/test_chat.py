"""Tests for chat API endpoints"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient


class TestChatEndpoints:
    """Test chat API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from app.main import create_app

        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_chat_service(self):
        """Mock chat service"""
        service = Mock()
        service.process_message = AsyncMock(
            return_value=Mock(
                content="Hello! How can I help?",
                session_id=1,
                intent="greeting",
                sources=None,
                metadata={"tokens": 20}
            )
        )
        service.get_chat_history = AsyncMock(return_value=[])
        service.clear_chat_history = AsyncMock()
        return service

    def test_chat_endpoint(self, client, mock_chat_service):
        """Test POST /chat endpoint"""
        # Arrange
        with patch("app.api.v1.chat.get_chat_service", return_value=mock_chat_service):
            request_data = {
                "message": "Hello",
                "session_id": 1,
            }

            # Act
            response = client.post("/api/v1/chat", json=request_data)

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["content"] == "Hello! How can I help?"
            assert data["session_id"] == 1
            assert data["intent"] == "greeting"
            mock_chat_service.process_message.assert_called_once()

    def test_chat_endpoint_with_user_id(self, client, mock_chat_service):
        """Test chat endpoint with user_id"""
        # Arrange
        with patch("app.api.v1.chat.get_chat_service", return_value=mock_chat_service):
            request_data = {
                "message": "Hello",
                "session_id": 1,
                "user_id": 1,
            }

            # Act
            response = client.post("/api/v1/chat", json=request_data)

            # Assert
            assert response.status_code == 200
            mock_chat_service.process_message.assert_called_once()

    def test_chat_endpoint_validation_error(self, client):
        """Test chat endpoint with invalid request"""
        # Arrange
        request_data = {
            # Missing "message" field
            "session_id": 1,
        }

        # Act
        response = client.post("/api/v1/chat", json=request_data)

        # Assert
        assert response.status_code == 422  # Validation error

    def test_chat_history_endpoint(self, client, mock_chat_service):
        """Test GET /chat/history endpoint"""
        # Arrange
        mock_chat_service.get_chat_history = AsyncMock(
            return_value=[
                Mock(role="user", content="Hello"),
                Mock(role="assistant", content="Hi there!"),
            ]
        )

        with patch("app.api.v1.chat.get_chat_service", return_value=mock_chat_service):
            # Act
            response = client.get("/api/v1/chat/history?session_id=1")

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert "messages" in data
            assert len(data["messages"]) == 2
            mock_chat_service.get_chat_history.assert_called_once_with(session_id=1)

    def test_chat_history_with_limit(self, client, mock_chat_service):
        """Test chat history with limit parameter"""
        # Arrange
        mock_chat_service.get_chat_history = AsyncMock(return_value=[])

        with patch("app.api.v1.chat.get_chat_service", return_value=mock_chat_service):
            # Act
            response = client.get("/api/v1/chat/history?session_id=1&limit=10")

            # Assert
            assert response.status_code == 200
            mock_chat_service.get_chat_history.assert_called_once_with(
                session_id=1,
                limit=10,
            )

    def test_chat_history_missing_session_id(self, client):
        """Test chat history without session_id"""
        # Act
        response = client.get("/api/v1/chat/history")

        # Assert
        assert response.status_code == 422  # Validation error

    def test_clear_chat_history_endpoint(self, client, mock_chat_service):
        """Test DELETE /chat/history endpoint"""
        # Arrange
        with patch("app.api.v1.chat.get_chat_service", return_value=mock_chat_service):
            # Act
            response = client.delete("/api/v1/chat/history?session_id=1")

            # Assert
            assert response.status_code == 204
            mock_chat_service.clear_chat_history.assert_called_once_with(session_id=1)

    def test_chat_stream_endpoint(self, client, mock_chat_service):
        """Test streaming chat endpoint"""
        # Arrange
        async def mock_stream():
            yield "Hello"
            yield " there"
            yield "!"

        mock_chat_service.process_message_stream = AsyncMock(return_value=mock_stream())

        with patch("app.api.v1.chat.get_chat_service", return_value=mock_chat_service):
            request_data = {
                "message": "Hello",
                "session_id": 1,
            }

            # Act
            response = client.post("/api/v1/chat/stream", json=request_data)

            # Assert
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            # Consume stream
            chunks = []
            for line in response.iter_lines():
                if line:
                    chunks.append(line.decode())

            # Verify streaming chunks
            assert len(chunks) >= 3

    def test_chat_endpoint_with_retrieval(self, client, mock_chat_service):
        """Test chat with retrieval sources"""
        # Arrange
        mock_chat_service.process_message = AsyncMock(
            return_value=Mock(
                content="Python is a programming language",
                session_id=1,
                intent="question",
                sources=["doc1", "doc2"],
                metadata={"tokens": 50}
            )
        )

        with patch("app.api.v1.chat.get_chat_service", return_value=mock_chat_service):
            request_data = {
                "message": "What is Python?",
                "session_id": 1,
            }

            # Act
            response = client.post("/api/v1/chat", json=request_data)

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert "sources" in data
            assert len(data["sources"]) == 2
            assert data["sources"][0] == "doc1"

    def test_chat_endpoint_error_handling(self, client, mock_chat_service):
        """Test chat endpoint error handling"""
        # Arrange
        from app.core.exceptions import BaseServiceError

        mock_chat_service.process_message = AsyncMock(
            side_effect=BaseServiceError("Processing failed")
        )

        with patch("app.api.v1.chat.get_chat_service", return_value=mock_chat_service):
            request_data = {
                "message": "Hello",
                "session_id": 1,
            }

            # Act
            response = client.post("/api/v1/chat", json=request_data)

            # Assert
            assert response.status_code == 500
            data = response.json()
            assert "error" in data or "detail" in data

    def test_chat_endpoint_unauthorized(self, client):
        """Test chat endpoint without authentication"""
        # Arrange
        request_data = {
            "message": "Hello",
            "session_id": 1,
        }

        # Act
        response = client.post("/api/v1/chat", json=request_data)

        # Note: This will depend on auth configuration
        # If auth is disabled, will return 200
        # If auth is enabled, will return 401
        assert response.status_code in [200, 401]


class TestChatSchemas:
    """Test chat request/response schemas"""

    def test_chat_request_schema(self):
        """Test ChatRequest schema validation"""
        # Arrange
        from app.api.v1.chat import ChatRequest

        # Act
        request = ChatRequest(
            message="Hello",
            session_id=1,
            user_id=1,
            max_tokens=100,
        )

        # Assert
        assert request.message == "Hello"
        assert request.session_id == 1
        assert request.user_id == 1
        assert request.max_tokens == 100

    def test_chat_request_optional_fields(self):
        """Test ChatRequest with optional fields"""
        # Arrange
        from app.api.v1.chat import ChatRequest

        # Act
        request = ChatRequest(
            message="Hello",
            session_id=1,
        )

        # Assert
        assert request.message == "Hello"
        assert request.user_id is None
        assert request.max_tokens is None

    def test_chat_response_schema(self):
        """Test ChatResponse schema"""
        # Arrange
        from app.api.v1.chat import ChatResponse

        # Act
        response = ChatResponse(
            content="Hello!",
            session_id=1,
            intent="greeting",
            sources=None,
            metadata={"tokens": 20}
        )

        # Assert
        assert response.content == "Hello!"
        assert response.session_id == 1
        assert response.intent == "greeting"

    def test_chat_history_response_schema(self):
        """Test ChatHistoryResponse schema"""
        # Arrange
        from app.api.v1.chat import ChatHistoryResponse, ChatMessageResponse

        # Act
        response = ChatHistoryResponse(
            messages=[
                ChatMessageResponse(
                    role="user",
                    content="Hello",
                ),
                ChatMessageResponse(
                    role="assistant",
                    content="Hi there!",
                ),
            ]
        )

        # Assert
        assert len(response.messages) == 2
        assert response.messages[0].role == "user"
        assert response.messages[1].role == "assistant"
