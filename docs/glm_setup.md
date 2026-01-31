# GLM (Zhipu AI) Setup Guide

This guide explains how to configure and use GLM (Zhipu AI) models with the RAG chatbot.

## Prerequisites

1. **Get GLM API Key**:
   - Visit [Zhipu AI Open Platform](https://open.bigmodel.cn/)
   - Register for an account
   - Generate your API key from the console

2. **Available GLM Models**:
   - `glm-4-plus` - Most capable model (128k context)
   - `glm-4.5-air` - Efficient model, recommended balance (128k context) ⭐
   - `glm-4-0520` - GLM-4 version from May 2020
   - `glm-4` - Standard GLM-4 model
   - `glm-4-air` - Efficient model
   - `glm-4-airx` - Enhanced efficient model
   - `glm-4-flash` - Fastest model
   - `glm-3-turbo` - Legacy turbo model (128k context)

## Configuration

### 1. Environment Variables

Add the following to your `.env` file:

```bash
# GLM Configuration
GLM_API_KEY=your-glm-api-key-here
GLM_MODEL=glm-4.5-air
```

### 2. Direct Usage

```python
from app.services.llm import GLMClient, LLMMessage

# Initialize GLM client
client = GLMClient(
    api_key="your-api-key",
    model="glm-4.5-air",
    max_tokens=2000,
    temperature=0.7
)

# Create messages
messages = [
    LLMMessage(role="system", content="You are a helpful assistant."),
    LLMMessage(role="user", content="Hello! How are you?"),
]

# Generate response
response = await client.generate(messages)
print(response.content)

# Clean up
await client.close()
```

### 3. Using the Factory

```python
from app.services.llm import LLMFactory, LLMMessage

# Create GLM client (reads from environment variables)
llm = LLMFactory.create(provider="glm")

# Or with explicit parameters
llm = LLMFactory.create(
    provider="glm",
    api_key="your-api-key",
    model="glm-4.5-air"
)

# Use the client
messages = [
    LLMMessage(role="user", content="Explain quantum computing"),
]
response = await llm.generate(messages)
print(response.content)
```

### 4. Streaming Responses

```python
from app.services.llm import LLMFactory, LLMMessage

llm = LLMFactory.create(provider="glm")

messages = [
    LLMMessage(role="user", content="Tell me a story"),
]

# Stream response
async for chunk in llm.generate_stream(messages):
    print(chunk, end="", flush=True)

print()  # New line after streaming
```

### 5. Integration with Chat Service

```python
from app.services.llm import LLMFactory
from app.services.chat import ChatServiceFactory
from app.repositories import MessageRepository, SessionRepository

# Create GLM LLM service
llm_service = LLMFactory.create(provider="glm")

# Create chat service with GLM
chat_service = ChatServiceFactory.create_with_defaults(
    llm_service=llm_service,
    message_repo=message_repository,
    session_repo=session_repository,
    memory_type="sliding_window",
    intent_type="llm_based"
)

# Process messages
response = await chat_service.process_message(
    session_id=1,
    message="What is RAG?",
    user_id=1
)

print(response.content)
```

## API Endpoint Configuration

To use GLM in your API endpoints, update the dependency injection:

```python
# In app/api/deps.py or app/api/v1/chat.py

async def get_chat_service() -> ChatService:
    """Get chat service with GLM."""
    from app.services.llm import LLMFactory
    from app.services.chat import ChatServiceFactory
    from app.api.deps import get_message_repository, get_session_repository

    # Create GLM LLM service
    llm_service = LLMFactory.create(provider="glm")

    # Get repositories
    message_repo = await get_message_repository()
    session_repo = await get_session_repository()

    # Create chat service
    return ChatServiceFactory.create_with_defaults(
        llm_service=llm_service,
        message_repo=message_repo,
        session_repo=session_repo
    )
```

## Model Selection Guide

| Model | Use Case | Context Window | Speed |
|-------|----------|----------------|-------|
| `glm-4-plus` | Complex reasoning, analysis | 128k | Medium |
| `glm-4` | General purpose | 128k | Medium |
| `glm-4-air` | Balanced performance | 128k | Fast |
| `glm-4-flash` | Real-time responses | 128k | Very Fast |
| `glm-3-turbo` | Legacy compatibility | 128k | Fast |

## Token Estimation

GLM uses a tokenization similar to GPT models. You can estimate tokens:

```python
from app.services.llm import LLMFactory

llm = LLMFactory.create(provider="glm")

# Estimate tokens
text = "Your text here"
estimated = llm.estimate_tokens(text)
print(f"Estimated tokens: {estimated}")

# Count exact tokens for messages
from app.services.llm import LLMMessage
messages = [
    LLMMessage(role="user", content="Your message"),
]
actual = await llm.count_tokens(messages)
print(f"Actual tokens: {actual}")
```

## Error Handling

```python
from app.core.exceptions import ExternalServiceError

try:
    response = await llm.generate(messages)
except ExternalServiceError as e:
    if "GLM" in str(e.service):
        print(f"GLM API error: {e.message}")
        # Handle error (retry, fallback, etc.)
    else:
        raise
```

## Best Practices

1. **API Key Security**: Never commit API keys to version control
2. **Context Management**: GLM-4 supports 128k tokens, but optimize for cost
3. **Temperature Settings**:
   - 0.0-0.3: Deterministic, factual responses
   - 0.4-0.7: Balanced creativity
   - 0.8-1.0: Creative, varied responses
4. **Streaming**: Use streaming for better user experience in chat applications
5. **Error Handling**: Always wrap API calls in try-except blocks

## Troubleshooting

### "GLM API key not configured"
- Ensure `GLM_API_KEY` is set in `.env` file
- Restart the application after updating environment variables

### "HTTP error occurred: 401"
- Verify your API key is valid
- Check if the key has sufficient credits

### Connection Timeout
- Check your internet connection
- Verify GLM API status at https://open.bigmodel.cn/
- Increase timeout in client initialization if needed

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify httpx is installed: `pip install httpx>=0.26.0`

## Testing

Test your GLM setup:

```python
import asyncio
from app.services.llm import LLMFactory, LLMMessage

async def test_glm():
    # Create client
    llm = LLMFactory.create(provider="glm")

    # Test simple generation
    messages = [LLMMessage(role="user", content="Hello!")]
    response = await llm.generate(messages)
    print(f"Response: {response.content}")
    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage}")

    # Test streaming
    print("\nStreaming test:")
    async for chunk in llm.generate_stream(messages):
        print(chunk, end="", flush=True)
    print()

asyncio.run(test_glm())
```

## Additional Resources

- [Zhipu AI Documentation](https://open.bigmodel.cn/dev/api)
- [GLM Model Pricing](https://open.bigmodel.cn/pricing)
- [API Reference](https://open.bigmodel.cn/dev/api)
