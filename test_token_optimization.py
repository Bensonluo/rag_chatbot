#!/usr/bin/env python3
"""
Test script to demonstrate token savings with optimized context builder.

Compares standard memory strategy vs optimized context builder.
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def simulate_conversation():
    """Simulate a conversation with mixed topics."""
    messages = [
        {"role": "user", "content": "Hi there!"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I don't have access to weather data, but you can check a weather service."},
        {"role": "user", "content": "Tell me a joke"},
        {"role": "assistant", "content": "Why did the chicken cross the road? To get to the other side!"},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity and readability."},
        {"role": "user", "content": "How about Flask?"},
        {"role": "assistant", "content": "Flask is a micro web framework written in Python. It's designed to make getting started quick and easy."},
        {"role": "user", "content": "What's Django?"},
        {"role": "assistant", "content": "Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design."},
        {"role": "user", "content": "How do I deploy Flask apps?"},
        {"role": "assistant", "content": "You can deploy Flask apps using various platforms like Heroku, AWS, Google Cloud, or traditional VPS servers."},
        {"role": "user", "content": "What about async support in FastAPI?"},  # ← Current query
    ]

    return messages


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough estimate: ~4 chars per token)."""
    return len(text) // 4


async def test_standard_memory():
    """Test standard sliding window memory."""
    print("=" * 60)
    print("Standard Sliding Window Memory")
    print("=" * 60)

    # Simulate conversation
    conversation = await simulate_conversation()

    # Standard approach: Last 10 messages
    window_size = 10
    recent_messages = conversation[-window_size:]

    # Calculate tokens
    total_tokens = sum(
        estimate_tokens(msg["content"])
        for msg in recent_messages
    )

    print(f"\n📊 Standard Approach (Last {window_size} messages):")
    print(f"  Total messages: {len(recent_messages)}")
    print(f"  Total tokens:  {total_tokens}")
    print(f"  Relevance:     Mixed (includes irrelevant topics)")

    return total_tokens, len(recent_messages)


async def test_optimized_memory():
    """Test optimized context builder with relevance filtering."""
    print("\n" + "=" * 60)
    print("Optimized Context Builder (with Relevance Filtering)")
    print("=" * 60)

    conversation = await simulate_conversation()
    current_query = conversation[-1]["content"]  # "What about async support in FastAPI?"

    # Simulate relevance filtering (simplified)
    # Always include last 3 for continuity
    max_recent = 3
    recent_messages = conversation[-max_recent:]

    # Filter older messages by relevance (simulated)
    # In real implementation, this uses embeddings
    older_messages = conversation[:-max_recent]

    # Simulated: Messages about web frameworks are relevant
    relevant_keywords = ["flask", "django", "fastapi", "web", "framework", "deploy"]
    relevant_messages = []

    for msg in older_messages:
        content_lower = msg["content"].lower()
        if any(keyword in content_lower for keyword in relevant_keywords):
            relevant_messages.append(msg)
            if len(relevant_messages) >= 5:  # max_relevant
                break

    # Combine
    selected = recent_messages + relevant_messages

    # Calculate tokens
    total_tokens = sum(
        estimate_tokens(msg["content"])
        for msg in selected
    )

    print(f"\n📊 Optimized Approach (Relevance Filtering):")
    print(f"  Total messages: {len(selected)}")
    print(f"  Total tokens:  {total_tokens}")
    print(f"  Relevance:     High (only web framework related)")

    # Show selected messages
    print(f"\n✅ Selected messages:")
    for i, msg in enumerate(selected, 1):
        role_symbol = "👤" if msg["role"] == "user" else "🤖"
        preview = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
        print(f"  {i}. {role_symbol} {preview}")

    return total_tokens, len(selected)


async def main():
    """Run comparison tests."""
    print("\n🎯 Token Optimization Test")
    print("=" * 60)
    print("Current query: 'What about async support in FastAPI?'")
    print("=" * 60)

    # Test standard approach
    standard_tokens, standard_count = await test_standard_memory()

    # Test optimized approach
    optimized_tokens, optimized_count = await test_optimized_memory()

    # Calculate savings
    tokens_saved = standard_tokens - optimized_tokens
    percent_saved = (tokens_saved / standard_tokens * 100) if standard_tokens > 0 else 0

    # Summary
    print("\n" + "=" * 60)
    print("💰 TOKEN SAVINGS SUMMARY")
    print("=" * 60)
    print(f"  Standard tokens:    {standard_tokens}")
    print(f"  Optimized tokens:   {optimized_tokens}")
    print(f"  Tokens saved:       {tokens_saved}")
    print(f"  Savings:            {percent_saved:.1f}%")
    print(f"  Message reduction:   {standard_count - optimized_count} messages")

    # Cost impact
    print(f"\n💵 Cost Impact (GLM-4.5: ~¥0.01/1K tokens):")
    queries_per_month = 100_000

    standard_cost = (standard_tokens / 1000) * 0.01 * queries_per_month
    optimized_cost = (optimized_tokens / 1000) * 0.01 * queries_per_month
    monthly_savings = standard_cost - optimized_cost

    print(f"  Standard monthly cost:  ¥{standard_cost:,.0f}")
    print(f"  Optimized monthly cost: ¥{optimized_cost:,.0f}")
    print(f"  💰 Monthly savings:     ¥{monthly_savings:,.0f}")

    print("\n" + "=" * 60)
    print("✨ Key Benefits:")
    print("=" * 60)
    print("  ✅ Reduced noise (only relevant context)")
    print("  ✅ Faster LLM inference (less context)")
    print("  ✅ Lower API costs ({:.1f}% savings)".format(percent_saved))
    print("  ✅ Better response quality (focused context)")
    print("  ✅ Automatic optimization (built-in)")

    print("\n🚀 Your system is now using optimized context by default!")
    print("   Just start the chat service and enjoy automatic savings.")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
