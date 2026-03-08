#!/usr/bin/env python3
"""
Test script to verify GLM setup.

This script tests the GLM integration with your API key.
Run this after setting up your GLM_API_KEY to verify everything works.
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_glm_client():
    """Test GLM client with direct instantiation."""
    print("=" * 60)
    print("Testing GLM Client (Direct Instantiation)")
    print("=" * 60)

    from app.services.llm import GLMClient, LLMMessage
    from app.config.settings import get_settings

    settings = get_settings()

    # Check if API key is configured
    if not settings.GLM_API_KEY:
        print("\n❌ ERROR: GLM_API_KEY not found in environment variables")
        print("\nPlease set GLM_API_KEY in your .env file:")
        print("GLM_API_KEY=your-api-key-here")
        return False

    print(f"\n✅ GLM_API_KEY found")
    print(f"📝 Model: {settings.GLM_MODEL}")

    try:
        # Create client 
        print("\n🔧 Initializing GLM client...")
        client = GLMClient(
            api_key=settings.GLM_API_KEY,
            model=settings.GLM_MODEL,
        )
        print("✅ Client initialized successfully")

        # Test simple generation
        print("\n💬 Testing simple message generation...")
        messages = [
            LLMMessage(role="user", content="你好！请用一句话介绍你自己。"),
        ]

        print("⏳ Sending request to GLM API...")
        response = await client.generate(messages)

        print(f"\n✅ Response received!")
        print(f"📊 Model: {response.model}")
        print(f"📝 Content: {response.content}")
        print(f"🔢 Tokens: {response.usage}")
        print(f"🏁 Finish Reason: {response.finish_reason}")

        # Test streaming
        print("\n💬 Testing streaming generation...")
        messages = [
            LLMMessage(role="user", content="Count from 1 to 5"),
        ]

        print("⏳ Streaming response:")
        print("-" * 40)
        async for chunk in client.generate_stream(messages):
            print(chunk, end="", flush=True)
        print("\n" + "-" * 40)
        print("✅ Streaming completed successfully")

        # Clean up
        await client.close()
        print("\n✅ Client closed successfully")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_glm_factory():
    """Test GLM client using factory pattern."""
    print("\n\n" + "=" * 60)
    print("Testing GLM Client (Factory Pattern)")
    print("=" * 60)

    from app.services.llm import LLMFactory, LLMMessage

    try:
        print("\n🔧 Creating GLM client via factory...")
        llm = LLMFactory.create(provider="glm")
        print("✅ Client created via factory")

        print("\n💬 Testing message generation...")
        messages = [
            LLMMessage(role="user", content="What is 2+2?"),
        ]

        response = await llm.generate(messages)
        print(f"✅ Response: {response.content}")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False


async def main():
    """Run all tests."""
    print("\n🚀 GLM Setup Test Script")
    print("=" * 60)

    # Test 1: Direct client
    test1_passed = await test_glm_client()

    # Test 2: Factory pattern
    test2_passed = await test_glm_factory()

    # Summary
    print("\n\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Direct Client Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Factory Pattern Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! GLM is configured correctly.")
        print("\nYou can now use GLM in your RAG chatbot:")
        print("  from app.services.llm import LLMFactory")
        print("  llm = LLMFactory.create(provider='glm')")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    ## create an event loop and run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
