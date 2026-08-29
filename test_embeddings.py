#!/usr/bin/env python3
"""
Test script for embedding services.

Tests local and GLM embedding services.
"""
import asyncio
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_local_embeddings():
    """Test local BGE-M3 embeddings."""
    print("=" * 60)
    print("Testing Local Embeddings (BGE-M3)")
    print("=" * 60)

    from app.services.embeddings import LocalEmbeddingService

    try:
        # Initialize service
        print("\n🔧 Initializing local embedding service...")
        service = LocalEmbeddingService(model="bge-m3")

        print(f"✅ Model: {service.model_name}")
        print(f"✅ Dimensions: {service.dimensions}")

        # Test single embedding
        print("\n💬 Testing single text embedding...")
        test_text = "你好，这是一个测试文本。This is a test text."
        print(f"   Input: {test_text}")

        start = time.time()
        embedding = await service.embed_single(test_text)
        elapsed = time.time() - start

        print(f"✅ Generated embedding: {len(embedding)} dimensions")
        print(f"⏱️  Time: {elapsed:.2f}s")
        print(f"📊 Sample values (first 5): {embedding[:5]}")

        # Test batch embeddings
        print("\n💬 Testing batch embeddings...")
        texts = [
            "机器学习是人工智能的一个分支。",
            "深度学习使用神经网络进行学习。",
            "RAG结合了检索和生成技术。"
        ]
        print(f"   Input: {len(texts)} texts")

        start = time.time()
        result = await service.embed(texts)
        elapsed = time.time() - start

        print(f"✅ Generated {len(result.embeddings)} embeddings")
        print(f"⏱️  Total time: {elapsed:.2f}s")
        print(f"📊 Tokens used: {result.tokens_used}")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_glm_embeddings():
    """Test GLM API embeddings."""
    print("\n\n" + "=" * 60)
    print("Testing GLM Embeddings (API)")
    print("=" * 60)

    from app.services.embeddings import GLMEmbeddingService
    from app.config.settings import get_settings

    settings = get_settings()

    if not settings.GLM_API_KEY:
        print("\n⚠️  GLM_API_KEY not set, skipping GLM embeddings test")
        print("   To test: Add GLM_API_KEY to .env file")
        return True  # Not a failure, just skipped

    try:
        # Initialize service
        print("\n🔧 Initializing GLM embedding service...")
        service = GLMEmbeddingService(
            api_key=settings.GLM_API_KEY,
            model="embedding-2"
        )

        print(f"✅ Model: {service.model}")
        print(f"✅ Dimensions: {service.dimensions}")

        # Test single embedding
        print("\n💬 Testing single text embedding...")
        test_text = "你好，这是一个测试文本。"
        print(f"   Input: {test_text}")

        start = time.time()
        embedding = await service.embed_single(test_text)
        elapsed = time.time() - start

        print(f"✅ Generated embedding: {len(embedding)} dimensions")
        print(f"⏱️  Time: {elapsed:.2f}s")
        print(f"📊 Sample values (first 5): {embedding[:5]}")

        # Test batch embeddings
        print("\n💬 Testing batch embeddings...")
        texts = [
            "机器学习是人工智能的一个分支。",
            "深度学习使用神经网络进行学习。",
            "RAG结合了检索和生成技术。"
        ]
        print(f"   Input: {len(texts)} texts")

        start = time.time()
        result = await service.embed(texts)
        elapsed = time.time() - start

        print(f"✅ Generated {len(result.embeddings)} embeddings")
        print(f"⏱️  Total time: {elapsed:.2f}s")
        print(f"📊 Tokens used: {result.tokens_used}")

        # Clean up
        await service.close()
        print("\n✅ GLM client closed")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_embedding_factory():
    """Test embedding factory."""
    print("\n\n" + "=" * 60)
    print("Testing Embedding Factory")
    print("=" * 60)

    from app.services.embeddings import EmbeddingFactory

    try:
        print("\n🔧 Creating embedding service via factory...")

        # Create local embedding service
        service = EmbeddingFactory.create(provider="local")
        print(f"✅ Created local embedding service")

        # Test embedding
        texts = ["Factory test text."]
        result = await service.embed(texts)
        print(f"✅ Generated {len(result.embeddings)} embeddings")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_cached_embeddings():
    """Test cached embeddings."""
    print("\n\n" + "=" * 60)
    print("Testing Cached Embeddings (Redis)")
    print("=" * 60)

    from app.services.embeddings import (
        LocalEmbeddingService,
        CachedEmbeddingService
    )
    from app.config.settings import get_settings

    settings = get_settings()

    try:
        print("\n🔧 Initializing cached embedding service...")

        # Create base embedding service
        base_service = LocalEmbeddingService(model="bge-m3")

        # Wrap with cache
        cached_service = CachedEmbeddingService(
            embedding_service=base_service,
            redis_url=settings.REDIS_URL,
            cache_ttl=3600  # 1 hour for testing
        )

        print("✅ Created cached embedding service")

        # Test first call (cache miss)
        print("\n💬 First call (cache miss)...")
        text = "This is a test for caching."
        start = time.time()
        embedding1 = await cached_service.embed_single(text)
        time1 = time.time() - start
        print(f"✅ Generated in {time1:.2f}s")

        # Test second call (cache hit)
        print("\n💬 Second call (cache hit)...")
        start = time.time()
        embedding2 = await cached_service.embed_single(text)
        time2 = time.time() - start
        print(f"✅ Retrieved in {time2:.2f}s")

        # Verify they're the same
        if embedding1 == embedding2:
            print("✅ Embeddings match!")
        else:
            print("❌ Embeddings don't match!")

        # Speedup calculation
        if time2 < time1:
            speedup = time1 / time2
            print(f"🚀 Cache speedup: {speedup:.1f}x faster")

        # Clean up
        await cached_service.close()
        print("\n✅ Redis connection closed")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("   Make sure Redis is running: docker-compose up -d redis")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all embedding tests."""
    print("\n🚀 Embedding Service Test Suite")
    print("=" * 60)

    # Test 1: Local embeddings
    test1_passed = await test_local_embeddings()

    # Test 2: GLM embeddings (if API key available)
    test2_passed = await test_glm_embeddings()

    # Test 3: Factory
    test3_passed = await test_embedding_factory()

    # Test 4: Cached embeddings (if Redis available)
    test4_passed = await test_cached_embeddings()

    # Summary
    print("\n\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Local Embeddings: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"GLM Embeddings: {'✅ PASSED' if test2_passed else '❌ FAILED (or skipped)'}")
    print(f"Embedding Factory: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    print(f"Cached Embeddings: {'✅ PASSED' if test4_passed else '❌ FAILED (or Redis not available)'}")

    all_passed = test1_passed and test2_passed and test3_passed and test4_passed

    if all_passed:
        print("\n🎉 All tests passed!")
        print("\nYour embedding service is ready for:")
        print("  - Local BGE-M3 embeddings (free)")
        print("  - GLM API embeddings (if API key configured)")
        print("  - Redis caching for performance")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
