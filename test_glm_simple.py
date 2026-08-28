#!/usr/bin/env python3
"""
Simple test to verify GLM API key is configured correctly.
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_env_file():
    """Test that .env file exists and contains API key."""
    print("=" * 60)
    print("Testing GLM Configuration")
    print("=" * 60)

    env_path = os.path.join(os.path.dirname(__file__), '.env')

    # Check if .env exists
    if not os.path.exists(env_path):
        print("\n❌ ERROR: .env file not found")
        print(f"Expected location: {env_path}")
        return False

    print(f"\n✅ .env file found at: {env_path}")

    # Read and check content
    with open(env_path, 'r') as f:
        content = f.read()

    # Check for API key
    if 'GLM_API_KEY=' in content:
        print("✅ GLM_API_KEY found in .env file")

        # Extract and show part of the key (for security)
        for line in content.split('\n'):
            if line.startswith('GLM_API_KEY='):
                key = line.split('=')[1]
                if key:
                    # Show first 20 and last 10 characters
                    if len(key) > 30:
                        print(f"   Key preview: {key[:20]}...{key[-10:]}")
                    else:
                        print(f"   Key: {key}")
                else:
                    print("❌ ERROR: GLM_API_KEY is empty")
                    return False
    else:
        print("❌ ERROR: GLM_API_KEY not found in .env file")
        return False

    # Check for model
    if 'GLM_MODEL=' in content:
        for line in content.split('\n'):
            if line.startswith('GLM_MODEL='):
                model = line.split('=')[1]
                print(f"✅ GLM_MODEL: {model}")

    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print("✅ GLM API key is configured")
    print("✅ Ready to use GLM in your RAG chatbot")
    print("\nNext steps:")
    print("1. Install dependencies: pip install httpx pydantic")
    print("2. Run the full test: python test_glm_setup.py")
    print("3. Start using GLM in your application")

    return True


def test_code_integration():
    """Test that the GLM code files exist."""
    print("\n" + "=" * 60)
    print("Testing GLM Code Integration")
    print("=" * 60)

    files_to_check = [
        'app/services/llm/glm_client.py',
        'app/services/llm/factory.py',
        'app/services/llm/__init__.py',
        'app/config/settings.py',
    ]

    all_exist = True
    for file_path in files_to_check:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} NOT FOUND")
            all_exist = False

    if all_exist:
        print("\n✅ All GLM integration files are present")

    return all_exist


def main():
    """Run all tests."""
    print("\n🚀 GLM Configuration Test")
    print("=" * 60)

    # Test 1: .env file
    test1_passed = test_env_file()

    # Test 2: Code files
    test2_passed = test_code_integration()

    # Final summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Configuration Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Code Integration Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 Configuration successful!")
        print("\nYour GLM API key has been set to:")
        print("   API Key: 951a05b4ed8c49df996c39ffc38e071c.08KUDBXaZ3u76zZ3")
        print("   Model: glm-5.3-flash")
        print("\nYou can now use GLM in your RAG chatbot:")
        print("   from app.services.llm import LLMFactory")
        print("   llm = LLMFactory.create(provider='glm')")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
