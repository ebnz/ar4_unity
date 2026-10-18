#!/usr/bin/env python3
"""
Test script for simplified conversation logging system
"""

import os
import sys
import tempfile

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_conversation_utils():
    """Test conversation saving utilities"""
    print("\nTesting conversation utilities...")

    try:
        from eureka_unity.conversation_utils import save_conversation

        # Test basic conversation saving
        test_dir = tempfile.mkdtemp(prefix="test_conv_")

        test_messages = [
            {"role": "user", "content": "Generate reward function"},
            {"role": "assistant", "content": "float reward = 1.0f;"},
        ]

        filepath = save_conversation(
            conversations_dir=test_dir,
            iteration=0,
            run_id=1,
            attempt=1,
            conv_type="initial",
            messages=test_messages,
            generated_content="float reward = 1.0f;",
            metadata={"test": "data"},
        )

        # Verify file was created
        if os.path.exists(filepath):
            print(f"✓ Conversation saved to: {os.path.basename(filepath)}")
            return True
        else:
            print("✗ Conversation file not created")
            return False

    except Exception as e:
        print(f"✗ Conversation utils test failed: {e}")
        return False


def test_llm_classes():
    """Test LLM class structure"""
    print("\nTesting LLM classes...")

    try:
        from reward_llm import LLMMetadata, LLMResult

        # Test metadata creation
        metadata = LLMMetadata(
            model="test-model",
            temperature=1.0,
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=1500.0,
        )

        print(
            f"✓ LLMMetadata created: {metadata.model}, {metadata.total_tokens} tokens"
        )

        # Test result creation
        result = LLMResult(
            content="Test content",
            messages=[{"role": "user", "content": "test"}],
            metadata=metadata,
        )

        print(f"✓ LLMResult created with content length: {len(result.content)}")
        return True

    except Exception as e:
        print(f"✗ LLM classes test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("SIMPLIFIED CONVERSATION LOGGING SYSTEM TEST")
    print("=" * 50)

    tests = [
        ("Conversation Utils Test", test_conversation_utils),
        ("LLM Classes Test", test_llm_classes),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"✗ {test_name} FAILED")

    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Simplified system is ready.")
    else:
        print("⚠ Some tests failed. Check output above.")

    print("=" * 50)


if __name__ == "__main__":
    main()
