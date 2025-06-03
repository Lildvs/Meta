import pytest
from forecasting_tools.ai_models.general_llm import GeneralLlm

def test_perplexity_message_ordering():
    # Create a GeneralLlm instance with a Perplexity model
    llm = GeneralLlm(model="perplexity/sonar")

    # Test case 1: Consecutive user messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm good"},
        {"role": "user", "content": "What's the weather?"}
    ]

    ordered_messages = llm.model_input_to_message(messages)

    # Verify system message is first
    assert ordered_messages[0]["role"] == "system"

    # Verify alternating pattern after system message
    assert ordered_messages[1]["role"] == "user"
    assert ordered_messages[2]["role"] == "assistant"
    assert ordered_messages[3]["role"] == "user"

    # Verify consecutive user messages are combined
    assert "Hello" in ordered_messages[1]["content"]
    assert "How are you?" in ordered_messages[1]["content"]

    # Test case 2: Consecutive assistant messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "assistant", "content": "How can I help?"},
        {"role": "user", "content": "What's the weather?"}
    ]

    ordered_messages = llm.model_input_to_message(messages)

    # Verify system message is first
    assert ordered_messages[0]["role"] == "system"

    # Verify alternating pattern after system message
    assert ordered_messages[1]["role"] == "user"
    assert ordered_messages[2]["role"] == "assistant"
    assert ordered_messages[3]["role"] == "user"

    # Verify consecutive assistant messages are combined
    assert "Hi there" in ordered_messages[2]["content"]
    assert "How can I help?" in ordered_messages[2]["content"]

    # Test case 3: No system message
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm good"},
        {"role": "user", "content": "What's the weather?"}
    ]

    ordered_messages = llm.model_input_to_message(messages)

    # Verify alternating pattern
    assert ordered_messages[0]["role"] == "user"
    assert ordered_messages[1]["role"] == "assistant"
    assert ordered_messages[2]["role"] == "user"

    # Verify consecutive user messages are combined
    assert "Hello" in ordered_messages[0]["content"]
    assert "How are you?" in ordered_messages[0]["content"]