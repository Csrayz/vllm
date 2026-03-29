# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Simplified integration tests for usage policy configuration.

Tests verify two scenarios:
1. Default behavior (no usage policy):
   - Non-streaming: Should return usage in response
   - Streaming without stream_options: Should NOT return usage in chunks
   - Streaming with stream_options.include_usage: Should return usage in final chunk

2. Always policy (--include-usage-policy=always --continuous-usage-policy=always):
   - Non-streaming: Should return usage in response
   - Streaming: Every chunk should have usage (continuous)

Based on test_serving_chat.py pattern.
"""

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"

BASE_ARGS = [
    "--max-model-len",
    "4096",
    "--gpu-memory-utilization",
    "0.8",
    "--enforce-eager",
]


@pytest.fixture(scope="module")
def server_args():
    return BASE_ARGS


@pytest.fixture(scope="module")
def always_server_args():
    return BASE_ARGS + [
        "--include-usage-policy",
        "always",
        "--continuous-usage-policy",
        "always",
    ]


@pytest.fixture(scope="class")
def server(server_args):
    with RemoteOpenAIServer(MODEL_NAME, server_args) as remote_server:
        yield remote_server


@pytest.fixture(scope="class")
def always_server(always_server_args):
    with RemoteOpenAIServer(
        MODEL_NAME, always_server_args, max_wait_seconds=480
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest_asyncio.fixture
async def always_client(always_server):
    async with always_server.get_async_client() as async_client:
        yield async_client


class TestUsagePolicyDefault:
    @pytest.mark.asyncio
    async def test_non_streaming(self, client: openai.AsyncOpenAI):
        """Non-streaming should always return usage in response."""
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=5,
            temperature=0.0,
            stream=False,
        )

        assert response.usage is not None, "Non-streaming should have usage"
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    @pytest.mark.asyncio
    async def test_streaming_no_usage(self, client: openai.AsyncOpenAI):
        """Streaming without stream_options should NOT return usage in chunks."""
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=5,
            temperature=0.0,
            stream=True,
        )

        chunk_count = 0
        usage_in_chunks = False

        async for chunk in stream:
            chunk_count += 1
            if chunk.usage is not None and not chunk.choices:
                usage_in_chunks = True

        assert not usage_in_chunks, (
            "Streaming without stream_options should NOT have usage chunk"
        )

    @pytest.mark.asyncio
    async def test_streaming_with_usage_option(self, client: openai.AsyncOpenAI):
        """Streaming with stream_options.include_usage=True should return usage."""
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=5,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )

        chunk_count = 0
        final_chunk_with_usage = False

        async for chunk in stream:
            chunk_count += 1
            if chunk.usage is not None and not chunk.choices:
                final_chunk_with_usage = True
                assert chunk.usage.prompt_tokens > 0

        assert final_chunk_with_usage, (
            "Should have final usage chunk when stream_options.include_usage=True"
        )


class TestUsagePolicyAlways:
    @pytest.mark.asyncio
    async def test_non_streaming(self, always_client: openai.AsyncOpenAI):
        """Always policy: Non-streaming should return usage."""
        response = await always_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=5,
            temperature=0.0,
            stream=False,
        )

        assert response.usage is not None, (
            "Always policy non-streaming should have usage"
        )
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    @pytest.mark.asyncio
    async def test_streaming_continuous_usage(self, always_client: openai.AsyncOpenAI):
        """Always policy: Streaming should return usage in EVERY chunk."""
        stream = await always_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=5,
            temperature=0.0,
            stream=True,
        )

        chunk_count = 0
        completion_tokens = 0

        async for chunk in stream:
            chunk_count += 1
            assert chunk.usage is not None, "Every chunk must have usage"
            assert chunk.usage.prompt_tokens > 0
            # completion_tokens is cumulative
            assert chunk.usage.completion_tokens >= completion_tokens
            completion_tokens = chunk.usage.completion_tokens

        assert chunk_count > 0, "Should have received at least one chunk"
