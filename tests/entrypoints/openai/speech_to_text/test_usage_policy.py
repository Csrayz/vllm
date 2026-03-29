# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Simplified integration tests for usage policy configuration in Speech-to-Text API.

Tests verify default behavior (no usage policy):
1. Non-streaming: Should return usage in response
2. Streaming without stream_options: Should NOT return usage in chunks
3. Streaming with stream_options.include_usage: Should return usage in final chunk

Based on test_serving_chat.py pattern.
"""

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "openai/whisper-large-v3-turbo"

BASE_ARGS = [
    "--gpu-memory-utilization",
    "0.8",
    "--enforce-eager",
]


@pytest.fixture(scope="module")
def server_args():
    return BASE_ARGS


@pytest.fixture(scope="class")
def server(server_args):
    with RemoteOpenAIServer(MODEL_NAME, server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


class TestUsagePolicyDefault:
    @pytest.mark.asyncio
    async def test_non_streaming_transcriptions(
        self, client: openai.AsyncOpenAI, winning_call
    ):
        """Non-streaming should always return usage in response."""
        response = await client.audio.transcriptions.create(
            model=MODEL_NAME,
            file=winning_call,
            language="en",
            temperature=0.0,
            response_format="json",
        )

        assert response.usage is not None, "Non-streaming should have usage in response"
        assert response.usage.seconds > 0

    @pytest.mark.asyncio
    async def test_streaming_transcriptions(
        self, client: openai.AsyncOpenAI, winning_call
    ):
        """Streaming without stream_options should NOT return usage in chunks."""
        stream = await client.audio.transcriptions.create(
            model=MODEL_NAME,
            file=winning_call,
            language="en",
            temperature=0.0,
            response_format="json",
            stream=True,
        )

        chunk_count = 0
        usage_in_chunks = False

        async for chunk in stream:
            chunk_count += 1
            has_usage = hasattr(chunk, "usage") and chunk.usage is not None
            if not chunk.choices and has_usage:
                usage_in_chunks = True

        assert not usage_in_chunks, (
            "Streaming without stream_options should NOT have usage chunk"
        )
