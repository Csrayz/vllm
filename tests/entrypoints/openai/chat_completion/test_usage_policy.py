# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Integration tests for usage policy configuration.

Tests verify two scenarios:
1. Default behavior (no policy): Ensures backward compatibility with OpenAI API
2. Always policy: Tests that usage is returned correctly when policy is set

Note: Two vLLM servers cannot start simultaneously due to GPU memory constraints.
Each test uses module-scoped fixtures to ensure sequential server startup/shutdown.
"""

import logging
import sys

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


BASE_ARGS = [
    "--max-model-len",
    "4096",
    "--gpu-memory-utilization",
    "0.8",
    "--enforce-eager",
]


# =============================================================================
# Scenario 1: Default behavior (no usage policy)
# =============================================================================
DEFAULT_SERVER_ARGS = BASE_ARGS + [
    "--port",
    "55861",
]

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def default_server():
    """Start vLLM server with default settings (no usage policy)."""
    logger.info("=" * 60)
    logger.info("Starting DEFAULT server (no usage policy)...")
    logger.info("Args: %s", DEFAULT_SERVER_ARGS)
    logger.info("=" * 60)

    import time

    logger.info("Waiting 5s for any previous server to release GPU memory...")
    time.sleep(5)

    server = None
    try:
        with RemoteOpenAIServer(
            DEFAULT_MODEL, DEFAULT_SERVER_ARGS, auto_port=False
        ) as srv:
            server = srv
            logger.info("DEFAULT server started on port %s", srv.port)
            yield srv
    finally:
        if server:
            logger.info("DEFAULT server shutting down...")
        logger.info("DEFAULT server stopped")
        logger.info("Waiting 5s for GPU memory release...")
        time.sleep(5)


@pytest_asyncio.fixture
async def default_client(default_server):
    """Create async client for default server."""
    logger.info("Creating async client for DEFAULT server...")
    async with default_server.get_async_client() as client:
        logger.info("DEFAULT async client created")
        yield client


@pytest.mark.asyncio
async def test_default_non_streaming(default_client: openai.AsyncOpenAI):
    """Default behavior: Non-streaming should always return usage."""
    logger.info("TEST: Default non-streaming")

    response = await default_client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_completion_tokens=5,
        temperature=0.0,
        stream=False,
    )

    logger.info("Response usage: %s", response.usage)
    assert response.usage is not None, "Default non-streaming should have usage"
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    logger.info("TEST: Default non-streaming PASSED")


@pytest.mark.asyncio
async def test_default_streaming_no_usage(default_client: openai.AsyncOpenAI):
    """Default behavior: Streaming without stream_options should NOT
    return usage in chunks."""
    logger.info("TEST: Default streaming (no stream_options)")

    chunk_count = 0
    usage_in_chunks = False

    stream = await default_client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_completion_tokens=5,
        temperature=0.0,
        stream=True,
    )

    async for chunk in stream:
        chunk_count += 1
        logger.info(
            "Chunk %s: choices=%s, usage=%s",
            chunk_count,
            len(chunk.choices),
            chunk.usage,
        )
        if chunk.usage is not None and not chunk.choices:
            usage_in_chunks = True

    logger.info(
        "Total chunks: %s, Final usage chunk: %s",
        chunk_count,
        usage_in_chunks,
    )
    assert not usage_in_chunks, (
        "Default streaming should NOT have usage chunk without stream_options"
    )
    logger.info("TEST: Default streaming (no stream_options) PASSED")


@pytest.mark.asyncio
async def test_default_streaming_with_usage_option(
    default_client: openai.AsyncOpenAI,
):
    """Default behavior: Streaming with stream_options.include_usage=True
    should return usage."""
    logger.info("TEST: Default streaming with stream_options.include_usage=True")

    stream = await default_client.chat.completions.create(
        model=DEFAULT_MODEL,
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
        logger.info(
            "Chunk %s: choices=%s, usage=%s",
            chunk_count,
            len(chunk.choices),
            chunk.usage,
        )
        if chunk.usage is not None and not chunk.choices:
            final_chunk_with_usage = True
            assert chunk.usage.prompt_tokens > 0

    logger.info(
        "Total chunks: %s, Final usage chunk: %s",
        chunk_count,
        final_chunk_with_usage,
    )
    assert final_chunk_with_usage, (
        "Should have final usage chunk when stream_options.include_usage=True"
    )
    logger.info("TEST: Default streaming with stream_options.include_usage=True PASSED")


# =============================================================================
# Scenario 2: Always policy (include-usage-policy=always,
#              continuous-usage-policy=always)
# =============================================================================
ALWAYS_SERVER_ARGS = BASE_ARGS + [
    "--port",
    "55862",
    "--include-usage-policy",
    "always",
    "--continuous-usage-policy",
    "always",
]

ALWAYS_MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def always_server():
    """Start vLLM server with always usage policy."""
    logger.info("=" * 60)
    logger.info("Starting ALWAYS server (include+continuous usage policy)...")
    logger.info("Args: %s", ALWAYS_SERVER_ARGS)
    logger.info("=" * 60)

    import time

    logger.info("Waiting 5s for any previous server to release GPU memory...")
    time.sleep(5)

    server = None
    try:
        with RemoteOpenAIServer(
            ALWAYS_MODEL, ALWAYS_SERVER_ARGS, auto_port=False
        ) as srv:
            server = srv
            logger.info("ALWAYS server started on port %s", srv.port)
            yield srv
    finally:
        if server:
            logger.info("ALWAYS server shutting down...")
        logger.info("ALWAYS server stopped")
        logger.info("Waiting 5s for GPU memory release...")
        time.sleep(5)


@pytest_asyncio.fixture
async def always_client(always_server):
    """Create async client for always server."""
    logger.info("Creating async client for ALWAYS server...")
    async with always_server.get_async_client() as client:
        logger.info("ALWAYS async client created")
        yield client


@pytest.mark.asyncio
async def test_always_non_streaming(always_client: openai.AsyncOpenAI):
    """Always policy: Non-streaming should return usage."""
    logger.info("TEST: Always policy non-streaming")

    response = await always_client.chat.completions.create(
        model=ALWAYS_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_completion_tokens=5,
        temperature=0.0,
        stream=False,
    )

    logger.info("Response usage: %s", response.usage)
    assert response.usage is not None, "Always policy non-streaming should have usage"
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    logger.info("TEST: Always policy non-streaming PASSED")


@pytest.mark.asyncio
async def test_always_streaming_continuous_usage(always_client: openai.AsyncOpenAI):
    """Always policy: Streaming should return usage in EVERY chunk."""
    logger.info("TEST: Always policy streaming (continuous usage)")

    stream = await always_client.chat.completions.create(
        model=ALWAYS_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_completion_tokens=5,
        temperature=0.0,
        stream=True,
    )

    chunk_count = 0
    content_chunks_with_usage = 0
    final_chunk_count = 0

    async for chunk in stream:
        chunk_count += 1
        logger.info(
            "Chunk %s: choices=%s, usage=%s",
            chunk_count,
            len(chunk.choices),
            chunk.usage,
        )

        if not chunk.choices:
            # Final chunk
            final_chunk_count += 1
            assert chunk.usage is not None, "Final chunk must have usage"
            assert chunk.usage.prompt_tokens > 0
            assert chunk.usage.completion_tokens > 0
        else:
            # Content chunk - should also have usage when continuous_usage_policy=always
            assert chunk.usage is not None, (
                "Content chunk must have usage when continuous_usage_policy=always"
            )
            assert chunk.usage.prompt_tokens > 0
            assert chunk.usage.completion_tokens >= 0, (
                "completion_tokens should be >= 0 (may be 0 for first chunks)"
            )
            content_chunks_with_usage += 1

    logger.info(
        "Total chunks: %s, Content with usage: %s, Final: %s",
        chunk_count,
        content_chunks_with_usage,
        final_chunk_count,
    )
    assert final_chunk_count == 1, "Should have exactly 1 final chunk"
    assert content_chunks_with_usage > 0, "Should have content chunks with usage"
    logger.info("TEST: Always policy streaming (continuous usage) PASSED")


@pytest.mark.asyncio
async def test_always_streaming_final_chunk_completion_tokens(
    always_client: openai.AsyncOpenAI,
):
    """Always policy: Final chunk should have correct completion_tokens."""
    logger.info("TEST: Always policy streaming (verify final completion_tokens)")

    stream = await always_client.chat.completions.create(
        model=ALWAYS_MODEL,
        messages=[{"role": "user", "content": "What is 1+1?"}],
        max_completion_tokens=10,
        temperature=0.0,
        stream=True,
    )

    final_usage = None
    async for chunk in stream:
        if chunk.usage is not None and not chunk.choices:
            final_usage = chunk.usage
            logger.info("Final chunk usage: %s", final_usage)

    assert final_usage is not None, "Should have final usage"
    assert final_usage.completion_tokens > 0, (
        f"Final completion_tokens should be > 0, got {final_usage.completion_tokens}"
    )
    assert (
        final_usage.total_tokens
        == final_usage.prompt_tokens + final_usage.completion_tokens
    )
    logger.info("TEST: Always policy streaming (verify final completion_tokens) PASSED")
