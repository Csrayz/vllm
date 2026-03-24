# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.chat_utils import UsagePolicy
from vllm.entrypoints.openai.engine.protocol import StreamOptions
from vllm.entrypoints.utils import (
    get_max_tokens,
    sanitize_message,
    should_include_usage,
)


def test_sanitize_message():
    assert (
        sanitize_message("<_io.BytesIO object at 0x7a95e299e750>")
        == "<_io.BytesIO object>"
    )


class TestGetMaxTokens:
    """Tests for get_max_tokens() to ensure generation_config's max_tokens
    acts as a default when from model author, and as a ceiling when
    explicitly set by the user."""

    def test_default_sampling_params_used_when_no_request_max_tokens(self):
        """When user doesn't specify max_tokens, generation_config default
        should apply."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2048

    def test_request_max_tokens_not_capped_by_default_sampling_params(self):
        """When user specifies max_tokens in request, model author's
        generation_config max_tokens must NOT cap it (fixes #34005)."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 5000

    def test_override_max_tokens_caps_request(self):
        """When user explicitly sets max_tokens, it acts as a ceiling."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_override_max_tokens_used_as_default(self):
        """When no request max_tokens, override still applies as default."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_max_model_len_still_caps_output(self):
        """max_model_len - input_length is always the hard ceiling."""
        result = get_max_tokens(
            max_model_len=3000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2900  # 3000 - 100

    def test_request_max_tokens_smaller_than_default(self):
        """When user explicitly requests fewer tokens than gen_config default,
        that should be respected."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=512,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 512

    def test_input_length_exceeds_max_model_len(self):
        with pytest.raises(
            ValueError,
            match="Input length .* exceeds model's maximum context length .*",
        ):
            get_max_tokens(
                max_model_len=100,
                max_tokens=50,
                input_length=150,
                default_sampling_params={"max_tokens": 2048},
            )


class TestShouldIncludeUsage:
    """Tests for should_include_usage() function."""

    def test_no_usage_policy_no_stream_options(self):
        """Both usage_policy and stream_options are None."""
        include_usage, include_continuous = should_include_usage(None, None)
        assert include_usage is False
        assert include_continuous is False

    def test_no_usage_stream_options(self):
        """stream_options controls behavior."""
        stream_options = StreamOptions(
            include_usage=False, continuous_usage_stats=False
        )
        include_usage, include_continuous = should_include_usage(stream_options, None)
        assert include_usage is False
        assert include_continuous is False

    def test_include_usage_with_stream_options(self):
        """stream_options controls behavior."""
        stream_options = StreamOptions(include_usage=True, continuous_usage_stats=True)
        include_usage, include_continuous = should_include_usage(stream_options, None)
        assert include_usage is True
        assert include_continuous is True

    def test_include_usage_stream_options(self):
        """stream_options controls behavior."""
        stream_options = StreamOptions(include_usage=True, continuous_usage_stats=False)
        include_usage, include_continuous = should_include_usage(stream_options, None)
        assert include_usage is True
        assert include_continuous is False

    def test_no_usage_with_stream_options(self):
        """stream_options controls behavior."""
        stream_options = StreamOptions(include_usage=False, continuous_usage_stats=True)
        include_usage, include_continuous = should_include_usage(stream_options, None)
        assert include_usage is False
        assert include_continuous is False

    def test_always_usage_policy_no_stream_options(self):
        """usage_policy always controls behavior."""
        stream_options = None
        usage_policy = UsagePolicy(include_usage="always")
        include_usage, include_continuous = should_include_usage(
            stream_options, usage_policy
        )
        assert include_usage is True
        assert include_continuous is False

    def test_always_usage_policy_with_stream_options(self):
        """usage_policy always controls behavior."""
        stream_options = StreamOptions(
            include_usage=False, continuous_usage_stats=False
        )
        usage_policy = UsagePolicy(include_usage="always")
        include_usage, include_continuous = should_include_usage(
            stream_options, usage_policy
        )
        assert include_usage is True
        assert include_continuous is False

    def test_default_include_usage_no_stream_options(self):
        """usage_policy default_include_usage follows stream_options."""
        stream_options = None
        usage_policy = UsagePolicy(include_usage="default_include_usage")
        include_usage, include_continuous = should_include_usage(
            stream_options, usage_policy
        )
        assert include_usage is True
        assert include_continuous is False

    def test_default_include_usage_with_stream_options(self):
        """usage_policy default_include_usage follows stream_options."""
        stream_options = StreamOptions(
            include_usage=False, continuous_usage_stats=False
        )
        usage_policy = UsagePolicy(include_usage="default_include_usage")
        include_usage, include_continuous = should_include_usage(
            stream_options, usage_policy
        )
        assert include_usage is False
        assert include_continuous is False

    def test_always_continuous_usage_no_stream_options(self):
        """usage_policy always controls behavior."""
        stream_options = StreamOptions(
            include_usage=False, continuous_usage_stats=False
        )
        usage_policy = UsagePolicy(continuous_usage="always")
        include_usage, include_continuous = should_include_usage(
            stream_options, usage_policy
        )
        assert include_usage is True
        assert include_continuous is True
