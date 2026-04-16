"""Tests for tools/image_generation_tool.py — FAL multi-model support.

Covers the pure logic of the new wrapper: catalog integrity, the three size
families (image_size_preset / aspect_ratio / gpt_literal), the supports
whitelist, default merging, GPT quality override, and model resolution
fallback. Does NOT exercise fal_client submission — that's covered by
tests/tools/test_managed_media_gateways.py.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def image_tool():
    """Fresh import of tools.image_generation_tool per test."""
    import importlib
    import tools.image_generation_tool as mod
    return importlib.reload(mod)


# ---------------------------------------------------------------------------
# Catalog integrity
# ---------------------------------------------------------------------------

class TestFalCatalog:
    """Every FAL_MODELS entry must have a consistent shape."""

    def test_default_model_is_klein(self, image_tool):
        assert image_tool.DEFAULT_MODEL == "fal-ai/flux-2/klein/9b"

    def test_default_model_in_catalog(self, image_tool):
        assert image_tool.DEFAULT_MODEL in image_tool.FAL_MODELS

    def test_all_entries_have_required_keys(self, image_tool):
        required = {
            "display", "speed", "strengths", "price",
            "size_style", "sizes", "defaults", "supports", "upscale",
        }
        for mid, meta in image_tool.FAL_MODELS.items():
            missing = required - set(meta.keys())
            assert not missing, f"{mid} missing required keys: {missing}"

    def test_size_style_is_valid(self, image_tool):
        valid = {"image_size_preset", "aspect_ratio", "gpt_literal"}
        for mid, meta in image_tool.FAL_MODELS.items():
            assert meta["size_style"] in valid, \
                f"{mid} has invalid size_style: {meta['size_style']}"

    def test_sizes_cover_all_aspect_ratios(self, image_tool):
        for mid, meta in image_tool.FAL_MODELS.items():
            assert set(meta["sizes"].keys()) >= {"landscape", "square", "portrait"}, \
                f"{mid} missing a required aspect_ratio key"

    def test_supports_is_a_set(self, image_tool):
        for mid, meta in image_tool.FAL_MODELS.items():
            assert isinstance(meta["supports"], set), \
                f"{mid}.supports must be a set, got {type(meta['supports'])}"

    def test_prompt_is_always_supported(self, image_tool):
        for mid, meta in image_tool.FAL_MODELS.items():
            assert "prompt" in meta["supports"], \
                f"{mid} must support 'prompt'"

    def test_only_flux2_pro_upscales_by_default(self, image_tool):
        """Upscaling should default to False for all new models to preserve
        the <1s / fast-render value prop. Only flux-2-pro stays True for
        backward-compat with the previous default."""
        for mid, meta in image_tool.FAL_MODELS.items():
            if mid == "fal-ai/flux-2-pro":
                assert meta["upscale"] is True, \
                    "flux-2-pro should keep upscale=True for backward-compat"
            else:
                assert meta["upscale"] is False, \
                    f"{mid} should default to upscale=False"


# ---------------------------------------------------------------------------
# Payload building — three size families
# ---------------------------------------------------------------------------

class TestImageSizePresetFamily:
    """Flux, z-image, qwen, recraft, ideogram all use preset enum sizes."""

    def test_klein_landscape_uses_preset(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/flux-2/klein/9b", "hello", "landscape")
        assert p["image_size"] == "landscape_16_9"
        assert "aspect_ratio" not in p

    def test_klein_square_uses_preset(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/flux-2/klein/9b", "hello", "square")
        assert p["image_size"] == "square_hd"

    def test_klein_portrait_uses_preset(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/flux-2/klein/9b", "hello", "portrait")
        assert p["image_size"] == "portrait_16_9"


class TestAspectRatioFamily:
    """Nano-banana uses aspect_ratio enum, NOT image_size."""

    def test_nano_banana_landscape_uses_aspect_ratio(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/nano-banana", "hello", "landscape")
        assert p["aspect_ratio"] == "16:9"
        assert "image_size" not in p

    def test_nano_banana_square_uses_aspect_ratio(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/nano-banana", "hello", "square")
        assert p["aspect_ratio"] == "1:1"

    def test_nano_banana_portrait_uses_aspect_ratio(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/nano-banana", "hello", "portrait")
        assert p["aspect_ratio"] == "9:16"


class TestGptLiteralFamily:
    """GPT-Image 1.5 uses literal size strings."""

    def test_gpt_landscape_is_literal(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/gpt-image-1.5", "hello", "landscape")
        assert p["image_size"] == "1536x1024"

    def test_gpt_square_is_literal(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/gpt-image-1.5", "hello", "square")
        assert p["image_size"] == "1024x1024"

    def test_gpt_portrait_is_literal(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/gpt-image-1.5", "hello", "portrait")
        assert p["image_size"] == "1024x1536"


# ---------------------------------------------------------------------------
# Supports whitelist — the main safety property
# ---------------------------------------------------------------------------

class TestSupportsFilter:
    """No model should receive keys outside its `supports` set."""

    def test_payload_keys_are_subset_of_supports_for_all_models(self, image_tool):
        for mid, meta in image_tool.FAL_MODELS.items():
            payload = image_tool._build_fal_payload(mid, "test", "landscape", seed=42)
            unsupported = set(payload.keys()) - meta["supports"]
            assert not unsupported, \
                f"{mid} payload has unsupported keys: {unsupported}"

    def test_gpt_image_has_no_seed_even_if_passed(self, image_tool):
        # GPT-Image 1.5 does not support seed — the filter must strip it.
        p = image_tool._build_fal_payload("fal-ai/gpt-image-1.5", "hi", "square", seed=42)
        assert "seed" not in p

    def test_gpt_image_strips_unsupported_overrides(self, image_tool):
        p = image_tool._build_fal_payload(
            "fal-ai/gpt-image-1.5", "hi", "square",
            overrides={"guidance_scale": 7.5, "num_inference_steps": 50},
        )
        assert "guidance_scale" not in p
        assert "num_inference_steps" not in p

    def test_recraft_has_minimal_payload(self, image_tool):
        # Recraft supports prompt, image_size, style only.
        p = image_tool._build_fal_payload("fal-ai/recraft-v3", "hi", "landscape")
        assert set(p.keys()) <= {"prompt", "image_size", "style"}

    def test_nano_banana_never_gets_image_size(self, image_tool):
        # Common bug: translator accidentally setting both image_size and aspect_ratio.
        p = image_tool._build_fal_payload("fal-ai/nano-banana", "hi", "landscape", seed=1)
        assert "image_size" not in p
        assert p["aspect_ratio"] == "16:9"


# ---------------------------------------------------------------------------
# Default merging
# ---------------------------------------------------------------------------

class TestDefaults:
    """Model-level defaults should carry through unless overridden."""

    def test_klein_default_steps_is_4(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/flux-2/klein/9b", "hi", "square")
        assert p["num_inference_steps"] == 4

    def test_flux_2_pro_default_steps_is_50(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/flux-2-pro", "hi", "square")
        assert p["num_inference_steps"] == 50

    def test_override_replaces_default(self, image_tool):
        p = image_tool._build_fal_payload(
            "fal-ai/flux-2-pro", "hi", "square", overrides={"num_inference_steps": 25}
        )
        assert p["num_inference_steps"] == 25

    def test_none_override_does_not_replace_default(self, image_tool):
        """None values from caller should be ignored (use default)."""
        p = image_tool._build_fal_payload(
            "fal-ai/flux-2-pro", "hi", "square",
            overrides={"num_inference_steps": None},
        )
        assert p["num_inference_steps"] == 50


# ---------------------------------------------------------------------------
# GPT-Image quality_setting wiring
# ---------------------------------------------------------------------------

class TestGptQualitySetting:

    def test_default_quality_is_medium(self, image_tool):
        with patch("hermes_cli.config.load_config", return_value={}):
            assert image_tool._resolve_gpt_quality() == "medium"

    def test_config_quality_low_applied(self, image_tool):
        with patch("hermes_cli.config.load_config",
                   return_value={"image_gen": {"quality_setting": "low"}}):
            assert image_tool._resolve_gpt_quality() == "low"

    def test_config_quality_high_applied(self, image_tool):
        with patch("hermes_cli.config.load_config",
                   return_value={"image_gen": {"quality_setting": "high"}}):
            assert image_tool._resolve_gpt_quality() == "high"

    def test_invalid_quality_falls_back_to_medium(self, image_tool):
        with patch("hermes_cli.config.load_config",
                   return_value={"image_gen": {"quality_setting": "bogus"}}):
            assert image_tool._resolve_gpt_quality() == "medium"

    def test_gpt_payload_includes_quality(self, image_tool):
        with patch("hermes_cli.config.load_config",
                   return_value={"image_gen": {"quality_setting": "low"}}):
            p = image_tool._build_fal_payload("fal-ai/gpt-image-1.5", "hi", "square")
        assert p["quality"] == "low"

    def test_non_gpt_model_ignores_quality_setting(self, image_tool):
        # honors_quality_setting is only set on gpt-image-1.5 — other
        # models should not inject a `quality` key.
        with patch("hermes_cli.config.load_config",
                   return_value={"image_gen": {"quality_setting": "high"}}):
            p = image_tool._build_fal_payload("fal-ai/flux-2-pro", "hi", "square")
        assert "quality" not in p


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

class TestModelResolution:

    def test_no_config_falls_back_to_default(self, image_tool):
        with patch("hermes_cli.config.load_config", return_value={}):
            mid, meta = image_tool._resolve_fal_model()
        assert mid == "fal-ai/flux-2/klein/9b"

    def test_valid_config_model_is_used(self, image_tool):
        with patch("hermes_cli.config.load_config",
                   return_value={"image_gen": {"model": "fal-ai/flux-2-pro"}}):
            mid, meta = image_tool._resolve_fal_model()
        assert mid == "fal-ai/flux-2-pro"
        assert meta["upscale"] is True  # flux-2-pro keeps backward-compat upscaling

    def test_unknown_model_falls_back_to_default_with_warning(self, image_tool, caplog):
        with patch("hermes_cli.config.load_config",
                   return_value={"image_gen": {"model": "fal-ai/nonexistent-9000"}}):
            mid, _ = image_tool._resolve_fal_model()
        assert mid == "fal-ai/flux-2/klein/9b"

    def test_env_var_fallback_when_no_config(self, image_tool, monkeypatch):
        monkeypatch.setenv("FAL_IMAGE_MODEL", "fal-ai/z-image/turbo")
        with patch("hermes_cli.config.load_config", return_value={}):
            mid, _ = image_tool._resolve_fal_model()
        assert mid == "fal-ai/z-image/turbo"

    def test_config_wins_over_env_var(self, image_tool, monkeypatch):
        monkeypatch.setenv("FAL_IMAGE_MODEL", "fal-ai/z-image/turbo")
        with patch("hermes_cli.config.load_config",
                   return_value={"image_gen": {"model": "fal-ai/nano-banana"}}):
            mid, _ = image_tool._resolve_fal_model()
        assert mid == "fal-ai/nano-banana"


# ---------------------------------------------------------------------------
# Aspect ratio handling
# ---------------------------------------------------------------------------

class TestAspectRatioNormalization:

    def test_invalid_aspect_defaults_to_landscape(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/flux-2/klein/9b", "hi", "cinemascope")
        assert p["image_size"] == "landscape_16_9"

    def test_uppercase_aspect_is_normalized(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/flux-2/klein/9b", "hi", "PORTRAIT")
        assert p["image_size"] == "portrait_16_9"

    def test_empty_aspect_defaults_to_landscape(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/flux-2/klein/9b", "hi", "")
        assert p["image_size"] == "landscape_16_9"


# ---------------------------------------------------------------------------
# Schema + registry integrity
# ---------------------------------------------------------------------------

class TestRegistryIntegration:

    def test_schema_exposes_only_prompt_and_aspect_ratio_to_agent(self, image_tool):
        """The agent-facing schema must stay tight — model selection is a
        user-level config choice, not an agent-level arg."""
        props = image_tool.IMAGE_GENERATE_SCHEMA["parameters"]["properties"]
        assert set(props.keys()) == {"prompt", "aspect_ratio"}

    def test_aspect_ratio_enum_is_three_values(self, image_tool):
        enum = image_tool.IMAGE_GENERATE_SCHEMA["parameters"]["properties"]["aspect_ratio"]["enum"]
        assert set(enum) == {"landscape", "square", "portrait"}
