"""Unit tests for ConfigurationService with fake store and probe."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from captioner.core.application.configuration import (
    ConfigurationIssue,
    ConfigurationService,
    ConfigurationSnapshot,
    ExecutionPreset,
    GlobalSettings,
    ProviderConnectionResult,
    ProviderPublicSettings,
    ProviderSettingsUpdate,
    built_in_presets,
    default_configuration_snapshot,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.stage import PipelineProfile
from captioner.core.ports.configuration_store import ProviderRuntimeProbeSettings


@dataclass
class FakeStore:
    snapshot: ConfigurationSnapshot = field(default_factory=default_configuration_snapshot)
    save_global_calls: list[GlobalSettings] = field(default_factory=lambda: [])
    save_provider_calls: list[ProviderSettingsUpdate] = field(default_factory=lambda: [])
    save_preset_calls: list[ExecutionPreset] = field(default_factory=lambda: [])
    delete_preset_calls: list[str] = field(default_factory=lambda: [])
    fail_save: bool = False

    def load_snapshot(self) -> ConfigurationSnapshot:
        return self.snapshot

    def save_global(self, settings: GlobalSettings) -> None:
        if self.fail_save:
            raise AppError("config.write_failed")
        self.save_global_calls.append(settings)
        self.snapshot = ConfigurationSnapshot(
            global_settings=settings,
            presets=self.snapshot.presets,
            provider=self.snapshot.provider,
            issues=(),
        )

    def save_provider(self, update: ProviderSettingsUpdate) -> None:
        if self.fail_save:
            raise AppError("config.write_failed")
        self.save_provider_calls.append(update)
        self.snapshot = ConfigurationSnapshot(
            global_settings=self.snapshot.global_settings,
            presets=self.snapshot.presets,
            provider=ProviderPublicSettings(
                profile_name=update.profile_name,
                base_url=update.base_url,
                model=update.model,
                max_concurrency=update.max_concurrency,
                request_timeout_sec=update.request_timeout_sec,
                max_retries=update.max_retries,
                temperature=update.temperature,
                tokenizer=update.tokenizer,  # type: ignore[arg-type]
                credential_source="config" if update.api_key else "missing",
            ),
            issues=(),
        )

    def save_user_preset(self, preset: ExecutionPreset) -> None:
        if self.fail_save:
            raise AppError("config.write_failed")
        self.save_preset_calls.append(preset)
        builtins = built_in_presets()
        users = tuple(
            item for item in self.snapshot.presets if not item.built_in and item.name != preset.name
        )
        self.snapshot = ConfigurationSnapshot(
            global_settings=self.snapshot.global_settings,
            presets=builtins + users + (preset,),
            provider=self.snapshot.provider,
            issues=(),
        )

    def delete_user_preset(self, name: str) -> None:
        if self.fail_save:
            raise AppError("config.write_failed")
        self.delete_preset_calls.append(name)
        remaining = tuple(
            item for item in self.snapshot.presets if item.name != name or item.built_in
        )
        self.snapshot = ConfigurationSnapshot(
            global_settings=self.snapshot.global_settings,
            presets=remaining,
            provider=self.snapshot.provider,
            issues=(),
        )

    def resolve_provider_for_test(
        self,
        update: ProviderSettingsUpdate,
    ) -> ProviderRuntimeProbeSettings:
        key = update.api_key or "resolved-key"
        return ProviderRuntimeProbeSettings(
            base_url=update.base_url,
            api_key=key,
            timeout_sec=update.request_timeout_sec,
        )


@dataclass
class FakeProbe:
    result: ProviderConnectionResult = field(
        default_factory=lambda: ProviderConnectionResult(True, "llm.connection_ok")
    )
    error: AppError | None = None
    calls: list[ProviderRuntimeProbeSettings] = field(default_factory=lambda: [])

    def test(self, settings: ProviderRuntimeProbeSettings) -> ProviderConnectionResult:
        self.calls.append(settings)
        if self.error is not None:
            raise self.error
        return self.result


def test_load_defaults_and_invalid_issue() -> None:
    store = FakeStore(
        snapshot=default_configuration_snapshot(
            issues=(ConfigurationIssue(code="config.settings_invalid"),)
        )
    )
    service = ConfigurationService(store=store, provider_probe=FakeProbe())
    snapshot = service.load()
    assert snapshot.global_settings.locale == "en"
    assert [p.name for p in snapshot.presets[:3]] == ["deterministic", "fast", "quality"]
    assert snapshot.issues[0].code == "config.settings_invalid"


def test_save_global_provider_and_presets() -> None:
    store = FakeStore()
    service = ConfigurationService(store=store, provider_probe=FakeProbe())
    settings = GlobalSettings(locale="zh-CN", default_preset_name="fast")
    snapshot = service.save_global(settings)
    assert snapshot.global_settings.locale == "zh-CN"

    update = ProviderSettingsUpdate(
        profile_name="default",
        base_url="https://example.com/v1",
        model="m",
        api_key="secret-key",
    )
    snapshot = service.save_provider(update)
    assert snapshot.provider.model == "m"
    assert "secret-key" not in repr(update)
    assert "secret-key" not in repr(snapshot)

    preset = ExecutionPreset(
        name="my preset",
        display_name="My Preset",
        built_in=False,
        pipeline_profile=PipelineProfile.FAST,
        model_ref="tiny",
        device="auto",
        compute_type="default",
        source_language=None,
        target_language="zh-CN",
        provider_profile="default",
    )
    snapshot = service.save_user_preset(preset)
    assert any(item.name == "my preset" for item in snapshot.presets)
    snapshot = service.delete_user_preset("my preset")
    assert all(item.name != "my preset" for item in snapshot.presets)


def test_builtin_mutation_rejected() -> None:
    service = ConfigurationService(store=FakeStore(), provider_probe=FakeProbe())
    builtin = built_in_presets()[0]
    with pytest.raises(AppError, match=r"config\.preset_builtin_immutable"):
        service.save_user_preset(builtin)
    with pytest.raises(AppError, match=r"config\.preset_builtin_immutable"):
        service.delete_user_preset("deterministic")


@pytest.mark.parametrize(
    ("timeout", "temperature"),
    [
        (-1.0, 0.1),
        (float("nan"), 0.1),
        (float("inf"), 0.1),
        (120.0, -0.1),
        (120.0, float("nan")),
        (120.0, float("inf")),
    ],
)
def test_provider_update_rejects_non_finite_or_negative_timeout_temperature(
    timeout: float,
    temperature: float,
) -> None:
    with pytest.raises(AppError, match=r"config\.provider_invalid"):
        ProviderSettingsUpdate(
            profile_name="default",
            base_url="https://example.com/v1",
            model="m",
            request_timeout_sec=timeout,
            temperature=temperature,
        )


def test_failed_save_raises_and_provider_test() -> None:
    store = FakeStore(fail_save=True)
    previous = store.snapshot
    service = ConfigurationService(store=store, provider_probe=FakeProbe())
    with pytest.raises(AppError, match=r"config\.write_failed"):
        service.save_global(GlobalSettings(locale="zh-CN"))
    assert store.snapshot is previous

    probe = FakeProbe(
        error=AppError("llm.connection_auth_failed"),
    )
    service = ConfigurationService(store=FakeStore(), provider_probe=probe)
    with pytest.raises(AppError, match=r"llm\.connection_auth_failed"):
        service.test_provider(
            ProviderSettingsUpdate(
                profile_name="default",
                base_url="https://example.com/v1",
                model="m",
                api_key="k",
            )
        )
    ok_probe = FakeProbe()
    service = ConfigurationService(store=FakeStore(), provider_probe=ok_probe)
    result = service.test_provider(
        ProviderSettingsUpdate(
            profile_name="default",
            base_url="https://example.com/v1",
            model="m",
            api_key="k",
        )
    )
    assert result.ok is True
    assert result.code == "llm.connection_ok"
    assert "k" not in repr(ok_probe.calls[0])
