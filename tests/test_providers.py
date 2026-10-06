"""Adapter/factory pattern for LLM and TTS providers: the factory reads
LLM_PROVIDER / TTS_PROVIDER from the environment, every provider shares one
interface, and the old import names keep working. No network calls.
"""

import pytest

from app.llm import client as llm
from app.tts import engine as tts


# ---- LLM ------------------------------------------------------------------


def test_groq_is_the_default_llm_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with llm.get_llm_client() as c:
        assert isinstance(c, llm.GroqLLMClient)
        assert isinstance(c, llm.BaseLLMClient)


@pytest.mark.parametrize(
    "provider, cls, base_url_var, model_var",
    [
        ("vllm", llm.VLLMClient, "COMPANY_LLM_BASE_URL", "COMPANY_LLM_MODEL"),
        ("llamacpp", llm.LlamaCppClient, "LLAMACPP_BASE_URL", "LLAMACPP_MODEL"),
        ("GROQ", llm.GroqLLMClient, "LLM_BASE_URL", "LLM_MODEL"),
    ],
)
def test_llm_provider_is_selected_from_env(monkeypatch, provider, cls, base_url_var, model_var):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv(base_url_var, "http://example.test/v1")
    monkeypatch.setenv(model_var, "some-model")
    with llm.get_llm_client() as c:
        assert isinstance(c, cls)
        assert c.config.base_url == "http://example.test/v1"
        assert c.config.model == "some-model"


def test_unknown_llm_provider_names_the_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        llm.get_llm_client()


def test_vllm_requires_a_model_and_says_which_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("COMPANY_LLM_MODEL", "")
    with pytest.raises(ValueError, match="COMPANY_LLM_MODEL"):
        llm.get_llm_client()


def test_llm_backward_compatible_alias():
    assert llm.LLMClient is llm.GroqLLMClient


def test_all_llm_providers_share_the_interface():
    for cls in llm.LLM_PROVIDERS.values():
        assert issubclass(cls, llm.BaseLLMClient)
        for method in ("complete", "stream", "stream_measured", "close"):
            assert callable(getattr(cls, method))


def test_reasoning_fallback_when_content_is_empty(monkeypatch):
    import httpx

    def handler(request):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "", "reasoning": "from reasoning"},
                               "finish_reason": "stop"}]},
        )

    c = llm.VLLMClient(llm.LLMConfig(base_url="http://example.test/v1", model="m"))
    c._http = httpx.Client(base_url="http://example.test/v1", transport=httpx.MockTransport(handler))
    assert c.complete([{"role": "user", "content": "hi"}]).text == "from reasoning"


# ---- TTS ------------------------------------------------------------------


def test_tts_factory_returns_a_base_client_with_speak(monkeypatch):
    monkeypatch.delenv("TTS_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    service = tts.get_tts_client()
    assert isinstance(service, tts.BaseTTSClient)
    assert isinstance(service, tts.TTSService)
    assert callable(service.speak)
    assert isinstance(service.engines[0], tts.GroqOrpheusTTS)
    assert isinstance(service.engines[-1], tts.EspeakTTS)


def test_tts_groq_without_key_degrades_to_local_fallbacks(monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "groq")
    monkeypatch.setenv("LLM_API_KEY", "")
    service = tts.get_tts_client()
    assert not any(isinstance(e, tts.GroqOrpheusTTS) for e in service.engines)
    assert isinstance(service.engines[-1], tts.EspeakTTS)


def test_tts_kokoro_missing_package_does_not_crash(monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "kokoro")
    service = tts.get_tts_client()  # kokoro may or may not be installed
    assert isinstance(service, tts.TTSService)
    speech = service.synthesize("")
    assert not speech.ok


def test_tts_piper_provider_reads_model_path(monkeypatch, tmp_path):
    monkeypatch.setenv("TTS_PROVIDER", "piper")
    monkeypatch.setenv("PIPER_MODEL_PATH", str(tmp_path / "not-a-voice.onnx"))
    service = tts.get_tts_client()  # download of a bogus voice fails gracefully
    assert isinstance(service, tts.TTSService)
    assert not any(isinstance(e, tts.PiperTTS) for e in service.engines)


def test_unknown_tts_provider_names_the_env_var(monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "bogus")
    with pytest.raises(ValueError, match="TTS_PROVIDER"):
        tts.get_tts_client()


def test_tts_backward_compatible_aliases():
    assert tts.OrpheusEngine is tts.GroqOrpheusTTS
    assert tts.PiperEngine is tts.PiperTTS


def test_all_tts_providers_share_the_interface():
    for cls in tts.TTS_PROVIDERS.values():
        assert issubclass(cls, tts.BaseTTSClient)


def test_tts_service_falls_through_to_next_engine():
    class Broken(tts.BaseTTSClient):
        def synthesize(self, text, out_path=None):
            return tts.Speech(text=text, error="boom")

        def close(self):
            pass

    class Works(tts.BaseTTSClient):
        def synthesize(self, text, out_path=None):
            return tts.Speech(text=text, wav_path=tts.Path("x.wav"))

        def close(self):
            pass

    service = tts.TTSService(engines=[Broken(), Works()])
    speech = service.speak("hello", autoplay=False)
    assert speech.ok
