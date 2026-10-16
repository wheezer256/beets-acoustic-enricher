"""LLM providers.

Interface: ``analyze(audio_path, metadata) -> dict`` returning the raw result
(mood/style/elements/energy_arc). Each provider also sets ``last_usage`` to a
dict of token counts when the API reports them, for cost accounting.

Audio providers (gemini, openai, ollama, llamacpp) receive the clip MP3.
The anthropic provider is metadata-only: Claude has no audio input API.
"""

from __future__ import annotations

import base64
import json
import os

from . import schema as S


class QuotaError(Exception):
    """Raised when a provider's quota, rate limit or spend cap is exceeded."""


def is_quota_error(exc) -> bool:
    try:
        if hasattr(exc, 'response') and exc.response is not None \
                and exc.response.status_code in (402, 429):
            return True
    except Exception:
        pass
    if type(exc).__name__ in ('RateLimitError', 'ResourceExhausted', 'TooManyRequests'):
        return True
    if getattr(exc, 'status_code', None) in (402, 429) or getattr(exc, 'code', None) in (402, 429):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in ('quota', 'resource_exhausted', 'budget', 'spend limit'))


def parse_json(text: str) -> dict:
    text = (text or '').strip()
    if text.startswith('```'):
        text = '\n'.join(text.split('\n')[1:])
        if text.endswith('```'):
            text = text[:text.rfind('```')]
        text = text.strip()
    return json.loads(text)


def gemini_thinking(model: str, budget: int, level: str) -> dict | None:
    """Gemini thinking control, as REST camelCase JSON.

    Gemini 3.x models take a `thinkingLevel` (MINIMAL gives no thinking tokens);
    Gemini 2.x models take a `thinkingBudget` (0 disables thinking). Returns None
    to leave the model default."""
    name = model.split('/')[-1]
    if name.startswith('gemini-3'):
        return {'thinkingLevel': level.upper()} if level else None
    return {'thinkingBudget': budget} if budget >= 0 else None


def temperature(cfg) -> float | None:
    """Sampling temperature from config; None (key absent or '') = provider default."""
    if 'temperature' not in cfg:
        return None
    val = cfg['temperature'].get()
    return None if val in (None, '') else float(val)


def _b64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def _secret(cfg, key: str, env: str) -> str:
    """Config value, else `<key>_file` contents, else the environment variable."""
    val = cfg[key].get() if key in cfg else ''
    if val:
        return str(val)
    fkey = f'{key}_file'
    if fkey in cfg and cfg[fkey].get():
        with open(os.path.expanduser(cfg[fkey].get())) as f:
            return f.read().strip()
    return os.environ.get(env, '')


class GeminiProvider:
    """Google Gemini (AI Studio key or Vertex AI), audio sent inline."""

    def __init__(self, cfg, log):
        from google import genai
        self._log = log
        model = cfg['model'].get()
        self._model = model if model.startswith('models/') else f'models/{model}'
        self._thinking = gemini_thinking(model, cfg['thinking_budget'].get(int),
                                         cfg['thinking_level'].get() or '')
        self._temperature = temperature(cfg)
        self._meta = cfg['include_metadata'].get(bool)
        if cfg['use_vertex'].get(bool):
            self._client = genai.Client(vertexai=True, project=cfg['project'].get(),
                                        location=cfg['location'].get())
        else:
            api_key = _secret(cfg, 'api_key', 'GEMINI_API_KEY')
            if not api_key:
                raise RuntimeError('No Gemini API key. Set GEMINI_API_KEY, '
                                   'acoustic_enricher.api_key or api_key_file.')
            self._client = genai.Client(api_key=api_key)
        self.last_usage = None

    def generation_config(self):
        from google.genai import types
        kw = dict(response_mime_type='application/json',
                  response_schema=S.gemini_schema())
        if self._temperature is not None:
            kw['temperature'] = self._temperature
        if self._thinking:
            if 'thinkingLevel' in self._thinking:
                kw['thinking_config'] = types.ThinkingConfig(thinking_level=self._thinking['thinkingLevel'])
            else:
                kw['thinking_config'] = types.ThinkingConfig(thinking_budget=self._thinking['thinkingBudget'])
        return types.GenerateContentConfig(**kw)

    def analyze(self, audio_path, metadata):
        from google.genai import types
        with open(audio_path, 'rb') as f:
            audio = types.Part.from_bytes(data=f.read(), mime_type='audio/mp3')
        instructions, context = S.request_parts(metadata if self._meta else None)
        try:
            resp = self._client.models.generate_content(
                model=self._model, contents=[instructions, audio] + ([context] if context else []),
                config=self.generation_config())
        except Exception as e:
            if is_quota_error(e):
                raise QuotaError(str(e)) from e
            raise
        um = getattr(resp, 'usage_metadata', None)
        if um is not None:
            self.last_usage = {
                'prompt_tokens': getattr(um, 'prompt_token_count', None),
                'output_tokens': getattr(um, 'candidates_token_count', None),
                'thinking_tokens': getattr(um, 'thoughts_token_count', None),
                'cached_tokens': getattr(um, 'cached_content_token_count', None),
            }
        return parse_json(resp.text)


class OpenAICompatProvider:
    """Any OpenAI-compatible chat/completions endpoint with `input_audio` support:
    LiteLLM proxy, vLLM, OpenRouter, llama.cpp's llama-server, ..."""

    url_key, url_default = 'openai_url', 'http://localhost:4000/v1/chat/completions'

    def __init__(self, cfg, log):
        self._log = log
        self._url = cfg[self.url_key].get() or self.url_default
        self._model = cfg['model'].get()
        self._timeout = cfg['timeout'].get(int)
        self._key = _secret(cfg, 'openai_api_key', 'OPENAI_API_KEY') \
            if self.url_key == 'openai_url' else ''
        self._reasoning = cfg['reasoning_effort'].get() if 'reasoning_effort' in cfg else None
        self._strict = cfg['structured_output'].get(bool) if 'structured_output' in cfg else True
        self._temperature = temperature(cfg)
        self._meta = cfg['include_metadata'].get(bool) if 'include_metadata' in cfg else False
        self.last_usage = None

    def _payload(self, audio_path, metadata=None):
        p = {
            'model': self._model,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': S.request_parts()[0]},
                {'type': 'input_audio', 'input_audio': {'data': _b64(audio_path), 'format': 'mp3'}},
            ] + ([{'type': 'text', 'text': S.context_text(metadata)}]
                 if self._meta and S.context_text(metadata) else [])}],
            'response_format': (
                {'type': 'json_schema',
                 'json_schema': {'name': 'acoustic_analysis', 'schema': S.json_schema(), 'strict': True}}
                if self._strict else {'type': 'json_object'}),
        }
        if self._reasoning:
            p['reasoning_effort'] = self._reasoning
        if self._temperature is not None:
            p['temperature'] = self._temperature
        return p

    def analyze(self, audio_path, metadata):
        import requests
        headers = {'Authorization': f'Bearer {self._key}'} if self._key else {}
        try:
            r = requests.post(self._url, json=self._payload(audio_path, metadata), headers=headers,
                              timeout=self._timeout)
        except Exception as e:
            if is_quota_error(e):
                raise QuotaError(str(e)) from e
            raise
        if r.status_code >= 400:
            # Proxies (e.g. LiteLLM) report spend caps as HTTP 400 with the reason in
            # the body, which raise_for_status() would hide.
            detail = f'HTTP {r.status_code}: {r.text[:300]}'
            if r.status_code in (402, 429) or is_quota_error(Exception(detail)):
                raise QuotaError(detail)
            raise RuntimeError(detail)
        body = r.json()
        u = body.get('usage') or {}
        self.last_usage = {
            'prompt_tokens': u.get('prompt_tokens'),
            'output_tokens': u.get('completion_tokens'),
            'audio_tokens': (u.get('prompt_tokens_details') or {}).get('audio_tokens'),
            'thinking_tokens': (u.get('completion_tokens_details') or {}).get('reasoning_tokens'),
        }
        return parse_json(body['choices'][0]['message']['content'])


class LlamaCppProvider(OpenAICompatProvider):
    """llama-server (OpenAI-compatible) with native audio input."""
    url_key, url_default = 'llama_url', 'http://localhost:8080/v1/chat/completions'


class OllamaProvider:
    """Ollama /api/chat; requires a model with audio/multimodal support."""

    def __init__(self, cfg, log):
        self._log = log
        self._url = cfg['ollama_url'].get() or 'http://localhost:11434/api/chat'
        self._model = cfg['model'].get()
        self._timeout = cfg['timeout'].get(int)
        self.last_usage = None

    def analyze(self, audio_path, metadata):
        import requests
        payload = {
            'model': self._model,
            'messages': [{'role': 'user', 'content': S.audio_prompt(), 'images': [_b64(audio_path)]}],
            'format': S.json_schema(),
            'stream': False,
        }
        try:
            r = requests.post(self._url, json=payload, timeout=self._timeout)
            r.raise_for_status()
        except Exception as e:
            if is_quota_error(e):
                raise QuotaError(str(e)) from e
            raise
        return parse_json(r.json()['message']['content'])


class AnthropicProvider:
    """Claude via the Anthropic SDK: metadata-only inference (no audio input)."""

    def __init__(self, cfg, log):
        import anthropic
        self._log = log
        api_key = _secret(cfg, 'anthropic_api_key', 'ANTHROPIC_API_KEY')
        if not api_key:
            raise RuntimeError('No Anthropic API key. Set ANTHROPIC_API_KEY or '
                               'acoustic_enricher.anthropic_api_key.')
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = cfg['model'].get()
        self.last_usage = None

    def analyze(self, audio_path, metadata):
        meta_lines = '\n'.join(f'{k.title()}: {v}' for k, v in metadata.items() if v)
        try:
            resp = self._client.messages.create(
                model=self._model, max_tokens=700,
                messages=[{'role': 'user', 'content': S.text_prompt(meta_lines)}])
        except Exception as e:
            if is_quota_error(e):
                raise QuotaError(str(e)) from e
            raise
        u = getattr(resp, 'usage', None)
        if u is not None:
            self.last_usage = {'prompt_tokens': u.input_tokens, 'output_tokens': u.output_tokens}
        return parse_json(resp.content[0].text)


PROVIDERS = {
    'gemini': GeminiProvider,
    'openai': OpenAICompatProvider,
    'llamacpp': LlamaCppProvider,
    'ollama': OllamaProvider,
    'anthropic': AnthropicProvider,
}
AUDIO_PROVIDERS = {'gemini', 'openai', 'llamacpp', 'ollama'}
