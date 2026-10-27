"""Thin LLM client. Pick a provider with environment variables (see .env.example).

Supported out of the box:
  LLM_PROVIDER=ollama     free, local (https://ollama.com) - no API key needed
  LLM_PROVIDER=anthropic  needs ANTHROPIC_API_KEY and `pip install anthropic`
  LLM_PROVIDER=openai     needs OPENAI_API_KEY and `pip install openai`

You may replace this file, add native tool calling, or use a framework
(LangGraph, LlamaIndex, etc.). Keep the provider configurable via env vars.
"""
import json
import os
import urllib.request
from pathlib import Path

# minimal .env loader (no extra dependency)
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.split("#", 1)[0].strip()
        if "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        self.model = model or os.getenv("LLM_MODEL", {
            "ollama": "llama3.1:8b", "anthropic": "claude-sonnet-4-5", "openai": "gpt-4o-mini"}[self.provider])
        self.calls = 0  # simple counter you can report for cost/latency

    def chat(self, messages: list[dict], system: str | None = None, max_tokens: int = 1024,
             temperature: float = 0.0, json_mode: bool = False) -> str:
        """messages: [{"role": "user"|"assistant", "content": str}] -> assistant text.

        json_mode=True asks Ollama/OpenAI to return valid JSON only (Anthropic ignores it; the agent
        validates the JSON itself either way).
        """
        self.calls += 1
        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(model=self.model, max_tokens=max_tokens, temperature=temperature,
                                          system=system or "", messages=messages)
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if self.provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("XAI_API_KEY"),base_url="https://api.x.ai/v1",timeout=30,max_retries=0,)
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            extra = {"response_format": {"type": "json_object"}} if json_mode else {}
            resp = client.chat.completions.create(model=self.model, messages=msgs, max_tokens=max_tokens,
                                                  temperature=temperature, **extra)
            return resp.choices[0].message.content or ""
        if self.provider == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            # Ollama's default context window (2048 tokens) is too small for tool results: make it configurable.
            options = {"temperature": temperature, "num_predict": max_tokens,
                       "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096"))}
            payload = {"model": self.model, "messages": msgs, "stream": False, "options": options}
            if json_mode:
                payload["format"] = "json"
            body = json.dumps(payload).encode()
            req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["message"]["content"]
        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")
