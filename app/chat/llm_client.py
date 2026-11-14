"""
LLM CLIENTS - the only code that talks to a model provider.

  GeminiClient  real Google Gemini over plain HTTPS (no extra library needed)
  MockLLM       scripted replies for tests and offline demos (no key, no network)

Every client has one method:   generate(system_prompt, user_prompt) -> str
and raises LLMError(code, message, retryable) on failure. Error messages never contain
the API key or the provider's raw response.

call_with_retries() adds exponential backoff for temporary problems (timeout, rate limit,
server error). A CircuitBreaker stops calling a provider that keeps failing, so a user
is not made to wait for a model that is down.
"""
import json
import socket
import time
import urllib.error
import urllib.request

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_OUTPUT_TOKENS = 1024
MAX_RESPONSE_CHARS = 20_000


class LLMError(Exception):
    def __init__(self, code, message, retryable=False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable

    def to_dict(self):
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


# ------------------------------------------------------------------ transport
def _urllib_post(url, headers, body, timeout):
    """Default HTTP function. Returns (status, text). Raises TimeoutError / OSError on network trouble."""
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:          # noqa: S310 (fixed https URL)
            return resp.status, resp.read(MAX_RESPONSE_CHARS * 4).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as err:
        return err.code, ""
    except (socket.timeout, TimeoutError):
        raise TimeoutError("timeout")
    except urllib.error.URLError as err:
        if isinstance(err.reason, (socket.timeout, TimeoutError)):
            raise TimeoutError("timeout")
        raise OSError("network error")


class GeminiClient:
    def __init__(self, api_key, model="gemini-3.5-flash", timeout=15.0, http_post=None):
        self._key = api_key
        self.model = model
        self.timeout = timeout
        self._post = http_post or _urllib_post

    def build_request(self, system_prompt, user_prompt):
        url = GEMINI_URL.format(model=self.model)
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._key}   # key in a header, never in the URL
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "maxOutputTokens": MAX_OUTPUT_TOKENS},
        }
        return url, headers, json.dumps(body).encode("utf-8")

    def generate(self, system_prompt, user_prompt):
        url, headers, body = self.build_request(system_prompt, user_prompt)
        try:
            status, text = self._post(url, headers, body, self.timeout)
        except TimeoutError:
            raise LLMError("TIMEOUT", "The model did not answer in time.", retryable=True)
        except OSError:
            raise LLMError("NETWORK", "Could not reach the model service.", retryable=True)
        if status == 429:
            raise LLMError("RATE_LIMIT", "The model service is rate limiting requests.", retryable=True)
        if status >= 500:
            raise LLMError("SERVER_ERROR", f"The model service had a problem (HTTP {status}).", retryable=True)
        if status in (401, 403):
            raise LLMError("AUTH", "The model service rejected the API key.", retryable=False)
        if status != 200:
            raise LLMError("BAD_REQUEST", f"The model service refused the request (HTTP {status}).", retryable=False)
        try:
            data = json.loads(text)
            parts = data["candidates"][0]["content"]["parts"]
            out = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise LLMError("EMPTY_RESPONSE", "The model returned no usable text.", retryable=False)
        if not out.strip():
            raise LLMError("EMPTY_RESPONSE", "The model returned no usable text.", retryable=False)
        return out


class MockLLM:
    """Scripted model. Each item is a string reply or an Exception to raise. Records every call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.model = "mock"

    def generate(self, system_prompt, user_prompt):
        self.calls.append({"system": system_prompt, "user": user_prompt})
        if not self.script:
            raise LLMError("EMPTY_RESPONSE", "mock script is empty")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ---------------------------------------------------------- retries + breaker
def call_with_retries(fn, max_retries=2, base_delay=0.5, max_delay=4.0, sleep=time.sleep):
    """Run fn(). On a retryable LLMError wait 0.5s, 1s, 2s... then try again. Returns (result, attempts)."""
    attempt = 0
    while True:
        try:
            return fn(), attempt + 1
        except LLMError as err:
            if not err.retryable or attempt >= max_retries:
                err.attempts = attempt + 1
                raise
            sleep(min(max_delay, base_delay * (2 ** attempt)))
            attempt += 1


class CircuitBreaker:
    """After `threshold` failures in a row the breaker opens for `cooldown` seconds."""

    def __init__(self, threshold=3, cooldown=60.0, clock=time.monotonic):
        self.threshold, self.cooldown, self.clock = threshold, cooldown, clock
        self.failures, self.opened_at = 0, None

    def is_open(self):
        if self.opened_at is None:
            return False
        if self.clock() - self.opened_at >= self.cooldown:
            self.opened_at, self.failures = None, self.threshold - 1       # allow one trial call
            return False
        return True

    def record_success(self):
        self.failures, self.opened_at = 0, None

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = self.clock()
