import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()
# Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise EnvironmentError("Missing OPENROUTER_API_KEY environment variable. Set it before running.")

URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "X-OpenRouter-Title": "AccidentPredictionApp"
}

ALLOWED_MODELS = {"openrouter/free"}
MAX_PROMPT_CHARS = 2000
RETRY_ATTEMPTS = 3
TIMEOUT = 10


def validate_prompt(prompt: str):
    if not prompt or len(prompt) > MAX_PROMPT_CHARS:
        return False, f"Prompt must be 1..{MAX_PROMPT_CHARS} characters"
    banned_terms = ["password", "credit card", "ssn"]
    low = prompt.lower()
    for term in banned_terms:
        if term in low:
            return False, f"Prompt contains disallowed term: {term}"
    return True, ""


def make_request(payload: dict):
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.post(URL, headers=HEADERS, json=payload, timeout=TIMEOUT)
            if resp.status_code == 200:
                return resp
            if 500 <= resp.status_code < 600:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
        except requests.RequestException:
            if attempt == RETRY_ATTEMPTS:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed to get successful response after retries")


def extract_json(text: Optional[str]) -> Optional[Any]:
    if not text:
        return None

    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    code_fence_match = re.search(r"```(?:json)?\s*(\{.*?|\[.*?])\s*```", text, re.DOTALL | re.IGNORECASE)
    if code_fence_match:
        try:
            return json.loads(code_fence_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_index = text.find("{")
    bracket_index = text.find("[")
    if brace_index == -1 and bracket_index == -1:
        return None

    start_index = min(index for index in [brace_index, bracket_index] if index != -1)
    if start_index != -1:
        for candidate in [text[start_index:], text[start_index:].strip("`\n")]:
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    return None


def validate_output(data: Any) -> bool:
    return isinstance(data, (dict, list))


def build_fallback_response(state_name: str, city_name: str, prediction: str) -> Dict[str, Any]:
    safe_state = state_name or "Unknown State"
    safe_city = city_name or "Unknown City"
    safe_prediction = prediction or "Accident hotspot"
    return {
        "state": safe_state,
        "city": safe_city,
        "prediction": safe_prediction,
        "results": [
            f"{safe_city} Main {safe_prediction} Zone",
            f"{safe_city} High Risk Junction",
            f"{safe_city} Priority Accident Corridor",
        ],
    }


def normalize_response(data: Any, state_name: str, city_name: str, prediction: str) -> Dict[str, Any]:
    fallback = build_fallback_response(state_name, city_name, prediction)
    if isinstance(data, dict):
        normalized_results = data.get("results") or []
        if isinstance(normalized_results, list) and all(isinstance(item, str) for item in normalized_results):
            return {
                "state": data.get("state") or state_name,
                "city": data.get("city") or city_name,
                "prediction": data.get("prediction") or prediction,
                "results": normalized_results,
            }
        return fallback

    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        fallback["results"] = data
        return fallback

    return fallback


def llm_finds_areas(state_name, city_name, prediction):
    if city_name == "Some City":
        city_name = "don't consider city list all areas in state"
    user_content = (
        f"You are a geographic information assistant. Return only valid JSON. "
        f"Provide a JSON object with the exact shape:"
        "{\n"
        "  \"state\": \"<state_name>\",\n"
        "  \"city\": \"<city_name>\",\n"
        "  \"prediction\": \"<prediction>\",\n"
        "  \"results\": [\n"
        "    \"<result1>\",\n"
        "    \"<result2>\"\n"
        "  ]\n"
        "}. "
        f"Suggest high-risk areas for {prediction} in {city_name}, {state_name}. "
        "Do not include markdown, commentary, or extra text."
    )

    ok, msg = validate_prompt(user_content)
    if not ok:
        print("Invalid prompt:", msg)
        return

    model = "openrouter/free"
    if model not in ALLOWED_MODELS:
        print("Model not allowed")
        return

    payload = {"model": model, "messages": [{"role": "user", "content": user_content}]}

    try:
        resp = make_request(payload)
    except Exception as exc:
        print("Request failed:", str(exc))
        return

    try:
        data = resp.json()
    except Exception:
        parsed = extract_json(resp.text)
        if parsed is None:
            print("Could not parse JSON from response")
            print(resp.text)
            return
        data = parsed

    assistant_text = None
    list_of_areas = []

    if isinstance(data, dict):
        choices = data.get("choices")
        if choices and isinstance(choices, list):
            message = choices[0].get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list):
                    assistant_text = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                else:
                    assistant_text = content

    if assistant_text:
        parsed = extract_json(assistant_text)
        if parsed and validate_output(parsed):
            normalized = normalize_response(parsed, state_name, city_name, prediction)
            list_of_areas = normalized.get("results", [])
            print(json.dumps(normalized, indent=2))
            return list_of_areas

        print("Assistant returned non-JSON or invalid shape. Falling back to default JSON response.")
        fallback = build_fallback_response(state_name, city_name, prediction)
        print(json.dumps(fallback, indent=2))
        return fallback.get("results", [])

    if validate_output(data):
        normalized = normalize_response(data, state_name, city_name, prediction)
        list_of_areas = normalized.get("results", [])
        print(json.dumps(normalized, indent=2))
        return list_of_areas

    print("Unknown response format. Raw:")
    print(data)
    fallback = build_fallback_response(state_name, city_name, prediction)
    print(json.dumps(fallback, indent=2))
    return fallback.get("results", [])


if __name__ == "__main__":
    llm_finds_areas(state_name="Tamil Nadu",city_name="Chennai",prediction="Intersection")
