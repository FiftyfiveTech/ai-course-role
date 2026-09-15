#!/usr/bin/env python3
"""
Env doctor: checks HF_TOKEN, Groq API key, and Ollama.
Prints PASS/FAIL per dependency. Exits non-zero if any check fails.
A silent skip is a FAIL — this script never skips a check.
"""

import os
import subprocess
import sys
import urllib.request
import urllib.error

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    line = f"  [{status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)


def check_hf_token() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        check("HF_TOKEN", False, "not set or empty")
        return
    if not token.startswith("hf_"):
        check("HF_TOKEN", False, f"unexpected format (got {token[:6]}…)")
        return
    # whoami works for classic tokens; fine-grained tokens use whoami-v2
    for url in [
        "https://huggingface.co/api/whoami-v2",
        "https://huggingface.co/api/whoami",
    ]:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                check("HF_TOKEN", resp.status == 200, "token accepted by HF API")
                return
        except urllib.error.HTTPError as e:
            if e.code == 401:
                continue
            check("HF_TOKEN", False, f"HF API returned {e.code}")
            return
        except Exception as e:
            check("HF_TOKEN", False, f"network error: {e}")
            return
    check("HF_TOKEN", False, "token rejected by both whoami endpoints (401)")


def check_groq() -> None:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        check("GROQ_API_KEY", False, "not set or empty")
        return
    if not key.startswith("gsk_"):
        check("GROQ_API_KEY", False, f"unexpected format (got {key[:6]}…)")
        return
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={
            "Authorization": f"Bearer {key}",
            "User-Agent": "python-groq/0.1",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            check("GROQ_API_KEY", resp.status == 200, "key accepted by Groq API")
    except urllib.error.HTTPError as e:
        check("GROQ_API_KEY", False, f"Groq API returned {e.code}")
    except Exception as e:
        check("GROQ_API_KEY", False, f"network error: {e}")


def check_ollama() -> None:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as resp:
            check("ollama", resp.status == 200, "daemon reachable at localhost:11434")
    except Exception:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=10,
            )
            check("ollama", result.returncode == 0, "CLI reachable")
        except FileNotFoundError:
            check("ollama", False, "binary not found — run: curl https://ollama.com/install.sh | sh")
        except subprocess.TimeoutExpired:
            check("ollama", False, "ollama list timed out")
        except Exception as e:
            check("ollama", False, str(e))


def main() -> int:
    print("=== env doctor ===")
    check_hf_token()
    check_groq()
    check_ollama()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n  {passed} PASS  {failed} FAIL")

    if failed:
        print("\nFill in missing values in ~/.config/ and re-run `make doctor`.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
