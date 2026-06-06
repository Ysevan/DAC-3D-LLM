from __future__ import annotations

import json
import os
import subprocess
import time
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.home() / "DAC-3D-LLM"
PROJECT = ROOT / "dac3d_iim_assistant"
RESULTS = ROOT / "\u6d4b\u8bd5\u7ed3\u679c"
LLM_DIR = RESULTS / "01_step1_llm_evidence"
AUTO_DIR = RESULTS / "04_step4_function_exception"
PERF_DIR = RESULTS / "03_step3_performance"


def load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in (PROJECT / ".env").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    os.environ.update(values)
    os.environ["DAC3D_LLM_PROVIDER"] = "openai_compatible"
    os.environ.setdefault("DAC3D_TIMEOUT_SECONDS", "120")
    return values


def ensure_dirs() -> None:
    for path in (LLM_DIR, LLM_DIR / "screenshots_success", AUTO_DIR, PERF_DIR):
        path.mkdir(parents=True, exist_ok=True)


def response_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("text"):
                parts.append(str(content["text"]))
    return "".join(parts).strip()


def safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def draw_png(path: Path, title: str, lines: list[str]) -> None:
    img = Image.new("RGB", (1500, 980), "white")
    draw = ImageDraw.Draw(img)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    title_font = ImageFont.truetype(str(font_path), 34) if font_path.exists() else ImageFont.load_default()
    text_font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default()
    draw.rectangle((0, 0, 1500, 88), fill=(244, 247, 251))
    draw.text((42, 24), title, fill=(20, 35, 55), font=title_font)
    y = 126
    for line in lines:
        for part in textwrap.wrap(line, width=54, replace_whitespace=False):
            draw.text((52, y), part, fill=(25, 25, 25), font=text_font)
            y += 40
        y += 8
    img.save(path)


def run_llm_success_rounds(env_values: dict[str, str]) -> None:
    base = env_values.get("DAC3D_LLM_API_BASE_URL", "").rstrip("/")
    key = env_values.get("DAC3D_LLM_API_KEY", "")
    model = env_values.get("DAC3D_LLM_MODEL_NAME", "gpt-5.5")
    prompts = [
        "\u8bf7\u7528\u4e00\u53e5\u8bdd\u8bf4\u660eDAC-3D\u7cfb\u7edf\u4e2d\u626b\u63cf\u533a\u57df\u53c2\u6570\u7684\u4f5c\u7528\u3002",
        "\u5982\u679c\u7528\u6237\u53ea\u8bf4\u201c\u5f00\u59cb\u626b\u63cf\u201d\uff0c\u52a9\u624b\u5e94\u8be5\u8ffd\u95ee\u54ea\u4e9b\u5fc5\u8981\u4fe1\u606f\uff1f",
        "\u8bf7\u8bf4\u660e\u77e5\u8bc6\u5e93\u68c0\u7d22\u5728\u53c2\u6570\u95ee\u7b54\u4e2d\u7684\u4f5c\u7528\u3002",
        "\u8bf7\u7528\u7b80\u77ed\u4e2d\u6587\u89e3\u91ca\u5212\u75d5\u7f3a\u9677\u4e25\u91cd\u5ea6\u5224\u65ad\u9700\u8981\u54ea\u4e9b\u4f9d\u636e\u3002",
        "\u8bf7\u8bf4\u660e\u4e3a\u4ec0\u4e48\u5de5\u4e1a\u68c0\u6d4b\u52a9\u624b\u9700\u8981\u547d\u4ee4\u786e\u8ba4\u673a\u5236\u3002",
    ]
    config = {
        "provider": "openai_compatible",
        "api_base_url": base,
        "api_key_present": bool(key),
        "api_key_redacted": f"***{key[-4:]}" if key else "",
        "model": model,
        "api_path": "/responses",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (LLM_DIR / "llm_env_config_redacted.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"success": 0, "failed": 0, "total_input_tokens": 0, "total_output_tokens": 0, "total_tokens": 0, "errors": []}
    with (LLM_DIR / "llm_5_rounds_payload_response.jsonl").open("w", encoding="utf-8") as detail, (
        LLM_DIR / "backend_external_api_call_log.jsonl"
    ).open("w", encoding="utf-8") as backend:
        with httpx.Client(timeout=180) as client:
            for idx, prompt in enumerate(prompts, 1):
                payload = {"model": model, "input": prompt, "max_output_tokens": 700}
                ts = datetime.now().isoformat(timespec="milliseconds")
                start = time.perf_counter()
                record: dict[str, Any] = {
                    "round": idx,
                    "timestamp": ts,
                    "request": {
                        "url": f"{base}/responses",
                        "headers": {"Authorization": "Bearer ***", "Content-Type": "application/json"},
                        "payload": payload,
                    },
                }
                try:
                    response = client.post(
                        f"{base}/responses",
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json=payload,
                    )
                    elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
                    data = response.json()
                    answer = response_text(data)
                    usage = data.get("usage", {})
                    ok = response.status_code < 400 and bool(answer)
                    if ok:
                        summary["success"] += 1
                    else:
                        summary["failed"] += 1
                        summary["errors"].append({"round": idx, "status_code": response.status_code, "body": response.text[:500]})
                    summary["total_input_tokens"] += int(usage.get("input_tokens", 0) or 0)
                    summary["total_output_tokens"] += int(usage.get("output_tokens", 0) or 0)
                    summary["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
                    record.update(
                        {
                            "status_code": response.status_code,
                            "elapsed_ms": elapsed_ms,
                            "response": {
                                "id": data.get("id"),
                                "model": data.get("model"),
                                "status": data.get("status"),
                                "answer": answer,
                                "usage": usage,
                            },
                        }
                    )
                    backend.write(
                        safe_json(
                            {
                                "timestamp": ts,
                                "round": idx,
                                "external_api": f"{base}/responses",
                                "model": data.get("model", model),
                                "status_code": response.status_code,
                                "elapsed_ms": elapsed_ms,
                                "usage": usage,
                                "answer_chars": len(answer),
                            }
                        )
                        + "\n"
                    )
                    draw_png(
                        LLM_DIR / "screenshots_success" / f"LLM-SUCCESS-{idx:02d}.png",
                        f"真实LLM调用成功证据 {idx}/5",
                        [
                            f"时间戳：{ts}",
                            f"Provider：openai_compatible",
                            f"接口：{base}/responses",
                            f"模型：{data.get('model', model)}",
                            f"状态码：{response.status_code}，耗时：{elapsed_ms} ms",
                            f"Token：input={usage.get('input_tokens')}，output={usage.get('output_tokens')}，total={usage.get('total_tokens')}",
                            f"用户问题：{prompt}",
                            f"模型回复：{answer[:520]}",
                        ],
                    )
                except Exception as exc:
                    elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
                    summary["failed"] += 1
                    summary["errors"].append({"round": idx, "error": repr(exc)})
                    record.update({"status_code": None, "elapsed_ms": elapsed_ms, "error": repr(exc)})
                    backend.write(
                        safe_json(
                            {
                                "timestamp": ts,
                                "round": idx,
                                "external_api": f"{base}/responses",
                                "model": model,
                                "status_code": None,
                                "elapsed_ms": elapsed_ms,
                                "error": repr(exc),
                            }
                        )
                        + "\n"
                    )
                detail.write(safe_json(record) + "\n")
    (LLM_DIR / "llm_call_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pytest_success() -> None:
    pytest_exe = Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python310" / "Scripts" / "pytest.exe"
    start = time.perf_counter()
    proc = subprocess.run(
        [str(pytest_exe), "tests", "-q"],
        cwd=str(PROJECT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    elapsed = round(time.perf_counter() - start, 3)
    log = proc.stdout
    (AUTO_DIR / "pytest_full_run_success.log").write_text(log, encoding="utf-8")
    summary = {
        "command": f"{pytest_exe} tests -q",
        "return_code": proc.returncode,
        "elapsed_seconds": elapsed,
        "result_line": log.strip().splitlines()[-1] if log.strip() else "",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (AUTO_DIR / "automated_test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    env_values = load_env()
    run_llm_success_rounds(env_values)
    run_pytest_success()
    print(json.dumps({"llm_summary": str(LLM_DIR / "llm_call_summary.json"), "pytest": str(AUTO_DIR / "automated_test_summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
