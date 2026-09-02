#!/usr/bin/env python3
"""model-atlas + HF config.json -> db/coefficients.json 의 models 섹션 생성.

계산기가 필요로 하는 건 아키텍처 필드뿐이고, 전부 config.json 에서 기계적으로
나온다. 벤치마크는 한 번도 돌리지 않는다.
"""
from __future__ import annotations
import json, os, re, sys, tomllib, urllib.error, urllib.request
from pathlib import Path

ATLAS = Path.home() / "personal" / "model-atlas" / "data" / "models"
OUT = Path(__file__).resolve().parent.parent / "db" / "models.json"
HF = "https://huggingface.co/{}/resolve/main/config.json"
SKIP_MODALITY = {"text-to-image", "image", "audio"}

# config.json 이 transformers 기본값에 의존해 필드를 생략하는 경우에만 손으로 보정한다.
# 추측 금지 — 근거를 note 에 남기고, 없으면 차라리 항목을 버린다.
OVERRIDES = {
    "google/gemma-3": {
        "attn": "hybrid", "global_every": 6, "window": 1024,
        "note": "config.json 에 sliding_window_pattern 이 없고 transformers 기본값 6 "
                "(로컬 5 : 글로벌 1)에 의존. window 는 config 값.",
    },
}


def apply_override(hf_id: str, arch: dict) -> dict:
    for prefix, ov in OVERRIDES.items():
        if hf_id.startswith(prefix):
            return {**arch, **ov}
    return arch


def fetch_config(hf_id: str) -> dict:
    h = {"User-Agent": "serving-capacity/1.0"}
    tok = os.environ.get("HF_TOKEN")          # gated 레포(gemma, llama)에 필요
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(HF.format(hf_id), headers=h)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def text_cfg(cfg: dict) -> dict:
    """멀티모달 config 는 언어 모델 설정을 text_config 아래 둔다."""
    for k in ("text_config", "llm_config", "language_config"):
        if isinstance(cfg.get(k), dict) and "num_hidden_layers" in cfg[k]:
            return cfg[k]
    return cfg


def classify(c: dict) -> dict:
    """어텐션 구조를 판정한다. KV 캐시 공식이 여기서 갈린다."""
    n_head = c.get("num_attention_heads")
    kv_head = c.get("num_key_value_heads", n_head)
    hidden = c.get("hidden_size")
    head_dim = c.get("head_dim") or (hidden // n_head if hidden and n_head else None)
    layers = c.get("num_hidden_layers")
    a: dict = {"layers": layers, "kv_heads": kv_head, "head_dim": head_dim}

    # MLA (DeepSeek 계열): 압축 잠재 KV + 분리된 RoPE 키. kv_heads×head_dim 개념이 안 맞는다.
    if c.get("kv_lora_rank"):
        a.update(attn="mla", kv_lora_rank=c["kv_lora_rank"],
                 qk_rope_head_dim=c.get("qk_rope_head_dim", 64))
        return a

    win = c.get("sliding_window") or c.get("attention_window_size")
    types = c.get("layer_types") or c.get("attn_type_list")
    if win and types:                      # 로컬/글로벌 교차 (gemma-3, gpt-oss)
        full = sum(1 for t in types if "full" in str(t) or "global" in str(t))
        if 0 < full < len(types):
            a.update(attn="hybrid", window=win, global_every=round(len(types) / full))
            return a
    if win and c.get("sliding_window_pattern"):
        a.update(attn="hybrid", window=win, global_every=c["sliding_window_pattern"])
        return a
    if win and c.get("use_sliding_window", True) and not types:
        a.update(attn="swa", window=win)
        return a

    a["attn"] = "mha" if kv_head == n_head else ("mqa" if kv_head == 1 else "gqa")
    return a


def active_params(entry: dict, hf_id: str, total: float) -> float:
    if entry.get("active_params"):
        return entry["active_params"] / 1e9
    m = re.search(r"[-_]A(\d+(?:\.\d+)?)B", hf_id, re.I)   # Qwen3-235B-A22B 규약
    return float(m.group(1)) if m else total


def main() -> int:
    out, failed = {}, []
    files = sorted(ATLAS.glob("*/*.toml"))
    for f in files:
        e = tomllib.load(f.open("rb"))
        hf_id, mod = e.get("hf_id"), e.get("modality", "")
        if not hf_id or mod in SKIP_MODALITY:
            continue
        try:
            arch = apply_override(hf_id, classify(text_cfg(fetch_config(hf_id))))
        except urllib.error.HTTPError as err:
            failed.append((hf_id, f"HTTP {err.code}" + (" (gated)" if err.code in (401, 403) else "")))
            continue
        except Exception as err:                                  # noqa: BLE001
            failed.append((hf_id, type(err).__name__))
            continue
        if not all(arch.get(k) for k in ("layers", "head_dim", "kv_heads")):
            failed.append((hf_id, "config 필드 부족"))
            continue
        total = (e.get("params") or 0) / 1e9
        out[e["id"]] = {
            "name": f"{e['name']} ({e.get('org','?')})",
            "hf_id": hf_id,
            "pTot": round(total, 2),
            "pAct": round(active_params(e, hf_id, total), 2),
            "context": e.get("context"),
            "src_as_of": e.get("source", {}).get("as_of"),
            **arch,
        }
        print(f"  ok  {hf_id:52s} {arch['attn']:7s} L{arch['layers']}", file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    print(f"\n{len(out)}개 기록 -> {OUT}", file=sys.stderr)
    if failed:
        print(f"실패 {len(failed)}건:", file=sys.stderr)
        for h, why in failed:
            print(f"  -- {h:52s} {why}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
