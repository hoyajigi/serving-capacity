#!/usr/bin/env python3
"""src 템플릿 + db/*.json + 인라인 폰트 -> 배포용 단일 HTML.

에어갭이 전제라 산출물은 외부 요청이 0건이어야 한다. 빌드가 그걸 검사한다.
동시에 db/coefficients.json 을 내보낸다 — HTML 재배포 없이 계수만 갱신하는 통로.
"""
from __future__ import annotations
import json, re, sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "capacity-calc.src.html"
FONTS = ROOT / "assets" / "fonts.css"
OUT = ROOT / "capacity-calc.html"
COEFF = ROOT / "db" / "coefficients.json"
DEFAULT_MODEL = "qwen3.8-27b"

TEMPLATES = {   # DB 에 없는 크기대를 손으로 잡아볼 때 쓰는 대표값
    "t7":   {"name": "7B 급 (dense)",   "pTot": 7,  "pAct": 7,  "layers": 32, "kv_heads": 8, "head_dim": 128, "attn": "gqa"},
    "t27":  {"name": "27B 급 (dense)",  "pTot": 27, "pAct": 27, "layers": 62, "kv_heads": 8, "head_dim": 128, "attn": "gqa"},
    "t70":  {"name": "70B 급 (dense)",  "pTot": 70, "pAct": 70, "layers": 80, "kv_heads": 8, "head_dim": 128, "attn": "gqa"},
}


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    models = {**load(ROOT / "db" / "models.json"), **TEMPLATES}
    calib_p = ROOT / "db" / "calibration.json"
    db = {
        "gpus": load(ROOT / "db" / "gpus.json"),
        "models": models,
        "workloads": load(ROOT / "db" / "workloads.json"),
        "calibration": load(calib_p) if calib_p.exists() else {},
        "meta": {"built": date.today().isoformat(),
                 "defaultModel": DEFAULT_MODEL if DEFAULT_MODEL in models else next(iter(models)),
                 "models": len([m for m in models.values() if not m.get("custom")])},
    }
    COEFF.write_text(json.dumps(db, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")

    html = SRC.read_text(encoding="utf-8")
    assert "<!--__FONTS__-->" in html and "/*__DB__*/{}" in html, "템플릿 표식 없음"
    html = html.replace("<!--__FONTS__-->", "<style>\n" + FONTS.read_text(encoding="utf-8") + "\n</style>")
    html = html.replace("/*__DB__*/{}", json.dumps(db, ensure_ascii=False, separators=(",", ":")))

    # 에어갭 검증: 외부 참조가 하나라도 남으면 빌드 실패
    ext = re.findall(r'(?:src|href)="(https?://[^"]+)"', html) + re.findall(r"url\((https?://[^)]+)\)", html)
    if ext:
        print("빌드 실패 — 외부 참조가 남아 있습니다:", file=sys.stderr)
        for u in sorted(set(ext)):
            print("  " + u, file=sys.stderr)
        return 1

    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT.name}  {len(html.encode())/1024:.0f} KB · 모델 {db['meta']['models']} · "
          f"캘리브레이션 {len(db['calibration'])} · 외부 참조 0")
    print(f"{COEFF.relative_to(ROOT)}  {COEFF.stat().st_size/1024:.0f} KB (오프라인 갱신용)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
