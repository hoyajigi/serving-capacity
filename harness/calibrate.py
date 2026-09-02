#!/usr/bin/env python3
"""계산기 계수를 실측으로 채운다. vLLM/SGLang 의 OpenAI 호환 엔드포인트만 있으면 된다.

측정 설계 — 두 항을 분리해서 재는 게 핵심이다.

  1) 디코드 프로브: 입력을 극단적으로 짧게(≈16 tok) 두고 긴 출력을 뽑는다.
     프리필 점유가 0에 수렴하므로 관측된 TPOT 가 곧 t_step(B) 이고,
     B 에 대한 최소제곱 직선이 a+오버헤드(절편)와 b(기울기)를 준다.
  2) 프리필 프로브: 긴 입력 + max_tokens=1 을 동시성 1로 던진다.
     TTFT 가 곧 프리필 시간이고 여기서 달성 TFLOPS 를 역산한다.

두 프로브를 섞으면 하나의 곡선에 두 병목이 겹쳐서 아무것도 분리되지 않는다.

안전 — 5090 은 고동시성에서 Xid 79(버스 이탈, 재부팅 필요)가 관측됐다.
동시성 상한이 32 로 걸려 있고, 넘기려면 --max-concurrency 를 명시해야 한다.
장시간 잡 전에 전력 상한을 먼저 걸 것:  sudo nvidia-smi -pl 400
"""
from __future__ import annotations
import argparse, json, statistics, sys, threading, time, urllib.request
from pathlib import Path

XID79_CAP = 32          # 이 값을 넘기려면 명시적 --max-concurrency 필요


def stream(url: str, model: str, prompt: str, max_tokens: int, key: str, timeout: float):
    """한 요청을 스트리밍하고 (TTFT, 토큰 도착 시각들)을 돌려준다."""
    body = json.dumps({
        "model": model, "stream": True, "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": True,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    hdr = {"Content-Type": "application/json"}
    if key:
        hdr["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body, headers=hdr)
    t0 = time.perf_counter()
    ttft, stamps = None, []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            chunk = raw[6:].strip()
            if chunk == b"[DONE]":
                break
            try:
                d = json.loads(chunk)["choices"][0].get("delta", {})
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if not d.get("content"):
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            stamps.append(now)
    return ttft, stamps


def closed_loop(cfg, conc: int, prompt: str, max_tokens: int, seconds: float):
    """동시성 conc 를 seconds 동안 유지하며 완료 요청들을 모은다."""
    out, lock, stop = [], threading.Lock(), time.perf_counter() + seconds

    def worker():
        while time.perf_counter() < stop:
            try:
                ttft, st = stream(cfg.url, cfg.model, prompt, max_tokens, cfg.key, cfg.timeout)
            except Exception as e:                                   # noqa: BLE001
                with lock:
                    out.append({"error": type(e).__name__})
                return
            if ttft is None or len(st) < 2:
                continue
            with lock:
                out.append({"ttft": ttft, "tpot": (st[-1] - st[0]) / (len(st) - 1), "n": len(st)})

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(conc)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(seconds + cfg.timeout)
    return out


def fit_line(xs, ys):
    """최소제곱 y = m·x + c."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    m = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    c = my - m * mx
    ss_res = sum((y - (m * x + c)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return m, c, (1 - ss_res / ss_tot if ss_tot else 1.0)


def main() -> int:
    p = argparse.ArgumentParser(description="서빙 용량 계산기 계수 실측")
    p.add_argument("--url", required=True, help="예: http://localhost:8000/v1")
    p.add_argument("--model", required=True, help="엔드포인트가 받는 모델 이름")
    p.add_argument("--gpu-key", required=True, help="db/gpus.json 의 키 (예: rtx5090)")
    p.add_argument("--model-key", required=True, help="db/models.json 의 키 (예: qwen3.8-27b)")
    p.add_argument("--key", default="", help="Bearer 토큰")
    p.add_argument("--concurrency", default="1,2,4,8,16,32", help="디코드 스윕 지점")
    p.add_argument("--max-concurrency", type=int, default=XID79_CAP,
                   help=f"안전 상한 (기본 {XID79_CAP}; 5090 Xid 79 회피)")
    p.add_argument("--prefill-tokens", type=int, default=4096)
    p.add_argument("--decode-tokens", type=int, default=192)
    p.add_argument("--seconds", type=float, default=25.0, help="지점당 유지 시간")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "db" / "calibration.json"))
    cfg = p.parse_args()

    levels = sorted({int(x) for x in cfg.concurrency.split(",") if x.strip()})
    over = [c for c in levels if c > cfg.max_concurrency]
    if over:
        print(f"중단 — 동시성 {over} 이(가) 상한 {cfg.max_concurrency}을 넘습니다.\n"
              f"5090 은 고동시성에서 Xid 79 로 버스를 이탈합니다(재부팅 필요).\n"
              f"정말 필요하면 --max-concurrency 로 명시하고, 먼저 전력 상한을 거세요: "
              f"sudo nvidia-smi -pl 400", file=sys.stderr)
        return 2

    print(f"# 대상 {cfg.url} · {cfg.model}", file=sys.stderr)

    # --- 1) 프리필 프로브 (동시성 1) ---
    long_prompt = "가 " * cfg.prefill_tokens          # 대략 토큰당 1 어절
    ttfts = []
    for i in range(5):
        try:
            t, _ = stream(cfg.url, cfg.model, long_prompt, 1, cfg.key, cfg.timeout)
        except Exception as e:                                       # noqa: BLE001
            print(f"  프리필 {i+1}/5  실패: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if t:
            ttfts.append(t)
            print(f"  프리필 {i+1}/5  TTFT {t:.3f}s", file=sys.stderr)
    if not ttfts:
        print("프리필 프로브 실패 — --url 과 --model, 그리고 서버가 스트리밍을 "
              "지원하는지 확인하세요.", file=sys.stderr)
        return 1
    t_pre = statistics.median(ttfts)
    r_prefill = cfg.prefill_tokens / t_pre

    # --- 2) 디코드 스윕 (프리필 점유 ≈ 0) ---
    rows, p95_ratios = [], []
    for c in levels:
        res = closed_loop(cfg, c, "안녕", cfg.decode_tokens, cfg.seconds)
        ok = [r for r in res if "tpot" in r]
        if len(ok) < 3:
            print(f"  동시성 {c:>3}  표본 부족({len(ok)}) — 건너뜀", file=sys.stderr)
            continue
        tpot = statistics.median(r["tpot"] for r in ok)
        tt = sorted(r["ttft"] for r in ok)
        p95 = tt[min(len(tt) - 1, int(0.95 * len(tt)))]
        mean = statistics.fmean(tt)
        if mean > 0:
            p95_ratios.append(p95 / mean)
        rows.append({"concurrency": c, "tpot_s": tpot, "tok_s_per_user": 1 / tpot,
                     "ttft_mean_s": mean, "ttft_p95_s": p95, "samples": len(ok)})
        print(f"  동시성 {c:>3}  TPOT {tpot*1000:6.2f} ms  "
              f"사용자당 {1/tpot:6.1f} tok/s  TTFT p95 {p95:.3f}s  n={len(ok)}", file=sys.stderr)

    if len(rows) < 3:
        print("스윕 표본이 3점 미만이라 적합할 수 없습니다.", file=sys.stderr)
        return 1

    slope, intercept, r2 = fit_line([r["concurrency"] for r in rows], [r["tpot_s"] for r in rows])

    entry = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "endpoint_model": cfg.model,
        "prefill_tok_s": round(r_prefill, 1),
        "prefill_probe_tokens": cfg.prefill_tokens,
        "decode_slope_ms_per_seq": round(slope * 1000, 4),      # = KV_seq / BW_eff
        "decode_intercept_ms": round(intercept * 1000, 3),      # = 가중치/BW_eff + 오버헤드
        "fit_r2": round(r2, 4),
        "ttftP95Factor": round(statistics.fmean(p95_ratios), 2) if p95_ratios else None,
        "sweep": rows,
        "note": "디코드 프로브는 입력 ≈0 이므로 프리필 점유가 분리돼 있음. "
                "stepOverheadMs 는 절편에서 가중치 스트리밍 시간을 뺀 값 — "
                "가중치/BW_eff 는 계산기가 스펙에서 구하므로 여기서는 절편만 넘긴다.",
    }
    out = Path(cfg.out)
    db = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    db[f"{cfg.gpu_key}|{cfg.model_key}"] = entry
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(db, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")

    print(f"\n적합  t_step(B) = {intercept*1000:.2f} + {slope*1000:.4f}·B ms   (R²={r2:.4f})", file=sys.stderr)
    print(f"프리필 {r_prefill:,.0f} tok/s · TTFT p95/평균 = {entry['ttftP95Factor']}", file=sys.stderr)
    print(f"기록 -> {out}   이제 `python3 tools/build.py` 로 계산기에 반영하세요.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
