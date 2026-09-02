# serving-capacity

고객사 앞에서 **"동시 몇 명, 몇 tok/s 나옵니까"** 에 근거를 대고 답하기 위한 도구.

SLO(TTFT p95 · 사용자당 출력 속도)를 고정한 상태에서 주어진 GPU 구성이 버티는
동시 요청 수, 시스템 처리량, 그리고 **무엇이 그 한계를 묶고 있는지**를 계산한다.
계산기는 단일 HTML 파일이고 외부 요청이 0건이라 에어갭 노트북에서 그대로 돌아간다.

```
capacity-calc.html          배포용 단일 파일 (306KB, 외부 요청 0)
db/                         계수 DB
  models.json                 HF config.json 에서 동기화 (46개)
  gpus.json                   손으로 관리 (7개)
  workloads.json              워크로드 프리셋
  calibration.json            하네스 산출물 (있으면 등급 A)
  coefficients.json           ↑ 전부 합친 배포본 — 오프라인 갱신 통로
src/capacity-calc.src.html  템플릿 (__DB__ / __FONTS__ 표식)
assets/fonts.css            IBM Plex 9개 페이스 base64 (latin 서브셋)
tools/sync_models.py        model-atlas + HF → db/models.json
tools/build.py              템플릿 + DB + 폰트 → capacity-calc.html
harness/calibrate.py        실측 계수 추출 (vLLM/SGLang OpenAI 엔드포인트)
```

## 쓰는 법

```bash
python3 tools/build.py          # 계산기 빌드 (외부 참조 남으면 실패시킴)
open capacity-calc.html         # 끝. 서버도 네트워크도 필요 없음
```

모델이 새로 나왔을 때:

```bash
python3 tools/sync_models.py    # HF config.json 파싱 (gated 레포는 HF_TOKEN 필요)
python3 tools/build.py
```

고객사 장비에서 실측할 때:

```bash
sudo nvidia-smi -pl 400         # 5090 은 먼저 전력 상한 (Xid 79 회피)
python3 harness/calibrate.py \
    --url http://localhost:8000/v1 --model <서빙명> \
    --gpu-key rtx5090 --model-key qwen3.8-27b
python3 tools/build.py          # 등급이 A 로 올라감
```

**HTML 재배포 없이 계수만 갱신**하려면 `db/coefficients.json`(15KB)만 전달하고
계산기의 「계수 불러오기」로 물리면 된다. 이게 에어갭 환경의 갱신 통로다.

## 계산 모델

```
디코드 스텝   t(B) = 가중치/BW_eff + KV_seq/BW_eff · B + 오버헤드
프리필        R    = FLOPS_eff / (2 · 활성파라미터)
정상상태      X    = 1 / (O · t(B)/B + P/R)          req/s
동시 사용자   N    = X · (TTFT + O · TPOT)            Little's law
```

한계점은 두 SLO를 모두 만족하는 마지막 지점이고, 병목은 그 지점을 실제로 묶은
자원(KV 용량 / 메모리 대역폭 / 프리필 연산)으로 판정한다.

### KV 캐시 — 어텐션 구조별로 갈린다

동기화된 46개 모델 중 **17개(37%)가 단순 GQA가 아니다.** 구조를 잘못 잡으면
데이터가 최신이어도 답이 틀린다. gemma-3-27b 를 24K 컨텍스트로 돌릴 때
GQA 공식을 쓰면 KV 를 **5.1배 과대추정**한다 (12,545 MB → 실제 2,460 MB).

| 구조 | 공식 | 동기화된 모델 |
|---|---|---|
| GQA / MHA / MQA | `2·L·kv_heads·head_dim·C` | 29 |
| MLA | `L·(kv_lora_rank + qk_rope_head_dim)·C` — 압축 잠재 1벌 | 6 |
| SWA | `C → min(C, window)` | 2 |
| 하이브리드 | 글로벌 레이어만 풀 컨텍스트, 나머지는 윈도우 | 6 |

검증: DeepSeek-V3.1 = 68.6 KB/token (61층 × 576 × 2B), 공개 수치 ~70KB 와 일치.

## 근거 등급

계산기가 자기 신뢰도를 화면에 밝힌다. 3자리 유효숫자를 자신 있게 뱉으면 사고 난다.

| 등급 | 조건 |
|---|---|
| A · 실측 | 이 장비+모델 캘리브레이션 있음 |
| B · 계열 추정 | GPU η 는 실측, 모델은 config 계산 (±30%) |
| C · 스펙 추정 | η 가 세대 기본값 — 자릿수 참고용, 구매 근거 불가 |

현재 η 를 실측한 장비는 RTX 5090 뿐(0.82, 자체 roofline 2026-08)이다.
나머지는 전부 C 로 뜬다. 이게 정상이고, 채우려면 하네스를 돌려야 한다.

## 알려진 한계

- **TTFT p95 는 M/M/1 근사(평균×3)** 로 이 모델에서 가장 약한 항이다. 한계점 위치는
  KV 상한과 선형 디코드 모델이 잡지만, 무릎을 넘어선 뒤의 곡선 모양은 캘리브레이션이
  필요하다. 하네스가 실측 p95/평균 비율을 뽑아 이 계수를 대체한다.
- **하네스는 실제 엔드포인트에 대해 아직 돌리지 않았다.** 안전 가드(동시성 32 상한)와
  실패 처리는 검증했지만, 적합 품질은 첫 실측 때 확인해야 한다.
- `google/gemma-3` 는 config.json 이 `sliding_window_pattern` 을 생략하고 transformers
  기본값(6)에 의존해서 `tools/sync_models.py` 의 `OVERRIDES` 로 보정한다. 근거는 코드의
  note 에 남겼다. 추측이 필요한 항목은 보정하지 않고 버린다.
- gemma-3-4b 는 config 에 헤드 수가 없어 제외, Llama-3.3-70B 는 라이선스 미동의로 403.

## 상류

모델 아키텍처는 [`model-atlas`](../model-atlas) 를 상류로 쓴다. 두 번째 모델 DB를
만들면 그게 먼저 썩는다. atlas 가 HF Hub 에서 모델 목록을 동기화하고,
`sync_models.py` 가 각 `hf_id` 의 `config.json` 을 읽어 아키텍처 필드를 채운다.
이 층은 **벤치마크가 전혀 필요 없다** — 가장 자주 바뀌는 층이 가장 싸게 갱신된다.
