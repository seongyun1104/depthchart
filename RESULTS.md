# Gemma-4-31B MTP × Spec × LMCache × DSD — 실측 결과 정본 (2026-07-08 ~ 07-10)

> 환경: H100 NVL 96GB 단일카드, target `prithivMLmods/gemma-4-31B-it-qat-FP8` + draft `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`(4L), KV fp8, max_len 32768, TRITON_ATTN.
> 프로토콜: temp=0, `ignore_eos` 실토큰 고정(출력 200), throughput=`vllm:generation_tokens_total` 델타(정상상태), 웜업 30s/창 120s(§2·3) 또는 45s(§5·6, 단문·짧은 램프), AR=spec 카운터 델타, Prometheus 교차검증.
> 하니스: `benchmarks/runner.py` + `benchmarks/configs/*.yaml`.

## 1. Crossover — 배치 축 (vLLM 0.23, 단문 ~150tok)

| c | K=0 (no-spec) | K=1 | K=2 | K=3 | 최선/K0 |
|---|---|---|---|---|---|
| 30 | 1,400 | 2,200 | 2,200 | **2,500** | 1.79x |
| 60 | 2,200 | 2,800 | — | **3,000** | 1.36x |
| 128 | 3,413 | 3,413 | — | 3,413 | **1.00x (수렴=compute 천장)** |

- TPOT p50(ms): c30 22.0/12.5/12.5/10.0 · c60 25.9/19.3/—/17.1 · c128 41.1/31.8/—/32.2
- per-pos(K=3): 86/66/50% — c30→128 불변. K=0은 speculative-config 미탑재 구성(각주).
- 방법론: ignore_eos 미사용 시 benign divergence 길이 오염으로 결론 반전(2회 오판 후 확정).

## 2. Dose-response — 컨텍스트 축 (0.23, hit 98~99% 매칭, c=256)

| 실측 입력(tok) | K=0 | K=3 | K3/K0 | K0 TPOT | K3 TPOT |
|---|---|---|---|---|---|
| ~460 | 3,107 | 3,640 | **1.17x** | 81.0 | 62.9 |
| ~970 | 2,320 | 2,897 | **1.25x** | 109.3 | 77.8 |
| ~1,990 | 2,110 | 2,915 | **1.38x** | 119.9 | **78.0** |
| ~4,096 (7/14) | 1,768 | 2,397 | ~1.36x | 137.8 | 96.9 |

- **핵심 증거: K=3 TPOT 평탄(970→1990: 77.8→78.0ms 정지) vs K=0 상승(109→120)** = KV-read 상각.
- **★ V4 완성(7/14, hit98 c256 각3런): 상각엔 knee가 있다.** K=3 TPOT가 **ctx 2k까지 78ms 완벽평탄 → 4k에서 97ms 붕괴(+24%)**, K=0은 계속상승(120→137). throughput 배율 곡선 = 1.17→1.25→1.38→**1.36**(4k에서 상승 멈춤/미세하락). **TPOT 배율 137/97=1.41×가 더 견고**(3런 안정; throughput 배율은 APC warming으로 1.28→1.32→1.36 미세상승 caveat). 원인=spec 자신의 장문 decode 비용 상승(SWA window/draft FLOPs가 KV-read 절감 잠식)이 상대이득 상한. **RFC 서사: 상각은 ctx비례하나 무한 아님 — ~2k knee 후 ~1.4× 포화(정직한 경계).** preempt0/kv≤48%=풀fit(hit0→0.98 config 수정 유효).
- c=192 병행: K0 2,060 / K3 2,978 = **1.45x** (TPOT 92.3→60.4). preemption 전 셀 0.
- in~200 셀은 hit 불일치(63%)로 제외.
- hit-축(in2048 K3): hit90/60/30 = 1,976/910/660 tok/s, kv 55/84/99% — hit 역할=용량(풀 fit), 저-hit 장문은 포화 붕괴. ⚠ **hit60/30 셀은 LMCache 자동개입 혼재(순수 APC 수치 아님)** — 인용 시 라벨 필수. 결론(hit=용량 조건)은 유지.

## 3. Drafter 계보 (0.23, K=3 c=30)

| drafter | AR | per-pos | tok/s |
|---|---|---|---|
| QAT-매칭 (셀A) | **67.7%** | 86/65/51 | **2,500** |
| 원본 non-QAT (셀B) | 51.6% | 78/47/30 | 2,200 |

→ 계보 일치만으로 +14% 처리량.

## 4. LMCache (0.23 + LMCache 0.5.1 MP 커넥터)

| 구성 | GPU KV pool | 비고 |
|---|---|---|
| V1 커넥터 단독 | 123,520 (−65%) | hybrid manager 강제 해제 |
| V1 + MTP | 크래시 | `KeyError: draft_model.layers.0.self_attn.attn` |
| **MP 단독** | **364,354** | SupportsHMA 작동, KV 6그룹(sliding5+full1) |
| MP + MTP | 331k대 | 4자 공존(+FP8 KV) 정상 |

- 배선 필수 2건: netns 공유(ZMQ localhost:5555 고정) + 서버 GPU 가시성(ptr=CUDA IPC).
- 정합: store→`reset_prefix_cache`→retrieve, external hit +8,704 = LMCache 서빙 확인. 장거리 무결(윈도우 1024 밖 인용).
- 용량: pool250(워킹셋 500k>331k) — 재계산 폭주 45건 → LMCache 572건.
- 속도: APC 2,684 vs LMCache-서빙 985 tok/s (c=64 비과부하, **원인 미분해**: 직렬화+36% miss 재계산+GPU 경합. max-gpu-workers 16 무효 — affinity 단일GPU 직렬화 추정).
- 안정성: 과부하 시 `scheduler assert req_id in self.requests` → EngineDead (업스트림 결함).

## 5. vLLM 0.24 이관 (7/10)

- 테스트①: 0.24×Gemma4-31B×MTP 기동 완주, 인자변경 0, KV pool 330,966(≈0.23), TRITON 유지 → **5090 12B 0.24 실패는 sm_120 특이로 격리**.

## 6. MTP × DSD (0.24, `num_speculative_tokens_per_batch_size:[[1,64,3],[65,128,1],[129,512,0]]`)

기동 시 능동 인식: `WARNING vllm.py:767 "Dynamic speculative decoding ... Overriding cudagraph_mode from FULL_AND_PIECEWISE to PIECEWISE"`. `VLLM_USE_V2_MODEL_RUNNER` env 불필요.

| 클라이언트 c | 엔진 running peak | 티어(집계) | tok/s | TPOT p50 |
|---|---|---|---|---|
| 30 | — | **K=3** (draft/step 3.0) | 2,533 | 10.3 |
| 100 | — | **K=1** (런타임 전환) | 3,089 | 31.5 |
| 192 | 192 | K=1 혼합 (AR 93.3%) | 2,870 | 70.5 |
| 224 | 224 | K=0 | 2,330 | 75.2 |
| 256 | 256 | K=1 혼합 (AR 84.1%) | 2,605 | 81.2 |
| 320 | 320 | K=0 | 1,740 | 110.5 |
| 400 | 400 | **K=0** (spec 0) | 1,880 | 129.5 |

- **판정**: MTP×DSD 완전 동작(공개 문서·이슈에서 선행 검증 사례를 확인하지 못한 조합 — vLLM 문서는 "Eagle/E3만 테스트").
- **경계 발견**: running≥129에서도 일부 셀 drafting = **DSD 인덱스(per-step 스케줄 요청 수)는 스텝 단위 요동**(wave 램프 구간이 하위 티어 통과). 경계 근처 거동 확률적 → 배포 경계는 running 분포 꼬리 밖 + 히스테리시스 필요.
- **Prometheus 볼륨 정밀화**: CAL 구간 draft/gen=0.02 (K=1 지배 시 ~0.52) → **c≥192 정상상태는 대부분 K=0, drafting은 램프 소량(~2%)**. "K=1 유지"는 집계 착시(draft한 스텝만의 K). 볼륨 기준 K=0 티어 사실상 정상 작동 — 티어혼합=소량 램프 누출. V1b 구간 d/g=0.99=순수 K=3 시그니처 교차확인.
- ⚠ **혼합 셀 AR(93.3%/84.1%)은 인용 금지** — draft 볼륨 ~2% 표본의 AR = 노이즈.
- ⚠ **고동시성 K=0 셀(c224/320/400: 1,740~2,330)은 이상치 의심 — V2 관문 대기**: 0.23 천장(c128 no-spec 3,413)의 51~68% + 비단조(c320<c400) + TPOT-환산 불일치(c224: 75.2ms→환산 2,979 vs 실측 2,330) = 45s 창 오염 시그니처. 원인 미분리: (a)창 오염 (b)DSD-K0 drafter sync 세금(산술 추정 ~7-10%라 -30~50% 설명 부족) (c)0.24 고배치 회귀. **실운영 워크로드 프로파일(sat256/peak769)가 이 티어라 분해 전 "실배포 가능" 판정 보류.** V2=120s 정본으로 c256 3자 대조(①DSD-K0@0.24 ②no-spec@0.24 ③참조 0.23 천장 3,413).

## 6b. V2 판정 (7/13) — DSD K=0 티어 ≠ no-spec, 세금 -25%

| c=256, 120s 정본 | tok/s | TPOT p50 | TTFT p50/p99 | 완료 |
|---|---|---|---|---|
| ② no-spec @0.24 | **3,413** | 73.7ms | 0.28s / 4.7s | 2,048 |
| ① DSD-K0 @0.24 | **2,560** | 78.6ms | **2.0s / 10.8s** | 1,536 |

- **②=③(0.23 천장) 정확 일치** → 0.24 회귀 없음 + 천장 c256까지 평탄 검증 + wave 인공물 기각.
- **①<② = -25%, 세금은 TTFT 집중**(TPOT +6.7%뿐) = prefill/스케줄링 비용. 45s 셀 저수치도 창 오염이 아니라 이 세금이 실체(45s 2,605 ≈ 120s 2,560).
- **배포 권고: 상시 고배치 서버는 K=0 티어가 아니라 spec config 자체 제거.** DSD는 c<128 구간용.
- 미분해: (i)PIECEWISE 고배치 비용 (ii)drafter prefill 의무 — 단 0.23 정적K3 c128=3,413이므로 drafter 탑재 자체는 무죄, **DSD 모드 특이** (iii)티어결정 오버헤드. 판별 실험=배치표 `[[1,512,0]]` 순수 인프라세금.
- **업스트림 4호 발견**: "DSD K=0 tier is not free at high batch" — 미문서 실측.

## 7. V1b — PIECEWISE 세금 (0.24, c=30 K=3 직접비교)

| | DSD(PIECEWISE) | 정적(FULL+PIECEWISE) |
|---|---|---|
| tok/s | 2,533 | 2,533 |
| TPOT p50 | 10.3ms | 10.5ms |
| 완료/AR | 570 / 68.1% | 570 / 68.2% |

→ **세금 ≈ 0**. **★ P4 갭 표 완성(7/14, 단문 B 61~128):** c90 K0/K3=2,445/**3,000**(1.23×, p99 960ms) · c110 K0/K1/K3=2,892/3,185/**3,300**(K3 최적 1.14×, **p99 796ms**) · c128=crossover(K0=K3=3,413, p99 4,044ms). **[1,110,3] 확장 승인** — c110 K3 TTFT p99=796ms로 건강(우려한 tail은 110~128 사이 폭발). **단문서도 K3>K1**(c110 3,300>3,185, pos2 53.7%) → 기존 [65,128,1] 티어 과보수, K3로 상향 가능(c128 직전까지). TPOT도 K3 전부 우위(27~30 vs K0 35~38). 4k 셀: K3 TPOT 78→97 = **평탄 깨짐**, K0@4k=137 측정 완료(7/14) → 배율 1.41×(TPOT)/1.36×(tput), 상각 knee 확정. **"실배포 가능" 현 근거는 c≤100뿐 — 고동시성 티어는 §6 V2 관문 후 판정.**

## 미측정 (부채)

- 정량 품질 게이트 — **(A) 엔진 정합성 완료 7/14**: greedy K=3 vs K=0 concurrency=1, control 30/30 결정적, K전환 0/30 bit-identical이나 **전부 benign reword(FP8 KV argmax flip, 부패 없음)**. bit-identity는 틀린 게이트 → **(B) 품질 채점 동등성**은 배포 파이프라인 scored eval 소관(미실시).
- ~~4k K=0 (SWA 점근 곡선 4점째)~~ — **완료 7/14** (§2, knee 확정).
- LMCache-서빙 × spec 결합 처리량 (복원 병목 분해 후).
- LMCache −63% 원인 분해 / 멀티-GPU 복원 병렬.
