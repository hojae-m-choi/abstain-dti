# exp2-comparing_models1 — BindingDB 3모델 비교

**질문:** 약물–단백질 상호작용을 토큰 수준에서 학습하는 것이, 요약 벡터 수준이나 전혀 학습하지 않는 것보다 얼마나 더 나은가? 그 이득이 cold split에서도 남는가?

세 모델 모두 같은 고정 인코더(ChemBERTa-77M, ESM-2 35M) 위에 올라갑니다. 따라서 차이는 결합 방식 하나뿐입니다.

| 모델 | 상호작용 학습 수준 |
| --- | --- |
| FLAML (AutoML) | 없음 — pooling된 벡터를 이어 붙여 tree 모델에 넘김 |
| Siamese | 요약 벡터 수준 — 공유 MLP 타워 + 쌍별 head |
| Fusion Transformer | 토큰 수준 — 원자와 잔기 사이 co-attention |

- 데이터: `bindingdb_large` (pAffinity), 층화 subsample
- split: `cold_drug`, `cold_target`, `cold_both`, 8:1:1 (엔티티 단위)
- 학습: pAffinity 회귀 1회. 분류 지표(AUROC/AUPRC)는 예측값을 기준값 6.0으로 이진화해 파생

설계 결정과 그 근거는 아래 "Phase 0이 바꾼 것"과 "결과"에 정리돼 있습니다.

## 진행 상황

| Phase | 내용 | 상태 |
| --- | --- | --- |
| 0 | 데이터 정찰 | 완료 — 설계 2건 변경 (아래) |
| 1 | 소규모 비교 (3만 쌍) | 건너뜀 — CUDA 장비 확보로 Phase 2 직행 |
| 2 | 본 실험 (10만 쌍, 4 split × 4 arm × 3 seed = 48 run) | 완료 (48/48) |
| 3 | 전체 1.9M | 보류 |

## 결과

평균 test RMSE (낮을수록 좋음, seed 3개). 전체 지표는 [`runs/summary.md`](runs/summary.md), run별 값은 `runs/all_runs.csv`.

| split | flaml-ecfp | flaml-chemberta | siamese | fusion |
| --- | --- | --- | --- | --- |
| cold_drug | **0.923** | 1.020 | 1.043 | 1.026 |
| scaffold | **0.959** | 1.052 | 1.093 | 1.071 |
| cold_target | 1.374 | 1.335 | 1.384 | **1.334** |
| cold_both | 1.393 | **1.372** | 1.390 | 1.414 |

### 가설은 기각됐습니다

"상호작용을 깊게 학습할수록 좋다(fusion > siamese > FLAML)"는 예상은 **어느 split에서도 나타나지 않았습니다.**
약물 쪽 split에서는 상호작용을 전혀 모델링하지 않는 FLAML+ECFP4가 0.09~0.10 차이로 가장 좋았고,
단백질 쪽 split에서는 네 arm이 사실상 구별되지 않았습니다.

### 아키텍처보다 약물 표현이 훨씬 중요했습니다

같은 FLAML 모델에 같은 단백질 feature를 주고 **약물 표현만 바꾼** 대조(ECFP4 − ChemBERTa):

| split | 차이 | 해석 |
| --- | --- | --- |
| cold_drug | **−0.097** | ECFP4가 크게 유리 |
| scaffold | **−0.093** | 〃 |
| cold_target | +0.038 | ChemBERTa가 약간 유리 |
| cold_both | +0.030 | 〃 |

약물 쪽 split에서 이 0.09~0.10 격차는 **어떤 아키텍처 차이보다도 큽니다**(fusion−siamese는 0.02 수준).
즉 이 실험이 실제로 측정한 것은 "상호작용 모델링의 가치"가 아니라 **"고정된 ChemBERTa 임베딩이 ECFP4보다 약한가"**였고, 답은 약물 일반화 상황에서 그렇다는 것입니다.

보조 FLAML 실행(ChemBERTa feature)을 넣어 둔 것이 여기서 결정적이었습니다. 그게 없었다면
"AutoML이 transformer를 이긴다"는 잘못된 결론을 냈을 것입니다. 실제로는 ECFP4가 고정 ChemBERTa를 이긴 것입니다.

### ECFP4의 우위는 단백질 split에서 뒤집힙니다

`cold_target`과 `cold_both`에서 flaml-ecfp는 오히려 **가장 나쁩니다**. 본 적 있는 단백질에 대해서는
ECFP4의 세밀한 구조 기술자로 구조–활성 관계를 메모리할 수 있지만, 처음 보는 단백질에는 그 기억이 전이되지 않습니다.

### co-attention은 작지만 일관된 이득을 줍니다

fusion − siamese: cold_drug −0.017, scaffold −0.022, cold_target −0.050, cold_both +0.024.
네 중 셋에서 fusion이 낫지만 폭이 작고, **계산 비용은 약 3배**입니다(159s/epoch 대 55s/epoch).

### 단백질 일반화는 아무도 못 합니다

`cold_target`/`cold_both`에서 RMSE 1.33~1.41, CI 0.61~0.65로 네 arm 모두 약합니다.
신경망 두 arm의 최적 에폭이 **1~3**이라는 점이 이를 뒷받침합니다 — 몇 에폭 만에 더 배울 것이 없어지고 과적합으로 넘어갑니다.
반면 약물 쪽 split에서는 최적 에폭이 20~30이었습니다.

### 결과를 읽을 때 반드시 함께 봐야 할 것

- `cold_drug`는 이 데이터셋에서 random과 거의 같습니다(약물의 66%가 한 번만 등장). 약물 쪽 진짜 어려운 split은 `scaffold`이고, 둘의 격차는 작았습니다(0.923 → 0.959)
- `cold_target`·`cold_both`는 test 단백질이 30개뿐이라 분산이 큽니다(FLAML의 seed 표준편차 0.04~0.06)
- `cold_both`는 test가 884쌍입니다
- 상위 300 단백질만 썼으므로 잘 연구된 단백질 쪽으로 편향돼 있습니다
- **인코더를 고정했으므로 "fine-tune했다면" 질문에는 답하지 못합니다.** 위 결론은 전부 "고정 ChemBERTa 기준"입니다
- 40 에폭 상한에 걸린 run은 24개 중 1개뿐이고(fusion cold_drug s1, best 38) 그 RMSE도 같은 조건 평균과 일치해, 절단은 결과를 왜곡하지 않았습니다

### 다음에 물어볼 것

1. ChemBERTa fine-tune — 고정 임베딩이 약했던 것이지 모델이 약한 것인지 분리
2. ECFP4를 신경망 arm에도 투입 — 아키텍처와 표현을 직교하게 비교
3. 단백질 표현 개선 — 현재 병목은 명백히 단백질 쪽입니다

### Phase 0이 바꾼 것

1. **`cold_drug`가 random과 거의 같습니다.** BindingDB는 약물의 66%가 단 한 번만 등장합니다(980,305 약물 / 1,569,547 쌍). 쌍을 무작위로 나눠도 test 약물 대부분이 train에 없습니다. 그래서 `scaffold`를 추가해 "처음 보는 화학 계열"을 제대로 묻고, `cold_drug`는 random의 대리 지표로 남겼습니다.
2. **단백질 중심 2단계 subsample.** 쌍 수 상위 300개 단백질만 남긴 뒤 그 안에서 pAffinity 십분위 층화 추출을 합니다. 균등 추출과 쌍 수·양성 비율은 같은데 고유 잔기가 3,340,441 → 198,738로 줄어 ESM-2 캐시가 17배 싸집니다. 대가는 test 단백질 30개와 잘 연구된 단백질 편향입니다.

`cold_both`는 교차 쌍 34%가 버려져 test가 884쌍까지 줄어듭니다. 유지하되 결과에 단서를 답니다.

## 실행

```bash
cd track_ml-baselines
uv run python experiments/exp2-comparing_models1/scripts/01_subsample.py   # 고정 subsample + split
uv run python experiments/exp2-comparing_models1/scripts/02_featurize.py   # ESM-2 / ChemBERTa 캐시

# 개별 run
uv run python experiments/exp2-comparing_models1/scripts/03_run_neural.py \
    --model siamese --split cold_target --n 100000 --seed 0
uv run --extra automl python -u experiments/exp2-comparing_models1/scripts/04_run_flaml.py \
    --drug-feature ecfp --split cold_target --n 100000 --seed 0

uv run python experiments/exp2-comparing_models1/scripts/06_summarize.py   # 결과 표
```

2-GPU 머신에서 전체 grid를 한 번에 돌리려면 `scripts/launch_phase2.sh` (상태는 `launch_phase2.sh status`,
중지는 `stop`). 이미 `metrics.json`이 있는 run은 건너뛰므로 중단 후 재개해도 안전합니다.

## 폴더

```
scripts/00_scout.py     데이터 정찰 → data/scout.json
scripts/01_subsample.py 2단계 subsample + split 고정
scripts/02_featurize.py ESM-2 / ChemBERTa 캐시
scripts/03_run_neural.py  fusion | siamese
scripts/04_run_flaml.py   flaml-ecfp | flaml-chemberta
scripts/06_summarize.py   all_runs.csv, summary.csv, summary.md
data/sub_*.parquet      고정된 subsample (이 파일이 비교의 기준)
data/splits/            split 인덱스 (고정)
src/                    features.py (ECFP4, pooling), siamese.py
runs/                   run별 metrics.json, predictions_*.csv
```

`data/sub_*.parquet`과 `data/splits/`가 비교의 기준입니다. 이 둘만 같으면 어느 머신에서 돌려도
**글자 그대로 같은 행**으로 학습하고 평가합니다. 특징 캐시는 이 파일들로부터 1~2분이면 다시 만들어집니다.

## 구현 메모

실험을 돌리며 상위 패키지에 들어간 변경과 그 이유입니다.

- **`dti_fusion.data.store`** — `TokenStore`를 `featurize.py`에서 분리했습니다. FLAML arm이 임베딩은
  읽어야 하지만 torch는 import하면 안 되기 때문입니다. torch와 LightGBM이 각자 OpenMP 런타임을 적재하면
  (macOS arm64) LightGBM이 스레드를 띄울 때 SIGSEGV로 죽습니다. 환경 변수로는 막히지 않습니다.
- **`TokenFeatures(max_len=...)`** — 약물 토큰 길이 상한. 중앙값 54, p99 118인데 0.78%가 256 상한에
  걸려 있고, collator가 배치 최댓값으로 padding하므로 그 하나가 배치 전체의 attention 비용을 정합니다.
  128로 자르면 MPS에서 OOM 없이 돕니다.
- **`train(cfg, prep=..., build_model=...)`** — 고정한 split과 다른 아키텍처로 같은 학습 루프를
  재사용하기 위한 주입 지점입니다.

데이터 로딩, split, 지표, Fusion 학습 루프는 상위 `dti_fusion` 패키지를 그대로 씁니다.
산출물 형식도 기존 run 폴더와 같게 맞춰서, 뒤에 올 conformal / risk-coverage 단계가 그대로 읽을 수 있습니다.
