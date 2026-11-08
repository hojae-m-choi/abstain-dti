# DTI affinity 파이프라인 (`dti_fusion`)

약물 SMILES와 단백질 서열을 받아 결합 친화도(DAVIS는 pKd, KIBA는 KIBA score)를 회귀로 예측합니다.
목표는 약물 표현 두 가지(처음부터 학습하는 Graph Transformer, 사전학습된 ChemBERTa)를 같은 조건에서 비교하는 것입니다.
나머지 부분(단백질 인코더, fusion, 학습 루프, 평가)은 두 방식이 공유합니다.

## 1. 전체 흐름

```
HF 데이터셋 ─► 정제 (SMILES canonical화, 중복 쌍 평균) ─► split (random / cold_* / scaffold)
                                                              │
단백질 서열 ─► ESM-2 (고정) ─► 잔기 임베딩 캐시 ─► 길이 pooling ─┐
                                                              ├─► Fusion Transformer ─► 회귀 head ─► affinity
SMILES ─┬─► 분자 그래프 ─► Graph Transformer (학습) ───────────┤
        └─► ChemBERTa 토큰 임베딩 (고정 캐시) ─► adapter (학습) ─┘
```

CLI 단계와 대응:

| 단계 | 명령 | 산출물 |
| --- | --- | --- |
| 데이터 받기 | `dti download <name>` | `data/processed/<name>.parquet` |
| 특징 캐시 | `dti featurize -c <cfg>` | `data/features/<name>/esm-*`, `chemberta-*` |
| 학습과 평가 | `dti train -c <cfg>` | `runs/<run_name>/` |
| 인코더 비교 | `dti compare -c <cfg> ...` | `<out_dir>/compare_summary.{md,csv}` |

`train`은 캐시가 없으면 스스로 만들기 때문에 `featurize`는 생략해도 됩니다.

## 2. 데이터

### 데이터셋 (`data/datasets.py`)

| 이름 | 출처 | 레이블 | 이진 기준값 | 라이선스 |
| --- | --- | --- | --- | --- |
| `davis` | HF 허브 | pKd | 7.0 | MIT |
| `kiba` | HF 허브 | KIBA score | 12.1 | Apache-2.0 |
| `bindingdb_large` | `jglaser/binding_affinity` (`data/all.parquet`) | −log10(M) | 6.0 | 재배포 금지 |

정제 규칙:

- RDKit으로 SMILES를 canonical 형태로 바꿉니다. 파싱에 실패한 행은 버립니다.
- (canonical SMILES, 서열) 쌍이 여러 번 나오면 레이블을 평균해 한 행으로 합칩니다.
- 약물과 표적의 키는 문자열의 SHA-1 앞 16자입니다. 특징 캐시는 이 키로 찾습니다.

정제 후 DAVIS는 25,772쌍(약물 68개, 표적 379개)입니다.

### split (`data/splits.py`)

| 종류 | 규칙 | 묻는 질문 |
| --- | --- | --- |
| `random` | 쌍 단위 무작위 | 본 약물과 본 표적의 새 조합 |
| `cold_drug` | 약물 단위로 나눔. test 약물은 train에 없음 | 처음 보는 약물 |
| `cold_target` | 표적 단위로 나눔 | 처음 보는 단백질 |
| `cold_both` | 약물과 표적 둘 다 겹치지 않음 | 둘 다 처음 |
| `scaffold` | Bemis–Murcko scaffold 단위 | 처음 보는 화학 계열 |

기본 비율은 0.8 / 0.1 / 0.1이고, split seed는 `data.split_seed`로 고정합니다. 실제 인덱스와 통계는 run 폴더의 `split.json`에 저장됩니다.
DAVIS는 약물이 68개뿐이라 `cold_drug`의 val과 test는 각각 약물 7개입니다. 이 split의 수치는 몇 개 약물의 성질에 크게 흔들립니다.

## 3. 모델

### 단백질: ESM-2 (`data/featurize.py`)

- Hugging Face `EsmModel`의 마지막 층 잔기 임베딩을 씁니다. BOS/EOS 토큰은 뺍니다. 가중치는 고정합니다.
- 1,022 잔기보다 긴 서열은 1,022 길이 창을 256씩 겹쳐 나눠 넣고, 겹친 위치는 평균합니다. 모든 잔기가 임베딩을 받습니다.
- 결과는 `TokenStore`(float16 memmap 하나와 `index.json`)에 저장하고 이후에는 메모리 매핑으로 읽습니다.
- fusion에 들어가기 전 길이를 `data.max_protein_len`으로 맞춥니다. 기본 정책 `pool`은 전체 서열을 adaptive average pooling으로 줄이고, `truncate`는 앞부분만 자릅니다.
- 모델 크기: M1 프리셋은 35M(`esm2_t12_35M_UR50D`, 480차원), CUDA 프리셋은 650M입니다.

### 약물 1: Graph Transformer (`models/graph_transformer.py`)

Graphormer 방식이며 처음부터 학습합니다.

- 원자 특징 8가지(원소, degree, 형식 전하, 수소 수, 혼성, 방향족성, 고리 여부, chirality)를 각각 임베딩해 더합니다. degree 임베딩이 centrality encoding 역할을 합니다.
- 구조 정보는 attention에 head별 가산 bias로 들어갑니다. 원자 쌍의 최단 경로 거리(12 초과는 한 구간, 끊긴 조각은 별도 구간)와 직접 결합한 쌍의 결합 종류를 씁니다.
- 가상 graph 토큰 하나가 모든 원자와 주고받으며, 방향별로 학습되는 bias를 따로 둡니다.
- 구현 메모: bias를 `nn.Embedding` 조회 대신 one-hot 행렬곱으로 계산합니다. MPS에서 Embedding backward가 B·N·N개의 기울기를 작은 표로 scatter하느라 약 30배 느렸기 때문입니다(배치당 374ms에서 5ms). 두 방식이 같은 값을 내는지는 `test_structural_bias_matches_embedding_lookup`이 확인합니다.

### 약물 2: ChemBERTa (`models/chemberta.py`)

- 모델은 `DeepChem/ChemBERTa-77M-MLM`입니다.
- `chemberta-frozen`(기본): 토큰 임베딩을 한 번 계산해 캐시하고, 그 위의 선형 투영과 transformer adapter 1층만 학습합니다. 학습 속도가 Graph Transformer와 비슷합니다.
- `chemberta-ft`: ChemBERTa 가중치까지 학습합니다. backbone은 별도 학습률(`drug.chemberta.lr`, 기본 2e-5)을 씁니다. SMILES를 매 step 토크나이즈하므로 느립니다. 이번 비교에서는 돌리지 않았습니다.

### Fusion Transformer (`models/fusion.py`)

- `cross`(기본): 약물 토큰과 단백질 토큰이 서로를 query로 보는 양방향 co-attention을 층마다 수행합니다. ESM 토큰은 이미 문맥을 담고 있어 단백질 쪽 self-attention은 기본으로 끕니다(`fusion.protein_self_attn`).
- `joint`: 두 토큰열을 이어 붙이고 type embedding을 더한 뒤 self-attention을 합니다.
- 각 쪽을 attention pooling으로 벡터 하나씩(hd, hp)으로 요약하고, `[hd, hp, hd·hp, |hd−hp|]`를 MLP에 넣어 값 하나를 냅니다.
- padding 위치는 attention mask로 막습니다. 배치에 다른 샘플이 섞여도 예측이 바뀌지 않는지는 `test_padding_invariance`가 확인합니다.
- `return_attention=True`를 주면 pooling 가중치(약물 원자별, 잔기별)를 돌려줘 해석에 쓸 수 있습니다.

## 4. 학습과 평가 (`train.py`, `metrics.py`)

- 레이블은 train 평균과 표준편차로 표준화해 학습하고, 예측은 원래 단위로 되돌려 평가합니다.
- AdamW(norm, bias, embedding에는 weight decay 없음), 5% warmup 뒤 cosine 감소, gradient clipping 1.0.
- 매 에폭 val RMSE로 checkpoint를 고르고, 10에폭 동안 나아지지 않으면 멈춥니다. 마지막에 best checkpoint로 test를 평가합니다.
- 지표: RMSE, MSE, MAE, Pearson, Spearman, CI(concordance index, 메모리 절약을 위해 덩어리로 나눠 정확히 계산), R², 그리고 데이터셋 기준값으로 이진화한 AUROC와 AUPRC.

run 폴더(`runs/<dataset>_<split>_<encoder>_t<esm 층수>_<fusion>_s<seed>/`)에 남는 파일:

| 파일 | 내용 |
| --- | --- |
| `config.yaml` | 실제로 쓴 전체 설정 |
| `split.json` | split 인덱스와 통계 |
| `history.csv` | 에폭별 손실, val 지표, 소요 시간 |
| `best.pt` | best checkpoint |
| `metrics.json` | test 지표 |
| `predictions_{val,test}.csv` | 쌍별 예측값. 보정(conformal) 단계의 입력 |

### 장치 처리 (`device.py`)

- 장치는 CUDA, MPS, CPU 순서로 고릅니다. `train.device`로 고정할 수 있습니다.
- CUDA에서는 bf16 autocast와 TF32를 켭니다. MPS와 CPU는 fp32로 학습합니다. bf16은 GradScaler가 필요 없어 학습 루프가 장치마다 같습니다.
- MPS에 없는 연산은 CPU로 넘깁니다(`PYTORCH_ENABLE_MPS_FALLBACK=1`).
- MPS 메모리 상한을 권장 작업 집합의 1.0배로 둡니다(`PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0`, LOW는 0.8). 기본값(1.7배)에서는 큰 run이 통합 메모리를 swap으로 밀어내 시스템 전체가 멈춘 적이 있습니다. 지금은 그 대신 OOM 오류로 끝납니다. 무거운 GPU 작업은 한 번에 하나만 돌리세요.

### M1 프리셋을 정한 근거

M1 Pro에서 단백질 1,000토큰, d_model 256으로 재 보니 MPS 메모리 8.3GB, step당 1.2초였습니다. 그래서 `configs/davis_*.yaml`은 단백질 512토큰(pooling), d_model 128(약물 인코더와 fusion 모두)로 줄였습니다. 이 설정에서 DAVIS 한 에폭이 약 85~95초, 메모리는 약 3GB입니다. CUDA 장비에서는 `configs/cuda_650m.yaml`(ESM-2 650M, d_model 384)을 쓰면 됩니다.

## 5. 결과: DAVIS, seed 0

설정은 `configs/davis_graph.yaml`(M1 프리셋)이고 장치는 M1 MPS입니다. 바꾼 것은 약물 인코더뿐입니다. 아래는 test 지표입니다.

| split | encoder | RMSE ↓ | Pearson ↑ | Spearman ↑ | CI ↑ | AUROC ↑ | AUPRC ↑ | best / 정지 에폭 | 학습 시간 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | graph | **0.457** | **0.829** | 0.650 | 0.869 | **0.969** | **0.734** | 41 / 51 | 73분 |
| random | chemberta-frozen | 0.460 | 0.825 | **0.667** | **0.879** | 0.965 | 0.708 | 30 / 40 | 61분 |
| cold_drug | graph | 0.741 | 0.305 | 0.174 | 0.625 | 0.762 | 0.203 | 2 / 12 | 17분 |
| cold_drug | chemberta-frozen | **0.683** | **0.390** | **0.260** | **0.688** | **0.799** | **0.299** | 16 / 26 | 40분 |
| cold_target | graph | 0.693 | 0.567 | 0.501 | 0.772 | 0.860 | 0.429 | 12 / 22 | 32분 |
| cold_target | chemberta-frozen | **0.651** | **0.625** | **0.548** | **0.801** | **0.882** | **0.486** | 14 / 24 | 37분 |

6개 run에 모두 약 4.4시간이 걸렸습니다. 원자료는 `runs/compare_davis/compare_runs.csv`, run별 곡선은 각 `history.csv`에 있습니다.

### 해석

- **random split에서는 두 인코더가 비슷합니다.** RMSE 차이는 0.003입니다. 같은 약물과 표적이 train에 이미 나오므로, 약물 표현이 어떤 방식이든 약물 정체성만 구분하면 충분합니다.
- **cold split에서는 ChemBERTa가 모든 지표에서 앞섭니다.** 처음 보는 약물(`cold_drug`)에서 RMSE가 0.058 낮고 CI가 0.063 높습니다. train 약물이 54개뿐인 상황에서 처음부터 학습한 그래프 인코더는 일반화할 화학 지식을 얻기 어렵습니다. 사전학습 표현이 이 부족을 메운 것으로 보입니다.
- **Graph Transformer는 `cold_drug`에서 2에폭 만에 val 최적에 도달했습니다.** 그 뒤로는 train 약물을 외우는 방향으로만 학습이 진행됐다는 뜻입니다. 정규화를 강화하거나(dropout, 작은 모델) 사전학습 그래프 인코더를 쓰면 나아질 여지가 있습니다.
- **`cold_target`에서도 ChemBERTa가 앞서는 것은 예상 밖입니다.** 이 split에서는 test 약물 68개가 모두 train에 있습니다. 단백질 쪽이 같으므로 약물 표현의 질이 fusion의 상호작용 학습에 영향을 준 것으로 볼 수 있지만, seed 하나로는 단정할 수 없습니다.
- 모든 cold split에서 성능이 크게 떨어집니다(Pearson 0.83에서 0.3~0.6). random split의 수치는 실제 신약이나 신규 표적 상황의 성능을 크게 과대평가합니다. 이 프로젝트의 기권(abstention)과 보정 작업이 필요한 이유이기도 합니다.

### 주의점

- seed가 하나라서 표준편차가 없습니다. 특히 `cold_drug`의 test는 약물 7개뿐이어서, 어떤 약물이 test에 들어가느냐에 따라 순위가 바뀔 수 있습니다. 결론을 내기 전에 seed(모델 seed와 split seed)를 3개 이상으로 늘려야 합니다.
- 두 인코더 모두 같은 M1 프리셋(d_model 128)을 썼습니다. 학습 가능한 파라미터 수는 graph 2.16M, chemberta-frozen 1.59M입니다. ChemBERTa는 고정된 77M 파라미터의 사전학습 표현을 추가로 씁니다.
- 하이퍼파라미터는 인코더별로 튜닝하지 않았습니다.

## 6. 다음 단계

1. seed 1, 2를 추가해 평균과 표준편차를 구합니다: `dti compare ... --seeds 0,1,2` (끝난 run은 건너뜀).
2. `cold_both`와 `scaffold` split, `chemberta-ft`를 추가합니다.
3. KIBA(`configs/kiba_graph.yaml`)에서 같은 비교를 반복합니다.
4. `predictions_*.csv`를 conformal prediction과 risk-coverage 평가(트랙 계획 W08)의 입력으로 씁니다.
5. CUDA 장비가 있으면 `configs/cuda_650m.yaml`로 ESM-2 크기의 효과를 확인합니다.

## 7. 코드 지도

| 파일 | 역할 |
| --- | --- |
| `config.py` | dataclass 설정, YAML 로드, `-s key=value` 덮어쓰기 |
| `device.py` | 장치 선택, autocast, MPS 환경 변수 |
| `data/datasets.py` | 데이터셋 목록, 다운로드, 정제 |
| `data/splits.py` | 5가지 split |
| `data/graph.py` | SMILES에서 원자 특징, 결합 행렬, 최단 경로 행렬 생성 |
| `data/featurize.py` | `TokenStore`, ESM-2와 ChemBERTa 임베딩 캐시 |
| `data/dataset.py` | 특징 제공, `PairDataset`, 배치 padding(`Collator`) |
| `models/layers.py` | attention, FFN, pooling 블록 |
| `models/graph_transformer.py` | Graph Transformer |
| `models/chemberta.py` | ChemBERTa 인코더(고정 또는 fine-tune) |
| `models/fusion.py` | Fusion Transformer와 회귀 head |
| `models/model.py` | 전체 모델 조립, 파라미터 그룹 |
| `metrics.py` | 회귀와 이진 지표, CI |
| `train.py` | 학습 루프, 평가, 산출물 저장 |
| `compare.py` | 인코더 비교 grid와 요약표 |
| `cli.py` | `dti` 명령 |
