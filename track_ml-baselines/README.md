# track_ml-baselines

**ML과 보정 트랙**의 작업 공간입니다. 조건 **A**(보정된 baseline)와 조건 **F**(Boltz-2 co-folding)를 만듭니다.
조건 E의 conformal 절차도 여기서 구현해 에이전트 트랙과 공유합니다.

> 상태: DTI affinity 회귀 파이프라인(`dti_fusion`)이 동작합니다. DAVIS seed 0으로 약물 인코더 비교(Graph Transformer와 ChemBERTa)를 마쳤습니다.
> 설계와 결과는 [`docs/PIPELINE.md`](docs/PIPELINE.md)에 있습니다. Boltz-2(`boltz2/`)는 실행 스크립트만 두고 진행하지 않습니다.

## 파이프라인 한눈에 보기

```
SMILES ──► Graph Transformer (처음부터 학습)  ─┐
       └─► ChemBERTa-77M (고정 또는 fine-tune) ─┤
                                                ├─► Fusion Transformer (co-attention) ─► pKd / KIBA score
서열   ──► ESM-2 잔기 임베딩 (고정, 디스크 캐시) ─┘
```

- 데이터: Hugging Face의 DAVIS, KIBA, BindingDB(`jglaser/binding_affinity`)
- split: `random`, `cold_drug`, `cold_target`, `cold_both`, `scaffold`
- 장치: CUDA, Apple Silicon(MPS), CPU 순서로 자동 선택. 같은 코드와 설정이 둘 다에서 돕니다.

## 빠른 시작

[uv](https://docs.astral.sh/uv/)를 씁니다. Python 3.12, PyTorch는 PyPI wheel을 받습니다(macOS는 MPS, Linux는 CUDA).

```bash
cd track_ml-baselines
uv sync --extra dev
uv run dti info                         # torch 버전과 선택된 장치 확인
uv run pytest -q                        # 단위 테스트 15개 (다운로드 없음)

uv run dti train -c configs/smoke.yaml  # 2천 쌍, 2 에폭으로 전체 경로 확인 (수 분)
```

DAVIS 한 번 학습:

```bash
uv run dti download davis
uv run dti featurize -c configs/davis_graph.yaml     # ESM-2 캐시 (처음 한 번)
uv run dti train -c configs/davis_graph.yaml -s data.split=cold_drug
```

인코더 비교 grid (이미 `metrics.json`이 있는 run은 건너뜀):

```bash
uv run dti compare -c configs/davis_graph.yaml -s train.out_dir=runs/compare_davis \
  --encoders graph,chemberta-frozen --splits random,cold_drug,cold_target --seeds 0
```

설정 값은 `-s key=value`(점 표기)로 덮어씁니다. 예: `-s drug.encoder=chemberta -s train.lr=1e-4`.

## 설정 파일

| 파일 | 용도 |
| --- | --- |
| `configs/smoke.yaml` | 2천 쌍, ESM-2 8M, 작은 모델. 동작 확인용 |
| `configs/davis_graph.yaml` | M1 프리셋. Graph Transformer + ESM-2 35M |
| `configs/davis_chemberta.yaml` | M1 프리셋. ChemBERTa 인코더 |
| `configs/kiba_graph.yaml` | KIBA용 M1 프리셋 |
| `configs/cuda_650m.yaml` | CUDA용. ESM-2 650M, 더 큰 모델 |

## DAVIS 결과 요약 (seed 0, test)

| split | encoder | RMSE | Pearson | CI |
| --- | --- | --- | --- | --- |
| random | graph | 0.457 | 0.829 | 0.869 |
| random | chemberta-frozen | 0.460 | 0.825 | 0.879 |
| cold_drug | graph | 0.741 | 0.305 | 0.625 |
| cold_drug | chemberta-frozen | **0.683** | **0.390** | **0.688** |
| cold_target | graph | 0.693 | 0.567 | 0.772 |
| cold_target | chemberta-frozen | **0.651** | **0.625** | **0.801** |

seed 하나의 결과라 차이의 유의성은 아직 말할 수 없습니다. 해석과 주의점은 [`docs/PIPELINE.md`](docs/PIPELINE.md#5-결과-davis-seed-0)를 보세요.

## 폴더 구조

```
src/dti_fusion/   패키지 (data/, models/, train.py, compare.py, cli.py)
configs/          YAML 설정
tests/            단위 테스트
docs/PIPELINE.md  파이프라인 설계 문서와 결과
boltz2/           Boltz-2 실행 스크립트 (보류)
data/, runs/      다운로드, 특징 캐시, 학습 결과 (git 제외)
```

## 트랙 계획

| 항목 | 내용 | 시기 |
| --- | --- | --- |
| 특징 캐시 | ESM-2 단백질 임베딩 (완료), Morgan fingerprint | W05 |
| 구조 feature | AlphaFold DB 구조 수집, pocket 검출(P2Rank / fpocket), pocket pLDDT·PAE·부피·잔기 조성 | W05 |
| 기준선 | Logistic regression / XGBoost, **4개 split 전부**. 딥러닝 기준선은 `dti_fusion` | W06 |
| 구조 ablation | pocket feature, SaProt / ESM-IF, 2D와 3D 비교 | W07 |
| 보정 | conformal prediction(MAPIE), risk-coverage 곡선, seed 고정 분산 | W08 |
| 조건 F (GPU 조건부) | Boltz-2 오픈 가중치 200쌍 층화 실행. 현재 보류 | W01 자원 게이트에서 결정 |

## 참고

- 평가 규약과 기권 정의: [`results/benchmark/EVALUATION.md`](../results/benchmark/EVALUATION.md)
- AlphaFold 3단계와 Boltz-2 배분 계획: 루트 [README](../README.md)의 "AlphaFold 활용 3단계"
- 이 트랙은 루트 `environment.yml` 대신 자체 uv 환경(`pyproject.toml`, `uv.lock`)을 씁니다.
- 의존성 라이선스 감사(`LICENSE-AUDIT.md`)는 W05에 이 트랙이 맡습니다. BindingDB 파생 데이터는 재배포하지 않습니다.
