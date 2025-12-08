# Lightweight ML Detector (Real vs AI-Generated Images)
전통적 머신러닝(scikit-learn/XGBoost/RandomForest/Linear SVM/MLP)로 **AI 생성 이미지 vs 실제 이미지**를 구분하기 위한 구현 가이드입니다.  
핵심은 “딥러닝 feature extractor 없이” 아래 4류의 **통계적 특징(feature engineering)**을 추출해 학습하는 것입니다.

- (1) **1D Power Spectrum** + **azimuthal integration(방위각 적분, radial average)** 기반 스펙트럼 특징  
- (2) **스펙트럼 왜곡(spectral distortions)** (업샘플링 유발 패턴) 기반 밴드 통계  
- (3) **Color cues**: saturation 통계 + 채널 간 상관 구조  
- (4) **Noise residual**: high-pass/denoise 후 residual에서 공분산/스펙트럼/co-occurrence 통계

## 0) 근거 논문(핵심 아이디어 출처)
- **Unmasking DeepFakes with simple Features**: 주파수 분석 + 간단 분류기로 deepfake 탐지 (FFT 기반 특징).  
  https://arxiv.org/abs/1911.00686 citeturn0search0  
- **Watch Your Up-Convolution (CVPR 2020)**: 생성모델 업샘플링이 스펙트럼 분포를 왜곡 → 이를 이용한 간단 탐지기.  
  https://openaccess.thecvf.com/content_CVPR_2020/papers/Durall_Watch_Your_Up-Convolution_CNN_Based_Generative_Deep_Neural_Networks_Are_CVPR_2020_paper.pdf citeturn0search4  
- **Leveraging Frequency Analysis for Deep Fake Image Recognition (ICML 2020)**: DCT/주파수 영역에서 GAN 아티팩트가 강하게 드러남(고전 분류기 성능 ↑).  
  https://proceedings.mlr.press/v119/frank20a/frank20a.pdf citeturn0search1  
- **Detecting GAN-generated Imagery using Color Cues**: 카메라 vs GAN의 색 처리 차이(두 가지 cue)로 구분.  
  https://arxiv.org/abs/1812.08247 citeturn0search2  
- **Detecting Generated Images by Real Images (ECCV 2022)**: “real 이미지의 noise pattern” 공통성을 활용 → 간단 분류기로도 광범위 생성기 탐지.  
  https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740089.pdf citeturn0search3  
  (ACM/스프링어 DOI 페이지) https://dl.acm.org/doi/10.1007/978-3-031-19781-9_6 citeturn0search7  

> 주의: 주파수/압축 특징은 “JPEG/리사이즈 편향”을 학습하기 쉬움. **real/fake를 동일한 전처리(리사이즈/재인코딩)**로 맞추는 프로토콜이 중요합니다. (아래 2-2 참고)

---

## 1) 프로젝트 구조 (CLI 환경 기준)
아래처럼 “단일 폴더 + 4개 스크립트” 구조로 가면 CLI 자동화가 쉽습니다.

```
detector/
  data/
    real/         # 실제 이미지들
    fake/         # AI 생성 이미지들
  out/
  extract_features.py
  train.py
  predict.py
  eval.py
  requirements.txt
  README.md
```

### 1-1) requirements.txt (예시)
```txt
numpy
opencv-python
scikit-learn
tqdm
pandas
joblib
matplotlib
# optional
xgboost
```

### 1-2) 설치 및 실행
```bash
cd detector
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2) 데이터 준비 규칙(중요)
### 2-1) 폴더 레이아웃
- `data/real/*.jpg|png`
- `data/fake/*.jpg|png`
- 파일명은 아무거나 가능(내부에서 index로 관리)

### 2-2) 편향(artifact) 줄이기 위한 “공정 전처리”
주파수 기반 탐지는 특히 **(a) 압축률, (b) 해상도, (c) 리사이즈 방식** 차이를 학습해버릴 수 있습니다.  
그래서 아래처럼 “모든 이미지를 동일 파이프라인으로 재인코딩”하는 것을 권장합니다.

- **Resize**: 예) 256×256 (Lanczos 또는 area 계열)
- **JPEG 재인코딩**: 예) Q=95로 통일(원본이 PNG여도 동일하게 JPEG로 저장)
- **Color space**: RGB로 통일

(이 전처리는 `extract_features.py`에서 옵션으로 수행하도록 설계합니다.)

---

## 3) 메인 로직: Feature Engineering (4 묶음)
각 이미지에서 feature vector 하나를 만듭니다. 최종 feature는 아래를 **concat**합니다.

```
F(x) = [F_spec1D(x), F_specDistort(x), F_color(x), F_residual(x)]
```

### 3-1) (A) 1D Power Spectrum + Azimuthal Integration (Radial Average)
**핵심 개념**  
- 이미지 FFT → magnitude^2(power spectrum) → 중심 쉬프트(fftshift)  
- 주파수 평면에서 반지름 r 별로 평균을 내면 **1D radial power spectrum**이 됩니다.  
- 이 “방위각 적분(azimuthal integration)”은 간단하지만 생성기 특유의 주파수 패턴을 잘 드러냅니다. citeturn0search0turn0search4

**추천 특징들**
- `P(r)` 값을 일정 bin 개수(예: 64/128)로 샘플링한 벡터
- low/mid/high band energy 비율
- log-log 스펙트럼의 선형 회귀 slope(자연이미지의 1/f 구조 대비 변화)
- peakness(주기적인 피크가 있는지: upsampling pattern 단서)

**구현 힌트(의사코드)**
```python
# img_gray: float32, [0,1], shape (H,W)
F = fft2(img_gray)
S = fftshift(abs(F)**2)                 # power spectrum
r, bins = radial_bins(H, W, n_bins=128)
P = radial_mean(S, r, bins)             # azimuthal integration
P = log(P + eps)
# feature: P[0:128] + summary stats
```

### 3-2) (B) “스펙트럼 왜곡” 밴드 통계 (Upsampling artifacts)
CVPR 2020 분석에 따르면, 업샘플링(특히 up-conv)이 **스펙트럼 분포를 현실과 다르게** 만들 수 있고, 이 왜곡은 탐지 단서가 됩니다. citeturn0search4

**여기서의 ‘왜곡 특징’은 (A)에서 얻는 P(r)를 요약해도 되고**, 아래처럼 직접 지표화해도 됩니다.

**추천 특징들**
- band 에너지:
  - `E_low = mean(P[r<r1])`, `E_mid`, `E_high`
  - `E_high/E_low`, `E_mid/E_low`
- “ring artifact index”: 특정 중간 주파수 구간에서 peak가 반복되는지(표준편차/첨도 등)
- (옵션) 2D 스펙트럼의 quadrant 별 에너지(방향성 차이)

**메인 로직**
- 생성 이미지에서 특정 band가 과/저평가되거나 peak가 생김 → band 통계로 분리 가능 citeturn0search4

### 3-3) (C) Color Cues: Saturation 통계 + 채널 상관 구조
McCloskey & Albright는 “GAN의 색 처리”가 카메라 파이프라인과 다르다는 점에서 **두 가지 cue**를 제시합니다. citeturn0search2

**추천 특징들**
- HSV에서 **S(saturation) 히스토그램**: 16~32 bins + mean/std/percentile
- RGB 채널 간 상관계수: corr(R,G), corr(R,B), corr(G,B)
- YCbCr/ Lab 변환 후 chroma 채널 분포(Cb/Cr 또는 a/b) 요약(스큐/첨도 등)

**메인 로직**
- 카메라-센서/ISP가 만들어내는 색 통계와 GAN의 “색 계산 방식”이 다르다 → 색 도메인 통계만으로도 구분 가능 citeturn0search2

### 3-4) (D) Noise residual 기반 특징 (High-pass / Denoise → Residual 통계)
ECCV 2022는 “real 이미지의 noise pattern 공통성”을 이용해 생성이미지를 검출하는 관점을 제시합니다. citeturn0search3turn0search7

여기서는 딥러닝 블록 대신 **간단한 residual**을 만들고 통계량을 뽑습니다.

**Residual 생성(가벼운 옵션 2개)**
1) **Denoise 기반 residual**
- `den = fastNlMeansDenoisingColored(img)` (OpenCV)  
- `res = img - den`

2) **High-pass 기반 residual**
- Laplacian(또는 Sobel)로 high-pass:
- `res = Laplacian(gray)`

**Residual에서 뽑을 특징**
- 공분산: `cov(res_R, res_G)`, `cov(res_R, res_B)`, 등
- residual의 1D power spectrum(= (A) 반복하지만 residual에서)
- co-occurrence(간단 버전):
  - residual을 양자화(예: -4..4로 클리핑 후 shift)
  - 인접 픽셀쌍 (dx=1, dy=0) 빈도 히트맵을 펼쳐서 feature로 사용

**메인 로직**
- 실제 사진은 센서/압축/광학에 의해 생기는 noise 패턴이 “일정한 구조”를 띠고, 생성 이미지는 그 구조가 다르거나 결여되기 쉬움 → residual 통계가 유효 citeturn0search3turn0search7

---

## 4) 코드 작성법 (CLI 기준, LLM이 작성/수정하기 쉽게)
아래는 “LLM이 이해하기 좋은” 방식으로 스크립트를 분할한 설계입니다.

### 4-1) extract_features.py (핵심)
**역할**
- 이미지 로딩 → (옵션) 공정 전처리(resize + JPEG 재저장) → feature vector 추출 → CSV 저장

**CLI**
```bash
python extract_features.py \
  --real_dir data/real \
  --fake_dir data/fake \
  --out_csv out/features.csv \
  --img_size 256 \
  --reencode_jpeg 95 \
  --bins 128
```

**출력 CSV 스키마(예)**
- `path,label`
- `spec_bin_0 ... spec_bin_127`
- `spec_E_low, spec_E_mid, spec_E_high, spec_slope, spec_peakness`
- `sat_hist_0 ... sat_hist_31, sat_mean, sat_std, corr_rg, corr_rb, corr_gb`
- `res_var, res_cov_rg, ... , res_spec_bin_0 ... , cooc_0 ...`

> 팁: feature 이름을 고정하면 `train.py`가 자동으로 모든 열을 읽어 학습 가능합니다.

### 4-2) train.py
**역할**
- CSV 로드 → train/val split → 스케일링(필요시) → 모델 학습 → joblib 저장

**CLI (예: RandomForest)**
```bash
python train.py \
  --csv out/features.csv \
  --model rf \
  --out_model out/model_rf.joblib
```

**CLI (예: Linear SVM)**
```bash
python train.py \
  --csv out/features.csv \
  --model linsvm \
  --out_model out/model_svm.joblib
```

**CLI (예: XGBoost, optional)**
```bash
python train.py \
  --csv out/features.csv \
  --model xgb \
  --out_model out/model_xgb.joblib
```

### 4-3) predict.py
**역할**
- 단일 이미지/폴더 입력 → feature 추출(동일 옵션) → model 로드 → 확률/라벨 출력

**CLI**
```bash
python predict.py \
  --model out/model_xgb.joblib \
  --input some.jpg \
  --img_size 256 \
  --reencode_jpeg 95
```

### 4-4) eval.py
**역할**
- test split 성능(Acc/F1/AUROC) + 혼동행렬 + ROC 저장
- “강건성 테스트” 옵션(blur/resize/jpeg)을 동일 적용해서 성능 변화 체크

**CLI**
```bash
python eval.py \
  --csv out/features.csv \
  --model out/model_xgb.joblib \
  --report out/report.json
```

---

## 5) 모델 선택 가이드(가벼움/성능/해석성)
- **Linear SVM / Logistic Regression**: 주파수 특징이 강하면 선형도 잘 나오는 경우가 있음(Frank ICML 2020 취지). citeturn0search1  
- **RandomForest**: 튜닝 부담 낮고 안정적(다만 고차원 feature에선 크기가 커질 수 있음)
- **XGBoost**: feature engineering과 궁합 좋고 성능 잘 나오는 경우가 많음
- **MLP**: feature가 충분하면 성능 좋지만 스케일링/튜닝 필요

추천: `linsvm`과 `xgb`를 베이스라인 2개로 두고 비교

---

## 6) 최소 구현 체크리스트(실패 방지)
1. real/fake 전처리 동일(resize, 재인코딩)  
2. 같은 이미지에서 **feature 추출 옵션이 train/test/predict에서 완전히 동일**  
3. `P(r)`는 **log 스케일**이 안정적  
4. co-occurrence는 **양자화/클리핑**이 없으면 차원이 폭발  
5. “dataset leakage” 방지: 비디오 프레임이라면 같은 비디오에서 나온 프레임이 train/test에 섞이지 않게 split

---

## 7) (선택) 추가 실험: 편향/강건성 리포트
- JPEG Q={70,85,95}, resize={192,256,384}, blur={0,1,2} 같은 변환을 **real/fake 모두 동일하게 적용**해 성능이 얼마나 유지되는지 측정
- 주파수 특징의 강점/취약점(압축/리사이즈)에 대한 해석을 논문/발표에 넣기 쉬움 citeturn0search4turn0search3

---

## Appendix A) 구현에 필요한 핵심 함수(설계 요약)
- `load_image(path) -> RGB float32`
- `preprocess(img, img_size, reencode_jpeg_q) -> RGB float32`
- `power_spectrum_1d(gray, n_bins) -> (P_bins, summary_stats)`  # azimuthal integration
- `color_cues(img_rgb) -> sat_hist + corr_stats`
- `residual_features(img_rgb) -> residual stats + cooc`
- `extract_all_features(img) -> 1D numpy array`
- `fit_model(X, y, model_name) -> sklearn estimator`
- `save(joblib)` / `load(joblib)`

---

## Appendix B) 추천 기본 하이퍼파라미터(출발점)
- Linear SVM: `C=1.0`, `class_weight='balanced'`, `StandardScaler`
- RF: `n_estimators=500`, `max_depth=None`, `min_samples_leaf=2`
- XGB: `n_estimators=800`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`
- MLP: `hidden_layer_sizes=(256,128)`, `alpha=1e-4`, `max_iter=200`, `early_stopping=True`

---

## References (BibTeX 간단 링크)
- Durall et al. Unmasking DeepFakes with simple Features. arXiv:1911.00686. citeturn0search0  
- Durall et al. Watch Your Up-Convolution. CVPR 2020. citeturn0search4  
- Frank et al. Leveraging Frequency Analysis for Deep Fake Image Recognition. ICML 2020. citeturn0search1  
- McCloskey & Albright. Detecting GAN-generated Imagery using Color Cues. arXiv:1812.08247. citeturn0search2  
- Liu et al. Detecting Generated Images by Real Images. ECCV 2022. citeturn0search3  
