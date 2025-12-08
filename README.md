# AI-Generated Image Detector

This project provides a lightweight machine learning-based detector to distinguish between real and AI-generated images. It avoids deep learning models and instead relies on traditional ML classifiers trained on statistical and frequency-domain features.

This approach is based on the idea that AI-generated images often contain subtle artifacts from the generation process (e.g., upsampling), which can be captured by analyzing features in the frequency domain, color space, and noise patterns.

---

## (KOR)

## AI 생성 이미지 탐지기

본 프로젝트는 AI 생성 이미지와 실제 이미지를 구분하는 경량 머신러닝 탐지기입니다. 딥러닝 모델을 사용하지 않고, 통계 및 주파수 기반 특징을 추출하여 전통적인 머신러닝 분류기를 학습시키는 방식을 사용합니다.

이 접근법은 AI 생성 이미지가 생성 과정(예: 업샘플링)에서 발생하는 미묘한 아티팩트를 포함하는 경우가 많으며, 이러한 아티팩트는 주파수 도메인, 색상 공간, 노이즈 패턴 분석을 통해 포착될 수 있다는 아이디어에 기반합니다.

### 핵심 특징

본 탐지기는 4가지 주요 통계적 특징 그룹을 추출하여 이미지를 분석합니다.

1.  **1D 파워 스펙트럼 (1D Power Spectrum)**: 이미지의 2D 푸리에 변환 후 방위각 적분(azimuthal integration)을 통해 얻은 1D 파워 스펙트럼은 생성 모델 특유의 주기적인 패턴을 탐지하는 데 효과적입니다.
2.  **스펙트럼 왜곡 (Spectral Distortions)**: 업샘플링 과정에서 발생하는 스펙트럼의 왜곡을 포착하기 위해 특정 주파수 대역의 에너지 비율 등을 계산합니다.
3.  **색상 단서 (Color Cues)**: 실제 카메라의 이미지 처리 파이프라인(ISP)과 생성 모델의 색상 처리 방식의 차이를 활용합니다. HSV 색 공간에서의 채도(Saturation) 분포, RGB 채널 간 상관 계수 등을 특징으로 사용합니다.
4.  **노이즈 잔여물 (Noise Residuals)**: 이미지에서 노이즈 제거 필터를 적용한 후 남는 잔여물(residual)을 분석합니다. 실제 이미지의 센서 노이즈와 생성된 노이즈 패턴의 통계적 차이를 활용합니다.

### 프로젝트 구조

```
deepdect/
├── data/                     # 데이터셋 루트 폴더
│   └── CIFAKE/
│       ├── train/
│       │   ├── real/
│       │   └── fake/
│       └── test/
│           ├── real/
│           └── fake/
├── CIFAKE_output/            # 파이프라인 실행 시 생성되는 결과물 폴더
│   ├── CIFAKE_train_features.csv
│   ├── CIFAKE_test_features.csv
│   ├── CIFAKE_rf.joblib
│   └── report_CIFAKE_rf/
│       ├── report.json
│       └── confusion_matrix.png
├── extract_features.py       # 이미지에서 특징을 추출하고 CSV로 저장
├── train.py                  # 특징 CSV 파일을 이용해 모델을 학습
├── eval.py                   # 학습된 모델을 평가
├── predict.py                # 단일 이미지 또는 폴더에 대해 예측 수행
├── run_pipeline.py           # 전체 파이프라인(특징 추출, 학습, 평가)을 실행
├── features.py               # 특징 추출 함수 모음
├── requirements.txt
└── README.md
```

### 설치 및 준비

1.  **저장소 복제:**
    ```bash
    git clone <repository-url>
    cd deepdect
    ```

2.  **가상 환경 생성 및 활성화:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **의존성 설치:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **데이터 준비:**
    `data` 폴더 아래에 데이터셋을 준비합니다. `run_pipeline.py`는 아래와 같은 구조를 예상합니다.
    - `data/<dataset_name>/train/real/`
    - `data/<dataset_name>/train/fake/`
    - `data/<dataset_name>/test/real/`
    - `data/<dataset_name>/test/fake/`

### 사용법

#### 전체 파이프라인 실행

`run_pipeline.py` 스크립트는 지정된 데이터셋에 대해 **특징 추출, 여러 모델 학습, 평가**까지의 전체 과정을 자동화합니다.

```bash
python run_pipeline.py --data_dir data/CIFAKE
```

- 이 명령은 `data/CIFAKE` 데이터셋을 사용하여, `CIFAKE_output` 폴더에 특징, 학습된 모델(`linsvm`, `rf`, `xgb`, `mlp`), 평가 리포트를 저장합니다.

#### 개별 스크립트 실행

**1. 특징 추출 (`extract_features.py`)**

```bash
python extract_features.py --real_dir data/CIFAKE/train/real --fake_dir data/CIFAKE/train/fake --out_csv CIFAKE_output/train_features.csv
```

**2. 모델 학습 (`train.py`)**

```bash
python train.py --csv CIFAKE_output/train_features.csv --model rf --out_model CIFAKE_output/model_rf.joblib
```

**3. 모델 평가 (`eval.py`)**

```bash
python eval.py --csv CIFAKE_output/test_features.csv --model CIFAKE_output/model_rf.joblib --report_dir CIFAKE_output/report_rf
```

**4. 예측 (`predict.py`)**

```bash
python predict.py --model CIFAKE_output/model_rf.joblib --input /path/to/your/image.jpg
```

---

## (ENG)

## AI-Generated Image Detector

This project provides a lightweight machine learning detector to distinguish between real and AI-generated images. It avoids deep learning models and instead uses traditional machine learning classifiers trained on statistical and frequency-based features.

This approach is based on the idea that AI-generated images often contain subtle artifacts from the generation process (e.g., upsampling), which can be captured by analyzing the frequency domain, color space, and noise patterns.

### Core Features

The detector analyzes images by extracting four main groups of statistical features:

1.  **1D Power Spectrum**: The 1D power spectrum, obtained through azimuthal integration of the 2D Fourier transform, is effective at detecting periodic patterns unique to generative models.
2.  **Spectral Distortions**: Calculates features like the energy ratio of different frequency bands to capture distortions caused by upsampling processes.
3.  **Color Cues**: Leverages the differences between the image processing pipeline (ISP) of real cameras and the color generation methods of models. Features include the saturation distribution in the HSV color space and correlation coefficients between RGB channels.
4.  **Noise Residuals**: Analyzes the residual patterns left after applying a denoising filter to an image. This utilizes the statistical differences between sensor noise in real images and generated noise patterns.

### Project Structure

```
deepdect/
├── data/                     # Dataset root folder
│   └── CIFAKE/
│       ├── train/
│       │   ├── real/
│       │   └── fake/
│       └── test/
│           ├── real/
│           └── fake/
├── CIFAKE_output/            # Output directory created when the pipeline runs
│   ├── CIFAKE_train_features.csv
│   ├── CIFAKE_test_features.csv
│   ├── CIFAKE_rf.joblib
│   └── report_CIFAKE_rf/
│       ├── report.json
│       └── confusion_matrix.png
├── extract_features.py       # Extracts features from images and saves them to a CSV
├── train.py                  # Trains a model using the feature CSV file
├── eval.py                   # Evaluates a trained model
├── predict.py                # Performs prediction on a single image or a folder
├── run_pipeline.py           # Runs the entire pipeline (extraction, training, evaluation)
├── features.py               # Collection of feature extraction functions
├── requirements.txt
└── README.md
```

### Setup and Preparation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd deepdect
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare data:**
    Place your dataset under the `data` folder. `run_pipeline.py` expects the following structure:
    - `data/<dataset_name>/train/real/`
    - `data/<dataset_name>/train/fake/`
    - `data/<dataset_name>/test/real/`
    - `data/<dataset_name>/test/fake/`

### Usage

#### Running the Full Pipeline

The `run_pipeline.py` script automates the entire process of **feature extraction, training multiple models, and evaluation** for a specified dataset.

```bash
python run_pipeline.py --data_dir data/CIFAKE
```

- This command uses the `data/CIFAKE` dataset and saves the features, trained models (`linsvm`, `rf`, `xgb`, `mlp`), and evaluation reports into the `CIFAKE_output` directory.

#### Running Individual Scripts

**1. Feature Extraction (`extract_features.py`)**

```bash
python extract_features.py --real_dir data/CIFAKE/train/real --fake_dir data/CIFAKE/train/fake --out_csv CIFAKE_output/train_features.csv
```

**2. Model Training (`train.py`)**

```bash
python train.py --csv CIFAKE_output/train_features.csv --model rf --out_model CIFAKE_output/model_rf.joblib
```

**3. Model Evaluation (`eval.py`)**

```bash
python eval.py --csv CIFAKE_output/test_features.csv --model CIFAKE_output/model_rf.joblib --report_dir CIFAKE_output/report_rf
```

**4. Prediction (`predict.py`)**

```bash
python predict.py --model CIFAKE_output/model_rf.joblib --input /path/to/your/image.jpg
```
