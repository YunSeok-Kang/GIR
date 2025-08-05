# GIR 논문 실험 재현을 위한 자동화 스크립트

이 저장소는 원본 "GIR: 3D Gaussian Inverse Rendering for Relightable Scene Factorization" 프로젝트의 실험을 쉽게 재현하기 위한 자동화 스크립트를 제공합니다.

**- 원본 프로젝트 저장소:** [https://github.com/guduxiaolang/GIR](https://www.google.com/search?q=https://github.com/guduxiaolang/GIR)

-----

## 자동화 워크플로우

이 저장소의 스크립트는 **설정 기반**으로 동작합니다. `run_experiment.py` 상단의 `EXPERIMENT_CONFIGS` 딕셔너리를 수정하여 새로운 종류의 실험을 쉽게 추가하고 관리할 수 있습니다.

### **1. 사전 준비: 파일 구조**

스크립트를 실행하기 전, 프로젝트의 파일 구조가 아래와 같이 구성되어 있는지 확인하세요.

  * **`data/` 폴더**: 모든 장면 데이터셋을 이 폴더 안에 위치시킵니다.
  * **최상위 폴더**: 환경맵(HDR) 폴더는 프로젝트 최상위에 그대로 둡니다.

<!-- end list -->

```
GIR/
|-- data/
|   |-- nerf_synthetic/
|   |-- shiny_blender_dataset/
|-- hdri/
|-- high_res_envmaps_2k/
|-- run_experiment.py   # 메인 실험 스크립트
|-- create_videos.py    # 후처리 스크립트
`-- ... (기타 원본 프로젝트 파일)
```

### **2. 스크립트 실행 방법**

워크플로우는 단 두 단계로 간소화됩니다. 모든 스크립트는 실행할 **설정의 이름**을 첫 번째 인자로 받습니다.

#### **1단계: 훈련 및 모든 이미지 렌더링 (`run_experiment.py`)**

이 스크립트 하나로 **모델 훈련, 원본(GT) 이미지 렌더링, 재조명 이미지 렌더링**까지 모든 데이터 생성 과정을 한 번에 수행합니다.

```bash
# 예시: Shiny Blender 데이터셋에 대한 모든 데이터 생성
# (실행 전 --dry-run 옵션으로 확인 권장)
python run_experiment.py shiny_blender
```

#### **2단계: 비디오 생성 및 통합 (`create_videos.py`)**

`run_experiment.py` 실행 후, 아래 명령어로 모든 이미지 시퀀스를 찾아 최종 비디오로 변환하고 한 폴더에 모읍니다.

**(주의: 이 스크립트를 실행하려면 시스템에 `ffmpeg`이 설치되어 있어야 합니다.)**

```bash
# 예시: Shiny Blender 실험 결과물로 비디오 생성
# (실행 전 --dry-run 옵션으로 확인 권장)
python create_videos.py shiny_blender
```

-----

### **3. 최종 결과물**

모든 과정이 끝나면, 각 실험 설정에 따라 생성된 최종 비디오들은 `output_videos_{설정이름}/` 폴더에 저장됩니다.

  * **`output_videos_blender/`**: `blender_synthetic` 실험의 모든 결과 비디오.
  * **`output_videos_shiny_blender/`**: `shiny_blender` 실험의 모든 결과 비디오.