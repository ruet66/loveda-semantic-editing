# 데이터·가중치 출처와 이용조건

이 저장소는 수업의 비상업적 연구·재현 실험을 공유하기 위한 자료입니다. 공개되어 있다는 사실이 모든 포함 자료의 상업 이용이나 무제한 재배포를 허용한다는 뜻은 아닙니다. 아래 자료의 조건은 자체 작성한 학습 코드와 구분해서 적용됩니다.

## LoveDA

- 저자: Junjue Wang, Zhuo Zheng, Ailong Ma, Xiaoyan Lu, Yanfei Zhong.
- 논문: *LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation*, NeurIPS Datasets and Benchmarks, 2021.
- 원본 및 공식 이용조건: https://github.com/Junjue-Wang/LoveDA#license
- 데이터 배포: https://doi.org/10.5281/zenodo.5706578
- 라이선스: **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.
- 라이선스 전문: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

`data/eval/`의 RGB·라벨·마스크는 LoveDA에서 가져온 개발 평가용 자료입니다. 원본 일부를 패치로 자르고, 실험용 목표 의미지도 및 편집 영역 마스크를 구성했습니다. 이 데이터 및 해당 데이터의 변형을 공유할 때 원저자와 출처, 변경 사실을 표시하고 비상업적 이용·동일조건 공유 요건을 따라야 합니다. 저장소 전체가 LoveDA 원저자의 공식 구현이거나 승인을 받은 결과라는 의미는 아닙니다.

LoveDA 원본 전체는 이 저장소에 포함하지 않으며, 학습 코드는 공식 배포처에서 Train ZIP을 다운로드합니다.

## TorchVision 및 사전학습 특징 추출기

- 사용 라이브러리: https://github.com/pytorch/vision
- 라이브러리 코드 라이선스: BSD-3-Clause. https://github.com/pytorch/vision/blob/main/LICENSE
- 사전학습 가중치 안내: https://docs.pytorch.org/vision/stable/models.html
- `squeezenet_features.pt`: TorchVision `SqueezeNet1_1_Weights.IMAGENET1K_V1`의 `features[:8]`를 추출한 고정 특징 추출기. 원본: https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth

TorchVision은 사전학습 가중치에 학습 데이터에서 유래한 별도 조건이 있을 수 있다고 안내합니다. 라이브러리 코드의 BSD 라이선스를 모든 사전학습 가중치에 일괄 적용하지 않습니다. 본 배포가 상업적 사용 또는 모든 형태의 재배포에 대한 권한 확인을 대신하지 않습니다.

## 프로젝트에서 학습한 가중치와 결과

- `semantic_04000.pt`: 프로젝트에서 LoveDA로 학습하고 추가 학습한 GAN 생성기·판별기.
- `semantic_teacher.pt`, `semantic_audit.pt`: 프로젝트에서 LoveDA로 학습한 DeepLabV3 MobileNetV3 분할 모델. 교사와 평가기는 별도 가중치입니다.
- 모델 출처 기록: `weights/provenance.json`.

생성 RGB는 합성 결과이며 실제 관측 사진으로 취급하지 마세요. 포함한 가중치 전체에 대해 독립적인 상업 이용 허락을 부여하지 않습니다. 코드·데이터·기존 가중치의 이용조건을 각각 확인해야 합니다. 편집 품질의 개선은 아직 본학습으로 검증하지 않았습니다.
