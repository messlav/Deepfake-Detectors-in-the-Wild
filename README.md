# Not All Deepfake Detectors Are Created Equal

This repository contains all the code for models used to generate our datasets, the datasets themselves, 
and the Deepfake Detector models.

## Datasets

[File with links to all datasets](https://docs.google.com/document/d/17-uU4Y3eaOv2HqtLsjBYJUyY1_s-GunFBP9sF5i-2A4/edit?usp=sharing)

## Generate datasets
GPEN example

### Installation
```bash
cd ~/NADDACE/data/GPEN/GPEN
pip install -r requirements.txt
```

### Download dataset
```bash
cd ~/NADDACE/data/data/lfw
gdown 1msWS3tVzMCTlK7vMoxTW8RJRmpWtCwSj
unzip -q lfw_SimSwap.zip
```

### Pretrained models

```bash
wget "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth" -O weights/RetinaFace-R50.pth
wget "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth" -O weights/GPEN-BFR-512.pth
wget "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-256.pth" -O weights/GPEN-BFR-256.pth
wget "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/realesrnet_x2.pth" -O weights/realesrnet_x2.pth
wget "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/ParseNet-latest.pth" -O weights/ParseNet-latest.pth
```

### Inference

```bash
cd ~/NADDACE/data/GPEN
python3 create_data.py \
        --input_dir ~/NADDACE/data/data/lfw/lfw_roop \
        --output_dir ~/NADDACE/data/data/lfw/GPEN_lfw_roop
```

## Test Models
SBI example

### Installation
```bash
cd ~/NADDACE/models/SBI/SelfBlendedImages
pip install -r requirements.txt
```
### Download Dataset
```bash
cd ~/NADDACE/data/data/CelebA_HQ
gdown 1xmSduyzHjywucxcvK9bl3CM7-oM5ksSR
unzip -q CelebA_HQ_roop.zip
```
### Pretrained Models
```bash
gdown 1X0-NYT8KPursLZZdxduRQju6E52hauV0 -O ~/NADDACE/models/SBI/SelfBlendedImages/weights/FFc23.tar
```
### Inference
```bash
python3 inference.py \
        --input_dir  ~/NADDACE/data/data/CelebA_HQ/CelebA_HQ_roop \
        --output_csv  ~/NADDACE/models/preds/SBI/SBI_CelebA_HQ_roop.csv \
        --max_size_image 1024
```

## Licenses

Our work uses a lot of third party libraries as well pre-trained models. The users should keep in mind that these third party components have their own license and terms, therefore our license is not being applied.

## Credits

- [GPEN](https://github.com/yangxy/GPEN)
- [Roop](https://github.com/s0md3v/roop)
- [SimSwap](https://github.com/neuralchen/SimSwap)
- [Detecting Deepfakes with Self-Blended Images - SBI](https://github.com/mapooon/SelfBlendedImages)
- [Implicit Identity Leakage: The Stumbling Block to Improving Deepfake Detection Generalization - CADDM](https://github.com/megvii-research/CADDM)
- [End-to-End Reconstruction-Classification Learning for Face Forgery Detection -
RECCE](https://github.com/VISION-SJTU/RECCE)
- [Multi-attentional Deepfake Detection - MAT](https://github.com/yoctta/multiple-attention)
- [FaceForensics++ - FF](https://github.com/ondyari/FaceForensics)
- [Multi-modal Multi-scale Transformers for Deepfake Detection - M2TR](https://github.com/wangjk666/M2TR-Multi-modal-Multi-scale-Transformers-for-Deepfake-Detection)
