# Training your OpenSphere Face Recognition Model using QMUL_SurvFace or any Custom Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hz4Vw3kqka55KDx8jeKLoOWOxJdVCzCt?usp=sharing) </br>
This Colab teach you how to train your OpenSPhere Face Recognition model on your custom dataset!

This repo is part of the **IEEE World AI IoT Congress 2023** </br>
Refer [YOLOv5 DeepSort Face](https://github.com/yjwong1999/Yolov5_DeepSort_Face) for the main implementation of the project. </br>
Simulation for Conference Proceedings [https://doi.org/10.1109/AIIoT58121.2023.10174362](https://doi.org/10.1109/AIIoT58121.2023.10174362) </br>
Refer [here](https://www.researchgate.net/publication/371315031_Multi-Camera_Face_Detection_and_Recognition_in_Unconstrained_Environment) for the preprint </br>
(Please note that this is an ongoing project)

## Abstract
Multi-camera face detection and recognition is an Artificial Intelligence (AI) based technology that leverages multiple cameras placed at different locations to detect and recognize human faces in real-world conditions accurately. While face detection and recognition technologies have exhibited high accuracy rates in controlled conditions, recognizing individuals in open environments remains challenging due to factors such as changes in illumination, movement, and occlusions. In this paper, we propose a multi-camera face detection and recognition (MCFDR) pipeline, which consists of three main parts - face detection, face recognition, and tracking. A series of model training is done with the open-source dataset to build a robust pipeline, and finally, the pipeline adopted trained YOLOv5n for face detection model with mAP 0.495, precision value of 0.868, and recall value of 0.781. The system also adopted the SphereFace SFNet20 model with an accuracy of 82.05% and a higher inference rate than SFNet64 for face recognition. These models are then fed into DeepSORT for multi-camera tracking. Our dataset has been applied to the pipeline and shows ideal outcomes with objectives achieved.

## Retrain OpenSphere Model

This tutorial below is mainly to 
1.   setup the conda environment for OpenSPhere
2.   retrain OpenSphere models on QMUL-SurvFace dataset </br>

For a **painless tutorial** without worrying the envrionment setup, please refer the colab link below for a more detailed tutorial to train your OpenSPhere model using your **custom dataset** </br>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hz4Vw3kqka55KDx8jeKLoOWOxJdVCzCt?usp=sharing)

- Clone the repo
```
git clone https://github.com/yjwong1999/opensphere.git
cd opensphere
```

- Create a new conda environment to train OpenSphere
```
conda deactivate # if you are in other conda environment
conda env create -f environment.yml
conda activate opensphere

# Choose one of the following
bash scripts/dataset_setup_validation_only.sh    # if you only want to train with your custom dataset
OR
bash scripts/dataset_setup.sh                     # if you want to train with VGG Face 2 dataet
OR
bash scripts/dataset_setup_ms1m.sh                # if you want to train with ms1m dataset
```

- Get QMUL-SurvFace dataset
```
cd customize
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13ch6BPaexlKt8gXB_I8aX7p1G3yPm2Bl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13ch6BPaexlKt8gXB_I8aX7p1G3yPm2Bl" -O QMUL-SurvFace.zip && rm -rf /tmp/cookies.txt
unzip QMUL-SurvFace.zip

python3 generate_annot.py --directory QMUL-SurvFace # generate annotation for this dataset

cd ../
```

- Train OpenSphere Model using QMUL-SurvFace dataset
```
# Train SFNet20 using SphereFace loss function
CUDA_VISIBLE_DEVICES=0 python train.py --config config/train/survface_sfnet20_sphereface.yml
# Train SFNet20 using SphereFaceR loss function
CUDA_VISIBLE_DEVICES=0 python train.py --config config/train/survface_sfnet20_spherefacer.yml
# Train SFNet20 using SphereFace2 loss function
CUDA_VISIBLE_DEVICES=0 python train.py --config config/train/survface_sfnet20_sphereface2.yml

# Train SFNet64 using SphereFace loss function
CUDA_VISIBLE_DEVICES=0 python train.py --config config/train/survface_sfnet64_sphereface.yml
# Train SFNet64 using SphereFaceR loss function
CUDA_VISIBLE_DEVICES=0 python train.py --config config/train/survface_sfnet64_spherefacer.yml
# Train SFNet64 using SphereFace2 loss function
CUDA_VISIBLE_DEVICES=0 python train.py --config config/train/survface_sfnet64_sphereface2.yml

# NOTE THAT:
# CUDA_VISIBLE_DEVICES=0 means use 1st CUDA device to train
# CUDA_VISIBLE_DEVICES=0,1 means use 1st and 2nd CUDA devices to train
# and so on...
```

- Test OpenSphere Model using QMUL-SurvFace dataset
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/test/survface.yml --proj_dir project/<dir name>
```

- Convert OpenSphere Model to ONNX (for future usage)
```
# install onnx api
pip install onnx==1.14.1

# convert to onnx with dynamic batch size
CUDA_VISIBLE_DEVICES=0 python onnx_exporter.py --config config/test/survface.yml --proj_dir project/<dir name> --dynamic

# convert to onnx with fixed batch size (32 for example)
CUDA_VISIBLE_DEVICES=0 python onnx_exporter.py --config config/test/survface.yml --proj_dir project/<dir name> --batch-size 32
```

- Convert ONNX to OpenVINO (for future usage)
```
# install openvino developer api
pip install -q "openvino-dev>=2023.0.0" "nncf>=2.5.0"

# convert to openvino file using model optimizer
mo --input_model project/<dir name>/models/backbone_<iteration_num>.onnx --input_shape [-1,3,112,112] --compress_to_fp16
```

## Known Issue
- [X] **Issue 1: Dynamic Batch Size for OpenVINO model** 
- [ ] **Issue 2: Dynamic Batch Size for ONNX model**

Issue 1 is solved by using ```--input_shape [-1,C,H,W]``` when using OpenVINO model optimizer (mo), where -1 indicates dynamic batch size

## Acknowledgement
This work was supported by the Greatech Integration (M) Sdn Bhd with project number 8084-0008.

### Reference Code
1. [OpenSphere Face Recognition](https://github.com/ydwen/opensphere) </br>
2. [YOLOv5 DeepSort Face](https://github.com/yjwong1999/Yolov5_DeepSort_Face) </br>

This repo is mainly based on ref work [1]. </br>
We intend to fork the repo from [1], however we previously have to combine the full source code with ref work [2]. </br>
Hence, we couldn't fork ref work [1] for acknowledgement purpose. </br>
Anyway, thanks a lot to [1] for the open-source + transparent + reproducible code! </br>
The codes allow us to train our own face recognition model. </br>


## Cite this repository
```
@INPROCEEDINGS{10174362,
  author={Wong, Yi Jie and Huang Lee, Kian and Tham, Mau-Luen and Kwan, Ban-Hoe},
  booktitle={2023 IEEE World AI IoT Congress (AIIoT)}, 
  title={Multi-Camera Face Detection and Recognition in Unconstrained Environment}, 
  year={2023},
  volume={},
  number={},
  pages={0548-0553},
  doi={10.1109/AIIoT58121.2023.10174362}}
```
