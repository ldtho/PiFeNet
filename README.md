## [PiFeNet: Accurate and Real-time 3D Pedestrian Detection Using an Efficient Attentive Pillar Network](https://arxiv.org/abs/2112.15458)

### Official implementation for KITTI/JRDB object detection
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pifenet-pillar-feature-network-for-real-time/birds-eye-view-object-detection-on-kitti-13)](https://paperswithcode.com/sota/birds-eye-view-object-detection-on-kitti-13?p=pifenet-pillar-feature-network-for-real-time)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pifenet-pillar-feature-network-for-real-time/3d-object-detection-on-kitti-pedestrian-2)](https://paperswithcode.com/sota/3d-object-detection-on-kitti-pedestrian-2?p=pifenet-pillar-feature-network-for-real-time)

![GuidePic](https://github.com/ldtho/PiFeNet/blob/main/images/JRDB22_viz.png?raw=true)

## Citation
If you find this repo useful, please consider citing us, appreciate it!
```bash
@ARTICLE{10003992,
  author={Le, Duy Tho and Shi, Hengcan and Rezatofighi, Hamid and Cai, Jianfei},
  journal={IEEE Robotics and Automation Letters}, 
  title={Accurate and Real-time 3D Pedestrian Detection Using an Efficient Attentive Pillar Network}, 
  year={2022},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/LRA.2022.3233234}}
```

## Introduction
This repository is based on [SECOND](https://github.com/traveller59/second.pytorch) project.

ONLY support python 3.6+, pytorch 1.0.0+. Tested in Ubuntu 16.04/18.04/20.04/22.04, Windows 10.

This repo is not optimal on nuScenes dataset, consider using [Det3D](https://github.com/poodarchu/Det3D)

If you want to train nuScenes dataset, see [this](NUSCENES-GUIDE.md).

## News
12/2022: The paper has been accepted for publication at IEEE Robotics and Automation Letters (RA-L)

02/2022: JRDB dataset supported

_WARNING_: you should rerun info generation after every code update.

### Performance in KITTI test set

```PiFeNet/KITTI/xyres_16_submission.config``` + 150 epochs:

```
Benchmark	                Easy   Moderate	 Hard
Pedestrian (Detection)	        72.74%	62.35%	59.29%
Pedestrian (Orientation)	55.11%	46.59%	44.14%
Pedestrian (3D Detection)	56.39%	46.71%	42.71%
Pedestrian (Bird's Eye View)	63.25%	53.92%	50.53%
```

### Performance in JRDB test set

```PiFeNet/jrdb22/xyres_16_largea_JRDB2022.config``` + 40 epochs:

**_JRDB 2019_**:
```
         AP@0.3      AP@0.5     AP@0.7
PiFeNet  74.284      42.617      4.886
```
**_JRDB 2022_**:
```
         AP@0.3      AP@0.5     AP@0.7
PiFeNet  70.724      39.838      4.59
```

## Install

### 1. Clone code

```bash
git clone https://github.com/ldtho/PiFeNet.git --recursive
cd ./PiFeNet/second
```

### 2. Install dependence python packages

It is recommend to use Anaconda package manager.

Create environment:
```bash
conda create --name pifenet python=3.8.6 pytorch=1.7.1 cudatoolkit=11.0.221 cudatoolkit-dev cmake=3.18.2 cuda-nvcc cudnn boost -c pytorch -c conda-forge -c nvidia
conda activate pifenet
```
Install dependencies

```bash
conda install addict einops fire jupyterlab jupyter-packaging tensorboard libboost matplotlib numba numpy open3d addict scikit-image psutil boost einops scikit-learn fire jupyterlab tensorboardx libboost matplotlib numba numpy open3d pandas pillow protobuf scipy seaborn tqdm yaml -c pytorch -c conda-forge -c nvidia -c numba -c open3d-admin
pip install opencv-python 
```


Follow instructions in [spconv](https://github.com/traveller59/spconv) to install spconv or you can try:
```bash
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv
git checkout fad3000249d27ca918f2655ff73c41f39b0f3127 && git submodule update --recursive    
python setup.py bdist_wheel
cd dist && pip install .
```

Sample conda environment [PiFeNet.yml](https://github.com/ldtho/PiFeNet/blob/main/PiFeNet.yml) is available for your reference. 

If you want to train with fp16 mixed precision (train faster in RTX series, Titan V/RTX and Tesla V100, but I only have 1080Ti), you need to install [apex](https://github.com/NVIDIA/apex).

[//]: # (If you want to use NuScenes dataset, you need to install [nuscenes-devkit]&#40;https://github.com/nutonomy/nuscenes-devkit&#41;.)

[//]: # (### 3. Setup cuda for numba &#40;will be removed in 1.6.0 release&#41;)

[//]: # ()
[//]: # (you need to add following environment variable for numba.cuda, you can add them to ~/.bashrc:)

[//]: # ()
[//]: # (```bash)

[//]: # (export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so)

[//]: # (export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so)

[//]: # (export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice)

[//]: # (```)

### 4. add second.pytorch/ to PYTHONPATH
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/PiFeNet
```
## Prepare dataset

* KITTI Dataset preparation

Download KITTI dataset and create some directories first:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory
```

Then run
```bash
python create_data.py kitti_data_prep --root_path=KITTI_DATASET_ROOT
```

* [JRDB](https://www.nuscenes.org) Dataset preparation.

Download NuScenes dataset:
```plain
└── train_dataset_with_activity
       ├── calibration       
       ├── images          <-- frames without annotation
       ├── detections      <-- sample test detection - unused
       └── pointclouds     <-- point cloud files
└── test_dataset_without_labels
       ├── calibration       
       ├── images          <-- frames without annotation
       ├── detections      <-- sample train detection - unused
       ├── labels          <-- train set annotations
       └── pointclouds     <-- point cloud files
```

Then run
```bash
python create_data.py jrdb22_data_prep --root_path=JRDB_DATASET_ROOT
```

[//]: # (This will create gt database **without velocity**. to add velocity, use dataset name ```NuscenesDatasetVelo```.)

* Modify config file

There is some path need to be configured in config file:

```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/dataset_dbinfos_train.pkl"
    ...
  }
  dataset: {
    dataset_class_name: "DATASET_NAME"
    kitti_info_path: "/path/to/dataset_infos_train.pkl"
    kitti_root_path: "DATASET_ROOT"
  }
}
...
eval_input_reader: {
  ...
  dataset: {
    dataset_class_name: "DATASET_NAME"
    kitti_info_path: "/path/to/dataset_infos_val.pkl"
    kitti_root_path: "DATASET_ROOT"
  }
}
```

## Usage


### Train with single GPU

_**KITTI**_:
```bash
python ./pytorch/train.py train --config_path=./configs/PiFeNet/KITTI/xyres_16_submission.config --model_dir=/your/save/dir
```
_**JRDB**_:
```bash
python ./pytorch/train.py train --config_path=./configs/PiFeNet/jrdb/xyres_16_largea_JRDB2022.config --model_dir=/your/save/dir
```
Note: add ```--resume=True``` if you want to continue training where you left off.

#### train with multiple GPU (need test, I only have one GPU)

Assume you have 4 GPUs and want to train with 3 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,3 python ./pytorch/train.py train --config_path=./configs/PiFeNet/KITTI/xyres_16_submission.config --model_dir=/your/save/dir --multi_gpu=True
```

Note: The batch_size and num_workers in config file is per-GPU, if you use multi-gpu, they will be multiplied by number of GPUs. Don't modify them manually.

You need to modify total step in config file. For example, 50 epochs = 15500 steps for car.lite.config and single GPU, if you use 4 GPUs, you need to divide ```steps``` and ```steps_per_eval``` by 4.

#### train with fp16 (mixed precision)

Modify config file, set enable_mixed_precision to true.

* Make sure "/path/to/model_dir" doesn't exist if you want to train new model. A new directory will be created if the model_dir doesn't exist, otherwise will read checkpoints in it.

* training process use batchsize=2 as default, you can increase or decrease depend on your GPU capability

### evaluate

```bash
python ./pytorch/train.py evaluate --config_path=./configs/PiFeNet/KITTI/xyres_16_submission.config --model_dir=/your/save/dir --measure_time=True --batch_size=1
```

* detection result will saved as a result.pkl file in model_dir/eval_results/step_xxx or save as official KITTI label format if you use --pickle_result=False.

### pretrained model

You can download pretrained models in [google drive](https://drive.google.com/drive/folders/1IGVV3oswFyBUxTZee10-Xzrx2rYPNyvJ?usp=sharing). The models' configs are the same as above

## Try Kitti Viewer Web

### Major step

1. run ```python ./kittiviewer/backend/main.py main --port=xxxx``` in your server/local.

2. run ```cd ./kittiviewer/frontend && python -m http.server``` to launch a local web server.

3. open your browser and enter your frontend url (e.g. http://127.0.0.1:8000, default]).

4. input backend url (e.g. http://127.0.0.1:16666)

5. input root path, info path and det path (optional)

6. click load, loadDet (optional), input image index in center bottom of screen and press Enter.

### Inference step

Firstly the load button must be clicked and load successfully.

1. input checkpointPath and configPath.

2. click buildNet.

3. click inference.

![GuidePic](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/viewerweb.png)



## Try Kitti Viewer (Deprecated)

You should use kitti viewer based on pyqt and pyqtgraph to check data before training.

run ```python ./kittiviewer/viewer.py```, check following picture to use kitti viewer:
![GuidePic](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/simpleguide.png)


## JRDB visualisation

First run the model, make predictions and convert is to KITTI-similar format
```bash
python ./pytorch/train.py evaluate --config_path=/path/to/config --model_dir=/pretrained_model/location --ckpt_path=/path/to/pretrained_model.tckpt
```
Follow the instructions in the [JRDB visualisation toolkit](https://github.com/JRDB-dataset/jrdb_toolkit/tree/main/visualisation)

### Result:
I also share the JRDB test set [detections in KITTI format](https://drive.google.com/drive/folders/1IGVV3oswFyBUxTZee10-Xzrx2rYPNyvJ?usp=sharing) so that you can test the visualisation script.

**_2D:_**

![GuidePic](https://github.com/ldtho/PiFeNet/blob/main/images/jrdb_2D_viz.png?raw=true)

**_3D:_**

![GuidePic](https://github.com/ldtho/PiFeNet/blob/main/images/jrdb_3D_viz.png?raw=true)




## Acknowledgement

* [second.pytorch](https://github.com/traveller59/second.pytorch)