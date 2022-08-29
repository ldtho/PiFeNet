# PiFeNet implementation for KITTI/JRDB object detection

This repository is based on [SECOND](https://github.com/traveller59/second.pytorch) project.

ONLY support python 3.6+, pytorch 1.0.0+. Tested in Ubuntu 16.04/18.04/20.04/22.04, Windows 10.

If you want to train nuScenes dataset, see [this](NUSCENES-GUIDE.md).

## News

02/2022: JRDB dataset supported

_WARNING_: you should rerun info generation after every code update.

### Performance in KITTI test 

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
```
JRDB 2019:
         AP@0.3      AP@0.5     AP@0.7
PiFeNet  74.284      42.617      4.886

JRDB 2022:
         AP@0.3      AP@0.5     AP@0.7
PiFeNet  70.724      39.838      4.59
```

## Install

### 1. Clone code

```bash
git clone https://github.com/traveller59/second.pytorch.git
cd ./second.pytorch/second
```

### 2. Install dependence python packages

It is recommend to use Anaconda package manager.

```bash
conda install scikit-image scipy numba pillow matplotlib
```

```bash
pip install fire tensorboardX protobuf opencv-python
```

If you don't have Anaconda:

```bash
pip install numba scikit-image scipy pillow
```

Follow instructions in [spconv](https://github.com/traveller59/spconv) to install spconv. 

If you want to train with fp16 mixed precision (train faster in RTX series, Titan V/RTX and Tesla V100, but I only have 1080Ti), you need to install [apex](https://github.com/NVIDIA/apex).

If you want to use NuScenes dataset, you need to install [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit).

### 3. Setup cuda for numba (will be removed in 1.6.0 release)

you need to add following environment variable for numba.cuda, you can add them to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

### 4. add second.pytorch/ to PYTHONPATH

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
python create_data.py kitti_data_prep --data_path=KITTI_DATASET_ROOT
```

* [NuScenes](https://www.nuscenes.org) Dataset preparation

Download NuScenes dataset:
```plain
└── NUSCENES_TRAINVAL_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       └── v1.0-trainval <-- metadata and annotations
└── NUSCENES_TEST_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       └── v1.0-test     <-- metadata
```

Then run
```bash
python create_data.py nuscenes_data_prep --data_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --max_sweeps=10
python create_data.py nuscenes_data_prep --data_path=NUSCENES_TEST_DATASET_ROOT --version="v1.0-test" --max_sweeps=10
--dataset_name="NuscenesDataset"
```
This will create gt database **without velocity**. to add velocity, use dataset name ```NuscenesDatasetVelo```.

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

### train

I recommend to use script.py to train and eval. see script.py for more details.

#### train with single GPU

```bash
python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=/path/to/model_dir
```

#### train with multiple GPU (need test, I only have one GPU)

Assume you have 4 GPUs and want to train with 3 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,3 python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=/path/to/model_dir --multi_gpu=True
```

Note: The batch_size and num_workers in config file is per-GPU, if you use multi-gpu, they will be multiplied by number of GPUs. Don't modify them manually.

You need to modify total step in config file. For example, 50 epochs = 15500 steps for car.lite.config and single GPU, if you use 4 GPUs, you need to divide ```steps``` and ```steps_per_eval``` by 4.

#### train with fp16 (mixed precision)

Modify config file, set enable_mixed_precision to true.

* Make sure "/path/to/model_dir" doesn't exist if you want to train new model. A new directory will be created if the model_dir doesn't exist, otherwise will read checkpoints in it.

* training process use batchsize=6 as default for 1080Ti, you need to reduce batchsize if your GPU has less memory.

* Currently only support single GPU training, but train a model only needs 20 hours (165 epoch) in a single 1080Ti and only needs 50 epoch to reach 78.3 AP with super converge in car moderate 3D in Kitti validation dateset.

### evaluate

```bash
python ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=/path/to/model_dir --measure_time=True --batch_size=1
```

* detection result will saved as a result.pkl file in model_dir/eval_results/step_xxx or save as official KITTI label format if you use --pickle_result=False.

### pretrained model

You can download pretrained models in [google drive](https://drive.google.com/open?id=1YOpgRkBgmSAJwMknoXmitEArNitZz63C). The ```car_fhd``` model is corresponding to car.fhd.config.

Note that this pretrained model is trained before a bug of sparse convolution fixed, so the eval result may slightly worse. 

## Docker (Deprecated. I can't push docker due to network problem.)

You can use a prebuilt docker for testing:
```
docker pull scrin/second-pytorch 
```
Then run:
```
nvidia-docker run -it --rm -v /media/yy/960evo/datasets/:/root/data -v $HOME/pretrained_models:/root/model --ipc=host second-pytorch:latest
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=/root/model/car
```

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

## Concepts


* Kitti lidar box

A kitti lidar box is consist of 7 elements: [x, y, z, w, l, h, rz], see figure.

![Kitti Box Image](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/kittibox.png)

All training and inference code use kitti box format. So we need to convert other format to KITTI format before training.

* Kitti camera box

A kitti camera box is consist of 7 elements: [x, y, z, l, h, w, ry].
