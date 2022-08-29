"""This backend now only support lidar. camera is no longer supported.
"""

import base64
import datetime
import io as sysio
import json
import pickle
import time
from pathlib import Path

import fire
import torch
import numpy as np
import skimage
from flask import Flask, jsonify, request
from flask_cors import CORS
from google.protobuf import text_format
from skimage import io

from second.data import kitti_common as kitti
from second.data.all_dataset import get_dataset_class
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.pytorch.train import build_network, example_convert_to_torch

app = Flask("second")
CORS(app)


class SecondBackend:
    def __init__(self):
        self.root_path = None
        self.image_idxes = None
        self.dt_annos = None
        self.dataset = None
        self.net = None
        self.device = None


BACKEND = SecondBackend()


def error_response(msg):
    response = {}
    response["status"] = "error"
    response["message"] = "[ERROR]" + msg
    print("[ERROR]" + msg)
    return response


@app.route('/api/readinfo', methods=['POST'])
def readinfo():
    global BACKEND
    instance = request.json
    root_path = Path(instance["root_path"])
    response = {"status": "normal"}
    BACKEND.root_path = root_path
    info_path = Path(instance["info_path"])
    dataset_class_name = instance["dataset_class_name"]
    BACKEND.dataset = get_dataset_class(dataset_class_name)(root_path=root_path, info_path=info_path)
    BACKEND.image_idxes = list(range(len(BACKEND.dataset)))
    response["image_indexes"] = BACKEND.image_idxes
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


@app.route('/api/read_detection', methods=['POST'])
def read_detection():
    global BACKEND
    instance = request.json
    det_path = Path(instance["det_path"])
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if Path(det_path).is_file():
        with open(det_path, "rb") as f:
            dt_annos = pickle.load(f)
    else:
        dt_annos = kitti.get_label_annos(det_path)
    BACKEND.dt_annos = dt_annos
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


@app.route('/api/get_pointcloud', methods=['POST'])
def get_pointcloud():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    image_idx = instance["image_idx"]
    enable_int16 = instance["enable_int16"]

    idx = BACKEND.image_idxes.index(image_idx)
    sensor_data = BACKEND.dataset.get_sensor_data(idx)

    # img_shape = image_info["image_shape"] # hw
    if 'annotations' in sensor_data["lidar"]:
        annos = sensor_data["lidar"]['annotations']
        gt_boxes = annos["boxes"].copy()
        response["locs"] = gt_boxes[:, :3].tolist()
        response["dims"] = gt_boxes[:, 3:6].tolist()
        rots = np.concatenate([np.zeros([gt_boxes.shape[0], 2], dtype=np.float32), -gt_boxes[:, 6:7]], axis=1)
        response["rots"] = rots.tolist()
        response["labels"] = annos["names"].tolist()
        response["bbox"] = sensor_data["cam"]["annotations"]["boxes"].tolist()
    # response["num_features"] = sensor_data["lidar"]["points"].shape[1]
    response["num_features"] = 3
    points = sensor_data["lidar"]["points"][:, :3]
    if enable_int16:
        int16_factor = instance["int16_factor"]
        points *= int16_factor
        points = points.astype(np.int16)
    pc_str = base64.b64encode(points.tobytes())
    response["pointcloud"] = pc_str.decode("utf-8")

    # if "score" in annos:
    #     response["score"] = score.tolist()
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    print("send response with size {}!".format(len(pc_str)))
    return response


@app.route('/api/get_image', methods=['POST'])
def get_image():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    image_idx = instance["image_idx"]
    idx = BACKEND.image_idxes.index(image_idx)
    query = {
        "lidar": {
            "idx": idx
        },
        "cam": {}
    }
    sensor_data = BACKEND.dataset.get_sensor_data(query)
    if "cam" in sensor_data and "data" in sensor_data["cam"] and sensor_data["cam"]["data"] is not None:
        image_str = sensor_data["cam"]["data"]
        response["image_b64"] = base64.b64encode(image_str).decode("utf-8")
        response["image_b64"] = 'data:image/{};base64,'.format(sensor_data["cam"]["datatype"]) + response["image_b64"]
        print("send an image with size {}!".format(len(response["image_b64"])))
    else:
        response["image_b64"] = ""
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


@app.route('/api/build_network', methods=['POST'])
def build_network_():
    global BACKEND
    instance = request.json
    cfg_path = Path(instance["config_path"])
    ckpt_path = Path(instance["checkpoint_path"])
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if not cfg_path.exists():
        return error_response("config file not exist.")
    if not ckpt_path.exists():
        return error_response("ckpt file not exist.")
    config = pipeline_pb2.TrainEvalPipelineConfig()

    with open(cfg_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_network(config.model.second).to(device).float().eval()
    net.load_state_dict(torch.load(ckpt_path))
    eval_input_cfg = config.eval_input_reader
    BACKEND.dataset = input_reader_builder.build(
        eval_input_cfg,
        config.model.second,
        training=False,
        voxel_generator=net.voxel_generator,
        target_assigner=net.target_assigner).dataset
    BACKEND.net = net
    BACKEND.config = config
    BACKEND.device = device
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    print("build_network successful!")
    return response


@app.route('/api/inference_by_idx', methods=['POST'])
def inference_by_idx():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    image_idx = instance["image_idx"]
    # remove_outside = instance["remove_outside"]
    idx = BACKEND.image_idxes.index(image_idx)
    example = BACKEND.dataset[idx]
    # don't forget to pad batch idx in coordinates
    example["coordinates"] = np.pad(
        example["coordinates"], ((0, 0), (1, 0)),
        mode='constant',
        constant_values=0)
    # don't forget to add newaxis for anchors
    example["anchors"] = example["anchors"][np.newaxis, ...]
    example_torch = example_convert_to_torch(example, device=BACKEND.device)
    pred = BACKEND.net(example_torch)[0]
    pred["metadata"] = {"image_idx": image_idx}

    box3d = pred["box3d_lidar"].detach().cpu().numpy()

    scores = pred["scores"].detach().cpu().numpy()
    labels = pred["label_preds"].detach().cpu().numpy()
    dt_bbox = BACKEND.dataset.convert_detection_to_kitti_annos([pred])[0]['bbox']
    idx = np.where(scores > 0.1)[0]
    box3d = box3d[idx, :]
    labels = np.take(labels, idx)
    scores = np.take(scores, idx)
    dt_bbox = dt_bbox[idx, :]

    locs = box3d[:, :3]
    dims = box3d[:, 3:6]
    rots = np.concatenate([np.zeros([locs.shape[0], 2], dtype=np.float32), -box3d[:, 6:7]], axis=1)
    response["dt_locs"] = locs.tolist()
    response["dt_dims"] = dims.tolist()
    response["dt_rots"] = rots.tolist()
    response["dt_labels"] = labels.tolist()
    response["dt_scores"] = scores.tolist()
    response["dt_bbox"] = dt_bbox.tolist()

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


def main(port=16666):
    app.run(host='127.0.0.1', threaded=True, port=port)


if __name__ == '__main__':
    fire.Fire()

    # [{'name': array(['Pedestrian', 'Cyclist', 'Cyclist', 'Cyclist', 'Cyclist',
    #                  'Cyclist', 'Pedestrian', 'Cyclist', 'Cyclist', 'Pedestrian',
    #                  'Cyclist', 'Cyclist', 'Pedestrian', 'Cyclist'], dtype='<U10'),
    #   'truncated': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    #   'occluded': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    #   'alpha': array([1.4115822, 2.009875, 4.62709215, 5.08006632, 2.17993653,
    #                   4.87488687, 2.17907313, 1.21800914, 4.03366184, 5.75548485,
    #                   3.95674694, 1.79757723, 5.21030703, -0.16130701]),
    #   'bbox': array([[329.49765664, 177.82383073, 357.58096446, 235.20920532],
    #                  [105.68452043, 208.88406019, 152.79075187, 274.04624912],
    #                  [654.44291116, 174.59671965, 662.58644842, 203.46315122],
    #                  [226.76254024, 200.36972776, 253.71354579, 247.15950066],
    #                  [41.96552263, 224.28920133, 106.0147678, 295.93893089],
    #                  [400.82948741, 185.07549668, 411.53549532, 212.30455337],
    #                  [406.22546381, 177.67021405, 423.22364705, 208.47331469],
    #                  [792.45420036, 151.05549374, 815.96890596, 191.31608981],
    #                  [294.5794994, 183.50952041, 339.04652875, 234.09794318],
    #                  [457.9305439, 180.48084068, 472.0745735, 208.21087805],
    #                  [421.82840995, 184.07319901, 455.0320453, 221.36111411],
    #                  [215.15697514, 194.88092131, 242.09126995, 240.62450441],
    #                  [418.46882649, 173.20958067, 432.95740433, 201.89926851],
    #                  [787.21781059, 150.42805345, 823.22962714, 190.99966708]]),
    #   'dimensions': array([[0.96082699, 1.77317202, 0.69853246],
    #                        [1.75476241, 1.73074102, 0.43018568],
    #                        [1.55860674, 1.62645268, 0.33443749],
    #                        [1.65284836, 1.69025958, 0.35132051],
    #                        [1.69639385, 1.6505636, 0.43074727],
    #                        [1.6018852, 1.66586769, 0.39683911],
    #                        [0.74729943, 1.80629408, 0.66882443],
    #                        [1.51697874, 1.76430941, 0.52405173],
    #                        [1.69170237, 1.75004077, 0.53804898],
    #                        [0.53659934, 1.67939448, 0.77553421],
    #                        [1.7003082, 1.83059859, 0.65114772],
    #                        [1.57108724, 1.61491406, 0.54967368],
    #                        [0.7173636, 1.79478455, 0.61895275],
    #                        [1.49159431, 1.7982434, 0.52062839]]),
    #   'location': array([[-8.47978604, 1.93556427, 22.9869466],
    #                      [-13.9884707, 2.82714074, 21.05751905],
    #                      [2.81345237, 1.72865165, 41.53112214],
    #                      [-14.26149377, 2.78599182, 27.89312629],
    #                      [-13.84137253, 3.04428246, 18.69857075],
    #                      [-12.87401902, 2.45372449, 45.69581395],
    #                      [-11.58708682, 2.09589321, 42.92121772],
    #                      [8.72843831, 0.80905009, 32.39877192],
    #                      [-10.50600665, 2.1418056, 25.88086638],
    #                      [-8.9011815, 2.15391308, 44.42368311],
    #                      [-8.68697281, 2.41132469, 36.60813807],
    #                      [-14.29255658, 2.46757324, 27.10116957],
    #                      [-11.60913504, 1.81745223, 45.5728407],
    #                      [8.75461717, 0.80426493, 32.31451399]]),
    #   'rotation_y': array([1.0614326, 1.42899942, 4.69375086, 4.61085796, 1.54922688,
    #                        4.60135269, 1.91654813, 1.4786849, 3.65116787, 5.55843163,
    #                        3.72484303, 1.31593919, 4.96191406, 0.10076371]),
    #   'score': array([0.46737525, 0.3266363, 0.30806524, 0.17042749, 0.08180686,
    #                   0.08074831, 0.07910713, 0.06854416, 0.0676259, 0.0673247,
    #                   0.0557738, 0.05510438, 0.05376802, 0.05077583], dtype=float32), 'metadata': {'image_idx': 3}}]
