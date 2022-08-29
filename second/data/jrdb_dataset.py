from collections import OrderedDict
from pathlib import Path
import pickle
import numpy as np

from second.core import box_np_ops
from second.core import preprocess as prep
from second.data import kitti_common as kitti
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.data.dataset import Dataset, register_dataset
from second.utils.progress_bar import progress_bar_iter as prog_bar
import os
import yaml
import torch
import pandas as pd


@register_dataset
class JRDBDataset(Dataset):
    NumPointFeatures = 4

    def __init__(self,
                 root_path,
                 info_path,
                 class_names=None,
                 prep_func=None,
                 num_point_features=None):
        assert info_path is not None
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        self._root_path = Path(root_path)
        self._jrdb_infos = infos

        print("remain number of infos:", len(self._jrdb_infos))
        self._class_names = class_names
        self._prep_func = prep_func

        global_config = os.path.join(root_path, 'training', 'calib', 'defaults.yaml')
        camera_config = os.path.join(root_path, 'training', 'calib', 'cameras.yaml')
        with open(global_config) as f:
            self.global_config_dict = yaml.safe_load(f)

        with open(camera_config) as f:
            self.camera_config_dict = yaml.safe_load(f)

        self.median_focal_length_y = self.calculate_median_param_value(param='f_y')
        self.median_optical_center_y = self.calculate_median_param_value(param='t_y')

    def __len__(self):
        return len(self._jrdb_infos)

    def calculate_median_param_value(self, param):
        if param == 'f_y':
            idx = 4
        elif param == 'f_x':
            idx = 0
        elif param == 't_y':
            idx = 5
        elif param == 't_x':
            idx = 2
        elif param == 's':
            idx = 1
        else:
            raise 'Wrong parameter!'

        omni_camera = ['sensor_0', 'sensor_2', 'sensor_4', 'sensor_6', 'sensor_8']
        parameter_list = []
        for sensor, camera_params in self.camera_config_dict['cameras'].items():
            if sensor not in omni_camera:
                continue
            K_matrix = camera_params['K'].split(' ')
            parameter_list.append(float(K_matrix[idx]))
        return np.median(parameter_list)

    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict=input_dict)
        example["metadata"] = {}
        if "image_idx" in input_dict["metadata"]:
            example["metadata"] = input_dict["metadata"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example

    def get_sensor_data(self, query):
        read_image = False
        idx = query
        if isinstance(query, dict):
            read_image = "cam" in query
            assert "lidar" in query
            idx = query["lidar"]["idx"]
        info = self._jrdb_infos[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "image_idx": info["image"]["image_idx"],
                "image_shape": info["image"]["image_shape"],
            },
            "calib": None,
            "cam": {}
        }

        pc_info = info["point_cloud"]
        velo_path = Path(pc_info['velodyne_path'])
        if not velo_path.is_absolute():
            velo_path = Path(self._root_path) / pc_info['velodyne_path']
        velo_reduced_path = velo_path.parent.parent / (
                velo_path.parent.stem + '_reduced') / velo_path.name
        if velo_reduced_path.exists():
            velo_path = velo_reduced_path
        # points = np.asarray(o3d.io.read_point_cloud(str(velo_path)).points).reshape([-1, self.NumPointFeatures])
        points = np.fromfile(
            str(velo_path), dtype=np.float32,
            count=-1).reshape([-1, self.NumPointFeatures])
        res["lidar"]["points"] = points
        image_info = info["image"]
        image_path = image_info['image_path']
        if read_image:
            image_path = self._root_path / image_path
            with open(str(image_path), 'rb') as f:
                image_str = f.read()
            res["cam"] = {
                "type": "camera",
                "data": image_str,
                "datatype": image_path.suffix[1:],
            }

        if 'annos' in info:
            annos = info['annos']
            # we need other objects to avoid collision when sample
            annos = kitti.remove_dontcare(annos)
            locs = annos["location"]
            dims = annos["dimensions"]
            rots = annos["rotation_y"]
            gt_names = annos["name"]
            gt_boxes = np.concatenate([locs, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)
            gt_boxes = box_np_ops.box_camera_to_lidar_jrdb(gt_boxes)
            box_np_ops.change_box3d_center_(gt_boxes, [0.5, 0.5, 0],
                                            [0.5, 0.5, 0.5])

            gt_boxes = gt_boxes
            gt_names = gt_names
            res["lidar"]["annotations"] = {
                'boxes': gt_boxes,
                'names': gt_names,
                'num_points': annos['num_points_in_gt']
            }
            res["cam"]["annotations"] = {
                'boxes': annos["bbox"],
                'names': gt_names,
            }

        return res

    def convert_detection_to_kitti_annos(self, detection, submission=False):
        class_names = self._class_names
        det_image_idxes = [det["metadata"]["image_idx"] for det in detection]
        gt_image_idxes = [
            info["image"]["image_idx"] for info in self._jrdb_infos
        ]
        annos = []
        for i in range(len(detection)):
            det_idx = det_image_idxes[i]
            det = detection[i]
            info = self._jrdb_infos[i]
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            if final_box_preds.shape[0] != 0:
                final_box_preds[:, 2] -= final_box_preds[:, 5] / 2
                if submission:
                    box3d_camera = box_np_ops.box_lidar_to_jrdb_camera(
                        final_box_preds)  # x', y', z', h', w', l', theta'
                else:
                    box3d_camera = box_np_ops.box_lidar_to_kitti_camera(
                        final_box_preds)  # x', y', z', h', w', l', theta'

                locs = box3d_camera[:, :3]
                dims = box3d_camera[:, 3:6]
                angles = box3d_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
            anno = kitti.get_start_result_anno()
            num_example = 0
            box3d_lidar = final_box_preds
            for j in range(box3d_lidar.shape[0]):
                anno["bbox"].append(np.array([0, 0, 50, 50]))
                anno["alpha"].append(-10)
                anno["dimensions"].append(box3d_camera[j, 3:6])
                anno["location"].append(box3d_camera[j, :3])
                anno["rotation_y"].append(box3d_camera[j, 6])

                anno["name"].append(class_names[int(label_preds[j])])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(scores[j])

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
            num_example = annos[-1]["name"].shape[0]
            annos[-1]["metadata"] = det["metadata"]
        return annos

    def evaluation(self, detections, output_dir):
        """
        detection
        When you want to eval your own dataset, you MUST set correct
        the z axis and box z center.
        If you want to eval by my KITTI eval function, you must
        provide the correct format annotations.
        ground_truth_annotations format:
        {
            bbox: [N, 4], if you fill fake data, MUST HAVE >25 HEIGHT!!!!!!
            alpha: [N], you can use -10 to ignore it.
            occluded: [N], you can use zero.
            truncated: [N], you can use zero.
            name: [N]
            location: [N, 3] center of 3d box.
            dimensions: [N, 3] dim of 3d box.
            rotation_y: [N] angle.
        }
        all fields must be filled, but some fields can fill
        zero.
        """
        if "annos" not in self._jrdb_infos[0]:
            return None
        gt_annos = [info["annos"] for info in self._jrdb_infos]
        dt_annos = self.convert_detection_to_kitti_annos(detections)

        # visualize predictions
        # firstly convert standard detection to kitti-format dt annos
        z_axis = 1  # KITTI camera format use y as regular "z" axis.
        z_center = 1.0  # KITTI camera box's center is [0.5, 1, 0.5]
        # for regular raw lidar data, z_axis = 2, z_center = 0.5.

        result_official_dict = get_official_eval_result(
            gt_annos,
            dt_annos,
            self._class_names,
            z_axis=z_axis,
            z_center=z_center)
        result_coco = get_coco_eval_result(
            gt_annos,
            dt_annos,
            self._class_names,
            z_axis=z_axis,
            z_center=z_center)
        return {
            "results": {
                "official": result_official_dict["result"],
                "coco": result_coco["result"],
            },
            "detail": {
                "eval.kitti": {
                    "official": result_official_dict["detail"],
                    "coco": result_coco["detail"]
                }
            },
        }

    def project_ref_to_image_torch(self, pointcloud):

        theta = (torch.atan2(pointcloud[:, 0], pointcloud[:, 2]) + np.pi) % (2 * np.pi)
        horizontal_fraction = theta / (2 * np.pi)
        x = (horizontal_fraction * self.img_shape[2]) % self.img_shape[2]
        y = -self.median_focal_length_y * (
                pointcloud[:, 1] * torch.cos(theta) / pointcloud[:, 2]) + self.median_optical_center_y
        pts_2d = torch.stack([x, y], dim=1)

        return pts_2d


def kitti_anno_to_jrdb_label_file(annos, folder, config):
    root_dir = config.eval_input_reader.dataset.kitti_root_path
    type = "training" if "_full_" in root_dir else "testing"
    imagesets_path = Path(root_dir) / type / "filelist.txt"
    filelist = pd.read_csv(imagesets_path, sep=' ', header=None)

    for anno in annos:
        image_idx = anno["metadata"]["image_idx"]
        seq_name, seq_idx = filelist.iloc[int(image_idx)]
        label_path = Path(folder) / seq_name
        label_path.mkdir(parents=True, exist_ok=True)
        label_lines = []
        for j in range(anno["bbox"].shape[0]):
            label_dict = {
                'name': anno["name"][j],
                'truncated': None,
                'occluded': None,
                'num_points': None,
                'alpha': None,
                'bbox': None,
                # lhw (camera)->hwl (camera) (label file format)
                'dimensions': [anno["dimensions"][j][1], anno["dimensions"][j][2], anno["dimensions"][j][0]],
                'location': anno["location"][j],
                'rotation_y': limit_rot_for_kitti(anno["rotation_y"][j]),
                'score': anno["score"][j],
            }
            label_line = jrdb_result_line(label_dict)
            label_lines.append(label_line)
        label_file = label_path / f"{get_image_index_str(seq_idx)}.txt"
        label_str = '\n'.join(label_lines)
        with open(label_file, 'w+') as f:
            f.write(label_str)


def get_image_index_str(img_idx):
    return "{:06d}".format(img_idx)


def limit_rot_for_kitti(rotation):
    rotation = (rotation % (2 * np.pi) + 2 * np.pi) % (2 * np.pi)  # scale to 0 -> 2pi
    rotation = rotation if rotation < np.pi else rotation - 2 * np.pi  # LIDAR 0->2pi => LIDAR -pi->pi
    return rotation


def jrdb_result_line(result_dict, precision=4):
    prec_float = "{" + ":.{}f".format(precision) + "}"
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', 0),
        ('occluded', 0),
        ('num_points', -1),
        ('alpha', 0),
        ('bbox', [-1, -1, -1, -1]),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError("you must specify a value for {}".format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score', 'num_points']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError("unknown key. supported key:{}".format(
                res_dict.keys()))
    return ' '.join(res_line)


