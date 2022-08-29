import os

import pandas as pd
import numpy as np

TRAIN = [
    'bytes-cafe-2019-02-07_0',
    'clark-center-2019-02-28_0',
    'clark-center-intersection-2019-02-28_0',
    'cubberly-auditorium-2019-04-22_0',
    'forbes-cafe-2019-01-22_0',
    'gates-159-group-meeting-2019-04-03_0',
    'gates-basement-elevators-2019-01-17_1',
    'gates-to-clark-2019-02-28_1',
    'hewlett-packard-intersection-2019-01-24_0',
    'huang-basement-2019-01-25_0',
    'huang-lane-2019-02-12_0',
    'jordan-hall-2019-04-22_0',
    'memorial-court-2019-03-16_0',
    'packard-poster-session-2019-03-20_0',
    'packard-poster-session-2019-03-20_1',
    'packard-poster-session-2019-03-20_2',
    'stlc-111-2019-04-19_0',
    'svl-meeting-gates-2-2019-04-08_0',
    'svl-meeting-gates-2-2019-04-08_1',
    'tressider-2019-03-16_0',
]
TEST = [
    'cubberly-auditorium-2019-04-22_1',
    'discovery-walk-2019-02-28_0',
    'discovery-walk-2019-02-28_1',
    'food-trucks-2019-02-12_0',
    'gates-ai-lab-2019-04-17_0',
    'gates-basement-elevators-2019-01-17_0',
    'gates-foyer-2019-01-17_0',
    'gates-to-clark-2019-02-28_0',
    'hewlett-class-2019-01-23_0',
    'hewlett-class-2019-01-23_1',
    'huang-2-2019-01-25_1',
    'huang-intersection-2019-01-22_0',
    'indoor-coupa-cafe-2019-02-06_0',
    'lomita-serra-intersection-2019-01-30_0',
    'meyer-green-2019-03-16_1',
    'nvidia-aud-2019-01-25_0',
    'nvidia-aud-2019-04-18_1',
    'nvidia-aud-2019-04-18_2',
    'outdoor-coupa-cafe-2019-02-06_0',
    'quarry-road-2019-02-28_0',
    'serra-street-2019-01-30_0',
    'stlc-111-2019-04-19_1',
    'stlc-111-2019-04-19_2',
    'tressider-2019-03-16_2',
    'tressider-2019-04-26_0',
    'tressider-2019-04-26_1',
    'tressider-2019-04-26_3'
]
VAL = [
    'clark-center-2019-02-28_1',
    'gates-ai-lab-2019-02-08_0',
    'huang-2-2019-01-25_0',
    'meyer-green-2019-03-16_0',
    'nvidia-aud-2019-04-18_0',
    'tressider-2019-03-16_1',
    'tressider-2019-04-26_2',

    # 'clark-center-2019-02-28_1',
    # 'jordan-hall-2019-04-22_0',
    # 'memorial-court-2019-03-16_0',
    # 'meyer-green-2019-03-16_0',
    # 'packard-poster-session-2019-03-20_2',
    # 'svl-meeting-gates-2-2019-04-08_0',
    # 'tressider-2019-03-16_1'
]

if __name__ == '__main__':
    base_dir = "/home/tho/datasets/JRDB2022_converted/"
    imgset_dir = "/home/tho/GoogleDrive/code/secondpytorch/second/data/ImageSets_jrdb22"
    for training in [True,False]:
        filelist_path = os.path.join(base_dir, 'training' if training else 'testing', "filelist.txt")
        data = pd.read_csv(filelist_path, sep=' ', header=None)
        if training:
            train_file = open(os.path.join(imgset_dir, 'train.txt'), "w")
            val_file = open(os.path.join(imgset_dir, 'val.txt'), "w")
            trainval_file = open(os.path.join(imgset_dir, 'trainval.txt'), "w")
        else:
            test_file = open(os.path.join(imgset_dir, 'test.txt'), "w")
        for idx, (seq, timestamp) in data.iterrows():
            if training:
                if seq in VAL:
                    val_file.write(f"{idx:06d}\n")
                else:
                    train_file.write(f"{idx:06d}\n")
                trainval_file.write(f"{idx:06d}\n")

            else:
                print(seq)
                test_file.write(f"{idx:06d}\n")
        if training:
            train_file.close()
            val_file.close()
            trainval_file.close()
        else:
            test_file.close()
