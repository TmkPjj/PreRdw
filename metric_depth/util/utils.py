import os
import re
import numpy as np
import logging
import cv2
logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def parse_dataset_txt(dataset_txt):
    with open(dataset_txt) as data_txt:
        gt_files = []
        image_files = []
        focals = []
        baselines = []
        calib_files = []

        for line in data_txt:
            values = line.split(" ")

            if len(values) == 2:
                basenames.append(values[0].strip())
                gt_files.append(values[1].strip())

            elif len(values) == 3:
                image_files.append(values[0].strip())
                gt_files.append(values[1].strip())
                calib_files.append(values[2].strip())

            else:
                print("Wrong format dataset txt file")
                exit(-1)
    
    dataset_dict = {}
    if gt_files: dataset_dict["gt_paths"] = gt_files
    if image_files: dataset_dict["image_paths"] = image_files
    if calib_files: dataset_dict["calib_paths"] = calib_files
    return dataset_dict



def read_calib_xml(calib_path, factor_baseline=0.001):
    cv_file = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    calib = cv_file.getNode("proj_matL").mat()[:3,:3]
    fx = calib[0,0]
    baseline = float(cv_file.getNode("baselineLR").real())*factor_baseline
    return fx, baseline