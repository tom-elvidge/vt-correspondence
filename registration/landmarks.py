from bs4 import BeautifulSoup
import numpy as np


def meshlab_picked_points_to_dict(filename):
    with open(filename, "r") as f:
        data = f.read()
    # Put picked points into dictionary.
    picked_points = {}
    for xml_point in BeautifulSoup(data, "xml").find_all("point"):
        picked_points[xml_point.get("name")] = [
            float(xml_point.get("x")),
            float(xml_point.get("y")),
            float(xml_point.get("z"))
        ]
    return picked_points


def correspond_picked_points(pps):
    # pps: List of dictionaries of picked points for each surface.
    # Initialise landmark matrix.
    landmarks = []
    for i in range(len(pps)):
        landmarks.append([])
    # Add all common landmarks to landmarks.
    for landmark_name in pps[0].keys():
        common_landmark = True
        # Ensure this landmark is common across all surfaces.
        for i in range(len(pps)):
            if landmark_name not in pps[i]:
                common_landmark = False
                break
        # Add landmark point to landmarks.
        if common_landmark:
            for i in range(len(pps)):
                landmarks[i].append(pps[i][landmark_name])
    return np.asarray(landmarks)


def test():
    s3_c1_points = meshlab_picked_points_to_dict(
        "data/surface_models/s3_contrast1_final_picked_points.pp")
    s3_n_points = meshlab_picked_points_to_dict(
        "data/surface_models/s3_neutralVT_final_picked_points.pp")
    landmarks = correspond_picked_points([s3_c1_points, s3_n_points])
    print(landmarks)
