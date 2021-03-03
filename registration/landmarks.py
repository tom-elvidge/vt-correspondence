from bs4 import BeautifulSoup
import numpy as np


def import_meshlab_pp_file(filename):
    """ 
    Loads a MeshLab picked points file into a dict.

    Parameters: 
        filename (string): Filename of the picked points file.

    Returns: 
        dict: Mapping of picked point names to points (list of length 3).
    """
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


def correspond_picked_points(pp_dicts):
    """ 
    Takes any number of picked point dicts, takes out the corresponding landmarks (so landmarks
    with the same name), and puts them into a different list for each dict where the corresponding
    landmarks have the same indicies.

    Parameters: 
        pp_dicts (list): A list of picked point dicts as defined in import_meshlab_pp_file.

    Returns: 
        list: A list containing a list of points for each passed dict. Corresponding landmarks have the
        same indicies.
    """
    # Initialise landmark matrix.
    landmarks = []
    for i in range(len(pp_dicts)):
        landmarks.append([])
    # Add all common landmarks to landmarks.
    for landmark_name in pp_dicts[0].keys():
        common_landmark = True
        # Ensure this landmark is common across all surfaces.
        for i in range(len(pp_dicts)):
            if landmark_name not in pp_dicts[i]:
                common_landmark = False
                break
        # Add landmark point to landmarks.
        if common_landmark:
            for i in range(len(pp_dicts)):
                landmarks[i].append(pp_dicts[i][landmark_name])
    return np.asarray(landmarks)
