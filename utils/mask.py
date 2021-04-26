import numpy as np
import cv2


def get_mask_instance(row):
    """Load instance masks for the given annotation row. Masks can be different types,
    mask is a binary true/false map of the same size as the image.
    """

    mask = np.zeros((int(row.height), int(row.height)), dtype=np.uint8)

    annotation_mode = row.annotationMode
    # print(annotation_mode)

    if annotation_mode == "bbox":
        # Bounding Box
        x = int(row["data"]["x"])
        y = int(row["data"]["y"])
        w = int(row["data"]["width"])
        h = int(row["data"]["height"])
        mask_instance = mask[:, :].copy()
        cv2.rectangle(mask_instance, (x, y), (x + w, y + h), 255, -1)
        mask[:, :] = mask_instance

    # FreeForm or Polygon
    elif annotation_mode == "freeform" or annotation_mode == "polygon":
        vertices = np.array(row["data"]["vertices"])
        vertices = vertices.reshape((-1, 2))
        mask_instance = mask[:, :].copy()
        cv2.fillPoly(mask_instance, np.int32([vertices]), (255, 255, 255))
        mask[:, :] = mask_instance

    # Line
    elif annotation_mode == "line":
        vertices = np.array(row["data"]["vertices"])
        vertices = vertices.reshape((-1, 2))
        mask_instance = mask[:, :].copy()
        cv2.polylines(mask_instance, np.int32([vertices]), False, (255, 255, 255), 12)
        mask[:, :] = mask_instance

    elif annotation_mode == "location":
        # Bounding Box
        x = int(row["data"]["x"])
        y = int(row["data"]["y"])
        mask_instance = mask[:, :].copy()
        cv2.circle(mask_instance, (x, y), 7, (255, 255, 255), -1)
        mask[:, :] = mask_instance

    elif annotation_mode is None:
        print("Not a local instance")

    return mask.astype(np.bool_)
