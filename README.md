###############################################################################################################
#                                                                                                             #
# References:                                                                                                 #
#  - https://wiki.cancerimagingarchive.net/display/Public/REST+API+Usage+Guide for complete list of API       #
#                                                                                                             #
# Credits:                                                                                                    #
#  - https://github.com/TCIA-Community/TCIA-API-SDK
#  - https://idiotdeveloper.com/unet-segmentation-with-pretrained-mobilenetv2-as-encoder/
#                                                                                                             #
###############################################################################################################
import concurrent.futures
import os
import shutil
from glob import glob

import cv2
import mdai
import numpy as np
import pandas as pd
import requests
from pydicom import dcmread
import pylibjpeg
from sklearn.model_selection import train_test_split
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
session = requests.session()

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


def array_to_rgb(x):
    x = np.load(x)

    # Converted the datatype to np.uint8
    array = x.astype(np.uint8) * 255

    # stack the channels in the new image
    rgb = np.dstack([array, array, array])
    return rgb


def download_image(url, img_path, params):
    img_bytes = session.get(url, params=params).content
    with open(img_path, 'wb') as img_file:
        img_file.write(img_bytes)


def read_image(path, IMAGE_SIZE):
    x = array_to_rgb(path)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x / 255.0
    return x


def read_mask(path, IMAGE_SIZE):
    x = np.load(path).astype(np.uint8) * 255.0
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)
    return x


class MIRDCRicord1aDataset(object):
    GET_SINGLE_IMAGE = "getSingleImage"
    GET_PATIENT_STUDY = "getPatientStudy"
    GET_SERIES = "getSeries"
    GET_PATIENT_BY_MODALITY = "PatientsByModality"
    GET_ALL_SOP_INSTANCES_UID = 'getSOPInstanceUIDs'

    PATIENT_ID = 'PatientID'
    STUDY_INSTANCE_UID = 'StudyInstanceUID'
    SERIES_INSTANCE_UID = 'SeriesInstanceUID'
    SOP_INSTANCE_UID = 'SOPInstanceUID'

    defaultParameters = {
        'Collection': 'MIDRC-RICORD-1a',
        'Modality': 'CT'
    }

    baseUrl = 'https://services.cancerimagingarchive.net/services/v4/TCIA/query/'

    def __init__(self, dataset_folder, annotations_json_path=None, clinical_data_path=None):
        self.IMAGE_SIZE = 128
        self.annotations_json_path = annotations_json_path
        self.clinical_data_path = clinical_data_path
        self.annotations = pd.DataFrame({
            'id': pd.Series([], dtype='int'),
            'StudyInstanceUID': pd.Series([], dtype='str'),
            'SeriesInstanceUID': pd.Series([], dtype='str'),
            'SOPInstanceUID': pd.Series([], dtype='str'),
            'labelName': pd.Series([], dtype='str'),
            'data': pd.Series([], dtype='object'),
            'annotationMode': pd.Series([], dtype='str')
        })
        self.patients = []
        self.patients_study = {}
        self.study_series = {}
        self.dataset_folder = dataset_folder

    def execute(self, url, queryParameters=None):
        if queryParameters is None:
            queryParameters = self.defaultParameters

        response = session.get(self.baseUrl + url, params=queryParameters)
        return response

    def fetch_patients(self, num_of_patients=None):
        resp = self.execute(self.GET_PATIENT_BY_MODALITY)
        if num_of_patients is None:
            self.patients += resp.json()
        self.patients += resp.json()[:num_of_patients]

    def fetch_patient_study(self, patient_id):
        parameters = self.defaultParameters
        parameters['PatientID'] = patient_id
        print('Retrieving the studies for patient ', patient_id)
        response = self.execute(self.GET_PATIENT_STUDY, parameters)
        self.patients_study[patient_id] = response.json()

    def fetch_study_series(self, patient_id, study_id):
        parameters = self.defaultParameters
        parameters['PatientID'] = patient_id
        parameters['StudyInstanceUID'] = study_id
        print('Retrieving the series of study {} for patient {}'.format(study_id, patient_id))
        response = self.execute(self.GET_SERIES, parameters)
        self.study_series[study_id] = response.json()

    def fetch_dicom_images_by_series(self, study_id, series_id, patient_id):
        parameters = self.defaultParameters
        parameters[self.SERIES_INSTANCE_UID] = series_id

        path = os.path.join(self.dataset_folder, 'dicoms', patient_id, study_id, series_id)

        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            print("Creation of the directory %s failed" % path)
            print("Could not save images of series ", series_id)
        else:
            print("Successfully created the directory %s " % path)

            url = self.baseUrl + self.GET_ALL_SOP_INSTANCES_UID

            # get all dicom file names with their SOPInstanceUID attached
            dicom_files = session.get(url, params=parameters).json()

            url = self.baseUrl + self.GET_SINGLE_IMAGE

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for dicom in dicom_files:
                    parameters[self.SOP_INSTANCE_UID] = dicom[self.SOP_INSTANCE_UID]
                    executor.submit(download_image, url, os.path.join(
                        path,
                        dicom[self.SOP_INSTANCE_UID] + '.dcm'),
                                    parameters)

    def fetch_single_dicom_image_by_sop_id(self, study_id, series_id, sop_id):
        path = os.path.join(self.dataset_folder, 'dicoms', study_id, series_id)
        parameters = self.defaultParameters
        parameters[self.SERIES_INSTANCE_UID] = series_id
        parameters[self.SOP_INSTANCE_UID] = sop_id

        url = self.baseUrl + self.GET_SINGLE_IMAGE
        download_image(url, os.path.join(path, sop_id + '.dcm'), parameters)

    def fetch_annotations(self, series_ids=None):
        """
        This method stores more information about the dataset like the patient ids etc for other preprocessing utilities
        but is slower and inefficient because we generate images that may not have masks
        """
        results = mdai.common_utils.json_to_dataframe(self.annotations_json_path)

        # Annotations dataframe
        annotations_df = results['annotations']
        # get only covid annotations
        # get only masks with particular series ids if specified
        """annotations_df = annotations_df.loc[annotations_df['labelName'].isin(['Typical',
                                                                              'Atypical',
                                                                              'Indeterminate'])]"""
        # Simplify table
        annotations_df = annotations_df[annotations_df.annotationNumber.notnull() &
                                        annotations_df.data.notnull() & annotations_df.height.notnull() &
                                        annotations_df.width.notnull()]

        columns_brief = ['id', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'labelName',
                         'data', 'annotationMode', 'height', 'width']
        annotations_df = annotations_df[columns_brief]

        # Shape annotations
        annotations_df = annotations_df[(annotations_df.annotationMode == 'bbox') |
                                        (annotations_df.annotationMode == 'polygon') |
                                        (annotations_df.annotationMode == 'freeform')]
        """(annotations_df.annotationMode == 'bbox') | (annotations_df.annotationMode == 'polygon')"""

        if series_ids is not None:
            # get only masks with particular series ids if specified
            annotations_df = annotations_df.loc[annotations_df['SeriesInstanceUID'].isin(series_ids)]

        self.annotations = self.annotations.append(annotations_df, ignore_index=True)

    def load_dataset_by_samples(self, num_of_samples_allowed=20):
        """
        This method is faster thn load_dataset_per_patients but we
        loose information about the the patientID in the directory structure
        """

        shutil.rmtree(self.dataset_folder, ignore_errors=True)
        print("Downloading the dataset. This may take some time. Please grab a cup of coffee and relax...")
        self.fetch_annotations()
        print('\n')

        annotations_per_image = self.annotations.groupby('SOPInstanceUID')

        max_samples = min(len(annotations_per_image), num_of_samples_allowed)

        for name, annotations in annotations_per_image:
            if num_of_samples_allowed < 1:
                # prevents the dataset to be more than what is requested
                break

            file_name = annotations.iloc[0][self.SOP_INSTANCE_UID]

            # fetch the original dicom image of each annotation annotation
            images_path = os.path.join(self.dataset_folder,
                                       'dicoms',
                                       annotations.iloc[0][self.STUDY_INSTANCE_UID],
                                       annotations.iloc[0][self.SERIES_INSTANCE_UID])
            os.makedirs(images_path, exist_ok=True)
            self.fetch_single_dicom_image_by_sop_id(annotations.iloc[0][self.STUDY_INSTANCE_UID],
                                                    annotations.iloc[0][self.SERIES_INSTANCE_UID],
                                                    file_name)

            masks_per_annotations = []
            for index, annotation in annotations.iterrows():
                masks_per_annotations.append(get_mask_instance(annotation))

            # generate masks from the annotations of each images
            masks_path = os.path.join(self.dataset_folder, 'masks')
            """annotations.iloc[0][self.STUDY_INSTANCE_UID],
            annotations.iloc[0][self.SERIES_INSTANCE_UID]"""

            # create directory of mask if it does not exists
            os.makedirs(masks_path, exist_ok=True)

            file_path = os.path.join(masks_path, file_name + '.npy')

            # combine all masks per annotation to a single
            # segmentation mask and save it
            mask = np.logical_or.reduce(masks_per_annotations)
            np.save(file_path, mask)

            num_of_samples_allowed -= 1
            print('{}/{} Image and Mask {} saved'.format(max_samples - num_of_samples_allowed,
                                                         max_samples,
                                                         file_name))

    def load_dataset_by_patients(self, num_of_patients=20):
        shutil.rmtree(self.dataset_folder, ignore_errors=True)

        print("Downloading the dataset. This may take some time. Please grab a cup of coffee and relax...\n")

        print('STEP 1/5: FETCHING PATIENTS')
        self.fetch_patients(num_of_patients)
        print('\n')

        print('STEP 2/5: GET ALL STUDIES FOR EACH PATIENTS')
        for patient in self.patients:
            self.fetch_patient_study(patient[self.PATIENT_ID])
        print('\n')

        print('STEP 3/5: GET ALL SERIES FOR EACH STUDY')
        for key in self.patients_study:
            for study in self.patients_study[key]:
                self.fetch_study_series(study[self.PATIENT_ID],
                                        study[self.STUDY_INSTANCE_UID])
        print('\n')

        print('STEP 4/5: GET ALL IMAGES FOR EACH SERIES')
        for key in self.study_series:
            for series in self.study_series[key]:
                self.fetch_dicom_images_by_series(series[self.PATIENT_ID],
                                                  series[self.STUDY_INSTANCE_UID],
                                                  series[self.SERIES_INSTANCE_UID])
        print('\n')

        print('STEP 5/5: GET ALL SEGMENTATION MASKS')
        series_ids = []
        # collect all the ids of series we could download to
        # only get the annotations of those series
        for key in self.study_series:
            for series in self.study_series[key]:
                series_ids.append(series[self.SERIES_INSTANCE_UID])
        self.fetch_annotations(series_ids)

        path = os.path.join(self.dataset_folder,
                            'masks',
                            )
        """annotations[self.STUDY_INSTANCE_UID]"""
        """annotations[self.SERIES_INSTANCE_UID])"""

        os.makedirs(path, exist_ok=True)
        for index, annotations in self.annotations.iterrows():
            np.save(get_mask_instance(annotations), os.path.join(path, annotations[self.SOP_INSTANCE_UID] + '.npy'))

    def generate_image_training_data(self):
        training_image_path = os.path.join(self.dataset_folder, 'images')
        shutil.rmtree(training_image_path, ignore_errors=True)
        os.makedirs(training_image_path, exist_ok=True)
        for subdir, dirs, files in os.walk(os.path.join(self.dataset_folder, 'dicoms')):
            for file in files:
                file_path = os.path.join(subdir, file)
                if file_path.endswith('.dcm'):
                    dicom = dcmread(file_path)
                    image = np.array(dicom.pixel_array)
                    np.save(os.path.join(training_image_path, os.path.splitext(file)[0]), image)

    def load_data(self, split=0.3):
        images = sorted(glob(os.path.join(self.dataset_folder, "images/*")))
        masks = sorted(glob(os.path.join(self.dataset_folder, "masks/*")))

        total_size = len(images)
        valid_size = int(split * total_size)
        test_size = int(split * total_size)

        train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
        train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

        train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
        train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

        return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

    def tf_dataset(self, x, y, batch=8):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.map(self.tf_parse)
        dataset = dataset.batch(batch)
        dataset = dataset.repeat()
        return dataset

    def tf_parse(self, x, y):
        IMAGE_SIZE = self.IMAGE_SIZE

        def _parse(image, mask):
            image = read_image(image, IMAGE_SIZE)
            mask = read_mask(mask, IMAGE_SIZE)
            return image, mask

        x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
        x.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        y.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])
        return x, y
# COVID-Segmentation-Pre-trained-Encoder-U-NET
