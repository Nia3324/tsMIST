import os
import pickle
from pathlib import Path

import numpy as np
import wfdb
from loguru import logger
from scipy.signal import filtfilt, firwin
from typing import List, Tuple, Union


# Utility constants related to the MIT-BIH loader
#MIT_BIH_PATH = "/net/sharedfolders/datasets/MOTION/mit-bih-arrhythmia-database-1.0.0/"
MIT_BIH_PATH = "mit-bih-arrhythmia-database-1.0.0"

AAMI = ["N", "SVEB", "VEB", "F", "Q"]
AAMI_N = ["N", "L", "R"]
AAMI_SVEB = ["A", "a", "J", "S", "e", "j"]
AAMI_VEB = ["V", "E"]
AAMI_F = ["F"]
AAMI_Q = ["P", "/", "f", "u"]
AAMI_ALL = AAMI_N + AAMI_SVEB + AAMI_VEB + AAMI_F + AAMI_Q

SUBJECTS = [
        "100",
        "101",
        "102",
        "103",
        "104",
        "105",
        "106",
        "107",
        "108",
        "109",
        "111",
        "112",
        "113",
        "114",
        "115",
        "116",
        "117",
        "118",
        "119",
        "121",
        "122",
        "123",
        "124",
        "200",
        "201",
        "202",
        "203",
        "205",
        "207",
        "208",
        "209",
        "210",
        "212",
        "213",
        "214",
        "215",
        "217",
        "219",
        "220",
        "221",
        "222",
        "223",
        "228",
        "230",
        "231",
        "232",
        "233",
        "234",
    ]


class MITBIHDataset:
    LEFT_WINDOW_SEC = 0.25
    RIGHT_WINDOW_SEC = 0.25
    BANDPASS_LOW = 4
    BANDPASS_HIGH = 22
    FILTER_ORDER = 2

    def __init__(
        self,
        lead_index: int = 0,
        apply_filter: bool = True,
        subjects: Union[List[str], None] = SUBJECTS,  # Set default value to SUBJECTS
        subsets: Union[List[str], str] = "aami"
    ):
        """Initialize the MITBIHDataset class. If a pickle file exists, it
        loads the dataset. Otherwise, it processes the data and saves it for
        future use.

        Args:
            lead_index (int): Index of the ECG lead to use.tuple[np.ndarray, np.ndarray, np.ndarray]:
            apply_filter (bool): Whether to apply bandpass filtering.
            subjects (List[str]): List of subject IDs to process.
            subsets (Union[List[str], str]): Subset of annotations to process.
        """
        self.path = Path("mit-bih-arrhythmia-database-1.0.0")
        self.pickle_path = Path(os.path.join("data", "MIT-BIH.pkl"))
        self.lead_index = lead_index
        self.apply_filter = apply_filter
        self.subjects = subjects
        self.subsets = subsets
        self.data = None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load the dataset, either from a pickle file or by processing the
        data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Features, labels, and subject IDs.
        """
        if self.data is not None:
            return self.data

        if self.pickle_path.exists():
            self._load()
            logger.info("Dataset loaded from pickle file.")
        else:
            self._process_and_save(self.subjects, self.subsets)
            logger.info("Dataset processed and saved.")
        return self.data

    @staticmethod
    def _bandpass_filter(signal: np.ndarray, low: float, high: float, sr: int, order: int) -> np.ndarray:
        """Apply a bandpass filter to the signal."""
        filter_coeffs = firwin(order, [low, high], fs=sr, pass_zero="bandpass")
        return filtfilt(filter_coeffs, 1, signal)

    def _construct_individual_trial(
        self,
        record: wfdb.Record,
        annotation: wfdb.Annotation,
        subsets: Union[List[str], str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a set of waves and annotation labels for a given record.

        The waves are fixed-size windows centered around the annotation
        index (assumed to be the R peak).
        """
        annotations = np.array(annotation.symbol).astype(str)
        logger.debug(annotations)
        annotation_ids = np.array(annotation.sample)

        sampling_rate = record.fs
        left_window = int(sampling_rate * self.LEFT_WINDOW_SEC)
        right_window = int(sampling_rate * self.RIGHT_WINDOW_SEC)
        window_size = left_window + right_window

        lead_signal = np.copy(record.p_signal[:, self.lead_index])
        if self.apply_filter:
            lead_signal = self._bandpass_filter(
                lead_signal,
                self.BANDPASS_LOW,
                self.BANDPASS_HIGH,
                sr=sampling_rate,
                order=self.FILTER_ORDER,
            )

        waves, labels = [], []

        if isinstance(subsets, list):
            for i, label in enumerate(annotations):
                if label in subsets:
                    segment = lead_signal[annotation_ids[i] - left_window : annotation_ids[i] + right_window]
                    if len(segment) != window_size:
                        logger.warning("Sample discarded due to incorrect beat length.")
                        continue
                    waves.append(segment)
                    labels.append(subsets.index(label))
        elif subsets == "aami":
            for i, label in enumerate(annotations):
                if label not in AAMI_ALL:
                    logger.debug(f"Beat does not belong to AAMI: {label}")
                    continue
                if i == 0 or i >= len(annotation_ids) - 1:
                    continue
                segment = lead_signal[annotation_ids[i] - left_window : annotation_ids[i] + right_window]
                if len(segment) != window_size:
                    logger.warning("Sample discarded due to incorrect beat length.")
                    continue

                waves.append(segment)
                labels.append(self._get_aami_class(label))

        return np.array(waves), np.array(labels)

    @staticmethod
    def _get_aami_class(label: str) -> int:
        """Map AAMI annotations to class indices."""
        if label in AAMI_N:
            return 0
        elif label in AAMI_SVEB:
            return 1
        elif label in AAMI_VEB:
            return 2
        elif label in AAMI_F:
            return 3
        elif label in AAMI_Q:
            return 4
        raise ValueError(f"Unknown AAMI label: {label}")

    def _load_dataset(
        self,
        subjects: Union[List[str]],
        subsets: Union[List[str], str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        subject_waves, labels, subject_ids = [], [], []

        for subject in subjects:
            try:
                record_path = os.path.join(self.path, subject)
                record = wfdb.rdrecord(record_path)
                annotation = wfdb.rdann(record_path, "atr")
                subject_features, subject_labels = self._construct_individual_trial(record, annotation, subsets)
                if len(subject_features) == 0:
                    logger.warning(f"Subject {subject} discarded due to lack of annotations.")
                    continue
                subject_waves.append(subject_features)
                labels.append(subject_labels)
                subject_ids.append(np.repeat(subjects.index(subject), len(subject_labels)))
            except Exception as e:
                logger.error(f"Error processing subject {subject}: {e}")

        logger.info(f"Total samples: {sum(len(f) for f in subject_waves)}")
        return np.vstack(subject_waves), np.hstack(labels), np.hstack(subject_ids)

    def _save(self):
        """Save the dataset to a pickle file."""
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.data, f)
        logger.info(f"Dataset saved to {self.pickle_path}.")

    def _load(self):
        """Load the dataset from a pickle file."""
        with open(self.pickle_path, "rb") as f:
            self.data = pickle.load(f)  # nosec

    def _process_and_save(
        self,
        subjects: Union[List[str]],
        subsets: Union[List[str], str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process the dataset and save it to a pickle file."""
        self.data = self._load_dataset(subjects, subsets=subsets)
        self._save()


if __name__ == "__main__":
    dataset = MITBIHDataset(
        lead_index=0,
        apply_filter=True,
        subjects=SUBJECTS,
        subsets=["N", "L"],
    )
    X, y, s = dataset.load_data()