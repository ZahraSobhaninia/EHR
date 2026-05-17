import os
from dataclasses import dataclass
from os.path import join
from typing import List

import pandas as pd
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm
import logging

from corebehrt.constants.data import (
    ABSPOS_FEAT,
    AGE_FEAT,
    ATTENTION_MASK,
    CONCEPT_FEAT,
    SEGMENT_FEAT,
    TARGET,
)
from corebehrt.modules.preparation.mask import ConceptMasker


@dataclass
class PatientData:
    pid: str
    concepts: List[int]  # or List[str], depending on your use
    abspos: List[float]  # or int, depends on your data
    segments: List[int]
    ages: List[float]  # e.g. age at each concept
    outcome: int = None


class PatientDataset:
    """A dataset class for managing patient data and vocabulary.

    This class provides functionality to store and process patient data along with their
    associated vocabulary. It supports parallel processing of patient data and saving/loading
    functionality.

    Attributes:
        patients (List[PatientData]): List of patient data objects containing medical concepts,
            positions, segments and ages.
    """

    def __init__(self, patients: List[PatientData]):
        """Initialize the PatientDataset.

        Args:
            patients (List[PatientData]): List of patient data objects.
        """
        self.patients = patients

    def __len__(self):
        """Get the number of patients in the dataset."""
        return len(self.patients)

    def __getitem__(self, idx: int):
        """Get a patient by index.

        Args:
            idx (int): Index of the patient to retrieve.

        Returns:
            PatientData: The patient data at the given index.
        """
        return self.patients[idx]

    def process_in_parallel(self, func, n_jobs=-1, chunk_size=1000, **kwargs):
        """Process all patients in parallel using the given function with chunking support.

        Args:
            func: Function to apply to each patient
            n_jobs (int): Number of parallel jobs. -1 means using all processors
            chunk_size (int): Size of patient chunks to process together
            **kwargs: Additional keyword arguments passed to the function

        Returns:
            list: Results of applying the function to each patient
        """
        # Get the chunk size
        n_jobs = 1 if len(self.patients) < 1000 else n_jobs
        loop = tqdm(
            self.patients,
            total=len(self.patients),
            desc=f"{func.__name__}",
            mininterval=10,
        )
        results = Parallel(n_jobs=n_jobs, batch_size=chunk_size, backend="threading")(
            delayed(func)(patient, **kwargs) for patient in loop
        )

        results = [r for r in results if r is not None]
        removed = len(self.patients) - len(results)
        if removed > 0:
            logging.warning(f"[process_in_parallel] Removed {removed} None results from {func.__name__}")

        return results

    def save(self, save_dir: str, suffix: str = ""):
        """Save patient data and vocabulary to disk.

        Args:
            save_dir (str): Directory path to save the files.
        """
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.patients, join(save_dir, f"patients{suffix}.pt"))

    def filter_by_pids(self, pids: List[str]) -> "PatientDataset":
        pids_set = set(pids)
        filtered = PatientDataset([p for p in self.patients if p.pid in pids_set])
        logging.info(f"[filter_by_pids] Before: {len(self.patients)} | After: {len(filtered)} | Removed: {len(self.patients) - len(filtered)}")
        return filtered

    def get_pids(self) -> List[str]:
        return [p.pid for p in self.patients]

    def get_outcomes(self) -> List[int]:
        return [p.outcome for p in self.patients]

    def assign_outcomes(self, outcomes: pd.Series):
        """Assigns binary outcomes to each patient in the dataset.

        Takes a pandas Series mapping patient IDs to outcomes absolute positions and assigns a binary outcome
        to each patient in the dataset.

        Args:
            outcomes (pd.Series): Series with patient IDs as index and outcomes as values.
                The actual outcome values are not used, only whether they are null or not.

        Returns:
            PatientDataset: Returns self for method chaining.
        """

        outcome_pids = set(outcomes.index)
        patient_pids = set(p.pid for p in self.patients)

        intersection = outcome_pids & patient_pids
        only_in_outcomes = outcome_pids - patient_pids
        only_in_patients = patient_pids - outcome_pids

        logging.info(f"[assign_outcomes] ___Outcome PIDs total: {len(outcome_pids)}")
        logging.info(f"[assign_outcomes] Patient PIDs total: {len(patient_pids)}")
        logging.info(f"[assign_outcomes] Outcome ∩ Patient: {len(intersection)}")
        logging.info(f"[assign_outcomes] PIDs only in outcomes (not in patients): {len(only_in_outcomes)}")
        logging.info(f"[assign_outcomes] ___PIDs only in patients (not in outcomes): {len(only_in_patients)}")

        missing_pids = 0
        is_multitask = isinstance(outcomes, pd.DataFrame)

        for p in self.patients:
            if p.pid in outcomes.index:
                if is_multitask:
                    p.outcome = outcomes.loc[p.pid].to_dict()
                else:
                    p.outcome = int(outcomes[p.pid])
            else:
                if is_multitask:
                    p.outcome = {col: 0 for col in outcomes.columns}
                else:
                    p.outcome = 0
                missing_pids += 1

        logging.info(f"[assign_outcomes] Total patients: {len(self.patients)}")
        logging.info(f"[assign_outcomes] Patients with missing outcome (set to 0): {missing_pids}")
        if is_multitask:
            for col in outcomes.columns:
                pos = sum(1 for p in self.patients if p.outcome.get(col, 0) == 1)
                logging.info(f"[assign_outcomes] Patients with {col}=1: {pos}")
        else:
            logging.info(f"[assign_outcomes] Patients with outcome=1: {sum(p.outcome for p in self.patients)}")


        return self

    @staticmethod
    def combine_datasets(datasets: List["PatientDataset"]) -> "PatientDataset":
        """Combine multiple PatientDataset objects into one.

        Args:
            datasets (List[PatientDataset]): List of PatientDataset objects to combine.

        Returns:
            PatientDataset: A new PatientDataset object with combined patients.
        """
        combined_patients = []
        for dataset in datasets:
            combined_patients.extend(dataset.patients)
        logging.info(f"[combine_datasets] Combined {len(datasets)} datasets with total {len(combined_patients)} patients")
        return PatientDataset(combined_patients)


class MLMDataset(Dataset):
    def __init__(
        self,
        patients: List[PatientData],
        vocabulary: dict,
        select_ratio: float,
        masking_ratio: float = 0.8,
        replace_ratio: float = 0.1,
        ignore_special_tokens: bool = True,
    ):
        self.patients = patients
        self.vocabulary = vocabulary
        self.masker = ConceptMasker(
            vocabulary,
            select_ratio,
            masking_ratio,
            replace_ratio,
            ignore_special_tokens,
        )

    def __getitem__(self, index: int) -> dict:
        """
        1. Retrieve the PatientData.
        2. Mask the 'concepts'.
        3. Convert everything to torch.Tensor.
        4. Return a dict that PyTorch can collate into a batch.
        """
        patient = self.patients[index]
        concepts = torch.tensor(patient.concepts, dtype=torch.long)
        masked_concepts, target = self.masker.mask_patient_concepts(concepts)
        attention_mask = torch.ones_like(masked_concepts)
        sample = {
            CONCEPT_FEAT: masked_concepts,
            TARGET: target,
            ABSPOS_FEAT: torch.tensor(patient.abspos, dtype=torch.float),
            SEGMENT_FEAT: torch.tensor(patient.segments, dtype=torch.long),
            AGE_FEAT: torch.tensor(patient.ages, dtype=torch.float),
            ATTENTION_MASK: attention_mask,
            "patient_id": torch.tensor(int(patient.pid), dtype=torch.long)

        }

        return sample

    def __len__(self):
        return len(self.patients)


class BinaryOutcomeDataset(Dataset):
    """
    outcomes: absolute position when outcome occured for each patient
    outcomes is a list of the outcome timestamps to predict
    """

    def __init__(self, patients: List[PatientData]):
        self.patients = patients

    def __getitem__(self, index: int) -> dict:
        patient = self.patients[index]
        attention_mask = torch.ones(
            len(patient.concepts), dtype=torch.long
        )  # Require attention mask for bi-gru head        
        if isinstance(patient.outcome, dict):
            target = torch.tensor(list(patient.outcome.values()), dtype=torch.float)
        else:
            target = torch.tensor(patient.outcome, dtype=torch.float)

        sample = {
            CONCEPT_FEAT: torch.tensor(patient.concepts, dtype=torch.long),
            ABSPOS_FEAT: torch.tensor(patient.abspos, dtype=torch.float),
            SEGMENT_FEAT: torch.tensor(patient.segments, dtype=torch.long),
            AGE_FEAT: torch.tensor(patient.ages, dtype=torch.float),
            ATTENTION_MASK: attention_mask,
            TARGET: target,
            "patient_id": torch.tensor(int(patient.pid), dtype=torch.long)
        }
        return sample

    def __len__(self):
        return len(self.patients)
