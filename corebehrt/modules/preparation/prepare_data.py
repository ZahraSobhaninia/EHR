import json
from collections import Counter
import logging
import os
from datetime import datetime
from os.path import join
from typing import List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from corebehrt.constants.data import ABSPOS_COL, PID_COL, TIMESTAMP_COL
from corebehrt.constants.paths import INDEX_DATES_FILE, OUTCOMES_FILE, PID_FILE
from corebehrt.functional.cohort_handling.outcomes import get_binary_outcomes
from corebehrt.functional.features.normalize import normalize_segments_for_patient
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.functional.io_operations.save import save_vocabulary
from corebehrt.functional.preparation.convert import dataframe_to_patient_list
from corebehrt.functional.preparation.filter import (
    censor_patient,
    censor_patient_with_delays,
    exclude_short_sequences,
)
from corebehrt.functional.preparation.truncate import (
    truncate_patient,
    truncate_patient_df,
)
from corebehrt.functional.preparation.utils import (
    get_background_length,
    get_background_length_pd,
    get_concept_id_to_delay,
    get_non_priority_tokens,
)
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.modules.cohort_handling.patient_filter import filter_df_by_pids
from corebehrt.modules.features.loader import ShardLoader
from corebehrt.modules.monitoring.logger import TqdmToLogger
from corebehrt.modules.preparation.dataset import PatientData, PatientDataset
from corebehrt.modules.setup.config import Config

logger = logging.getLogger(__name__)  # Get the logger for this module


# TODO: Add option to load test set only!
class DatasetPreparer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.processed_dir = cfg.paths.prepared_data
        self.vocab = load_vocabulary(cfg.paths.tokenized)
        self.predict_token = self.vocab["[CLS]"]

    def prepare_finetune_data(self, mode="tuning") -> PatientDataset:
        """
        Prepares and processes patient data for fine-tuning, including censoring, truncation, and outcome assignment.

        Loads patient data and outcomes, applies cohort and cutoff filtering, assigns binary outcomes, computes and validates censoring dates, applies censoring (optionally with concept-specific delays), excludes short sequences, truncates and normalizes patient records, and saves the processed dataset for downstream modeling.

        Args:
            mode: Specifies which data split to use for fine-tuning (default is "tuning").

        Returns:
            A PatientDataset object containing the processed and labeled patient data ready for fine-tuning.
        """

        outcome_cfg = self.cfg.outcome
        paths_cfg = self.cfg.paths
        data_cfg = self.cfg.data
        pids = self.load_cohort(paths_cfg)


    # --------------------
    # STEP 1 - Detect OOT vs Tuning mode
    # --------------------
      #  if mode == "tuning":
      #      tuning_path = os.path.join(paths_cfg.tokenized, "features_tuning")
      #      if not os.path.exists(tuning_path):
      #          logger.warning("⚠️ 'features_tuning' not found — assuming OOT mode, using 'features_train' instead.")
       #         mode = "train"

        split_path = os.path.join(paths_cfg.tokenized, f"features_{mode}")
        if not os.path.exists(split_path) or len(glob.glob(os.path.join(split_path, "*.parquet"))) == 0:
            logger.warning(f" 'features_{mode}' not found or empty — using 'features_train' instead.")
            mode = "train"    
        
    # --------------------
    # STEP 2 - Load index_dates
    # --------------------
        index_dates = pd.read_csv(
            join(paths_cfg.cohort, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
        )
        index_dates[PID_COL] = index_dates[PID_COL].astype(int)
        index_dates[ABSPOS_COL] = get_hours_since_epoch(index_dates[TIMESTAMP_COL])
            # --------------------
        # STEP 3 - Load outcomes
        # --------------------
        outcomes = pd.read_csv(paths_cfg.outcome)
        outcomes[PID_COL] = outcomes[PID_COL].astype(int)


       # --------------------
    # STEP 4 - Load tokenized data (ShardLoader)
    # --------------------
        loader = ShardLoader(
            data_dir=paths_cfg.tokenized,
            splits=[f"features_{mode}"],
            patient_info_path=None,
        )

        tokenized_dfs = []
        for df, _ in loader():
            tokenized_dfs.append(df)
        tokenized_data = pd.concat(tokenized_dfs, ignore_index=True)
        tokenized_data[PID_COL] = tokenized_data[PID_COL].astype(int)

        logger.info(f"[Finetune] index_dates: {len(index_dates)} | unique PIDs: {index_dates[PID_COL].nunique()}")
        logger.info(f"[Finetune] outcomes: {len(outcomes)} | unique PIDs: {outcomes[PID_COL].nunique()}")
        logger.info(f"[Finetune] tokenized: {len(tokenized_data)} | unique PIDs: {tokenized_data[PID_COL].nunique()}")

       
        # --------------------
        # STEP 6 - Compute intersection
        # --------------------
        pids_tokenized = set(tokenized_data[PID_COL].unique())
        pids_outcomes = set(outcomes[PID_COL].unique())
        pids_index = set(index_dates[PID_COL].unique())

        intersection_all = pids_tokenized & pids_index       # & pids_outcomes


        logger.info(f"[Finetune] Patients in ALL (tokenized ∩ outcomes ∩ index_dates): {len(intersection_all)}")
        logger.info(f"[Finetune] Only in tokenized: {len(pids_tokenized - intersection_all)}")
        logger.info(f"[Finetune] Only in outcomes: {len(pids_outcomes - intersection_all)}")
        logger.info(f"[Finetune] Only in index_dates: {len(pids_index - intersection_all)}")

        # --------------------
        # STEP 7 - Filter data to aligned PIDs
        # --------------------
        index_dates = index_dates[index_dates[PID_COL].isin(intersection_all)]
        outcomes = outcomes[outcomes[PID_COL].isin(intersection_all)]
        tokenized_data = tokenized_data[tokenized_data[PID_COL].isin(intersection_all)]

        logger.info(f"✅ After alignment, working with {len(intersection_all)} patients.")

      
        logger.info(f"[Sanity] tokenized_data shape: {tokenized_data.shape}")
        logger.info(f"[Sanity] unique patient IDs in tokenized: {tokenized_data[PID_COL].nunique()}")


        patient_list = []
        for df, _ in tqdm(
            loader(), desc="Batch Process Data", file=TqdmToLogger(logger)
        ):
            if pids is not None:
                df = filter_df_by_pids(df, pids)
            if data_cfg.get("cutoff_date"):
                df = self._cutoff_data(df, data_cfg.cutoff_date)
            # !TODO: if index date is the same for all patients, then we can censor here.
            self._check_sorted(df)
            batch_patient_list = dataframe_to_patient_list(df)
            patient_list.extend(batch_patient_list)
        logger.info(f"Number of patients: {len(patient_list)}")
        logger.info(f"[PatientList] Total patients constructed from tokenized: {len(patient_list)}")

        data = PatientDataset(patients=patient_list)

     #   # Loading and processing outcomes
     #   outcomes = pd.read_csv(paths_cfg.outcome)
     #   outcomes[PID_COL] = outcomes[PID_COL].astype(int)
     #   outcomes = filter_df_by_pids(outcomes, data.get_pids())
    #    logger.info("Handling outcomes")

        # Outcome Handler now only needs to do 1 thing: if outcome is in follow up window 1 else 0
     #   binary_outcomes = get_binary_outcomes(
     #       index_dates,
     #       outcomes,
      #      outcome_cfg.get("n_hours_start_follow_up", 0),
      #      outcome_cfg.get("n_hours_end_follow_up", None),
      #  )

      #  logger.info("Assigning outcomes")
      #  data = data.assign_outcomes(binary_outcomes)



        # -------------------------
        # Loading and processing outcomes
        logger.info("Handling outcomes")

        binary_outcomes = get_binary_outcomes(
            index_dates,
            outcomes,
            outcome_cfg.get("n_hours_start_follow_up", 0),
            outcome_cfg.get("n_hours_end_follow_up", None),
        )

        logger.info("Assigning outcomes")
        logger.info(f"Binary outcomes summary: {binary_outcomes.value_counts()}")

        if mode == "train":
            # فقط بیماران دارای outcome را نگه داریم
            valid_pids = set(binary_outcomes.index)
            data.patients = [p for p in data.patients if p.pid in valid_pids]
        logger.info(f"[Before Outcome Assignment] Patients in data: {len(data.patients)}")
        logger.info(f"[Before Outcome Assignment] Binary outcomes: {len(binary_outcomes)} | Positive outcomes: {(binary_outcomes==1).sum()}")

        data = data.assign_outcomes(binary_outcomes)

        # Align index_dates with data patients
        valid_pids = set(p.pid for p in data.patients)
        index_dates = index_dates[index_dates[PID_COL].isin(valid_pids)]

        missing_pids = valid_pids - set(index_dates[PID_COL])
        if missing_pids:
            logger.warning(f"[Censoring] {len(missing_pids)} patients in data but missing from index_dates! Adding placeholder dates.")
            
        
            placeholder_df = pd.DataFrame({
                PID_COL: list(missing_pids),
                ABSPOS_COL: index_dates[ABSPOS_COL].max()  # یا هر عدد بزرگ مثل max(abspos)+1
            })
            index_dates = pd.concat([index_dates, placeholder_df], ignore_index=True)

        index_dates[PID_COL] = index_dates[PID_COL].astype(int)  # اطمینان از نوع صحیح

        # اطمینان از نوع صحیح ایندکس در censor_dates
        censor_dates = (
            index_dates.set_index(PID_COL)[ABSPOS_COL] + self.cfg.outcome.n_hours_censoring
        )

        censor_dates.index = censor_dates.index.astype(int)


        valid_censor_pids = set(censor_dates.index)
        initial_patient_count = len(data.patients)
        data.patients = [p for p in data.patients if int(p.pid) in valid_censor_pids]
        logger.info(f" [Censoring] Removed {initial_patient_count - len(data.patients)} patients without censor_date.")
      

        remaining_pids = set([int(p.pid) for p in data.patients])
        missing_censor_pids = remaining_pids - valid_censor_pids
        if missing_censor_pids:
            logger.warning(f"❗ {len(missing_censor_pids)} patients still missing censor_date! Example: {list(missing_censor_pids)[:5]}")
        else:
            logger.info(" All patients have valid censor_date.")

        self._validate_censoring(data.patients, censor_dates, logger)
        ##zahra
        invalid_pids = [p.pid for p in data.patients if p.pid not in censor_dates.index]
        if invalid_pids:
            logger.warning(f"[Censoring] {len(invalid_pids)} patients missing from censor_dates (example: {invalid_pids[:5]})")

        if "concept_pattern_hours_delay" in self.cfg:
            concept_id_to_delay = get_concept_id_to_delay(
                self.cfg.concept_pattern_hours_delay, self.vocab
            )
            data.patients = data.process_in_parallel(
                censor_patient_with_delays,
                censor_dates=censor_dates,
                predict_token_id=self.predict_token,
                concept_id_to_delay=concept_id_to_delay,
            )
        else:
            data.patients = data.process_in_parallel(
                censor_patient,
                censor_dates=censor_dates,
                predict_token_id=self.predict_token,
            )

        background_length = get_background_length(data, self.vocab)
        # Exclude short sequences
        logger.info("Excluding short sequences")
        data.patients = exclude_short_sequences(
            data.patients,
            data_cfg.get("min_len", 1) + background_length,
        )
        logger.info(
            f"Number of patients after excluding short sequences: {len(data.patients)}"
        )

        # Truncation
        non_priority_tokens = (
            None
            if data_cfg.get("low_priority_prefixes", None) is None
            else get_non_priority_tokens(self.vocab, data_cfg.low_priority_prefixes)
        )

        # تأکید دوباره برای type match
        for patient in data.patients:
            patient.pid = int(patient.pid)

        data.patients = data.process_in_parallel(
            truncate_patient,
            max_len=data_cfg.truncation_len,
            background_length=background_length,
            sep_token=self.vocab["[SEP]"],
            non_priority_tokens=non_priority_tokens,
        )

        data.patients = data.process_in_parallel(normalize_segments_for_patient)
        # Check if max segment is larger than type_vocab b_size
        # TODO: pass pt_model_config and perform this check
        # max_segment(data, model_cfg.type_vocab_size)
        # Previously had issue with it
        logger.info(
            f"Max segment length: {max(max(patient.segments) for patient in data.patients)}"
        )
##

        finetune_stats = {
            'summary': {
                'total_patients': len(data.patients),
                'positive_patients': sum(1 for p in data.patients if p.outcome == 1),
                'negative_patients': sum(1 for p in data.patients if p.outcome == 0),
                'positive_rate': round(sum(1 for p in data.patients if p.outcome == 1) / len(data.patients) * 100, 3),
                'avg_sequence_length': round(sum(len(p.concepts) for p in data.patients) / len(data.patients), 2),
                'max_sequence_length': max(len(p.concepts) for p in data.patients),
                'min_sequence_length': min(len(p.concepts) for p in data.patients),
                'vocab_size': len(self.vocab),
            },
            'per_patient': {
                str(p.pid): {
                    'seq_length': len(p.concepts),
                    'unique_concepts': len(set(p.concepts)),
                    'outcome': int(p.outcome)
                }
                for p in data.patients
            }
        }
        
        with open(join(self.processed_dir, 'finetune_stats.json'), 'w') as f:
            json.dump(finetune_stats, f, indent=2)
        logger.info(f"Saved finetune stats: {finetune_stats['summary']}")

##
        # save
        os.makedirs(self.processed_dir, exist_ok=True)
        save_vocabulary(self.vocab, self.processed_dir)
        data.save(self.processed_dir)
        outcomes.to_csv(join(self.processed_dir, OUTCOMES_FILE), index=False)
        index_dates.to_csv(join(self.processed_dir, INDEX_DATES_FILE), index=False)

        return data

    def prepare_pretrain_data(self, save_data=False) -> Tuple[PatientDataset, dict]:
        data_cfg = self.cfg.data
        paths_cfg = self.cfg.paths

        pids = self.load_cohort(paths_cfg)
        # Load tokenized data + vocab
        loader = ShardLoader(
            data_dir=paths_cfg.tokenized,
            splits=["features_train"],
            patient_info_path=None,
        )
        patient_list = []
        for df, _ in tqdm(
            loader(), desc="Batch Process Data", file=TqdmToLogger(logger)
        ):
            if pids is not None:
                df = filter_df_by_pids(df, pids)
            df = df.set_index(PID_COL, drop=True)

            if data_cfg.get("cutoff_date"):
                df = self._cutoff_data(df, data_cfg.cutoff_date)
            df = self._truncate(df, self.vocab, data_cfg.truncation_len)
            df = df.reset_index(drop=False)
            self._check_sorted(df)
            batch_patient_list = dataframe_to_patient_list(df)
            patient_list.extend(batch_patient_list)

        logger.info(f"Number of patients: {len(patient_list)}")
        data = PatientDataset(patients=patient_list)

        logger.info("Excluding short sequences")
        background_length = get_background_length(data, self.vocab)
        data.patients = exclude_short_sequences(
            data.patients,
            data_cfg.get("min_len", 0) + background_length,
        )
        logger.info(
            f"Number of patients after excluding short sequences: {len(data.patients)}"
        )

# Normalize segments
        data.patients = data.process_in_parallel(normalize_segments_for_patient)
        logger.info(
            f"Max segment length: {max(max(patient.segments) for patient in data.patients)}"
        )
##
        # ===== Stats =====

        # inverse vocabulary
        inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # Codes distribution
        all_concepts = [c for p in data.patients for c in p.concepts]
        type_counts = Counter()
        code_counts = Counter()
        
        for concept_id in all_concepts:
            code = inv_vocab.get(concept_id, 'UNK')
            code_counts[code] += 1
            if '/' in code:
                prefix = code.split('/')[0]
                type_counts[prefix] += 1
            elif code.startswith('['):
                type_counts['SPECIAL'] += 1
            else:
                type_counts['OTHER'] += 1
        
        total_codes = sum(type_counts.values())
        
        full_stats = {
            'summary': {
                'total_patients': len(data.patients),
                'train_patients': int(len(data.patients) * 0.9),
                'val_patients': int(len(data.patients) * 0.1),
                'vocab_size': len(self.vocab),
                'avg_sequence_length': round(sum(len(p.concepts) for p in data.patients) / len(data.patients), 2),
                'max_sequence_length': max(len(p.concepts) for p in data.patients),
                'min_sequence_length': min(len(p.concepts) for p in data.patients),
            },
            'code_type_distribution': {
                'counts': dict(type_counts),
                'percentages': {k: round(v/total_codes*100, 2) for k, v in type_counts.items()}
            },
            'top_100_codes': dict(code_counts.most_common(100)),
            'per_patient': {
                str(p.pid): {
                    'seq_length': len(p.concepts),
                    'unique_concepts': len(set(p.concepts))
                }
                for p in data.patients
            }
        }
        
        os.makedirs(self.processed_dir, exist_ok=True)
        with open(join(self.processed_dir, 'dataset_stats.json'), 'w') as f:
            json.dump(full_stats, f, indent=2)
        logger.info(f"Saved dataset stats for {len(data.patients)} patients")
##
        # Save
        os.makedirs(self.processed_dir, exist_ok=True)
        save_vocabulary(self.vocab, self.processed_dir)
        if save_data:
            data.save(self.processed_dir)
        return data

    @staticmethod
    def _truncate(
        df: pd.DataFrame, vocab: dict, truncation_length: int
    ) -> pd.DataFrame:
        """
        Truncate the dataframe to the truncation length.
        """
        background_length = get_background_length_pd(df, vocab)

        df = df.groupby(PID_COL, group_keys=False).apply(
            truncate_patient_df,
            max_len=truncation_length,
            background_length=background_length,
            sep_token=vocab["[SEP]"],
        )
        return df

    @staticmethod
    def load_cohort(paths_cfg):
        pids = None
        if paths_cfg.get("cohort"):
            pids = torch.load(join(paths_cfg.cohort, PID_FILE))
        return pids

    @staticmethod
    def _check_sorted(df: pd.DataFrame, n_patients: int = 10):
        """Verify abspos sorting within each sampled patient"""
        sample_patients = df[PID_COL].unique()[:n_patients]
        for pid in sample_patients:
            patient_df = df[df[PID_COL] == pid]
            if not patient_df[ABSPOS_COL].is_monotonic_increasing:
                raise ValueError(f"Patient {pid} has unsorted abspos values")

    def _cutoff_data(self, df: pd.DataFrame, cutoff_date: dict) -> pd.DataFrame:
        """Cutoff data after a given date."""
        cutoff_abspos = get_hours_since_epoch(datetime(**cutoff_date))
        df = df[df[ABSPOS_COL] <= cutoff_abspos]
        return df

    @staticmethod
    def _validate_censoring(
        patients: List["PatientData"], censor_dates: pd.Series, logger: logging.Logger
    ) -> None:
        """Validate censoring dates and log basic statistics.

        Args:
            patients: List of patient data objects
            censor_dates: Series with censoring dates indexed by patient ID
            logger: Logger instance
        """
        patient_pids = set(p.pid for p in patients)
        censor_pids = set(censor_dates.index)

        missing_censor_dates = patient_pids - censor_pids
        if missing_censor_dates:
            logger.error(
                f"Missing censor dates for {len(missing_censor_dates)} patients"
            )
            raise ValueError("Some patients are missing censor dates")

        logger.info(f"Censoring validated for {len(patient_pids)} patients")

        # Check for NaN values in censor dates
        nan_censor_dates = censor_dates.isna().sum()
        if nan_censor_dates > 0:
            logger.error(f"Found {nan_censor_dates} NaN values in censor dates")
            raise ValueError("NaN values detected in censor dates")

        logger.info(f"Censoring validated for {len(patient_pids)} patients")
