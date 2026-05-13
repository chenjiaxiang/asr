import logging
import os
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
from omegaconf import DictConfig

from asr.data.audio.data_loader import AudioDataLoader
from asr.data.audio.dataset import SpeechToTextDataset
from asr.data.sampler import RandomSampler, SmartBatchingSampler
from asr.datasets import register_data_module


@register_data_module("timit")
class LightningTIMITDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the TIMIT corpus.

    TIMIT contains 630 speakers of American English reading phonetically
    balanced sentences.  This module uses the standard train / test split
    (``TRAIN/`` and ``TEST/`` directories) and reserves 10 % of the
    training utterances as a validation set.  SA (dialect) sentences are
    excluded to follow the standard evaluation protocol.

    Audio files are stored in NIST SPHERE format.  During :meth:`prepare_data`
    the preprocessing script converts each SPHERE ``.WAV`` file to a
    standard PCM ``.wav`` file so that librosa can load them; the original
    files are left untouched.

    Args:
        configs: Hydra / OmegaConf configuration object.  Expected keys
            under ``configs.dataset``:

            * ``dataset_path`` – root directory of the TIMIT corpus.
            * ``manifest_file_path`` – path where the manifest will be
              written / read.

            And under ``configs.tokenizer``:

            * ``unit`` – must be ``"timit_character"``.
            * ``vocab_path`` – path where the vocabulary CSV will be
              written / read.

    Example::

        module = LightningTIMITDataModule(configs)
        module.prepare_data()   # converts WAV files, builds manifest
        module.setup()          # creates SpeechToTextDataset instances
        train_loader = module.train_dataloader()
    """

    # Approximate split sizes derived from the standard TIMIT corpus
    # (4620 TRAIN utterances − 2×462 SA = 3696; 90 % / 10 % split)
    TIMIT_TRAIN_NUM: int = 3326   # floor(3696 * 0.9)
    TIMIT_VALID_NUM: int = 370    # 3696 - 3326
    TIMIT_TEST_NUM: int = 1344    # TEST set minus SA sentences

    def __init__(self, configs: DictConfig) -> None:
        super().__init__()
        self.configs = configs
        self.dataset: Dict[str, SpeechToTextDataset] = {}
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _parse_manifest_file(
        self, manifest_file_path: str
    ) -> Tuple[List[str], List[str]]:
        """Read a tab-separated manifest file into parallel lists.

        Each line is expected to have the form::

            <audio_path>\\t<transcript>\\t<label_ids>

        Args:
            manifest_file_path: Path to the manifest file.

        Returns:
            A tuple ``(audio_paths, transcripts)``.
        """
        audio_paths: List[str] = []
        transcripts: List[str] = []

        with open(manifest_file_path) as f:
            for line in f:
                audio_path, _, transcript = line.split("\t")
                transcripts.append(transcript.rstrip("\n"))
                audio_paths.append(audio_path)

        return audio_paths, transcripts

    # ------------------------------------------------------------------
    # Lightning interface
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Convert SPHERE WAV files and generate the manifest if needed.

        This method is called once per machine before :meth:`setup`.  It
        triggers SPHERE-to-WAV conversion for every training and test
        utterance (skipping files that have already been converted) and
        writes the tab-separated manifest file.

        Raises:
            ValueError: If the configured tokenizer unit is not
                ``"timit_character"``.
        """
        if self.configs.tokenizer.unit != "timit_character":
            raise ValueError(
                f"Unsupported tokenizer unit '{self.configs.tokenizer.unit}'. "
                "TIMIT data module only supports 'timit_character'."
            )

        from asr.datasets.timit.preprocess.character import generate_manifest_files

        if not os.path.exists(self.configs.dataset.manifest_file_path):
            self.logger.info(
                "Manifest file not found. Generating manifest files…"
            )
            generate_manifest_files(
                dataset_path=self.configs.dataset.dataset_path,
                manifest_file_path=self.configs.dataset.manifest_file_path,
                vocab_path=self.configs.tokenizer.vocab_path,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate :class:`SpeechToTextDataset` for each split.

        The manifest is read once and sliced by the pre-computed split
        boundaries.  Spec-augment and silence deletion are applied only
        to the training split.

        Args:
            stage: Ignored; all three splits are always prepared.
        """
        audio_paths, transcripts = self._parse_manifest_file(
            self.configs.dataset.manifest_file_path
        )

        valid_end = self.TIMIT_TRAIN_NUM + self.TIMIT_VALID_NUM

        split_audio = {
            "train": audio_paths[: self.TIMIT_TRAIN_NUM],
            "valid": audio_paths[self.TIMIT_TRAIN_NUM : valid_end],
            "test": audio_paths[valid_end:],
        }
        split_transcripts = {
            "train": transcripts[: self.TIMIT_TRAIN_NUM],
            "valid": transcripts[self.TIMIT_TRAIN_NUM : valid_end],
            "test": transcripts[valid_end:],
        }

        for split in ("train", "valid", "test"):
            self.dataset[split] = SpeechToTextDataset(
                configs=self.configs,
                dataset_path=self.configs.dataset.dataset_path,
                audio_paths=split_audio[split],
                transcripts=split_transcripts[split],
                apply_spec_augment=(
                    self.configs.audio.apply_spec_augment if split == "train" else False
                ),
                del_silence=(
                    self.configs.audio.del_silence if split == "train" else False
                ),
            )

    def train_dataloader(self) -> AudioDataLoader:
        """Return a DataLoader for the training split.

        Returns:
            An :class:`~asr.data.audio.data_loader.AudioDataLoader` backed
            by either a :class:`~asr.data.sampler.SmartBatchingSampler` or
            a :class:`~asr.data.sampler.RandomSampler` depending on the
            trainer configuration.
        """
        sampler_cls = (
            SmartBatchingSampler
            if self.configs.trainer.sampler == "smart"
            else RandomSampler
        )
        train_sampler = sampler_cls(
            data_source=self.dataset["train"],
            batch_size=self.configs.trainer.batch_size,
        )
        return AudioDataLoader(
            dataset=self.dataset["train"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=train_sampler,
        )

    def val_dataloader(self) -> AudioDataLoader:
        """Return a DataLoader for the validation split.

        Returns:
            An :class:`~asr.data.audio.data_loader.AudioDataLoader`.
        """
        sampler_cls = (
            SmartBatchingSampler
            if self.configs.trainer.sampler == "smart"
            else RandomSampler
        )
        valid_sampler = sampler_cls(
            data_source=self.dataset["valid"],
            batch_size=self.configs.trainer.batch_size,
        )
        return AudioDataLoader(
            dataset=self.dataset["valid"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=valid_sampler,
        )

    def test_dataloader(self) -> AudioDataLoader:
        """Return a DataLoader for the test split.

        Returns:
            An :class:`~asr.data.audio.data_loader.AudioDataLoader`.
        """
        sampler_cls = (
            SmartBatchingSampler
            if self.configs.trainer.sampler == "smart"
            else RandomSampler
        )
        test_sampler = sampler_cls(
            data_source=self.dataset["test"],
            batch_size=self.configs.trainer.batch_size,
        )
        return AudioDataLoader(
            dataset=self.dataset["test"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=test_sampler,
        )
