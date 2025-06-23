import logging
import os
import shutil
import tarfile
from typing import Optional, Tuple, List

import pytorch_lightning as pl
import wget
from omegaconf import DictConfig

from asr.data.audio.data_loader import AudioDataLoader
from asr.data.audio.dataset import SpeechToTextDataset
from asr.data.sampler import RandomSampler, SmartBatchingSampler
from asr.datasets import register_data_module
from asr.tokenizers.tokenizer import Tokenizer


@register_data_module("librispeech")
class LightningLibriSpeechDataModule(pl.LightningDataModule):
    LIBRISPEECH_TRAIN_NUM = 281241
    LIBRISPEECH_VALID_NUM = 5567
    LIBRISPEECH_TEST_NUM = 5559
    LIBRISPECCH_PARTS = [
        "dev-clean",
        "test-clean",
        "dev-other",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]

    def __init__(self, configs: DictConfig) -> None:
        super(LightningLibriSpeechDataModule, self).__init__()
        self.configs = configs
        self.dataset = dict()
        self.logger = logging.getLogger(__name__)

    def _parse_manifest_file(self, manifest_file_path: str) -> Tuple[List[str], List[str]]:
        """Parsing manifest file"""
        audio_paths = list()
        transcripts = list()

        with open(manifest_file_path, "r") as f:
            for idx, line in enumerate(f.readlines()):
                audio_path, _, transcript = line.split("\t")
                transcript = transcript.replace("\n", "")

                audio_paths.append(audio_path)
                transcripts.append(transcript)

        return audio_paths, transcripts

    def _download_datasets(self) -> None:
        # base_url = "http://www.openslr.org/resource/12"
        base_url = "https://openslr.magicdatatech.com/resources/12"
        train_dir = "LibriSpeech/train-960"

        # print(self.configs.dataset.dataset_path)
        # import pdb
        # pdb.set_trace()

        if not os.path.exists(self.configs.dataset.dataset_path):
            os.mkdir(self.configs.dataset.dataset_path)

        for part in self.LIBRISPECCH_PARTS:
            self.logger.info(f"LibriSpeech-{part} download...")
            url = f"{base_url}/{part}.tar.gz"
            # wget.download(url)
            # shutil.move(f"{part}.tar.gz", os.path.join(self.configs.dataset.dataset_path, f"{part}.tar.gz"))

            self.logger.info(f"Un-tarring archive {self.configs.dataset.dataset_path}/{part}.tar.gz")
            # tar = tarfile.open(f"{self.configs.dataset.dataset_path}/{part}.tar.gz", mode="r:gz")
            # tar.extractall(self.configs.dataset.dataset_path)
            # tar.close()
            # os.remove(f"{self.configs.dataset.dataset_path}")

        self.logger.info("Merge all train packs into one.")

        if not os.path.exists(self.configs.dataset.dataset_path):
            os.mkdir(os.path.join(self.configs.dataset.dataset_path, train_dir))

        for part in self.LIBRISPECCH_PARTS[-3:]:  # train
            path = os.path.join(self.configs.dataset.dataset_path, "LibriSpeech", part)
            subfolders = os.listdir(path)
            for subfolder in subfolders:
                shutil.move(
                    os.path.join(path, subfolder),
                    os.path.join(self.configs.dataset.dataset_path, train_dir, subfolder),
                )

    def prepare_data(self) -> None:
        if self.configs.tokenizer.unit == "libri_subword":
            from asr.datasets.librispeech.preprocess.subword import generate_manifest_files
        elif self.configs.tokenizer.unit == "libri_character":
            from asr.datasets.librispeech.preprocess.character import generate_manifest_files
        else:
            raise ValueError(f"Unsupported vocabulary unit: {self.configs.tokenizer.unit}. Support unit: (libri_subword, libri_character)")

        if self.configs.dataset.dataset_download:
            self._download_datasets()

        if not os.path.exists(self.configs.dataset.manifest_file_path):
            self.logger.info("Mainfest file is not exists !!\n" "Generate mainfest files...")

            if hasattr(self.configs.tokenizer, "vocab_size"):
                generate_manifest_files(
                    dataset_path=self.configs.dataset.dataset_path,
                    manifest_file_path=self.configs.dataset.manifest_file_path,
                    vocab_path=self.configs.tokenizer.vocab_path,
                    vocab_size=self.configs.tokenizer.vocab_size,
                )
            else:
                generate_manifest_files(
                    dataset_path=self.configs.dataset.dataset_path,
                    mainfest_file_path=self.configs.dataset.manifest_file_path,
                    vocab_path=self.configs.tokenizer.vocab_path,
                )
        
    def setup(self, stage: Optional[str] = None) -> None:
        r"""Split dataset into train, valid, and test."""
        valid_end_idx = self.LIBRISPEECH_TRAIN_NUM + self.LIBRISPEECH_VALID_NUM
        audio_paths, trascripts = self._parse_manifest_file(self.configs.dataset.manifest_file_path)
        
        audio_paths = {
            "train": audio_paths[: self.LIBRISPEECH_TRAIN_NUM],
            "valid": audio_paths[self.LIBRISPEECH_TRAIN_NUM: valid_end_idx],
            "test": audio_paths[valid_end_idx:],
        }
        trascripts = {
            "train": trascripts[: self.LIBRISPEECH_TRAIN_NUM],
            "valid": trascripts[self.LIBRISPEECH_TRAIN_NUM: valid_end_idx],
            "test": trascripts[valid_end_idx:],
        }

        for stage in audio_paths.keys():
            self.dataset[stage] = SpeechToTextDataset(
                configs=self.configs,
                dataset_path=os.path.join(self.configs.dataset.dataset_path, "LibriSpeech"),
                audio_paths=audio_paths[stage],
                transcripts=trascripts[stage],
                apply_spec_augment=self.configs.audio.apply_spec_augment if stage == "train" else False,
                del_silence=self.configs.audio.del_silence if stage == "train" else False,
            )

    def train_dataloader(self) -> AudioDataLoader:
        sampler = SmartBatchingSampler if self.configs.trainer.sampler == "smart" else RandomSampler
        train_sampler = sampler(data_source=self.dataset["train"], batch_size=self.configs.trainer.batch_size)
        return AudioDataLoader(
            dataset=self.dataset["train"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=train_sampler,
        )

    def val_dataloader(self) -> AudioDataLoader:
        sampler = SmartBatchingSampler if self.configs.trainer.sampler == "smart" else RandomSampler
        valid_sampler = sampler(data_source=self.dataset["valid"], batch_size=self.configs.trainer.batch_size)
        return AudioDataLoader(
            dataset=self.dataset["valid"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=valid_sampler,
        )

    def test_dataloader(self) -> AudioDataLoader:
        sampler = SmartBatchingSampler if self.configs.trainer.sampler == "smart" else RandomSampler
        test_sampler = sampler(data_source=self.dataset["test"], batch_size=self.configs.trainer.batch_size)
        return AudioDataLoader(
            dataset=self.dataset["test"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=test_sampler,
        )


