from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import torch.utils.data.sampler


def _collate_fn(batch, pad_id: int = 0):
    
    def seq_length_(p):
        return len(p[0])
    
    def target_length_(p):
        return len(p[1])

    # sort by sequence length for rnn.pack_padded_sequence()
    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(pad_id)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)
    target_lengths = torch.IntTensor(target_lengths)

    return seqs, targets, seq_lengths, target_lengths


class AudioDataLoader(DataLoader):
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            num_workers: int,
            batch_sampler: torch.utils.data.sampler.Sampler,
            **kwargs,
    ) -> None:
        super(AudioDataLoader, self).__init__(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            **kwargs,
        )
        self.collate_fn = _collate_fn
    

def load_dataset(manifest_file_path: str) -> Tuple[List, List]:
    audio_paths = list()
    transcripts = list()

    with open(manifest_file_path) as f:
        for idx, line in enumerate(f.readlines()):
            audio_path, korean_transcript, transcript = line.split("\t") # TODO, maybe wrong. Not compatiable with LibriSpeech
            transcript = transcript.replace("\n", "")

            audio_paths.append(audio_path)
            transcripts.append(transcript)
        
    return audio_paths, transcripts