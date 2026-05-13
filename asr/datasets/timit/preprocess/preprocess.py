import os
import struct
import wave
from typing import List, Tuple


def _parse_sphere_header(header_bytes: bytes) -> dict:
    """Parse NIST SPHERE header to extract audio metadata.

    Args:
        header_bytes: Raw 1024-byte SPHERE header.

    Returns:
        Dictionary with keys ``sample_rate``, ``channel_count``,
        ``sample_count``, and ``byte_format`` (``'<'`` little-endian
        or ``'>'`` big-endian).
    """
    header_str = header_bytes.decode("ascii", errors="replace")
    info = {}
    for line in header_str.splitlines():
        parts = line.split()
        if len(parts) == 3:
            key, _type, value = parts
            if key in ("sample_rate", "channel_count", "sample_count"):
                info[key] = int(value)
            elif key == "sample_byte_format":
                # '01' = little-endian, '10' = big-endian
                info["byte_format"] = "<" if value == "01" else ">"
    return info


def convert_sphere_to_wav(sphere_path: str, wav_path: str) -> None:
    """Convert a NIST SPHERE file to a standard PCM WAV file.

    TIMIT audio files use the SPHERE format with a fixed 1024-byte ASCII
    header followed by raw 16-bit PCM samples.  This function reads the
    header to extract metadata and writes a standard WAV file that librosa
    can load directly.

    Args:
        sphere_path: Path to the source SPHERE ``.WAV`` file.
        wav_path: Destination path for the converted ``.wav`` file.
    """
    with open(sphere_path, "rb") as f:
        header_bytes = f.read(1024)
        pcm_data = f.read()

    info = _parse_sphere_header(header_bytes)
    sample_rate = info.get("sample_rate", 16000)
    channel_count = info.get("channel_count", 1)
    byte_format = info.get("byte_format", "<")

    # Re-interpret bytes as int16 samples with the correct endianness
    n_samples = len(pcm_data) // 2
    samples = struct.unpack(f"{byte_format}{n_samples}h", pcm_data)

    with wave.open(wav_path, "w") as wf:
        wf.setnchannels(channel_count)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


def collect_transcripts(dataset_path: str) -> Tuple[List[str], List[str], List[str]]:
    """Walk the TIMIT directory tree and collect utterance metadata.

    Skips SA (dialect) sentences to match the standard TIMIT evaluation
    protocol.  SPHERE ``.WAV`` files are converted to standard ``.wav``
    files on the fly so that librosa can load them; the converted files are
    placed in the same directory as the originals.

    Args:
        dataset_path: Root directory of the TIMIT corpus, containing
            ``TRAIN/`` and ``TEST/`` sub-directories.

    Returns:
        A 3-tuple of parallel lists ``(audio_paths, transcripts, splits)``
        where

        * ``audio_paths`` – paths relative to *dataset_path* with a
          lowercase ``.wav`` extension.
        * ``transcripts`` – upper-cased transcript strings.
        * ``splits`` – one of ``"train"``, ``"valid"``, or ``"test"``
          indicating the canonical split for each utterance.

    Notes:
        The TRAIN set is partitioned 90 / 10 (by utterance) into
        ``"train"`` and ``"valid"`` splits after sorting paths
        deterministically.  All TEST utterances are assigned to
        ``"test"``.
    """
    train_entries: List[Tuple[str, str]] = []
    test_entries: List[Tuple[str, str]] = []

    for split_dir, target_list in [("TRAIN", train_entries), ("TEST", test_entries)]:
        split_root = os.path.join(dataset_path, split_dir)
        if not os.path.isdir(split_root):
            continue

        for dialect_dir in sorted(os.listdir(split_root)):
            dialect_path = os.path.join(split_root, dialect_dir)
            if not os.path.isdir(dialect_path):
                continue

            for speaker_dir in sorted(os.listdir(dialect_path)):
                speaker_path = os.path.join(dialect_path, speaker_dir)
                if not os.path.isdir(speaker_path):
                    continue

                for filename in sorted(os.listdir(speaker_path)):
                    if not filename.endswith(".TXT"):
                        continue
                    utt_id = filename[:-4]  # strip .TXT
                    # Skip SA (dialect) sentences
                    if utt_id.startswith("SA"):
                        continue

                    txt_path = os.path.join(speaker_path, filename)
                    sphere_path = os.path.join(speaker_path, utt_id + ".WAV")
                    wav_path = os.path.join(speaker_path, utt_id + ".wav")

                    # Convert SPHERE -> standard WAV if needed
                    if os.path.isfile(sphere_path) and not os.path.isfile(wav_path):
                        convert_sphere_to_wav(sphere_path, wav_path)

                    if not os.path.isfile(wav_path):
                        continue

                    with open(txt_path) as f:
                        line = f.readline().strip()
                    # Format: "<start> <end> <transcript>"
                    tokens = line.split(maxsplit=2)
                    if len(tokens) < 3:
                        continue
                    transcript = tokens[2].upper()

                    rel_path = os.path.join(
                        split_dir, dialect_dir, speaker_dir, utt_id + ".wav"
                    )
                    target_list.append((rel_path, transcript))

    # Sort deterministically so the 90/10 split is reproducible
    train_entries.sort(key=lambda x: x[0])
    test_entries.sort(key=lambda x: x[0])

    split_idx = int(len(train_entries) * 0.9)
    train_part = train_entries[:split_idx]
    valid_part = train_entries[split_idx:]

    audio_paths: List[str] = []
    transcripts: List[str] = []
    splits: List[str] = []

    for rel_path, transcript in train_part:
        audio_paths.append(rel_path)
        transcripts.append(transcript)
        splits.append("train")

    for rel_path, transcript in valid_part:
        audio_paths.append(rel_path)
        transcripts.append(transcript)
        splits.append("valid")

    for rel_path, transcript in test_entries:
        audio_paths.append(rel_path)
        transcripts.append(transcript)
        splits.append("test")

    return audio_paths, transcripts, splits
