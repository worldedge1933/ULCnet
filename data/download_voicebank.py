import argparse
import hashlib
import math
import shutil
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm


BASE_URL = "https://datashare.ed.ac.uk"
HANDLE_URL = "https://doi.org/10.7488/ds/2117"

DATASET_VARIANTS = {
    "28spk": {
        "train_clean": {
            "name": "clean_trainset_28spk_wav.zip",
            "path": "/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip?sequence=2&isAllowed=y",
            "md5": "d2d5a45ec32f8fcbf201bde0447e20ba",
        },
        "train_noisy": {
            "name": "noisy_trainset_28spk_wav.zip",
            "path": "/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip?sequence=6&isAllowed=y",
            "md5": "1fca9e8bafb8cd069f6653c6d92f9e9c",
        },
    },
    "56spk": {
        "train_clean": {
            "name": "clean_trainset_56spk_wav.zip",
            "path": "/bitstream/handle/10283/2791/clean_trainset_56spk_wav.zip?sequence=3&isAllowed=y",
            "md5": "c351cbc9db30e41686c988c19f0e1475",
        },
        "train_noisy": {
            "name": "noisy_trainset_56spk_wav.zip",
            "path": "/bitstream/handle/10283/2791/noisy_trainset_56spk_wav.zip?sequence=7&isAllowed=y",
            "md5": "b26382542716642b8f2845880af69cf0",
        },
    },
    "common": {
        "test_clean": {
            "name": "clean_testset_wav.zip",
            "path": "/bitstream/handle/10283/2791/clean_testset_wav.zip?sequence=1&isAllowed=y",
            "md5": "34eb1c0ba7ef667e9b966866c542fc16",
        },
        "test_noisy": {
            "name": "noisy_testset_wav.zip",
            "path": "/bitstream/handle/10283/2791/noisy_testset_wav.zip?sequence=5&isAllowed=y",
            "md5": "fb1b86caa31e8ba5b506c0c64da9aab5",
        },
    },
}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response, destination.open("wb") as handle:
        total = int(response.headers.get("Content-Length", 0))
        progress = tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {destination.name}",
        )
        try:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                progress.update(len(chunk))
        finally:
            progress.close()


def ensure_archive(spec: dict, raw_dir: Path) -> Path:
    archive_path = raw_dir / spec["name"]
    if archive_path.exists():
        current_md5 = md5sum(archive_path)
        if current_md5 == spec["md5"]:
            print(f"Using cached archive: {archive_path}")
            return archive_path
        archive_path.unlink()
    download_file(f"{BASE_URL}{spec['path']}", archive_path)
    current_md5 = md5sum(archive_path)
    if current_md5 != spec["md5"]:
        raise ValueError(
            f"MD5 mismatch for {archive_path.name}: expected {spec['md5']}, got {current_md5}."
        )
    return archive_path


def extract_archive(archive_path: Path, extract_root: Path) -> Path:
    target_dir = extract_root / archive_path.stem
    marker = target_dir / ".complete"
    if marker.exists():
        print(f"Using extracted archive: {target_dir}")
        return target_dir

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(target_dir)

    marker.write_text("ok", encoding="utf-8")
    return target_dir


def find_wav_files(root: Path) -> dict:
    wav_files = sorted(root.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found under {root}")
    mapping = {}
    for wav_path in wav_files:
        if wav_path.name in mapping:
            raise ValueError(f"Duplicate filename detected while indexing {root}: {wav_path.name}")
        mapping[wav_path.name] = wav_path
    return mapping


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported audio shape: {audio.shape!r}")


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    divisor = math.gcd(source_sr, target_sr)
    up = target_sr // divisor
    down = source_sr // divisor
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


def prepare_split(
    split_name: str,
    clean_root: Path,
    noisy_root: Path,
    output_root: Path,
    target_sr: int,
    overwrite: bool,
    limit: Optional[int],
) -> None:
    clean_files = find_wav_files(clean_root)
    noisy_files = find_wav_files(noisy_root)
    names = sorted(set(clean_files) & set(noisy_files))

    if not names:
        raise FileNotFoundError(f"No paired WAV files found for split {split_name}")
    if len(names) != len(clean_files) or len(names) != len(noisy_files):
        raise ValueError(
            f"Unpaired files detected for {split_name}: "
            f"{len(clean_files)} clean, {len(noisy_files)} noisy, {len(names)} paired."
        )
    if limit is not None:
        names = names[:limit]

    split_dir = output_root / split_name
    clean_out = split_dir / "clean"
    noisy_out = split_dir / "noisy"
    clean_out.mkdir(parents=True, exist_ok=True)
    noisy_out.mkdir(parents=True, exist_ok=True)

    for name in tqdm(names, desc=f"Preparing {split_name}", unit="file"):
        clean_target = clean_out / name
        noisy_target = noisy_out / name
        if not overwrite and clean_target.exists() and noisy_target.exists():
            continue

        clean_audio, clean_sr = sf.read(clean_files[name], always_2d=False)
        noisy_audio, noisy_sr = sf.read(noisy_files[name], always_2d=False)

        clean_audio = resample_audio(to_mono(clean_audio), clean_sr, target_sr)
        noisy_audio = resample_audio(to_mono(noisy_audio), noisy_sr, target_sr)

        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]

        sf.write(clean_target, clean_audio, target_sr)
        sf.write(noisy_target, noisy_audio, target_sr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and prepare the official VoiceBank+DEMAND dataset for this project."
    )
    parser.add_argument("--variant", choices=["28spk", "56spk"], default="28spk")
    parser.add_argument("--raw-dir", default=str(Path("data") / "raw" / "voicebank_demand"))
    parser.add_argument("--extract-dir", default=str(Path("data") / "raw" / "voicebank_demand_extracted"))
    parser.add_argument("--output-dir", default=str(Path("data") / "voicebank"))
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=None, help="Prepare only the first N paired files per split.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    extract_dir = Path(args.extract_dir)
    output_dir = Path(args.output_dir)

    print(f"Source dataset: {HANDLE_URL}")
    print(f"Preparing variant: {args.variant}")

    variant_specs = dict(DATASET_VARIANTS["common"])
    variant_specs.update(DATASET_VARIANTS[args.variant])

    archives = {name: ensure_archive(spec, raw_dir) for name, spec in variant_specs.items()}
    extracted = {name: extract_archive(path, extract_dir) for name, path in archives.items()}

    prepare_split(
        "train",
        extracted["train_clean"],
        extracted["train_noisy"],
        output_dir,
        args.target_sr,
        args.overwrite,
        args.limit,
    )
    prepare_split(
        "test",
        extracted["test_clean"],
        extracted["test_noisy"],
        output_dir,
        args.target_sr,
        args.overwrite,
        args.limit,
    )

    print(f"Prepared dataset at {output_dir.resolve()}")


if __name__ == "__main__":
    main()
