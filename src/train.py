
import os
import argparse
import csv
import torch
from torch.utils.data import DataLoader, Dataset
import glob
import soundfile as sf
import random
from .features.feature_extraction import FeatureExtractor
from .models.crn import CRN
from .models.cnn import CNN
from .utils.audio import ensure_mono
from .utils.metrics import LossFunction

class AudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, chunk_seconds=3, max_items=None):
        """
        Args:
            data_dir: Path to split directory (e.g., data/voicebank/train)
        """
        self.noisy_files = sorted(glob.glob(os.path.join(data_dir, "noisy", "*.wav")))
        self.clean_files = sorted(glob.glob(os.path.join(data_dir, "clean", "*.wav")))
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds

        if len(self.noisy_files) != len(self.clean_files):
            raise ValueError(
                f"Mismatched dataset pairs under {data_dir}: "
                f"{len(self.noisy_files)} noisy files vs {len(self.clean_files)} clean files."
            )
        if max_items is not None:
            self.noisy_files = self.noisy_files[:max_items]
            self.clean_files = self.clean_files[:max_items]

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy, sr_n = sf.read(self.noisy_files[idx])
        clean, sr_c = sf.read(self.clean_files[idx])

        noisy = ensure_mono(noisy)
        clean = ensure_mono(clean)

        if sr_n != sr_c:
            raise ValueError(
                f"Sample rate mismatch for pair {self.noisy_files[idx]} and {self.clean_files[idx]}: "
                f"{sr_n} vs {sr_c}."
            )
        if sr_n != self.sample_rate:
            raise ValueError(
                f"Expected {self.sample_rate} Hz audio, but found {sr_n} Hz in {self.noisy_files[idx]}."
            )
        
        # Normalize?
        # Usually float32 is already normalized -1 to 1.
        
        # Pad/Crop
        # For training, usually fixed length chunks are used.
        # Let's say 3 seconds.
        chunk_len = int(self.chunk_seconds * self.sample_rate)
        if len(noisy) < chunk_len:
            pad = chunk_len - len(noisy)
            noisy = torch.nn.functional.pad(torch.from_numpy(noisy), (0, pad))
            clean = torch.nn.functional.pad(torch.from_numpy(clean), (0, pad))
        else:
            # Random crop
            start = random.randint(0, len(noisy) - chunk_len)
            noisy = torch.from_numpy(noisy[start:start+chunk_len])
            clean = torch.from_numpy(clean[start:start+chunk_len])
            
        return noisy.float(), clean.float()

def train(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Loaders
    train_ds = AudioDataset(
        os.path.join(args.data_root, "train"),
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        max_items=args.max_items,
    )
    if len(train_ds) == 0:
        raise FileNotFoundError(
            f"No training WAV files were found under {args.data_root}/train/{{noisy,clean}}."
        )
    print(f"Training pairs: {len(train_ds)}")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    # Models
    feature_extractor = FeatureExtractor().to(device)
    crn = CRN().to(device)
    cnn = CNN().to(device)
    
    optimizer = torch.optim.Adam(list(crn.parameters()) + list(cnn.parameters()), lr=args.lr)
    criterion = LossFunction().to(device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    loss_csv = None
    loss_writer = None
    if args.loss_csv:
        loss_csv_dir = os.path.dirname(args.loss_csv)
        if loss_csv_dir:
            os.makedirs(loss_csv_dir, exist_ok=True)
        loss_csv = open(args.loss_csv, "w", newline="", encoding="utf-8")
        loss_writer = csv.DictWriter(
            loss_csv,
            fieldnames=["epoch", "step", "total_steps", "loss", "average_loss", "record_type"],
        )
        loss_writer.writeheader()
    
    try:
        for epoch in range(args.epochs):
            crn.train()
            cnn.train()
            total_loss = 0
            steps = 0
            
            for i, (noisy, clean) in enumerate(train_loader):
                noisy, clean = noisy.to(device), clean.to(device)
                
                # Forward
                # 1. Features
                noisy_mag, noisy_phase = feature_extractor(noisy) # (B, 1, T, F), (B, 1, T, F)
                clean_mag, clean_phase = feature_extractor(clean) # (B, 1, T, F), (B, 1, T, F)
                
                # Prepare clean complex ref
                # Clean Complex = clean_mag * exp(j * clean_phase)
                # Or just use the output of inverse? inverse takes compressed mag.
                # Let's assume loss computed on Compressed Mag and Resynthesized Complex.
                
                # 2. CRN (Stage 1)
                mag_mask = crn(noisy_mag)
                enhanced_mag = noisy_mag * mag_mask
                
                # 3. Intermediate Reconstruction
                # coarse_enhanced_complex = (enhanced_mag, noisy_phase)
                # But CNN takes (Real, Imag)
                
                # Convert (mag, phase) -> (Real, Imag)
                # Note: feature_extractor.inverse does decompression
                # We need the compressed complex representation for CNN? 
                # Or the decompressed?
                # Typically CNN operates on compressed feature domain or decompressed.
                # Let's assume it operates on the same domain as features (compressed mag).
                # So we create "Coarse Complex" from (enhanced_mag, noisy_phase)
                
                coarse_real = enhanced_mag * torch.cos(noisy_phase)
                coarse_imag = enhanced_mag * torch.sin(noisy_phase)
                cnn_input = torch.cat([coarse_real, coarse_imag], dim=1) # (B, 2, T, F)
                
                # 4. CNN (Stage 2)
                complex_mask = cnn(cnn_input) # (B, 2, T, F)
                
                # 5. Final Enhance
                # Additive or Multiplicative mask?
                # "cIRM" is multiplicative. "Residual" is additive.
                # Assuming Mask * Input (Multiplicative in complex domain) or Additive?
                # Complex multiplication: (a+bi)*(c+di) = (ac-bd) + i(ad+bc).
                # Let's assume Mask is Complex ratio.
                # enhanced_complex = cnn_input * complex_mask (Complex Mul)
                
                m_r = complex_mask[:, 0:1]
                m_i = complex_mask[:, 1:2]
                i_r = coarse_real
                i_i = coarse_imag
                
                final_real = i_r * m_r - i_i * m_i
                final_imag = i_r * m_i + i_i * m_r
                
                # Loss
                # We need Reference Complex (Clean) in the SAME domain (Compressed Mag + Phase)
                ref_real = clean_mag * torch.cos(clean_phase)
                ref_imag = clean_mag * torch.sin(clean_phase)
                ref_complex = torch.cat([ref_real, ref_imag], dim=1)
                
                est_complex = torch.cat([final_real, final_imag], dim=1)
                
                loss = criterion(enhanced_mag, clean_mag, est_complex, ref_complex)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_value = loss.item()
                total_loss += loss_value
                steps += 1
                
                if loss_writer is not None:
                    loss_writer.writerow({
                        "epoch": epoch + 1,
                        "step": i + 1,
                        "total_steps": len(train_loader),
                        "loss": f"{loss_value:.8f}",
                        "average_loss": "",
                        "record_type": "step",
                    })
                
                if (i+1) % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], "
                        f"Loss: {loss_value:.4f}",
                        end='\r',
                    )
                if args.max_steps_per_epoch is not None and steps >= args.max_steps_per_epoch:
                    break
                    
            average_loss = total_loss / max(steps, 1)
            print(f"\nEpoch [{epoch+1}/{args.epochs}] Average Loss: {average_loss:.4f}")
            if loss_writer is not None:
                loss_writer.writerow({
                    "epoch": epoch + 1,
                    "step": "",
                    "total_steps": len(train_loader),
                    "loss": "",
                    "average_loss": f"{average_loss:.8f}",
                    "record_type": "epoch",
                })
                loss_csv.flush()
            
            # Save checkpoint
            torch.save({
                'crn': crn.state_dict(),
                'cnn': cnn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': vars(args),
            }, os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pth"))
    finally:
        if loss_csv is not None:
            loss_csv.close()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join("data", "voicebank"))
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loss-csv", default=None, help="Optional path to write per-step and per-epoch losses as CSV.")
    return parser

if __name__ == "__main__":
    train(build_parser().parse_args())
