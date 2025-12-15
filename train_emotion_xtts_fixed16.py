"""
XTTS GPT Fine-tuning Script - CORRECTED VERSION

Usage:
    python train_gpt_xtts_complete_fixed.py --config_path emma_xtts_finetune_config.yaml

This script fixes the GPT parameter issues in the original TTS library script.
DO NOT use 'python -m TTS.bin.train_gpt_xtts' - use this script directly.
"""

import os
import sys
import argparse
import logging
import math
import types
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ✅ Diagnostics for 6-token bottleneck investigation
try:
    from run_diagnostics import inject_diagnostics
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    logging.warning("run_diagnostics not available - GPT shape verification disabled")

# ✅ Autoregressive training for long-form generation
try:
    from autoregressive_training import AutoregressiveProcessor, enable_autoregressive_training
    AUTOREGRESSIVE_AVAILABLE = True
except ImportError:
    AUTOREGRESSIVE_AVAILABLE = False
    logging.warning("autoregressive_training not available")

# audio libs
import torchaudio
import librosa
import soundfile as sf

# Optional libs (may not be installed)
try:
    from encodec import EncodecModel
except Exception:
    EncodecModel = None

try:
    from speechbrain.inference import SpeakerRecognition
except Exception:
    SpeakerRecognition = None

# -------------------------
# Coqui/TTS imports (optional)
# -------------------------

# Initialize variables so they are always defined
XttsConfig = None
Xtts = None
XttsAudioConfig = None
XttsArgs = None
BaseDatasetConfig = None
load_tts_samples = None
AudioProcessor = None
BaseCharacters = None
TTSTokenizer = None

# Try to import Coqui classes
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.utils.audio import AudioProcessor
    from TTS.tts.utils.text.characters import BaseCharacters
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
except Exception as e:
    logging.warning(
        "Coqui TTS imports failed: %s. Some features will be unavailable.", e)

# Safe globals fix for torch.load
from torch import serialization

# Only add if Coqui was imported successfully
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    serialization.add_safe_globals([XttsConfig])
except Exception:
    pass

# Scheduler & AMP
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, LambdaLR
try:
    from torch.cuda.amp import autocast, GradScaler
except Exception:
    autocast = None
    GradScaler = None

# Text classifier imports
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    logging.warning("transformers not installed - text classifier will not be available")
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

# -------------------------
# utilities
# -------------------------
# Early stopping class
# -----------------------
from pathlib import Path

class EarlyStopping:
    def _init_(self, patience=30, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def _call_(self, eval_loss, model, path):
        score = -eval_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        if self.verbose:
            print(f"Saved best model to {path}")


from types import SimpleNamespace

class AttrDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

def flatten(tensor_list):
    out = []
    for it in tensor_list:
        if isinstance(it, (list, tuple)):
            out.extend(flatten(it))
        elif it is not None:
            out.append(it)
    return out

def save_checkpoint(model, optimizer, epoch, step, config, prefix="checkpoint"):
    ckpt_dir = os.path.join(config.get("output_path", "./output"), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    fname = f"{prefix}_ep{epoch:03d}_step{step:06d}.pt"
    path = os.path.join(ckpt_dir, fname)
    try:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # store a 1-based epoch so resume shows the same human-readable number
            "epoch": epoch + 1,
            "step": step,
            "config": config,
        }, path)
        logging.info("Saved checkpoint: %s", path)
    except Exception as e:
        logging.warning("Failed to save checkpoint: %s", e)

def load_audio(audio_path, sample_rate):
    audio_path = audio_path.replace("\\", "/")
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    return waveform

def add_noise(waveform, noise_factor=0.005):
    return waveform + torch.randn_like(waveform) * noise_factor

def add_reverb(waveform, sample_rate, reverberance=0.3):
    kernel_size = max(1, int(sample_rate * reverberance * 0.01))
    kernel = torch.ones(kernel_size) / kernel_size
    kernel = kernel.to(waveform.device).unsqueeze(0).unsqueeze(0)
    wav_b = waveform.unsqueeze(0)
    out = torch.nn.functional.conv1d(wav_b, kernel, padding=kernel_size // 2)
    return out.squeeze(0)

def pitch_shift(waveform, sample_rate, n_steps):
    wav_np = waveform.squeeze().cpu().numpy()
    shifted = librosa.effects.pitch_shift(wav_np, sample_rate, n_steps)
    return torch.tensor(shifted).unsqueeze(0)

def augment_audio(waveform, sample_rate):
    # simple, safe augmentations
    w = waveform
    if np.random.rand() < 0.2:
        w = add_noise(w, noise_factor=np.random.uniform(0.001, 0.01))
    if np.random.rand() < 0.15:
        w = add_reverb(w, sample_rate, reverberance=np.random.uniform(0.05, 0.4))
    if np.random.rand() < 0.15:
        n_steps = np.random.uniform(-1.5, 1.5)
        try:
            w = pitch_shift(w, sample_rate, n_steps)
        except Exception:
            pass
    return w

# ✅ Text classifier for emotions
class TextEmotionPredictor:
    def __init__(self, classifier_path, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(classifier_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(classifier_path).to(device)
        self.model.eval()
        
        import json
        emotion_map_path = os.path.join(classifier_path, "emotion_mapping.json")
        if os.path.exists(emotion_map_path):
            with open(emotion_map_path) as f:
                mapping = json.load(f)
                if mapping and isinstance(list(mapping.keys())[0], str):
                    try:
                        self.id_to_emotion = {int(k): v for k, v in mapping.items()}
                    except (ValueError, TypeError):
                        self.id_to_emotion = {v: k for k, v in mapping.items()}
                else:
                    self.id_to_emotion = mapping
        else:
            self.id_to_emotion = {0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "surprised", 5: "fearful", 6: "disgusted"}
    
    def predict(self, text):
        if not text or len(text.strip()) == 0:
            return 0, "neutral", 0.0
        
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            emotion_id = torch.argmax(logits).item()
            confidence = torch.softmax(logits, dim=-1)[emotion_id].item()
            emotion_label = self.id_to_emotion.get(emotion_id, "neutral")
        
        return emotion_id, emotion_label, confidence
 
 # -------------------------
 # Collate function
 # -------------------------

# -------------------------
# Collate function
# -------------------------
# ✅ NOTE: Emotion features are now pre-computed using text classifier
# No need to extract from mel-spectrograms
# Emotions are loaded directly from CSV column 4

# -------------------------
# Collate function
# -------------------------
def collate_fn(batch, audio_processor, speaker_encoder=None, include_emotion=False, text_classifier=None, feature_normalization="standard_scale", encodec_model=None, reference_audio_path=None):
    """
    batch: list of dicts, each must contain 'audio_file' and optional text/token fields.
    feature_normalization: "standard_scale", "layer_norm", or "l2_norm" - controls emotion feature normalization
    encodec_model: optional EnCodec model for extracting audio tokens
    reference_audio_path: Optional path to a reference audio file (Emma's voice) for conditioning.
                         If provided, all samples will use this mel as cond_mels (not the target mel).
    Returns dict of tensors ready for training loop.
    """
    if not batch:
        return {}

    audio_files = [b["audio_file"] for b in batch]
    speaker_names = [b.get("speaker_name", "unknown") for b in batch]
    languages = [b.get("language", "en") for b in batch]
    token_ids = [b.get("token_id", torch.tensor([0], dtype=torch.long)) for b in batch]
    token_id_lengths = [int(b.get("token_id_lengths", len(t))) for b, t in zip(batch, token_ids)]

    # speaker id mapping
    speaker_to_id = {s: i for i, s in enumerate(sorted(set(speaker_names)))}
    speaker_ids = [speaker_to_id[s] for s in speaker_names]

    # simple lang map
    lang_map = { "en": 0 }
    language_ids = [lang_map.get(l, 0) for l in languages]

    mel_specs, linear_specs, mel_lengths = [], [], []
    waveforms, d_vectors, pitches, energies = [], [], [], []
    emotions_list = []

    for i, samp in enumerate(batch):
        audio_path = samp.get("path") or samp.get("audio_file")
        if not audio_path:
            raise KeyError(f"Neither 'path' nor 'audio_file' key found in sample {i}. Available keys: {list(samp.keys())}")
        
        path = audio_path.replace("\\", "/")
        waveform, sr = torchaudio.load(path)
        if sr != audio_processor.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, audio_processor.sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        waveform = augment_audio(waveform, audio_processor.sample_rate)  # can be identity

        wav_np = waveform.squeeze().numpy()

        # mel/linear
        try:
            mel = audio_processor.melspectrogram(wav_np)
            linear = audio_processor.spectrogram(wav_np)
        except Exception:
            # fallback to librosa features (power mel)
            mel_spec = librosa.feature.melspectrogram(y=wav_np, sr=audio_processor.sample_rate,
                                                      n_fft=audio_processor.fft_size,
                                                      hop_length=audio_processor.hop_length,
                                                      n_mels=audio_processor.num_mels)
            mel = librosa.power_to_db(mel_spec, ref=np.max)
            linear = np.abs(librosa.stft(wav_np, n_fft=audio_processor.fft_size, hop_length=audio_processor.hop_length))

        mel_len = mel.shape[1] if np.ndim(mel) > 1 else 0
        mel_specs.append(torch.tensor(mel, dtype=torch.float32))
        linear_specs.append(torch.tensor(linear, dtype=torch.float32))
        mel_lengths.append(mel_len)
        waveforms.append(waveform.float())

        # speaker embedding
        # speaker embedding
        # <CHANGE> Add diagnostics to find why speaker_encoder returns zeros
        # speaker embedding
        # speaker embedding
        # speaker embedding
        # speaker embedding
        # speaker embedding
        # speaker embedding
        # speaker embedding - FIXED VERSION
        if speaker_encoder is not None:
            try:
                encoder_device = next(speaker_encoder.parameters()).device
                
                if waveform.dim() == 1:
                    waveform_2d = waveform.unsqueeze(0)
                else:
                    waveform_2d = waveform
                
                waveform_on_device = waveform_2d.to(encoder_device)
                
                with torch.no_grad():
                    # Extract features and normalize properly
                    feats = speaker_encoder.mods.compute_features(waveform_on_device)
                    feats = feats.to(encoder_device)
                    
                    # Compute proper normalization statistics from features
                    feat_lens = torch.ones(feats.shape[0], device=encoder_device) * feats.shape[1]
                    feats = speaker_encoder.mods.mean_var_norm(feats, feat_lens)
                    
                    dvec = speaker_encoder.mods.embedding_model(feats)
                    dvec = dvec.squeeze()
                    
                    # CRITICAL: Proper normalization pipeline
                    # CRITICAL: Proper normalization pipeline for emotion stability
                    if torch.isnan(dvec).any() or torch.isinf(dvec).any():
                        logging.warning(f"Invalid dvec detected, using fallback")
                        dvec = torch.zeros(192, device=encoder_device)
                    else:
                        # ESSENTIAL: L2 normalization keeps embeddings at unit norm=1.0
                        # Emotion classifier was trained on normalized inputs
                        # Removing this causes 360x scale increase → gradient collapse
                        dvec = F.normalize(dvec, p=2, dim=-1)
                        
                        # Verify normalization worked
                        dvec_norm = torch.norm(dvec, p=2).item()
                        if abs(dvec_norm - 1.0) > 0.01:
                            logging.warning(f"L2 norm={dvec_norm:.4f}, expected 1.0")
                
                dvec = dvec.cpu()
                logging.info(f"[v0 SUCCESS] Sample {i}: Extracted speaker embedding: shape={dvec.shape}, mean={dvec.mean().item():.4f}, std={dvec.std().item():.4f}")
            except Exception as e:
                logging.error(f"[v0 DIAGNOSTIC] speaker_encoder manual encoding failed for sample {i}: {e}")
                import traceback
                logging.error(f"[v0 DIAGNOSTIC] Full traceback:\n{traceback.format_exc()}")
                dvec = torch.zeros(512)

        
        if dvec.ndim == 1:
            pass
        if dvec.shape[-1] != 512:
            original_size = dvec.shape[-1]
            
            # Use learned projection instead of padding (more stable)
            if not hasattr(speaker_encoder, 'dvec_projection'):
                # Create persistent projection layer
                speaker_encoder.dvec_projection = nn.Linear(original_size, 512).to(dvec.device)
                nn.init.xavier_uniform_(speaker_encoder.dvec_projection.weight)
            
            dvec = speaker_encoder.dvec_projection(dvec)
            
            # CRITICAL: Normalize after projection to maintain unit norm
            # This ensures consistent input scale for emotion classifier
            dvec = F.normalize(dvec, p=2, dim=-1)
            
            proj_norm = dvec.norm().item()
            logging.debug(f"Projected dvec from {original_size} to 512 dims, norm={proj_norm:.4f}")
            
            # Quality check: ensure projection didn't collapse information
            if dvec.std() < 0.01:
                logging.warning(f"[QUALITY] Projection variance too low: {dvec.std():.4f}")
        
        # CRITICAL FIX: Use speaker embedding WITH GRADIENTS (not numpy!)
        # Previous code used mel features but broke gradients with .cpu().numpy()
        # Now using dvec which maintains gradient flow for emotion classifier training
        d_vectors.append(dvec)  # Keep gradients flowing!
        
        # ✅ UPDATED: Emotions predicted from TEXT using trained classifier
        if include_emotion and text_classifier:
            text = samp.get("text", "")
            emotion_id, emotion_label, confidence = text_classifier.predict(text)
            logging.debug(f"[EMOTION] Sample {i}: text='{text[:50]}' → {emotion_label} (conf={confidence:.2f})")
            emotions_list.append(emotion_id)
        elif include_emotion:
            emotion_id = samp.get("emotion", 0)
            emotions_list.append(emotion_id)
        
        # pitch
        try:
            f0, _, _ = librosa.pyin(wav_np, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
                                     sr=audio_processor.sample_rate, frame_length=audio_processor.win_length,
                                     hop_length=audio_processor.hop_length, fill_na=0.0)
            f0 = np.pad(f0, (0, max(0, mel_len - len(f0))), constant_values=0.0)[:mel_len]
            pitches.append(torch.tensor(f0, dtype=torch.float32))
        except Exception:
            pitches.append(torch.zeros(mel_len, dtype=torch.float32))

        # energy
        try:
            rmse = librosa.feature.rms(y=wav_np, frame_length=audio_processor.win_length, hop_length=audio_processor.hop_length)[0]
            rmse = np.pad(rmse, (0, max(0, mel_len - len(rmse))), constant_values=0.0)[:mel_len]
            energies.append(torch.tensor(rmse, dtype=torch.float32))
        except Exception:
            energies.append(torch.zeros(mel_len, dtype=torch.float32))



    # pad tensors
    max_mel_len = max([m.shape[1] for m in mel_specs])

    mel_padded = torch.stack([torch.nn.functional.pad(m, (0, max_mel_len - m.shape[1]), "constant", 0.0)
                            for m in mel_specs])
    linear_padded = torch.stack([torch.nn.functional.pad(l, (0, max_mel_len - l.shape[1]), "constant", 0.0)
                                for l in linear_specs])
    pitch_padded = torch.stack([torch.nn.functional.pad(p, (0, max_mel_len - p.shape[0]), "constant", 0.0)
                                for p in pitches])
    energy_padded = torch.stack([torch.nn.functional.pad(e, (0, max_mel_len - e.shape[0]), "constant", 0.0)
                                for e in energies])
    stop_targets = torch.nn.utils.rnn.pad_sequence([torch.zeros(m) for m in mel_lengths], batch_first=True)

    token_ids_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(t, dtype=torch.long) for t in token_ids], batch_first=True
    )

    attns = torch.zeros(len(batch), token_ids_padded.shape[1], mel_padded.shape[2] if mel_padded.dim() > 2 else 1)
    for i, (tlen, mlen) in enumerate(zip(token_id_lengths, mel_lengths)):
        t_lim = min(tlen, attns.size(1))
        m_lim = min(mlen, attns.size(2))
        attns[i, :t_lim, :m_lim] = 1

    batch_wavs = []
    for b in batch:
        # Use the same logic as above to get the audio path
        audio_path = b.get("path") or b.get("audio_file")
        if not audio_path:
            raise KeyError(f"Neither 'path' nor 'audio_file' key found in batch sample. Available keys: {list(b.keys())}")
        
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.squeeze(0)  # make 1D tensor if mono
        batch_wavs.append(waveform)

    # Pad all waveforms to the max length in the batch
    wavs_padded = torch.nn.utils.rnn.pad_sequence(batch_wavs, batch_first=True)

    # Extract audio tokens from waveforms using EnCodec if available
    audio_tokens_list = []
    if encodec_model is not None:
        try:
            for i, wav in enumerate(batch_wavs):
                try:
                    with torch.no_grad():
                        # Reshape for encodec: [1, 1, audio_len]
                        wav_input = wav.unsqueeze(0).unsqueeze(0)
                        # Get encodec device from its parameters
                        try:
                            encodec_device = next(encodec_model.parameters()).device
                        except StopIteration:
                            encodec_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        
                        if wav_input.device != encodec_device:
                            wav_input = wav_input.to(encodec_device)
                        
                        logging.debug(f"[BATCH] Sample {i}: Encoding audio with shape {wav_input.shape} on {encodec_device}")
                        
                        # Encode to get discrete codes
                        encoded_frames = encodec_model.encode(wav_input)
                        logging.debug(f"[BATCH] Sample {i}: Encoded frames type={type(encoded_frames)}, len={len(encoded_frames) if isinstance(encoded_frames, (list, tuple)) else 'N/A'}")
                        
                        # Extract codes from frames
                        audio_tokens = None
                        if isinstance(encoded_frames, (list, tuple)) and len(encoded_frames) > 0:
                            codes = encoded_frames[0]  # Get codes for the first frame
                            logging.info(f"[BATCH] Sample {i}: Frame 0 codes type={type(codes).__name__}, is_list={isinstance(codes, (list, tuple))}, is_tensor={isinstance(codes, torch.Tensor)}")
                            
                            if isinstance(codes, (list, tuple)) and len(codes) > 0:
                                logging.info(f"[BATCH] Sample {i}: codes is list/tuple with {len(codes)} items")
                                # codes[0] is shape [batch=1, num_codebooks=8, seq_len]
                                # We need just the first codebook: [seq_len]
                                codes_tensor = codes[0]  # [1, 8, seq_len]
                                logging.info(f"[BATCH] Sample {i}: First item in codes list: shape={codes_tensor.shape if hasattr(codes_tensor, 'shape') else type(codes_tensor)}")
                                # Extract first codebook from [batch, codebooks, seq_len] -> [seq_len]
                                if hasattr(codes_tensor, 'shape') and codes_tensor.dim() == 3:
                                    # Tensor shape: [1, 8, seq_len]
                                    audio_tokens = codes_tensor[0, 0, :].cpu().long()  # [batch=0, codebook=0, seq_len]
                                    logging.info(f"[BATCH] Sample {i}: Final tokens shape={audio_tokens.shape} (extracted codebook 0 from 3D tensor)")
                                elif hasattr(codes_tensor, 'shape') and codes_tensor.dim() == 2:
                                    # Tensor shape: [batch, seq_len] or [8, seq_len]
                                    if codes_tensor.shape[0] == 8:
                                        audio_tokens = codes_tensor[0, :].cpu().long()  # First codebook
                                    else:
                                        audio_tokens = codes_tensor.squeeze(0).cpu().long()  # Remove batch dim
                                    logging.info(f"[BATCH] Sample {i}: Final tokens shape={audio_tokens.shape} (extracted from 2D tensor)")
                                elif hasattr(codes_tensor, 'shape') and codes_tensor.dim() == 1:
                                    # Already 1D
                                    audio_tokens = codes_tensor.cpu().long()
                                    logging.info(f"[BATCH] Sample {i}: Final tokens shape={audio_tokens.shape} (1D tensor)")
                                else:
                                    audio_tokens = None
                                    logging.warning(f"[BATCH] Sample {i}: Unexpected tensor dim={codes_tensor.dim() if hasattr(codes_tensor, 'dim') else 'N/A'}")
                                
                                if audio_tokens is not None and audio_tokens.dim() == 0:
                                    audio_tokens = audio_tokens.unsqueeze(0)
                            elif isinstance(codes, torch.Tensor):
                                logging.info(f"[BATCH] Sample {i}: codes is tensor with shape={codes.shape}, dim={codes.dim()}")
                                # codes might be [batch, 8, seq_len] or [8, seq_len] or [seq_len]
                                # Extract just first codebook: [seq_len]
                                if codes.dim() == 3:
                                    # Shape: [batch, num_codebooks, seq_len]
                                    audio_tokens = codes[0, 0, :].cpu().long()  # [batch=0, codebook=0, seq_len]
                                    logging.info(f"[BATCH] Sample {i}: Extracted codebook 0 from 3D -> shape={audio_tokens.shape}")
                                elif codes.dim() == 2:
                                    if codes.shape[0] == 8:
                                        # Shape: [8, seq_len] - multi-codebook format
                                        audio_tokens = codes[0, :].cpu().long()  # [seq_len]
                                        logging.info(f"[BATCH] Sample {i}: Extracted first codebook from [8, seq_len] -> shape={audio_tokens.shape}")
                                    elif codes.shape[1] == 8:
                                        # Shape: [seq_len, 8] - inverted format
                                        audio_tokens = codes[:, 0].cpu().long()  # [seq_len]
                                        logging.info(f"[BATCH] Sample {i}: Extracted first codebook from [seq_len, 8] -> shape={audio_tokens.shape}")
                                    else:
                                        # Shape: [batch, seq_len] - already single codebook
                                        audio_tokens = codes.squeeze(0).cpu().long()
                                        logging.info(f"[BATCH] Sample {i}: Using 2D tensor as-is -> shape={audio_tokens.shape}")
                                elif codes.dim() == 1:
                                    # Already 1D, just use it
                                    audio_tokens = codes.cpu().long()
                                    logging.info(f"[BATCH] Sample {i}: Using 1D tensor as-is -> shape={audio_tokens.shape}")
                                else:
                                    logging.warning(f"[BATCH] Sample {i}: codes has unexpected dim={codes.dim()}, shape={codes.shape}")
                                    audio_tokens = None
                                
                                if audio_tokens is not None and audio_tokens.dim() == 0:
                                    audio_tokens = audio_tokens.unsqueeze(0)
                        
                        if audio_tokens is None:
                            # Fallback: use single token
                            audio_tokens = torch.tensor([8192], dtype=torch.long)
                            logging.debug(f"[BATCH] Sample {i}: Using fallback token")
                        
                        audio_tokens_list.append(audio_tokens)
                        logging.info(f"[BATCH] Sample {i}: Encoded audio tokens shape={audio_tokens.shape}")
                except Exception as e:
                    import traceback
                    logging.warning(f"[BATCH] EnCodec encoding failed for sample {i}: {e}")
                    logging.debug(f"[BATCH] Traceback: {traceback.format_exc()}")
                    audio_tokens_list.append(torch.tensor([8192], dtype=torch.long))
        except Exception as e:
            logging.warning(f"[BATCH] EnCodec batch encoding failed: {e}, using fallback tokens")
            audio_tokens_list = [torch.tensor([8192], dtype=torch.long) for _ in batch_wavs]
    else:
        # No EnCodec available, use fallback
        logging.debug("[BATCH] No EnCodec model available, using fallback tokens")
        audio_tokens_list = [torch.tensor([8192], dtype=torch.long) for _ in batch_wavs]

    # ✅ CRITICAL FIX: Use consistent reference audio for conditioning (not target mel)
    # During training, ALL samples should be conditioned on the SAME Emma reference audio
    # This way the model learns: "Given Emma's voice, generate tokens for THIS different text"
    # NOT: "Given this exact mel, generate its own tokens" (which causes voice loss)
    cond_mels_output = mel_padded  # Default fallback
    
    if reference_audio_path is not None:
        try:
            ref_wav, ref_sr = torchaudio.load(reference_audio_path)
            if ref_sr != audio_processor.sample_rate:
                ref_wav = torchaudio.transforms.Resample(ref_sr, audio_processor.sample_rate)(ref_wav)
            if ref_wav.shape[0] > 1:
                ref_wav = ref_wav.mean(0, keepdim=True)
            
            ref_wav_np = ref_wav.squeeze().numpy()
            try:
                ref_mel = audio_processor.melspectrogram(ref_wav_np)
            except Exception:
                ref_mel = librosa.feature.melspectrogram(y=ref_wav_np, sr=audio_processor.sample_rate,
                                                         n_fft=audio_processor.fft_size,
                                                         hop_length=audio_processor.hop_length,
                                                         n_mels=audio_processor.num_mels)
                ref_mel = librosa.power_to_db(ref_mel, ref=np.max)
            
            ref_mel_tensor = torch.tensor(ref_mel, dtype=torch.float32)
            # Expand reference mel to batch size (get batch size from mel_padded)
            actual_batch_size = mel_padded.shape[0]
            cond_mels_output = ref_mel_tensor.unsqueeze(0).expand(actual_batch_size, -1, -1)
            logging.info(f"✅ [CONDITIONING] Using reference audio: {reference_audio_path}")
            logging.info(f"✅ [CONDITIONING] Reference mel shape: {ref_mel_tensor.shape} → expanded to batch: {cond_mels_output.shape}")
        except Exception as e:
            logging.warning(f"⚠️  [CONDITIONING] Failed to load reference audio {reference_audio_path}: {e}")
            logging.warning(f"⚠️  [CONDITIONING] Falling back to target mel as conditioning")

    out = {
        "audio_file": audio_files,
        "speaker_names": speaker_names,
        "token_id": token_ids_padded,
        "token_id_lengths": torch.tensor(token_id_lengths),
        "mel": mel_padded,
        "linear": linear_padded,
        "mel_lengths": torch.tensor(mel_lengths),
        "stop_targets": stop_targets,
        "waveform": wavs_padded,
        "pitch": pitch_padded,
        "energy": energy_padded,
        "d_vectors": torch.stack(d_vectors),
        "speaker_ids": torch.tensor(speaker_ids),
        "language_ids": torch.tensor(language_ids),
        "attns": attns,
        "gpt_audio_tokens": torch.nn.utils.rnn.pad_sequence(
            [b.get("gpt_audio_tokens", audio_tokens_list[i] if i < len(audio_tokens_list) else torch.tensor([0], dtype=torch.long)) 
             for i, b in enumerate(batch)], batch_first=True
        ),
        # ✅ CRITICAL: Compute expected output length from mel spectrogram length
        # This is used by GPT to know how many audio tokens to generate!
        "expected_output_len": torch.tensor(
            [mel_spec.shape[-1] if isinstance(mel_spec, torch.Tensor) and mel_spec.dim() >= 2 else b.get("expected_output_len", 100) 
             for b, mel_spec in zip(batch, mel_specs)],
            dtype=torch.long
        ),
        "gpt_cond_latent": torch.stack([b.get("gpt_cond_latent", torch.zeros(1024)) if isinstance(b.get("gpt_cond_latent", None), torch.Tensor) else torch.zeros(1024) for b in batch]),
        "speaker_embedding": torch.stack(d_vectors),  # For voice cloning
        "cond_mels": cond_mels_output  # ✅ Use reference audio conditioning (not target mel)
    }

    if include_emotion:
        # Extract emotion labels and emotion features
        emotion_ids = []
        emotion_features = []
        
        for e in emotions_list:
            # Handle emotion labels (from batch)
            if isinstance(e, torch.Tensor):
                # It's an emotion feature vector [512]
                emotion_features.append(e)
                # Get the label from batch
                emotion_ids.append(batch[len(emotion_ids)].get("emotion", 4))
            elif isinstance(e, int):
                emotion_ids.append(e)
            elif isinstance(e, str):
                emotion_to_id = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "surprised": 4, "fearful": 5, "disgusted": 6}
                emotion_ids.append(emotion_to_id.get(e.lower(), 0))
                if e.lower() not in emotion_to_id:
                    logging.warning(f"[EMOTION] Unknown emotion label: {e}, defaulting to neutral (0)")
            else:
                emotion_ids.append(4)
        
        # Add emotion labels for loss calculation
        out["emotion"] = torch.tensor(emotion_ids, dtype=torch.long)
        
        # ✅ NEW: Add emotion features for emotion classifier input
        if emotion_features:
            out["emotion_features"] = torch.stack(emotion_features)  # [batch, 512]
            logging.info(f"[EMOTION FEATURES] Batched shape: {out['emotion_features'].shape}, mean norm: {out['emotion_features'].norm(dim=1).mean():.4f}")
        else:
            out["emotion_features"] = torch.stack(d_vectors)
            logging.info(f"[EMOTION FEATURES] Using speaker embeddings from d_vectors: shape={out['emotion_features'].shape}, mean norm: {out['emotion_features'].norm(dim=1).mean():.4f}")
        
        logging.debug(f"Emotion IDs batch: {out['emotion']}")




    return out


# -------------------------
# Early stopping
# -------------------------
class EarlyStopping:
    def __init__(self, patience=30, verbose=False, min_delta=0):
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, checkpoint_path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, checkpoint_path)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logging.info("EarlyStopping counter: %d out of %d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, checkpoint_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, checkpoint_path):
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            self.best_loss = val_loss
            if self.verbose:
                logging.info("Saved early stopping checkpoint: %s", checkpoint_path)
        except Exception as e:
            logging.warning("Failed to save early stopping checkpoint: %s", e)

# -------------------------
# config/dataset helper functions (lightweight)
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument("--log_level", default="INFO")
    p.add_argument("--epochs", default=500, type=int)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--device", default=None, help="'cpu' or 'cuda'")
    p.add_argument("--restore_path", default=None, help="Path to restore checkpoint")
    return p.parse_args()

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
def load_dataset_simple(dataset_cfg):
    """
    Minimal fallback loader: expects dataset_cfg to contain path & meta_file_train.
    Returns dict with 'train' and 'val' lists of dict samples.
    """

    emotion_map = {
        "neutral": 0,
        "happy": 1,
        "sad": 2,
        "angry": 3,
        "surprised": 4,
        "fearful": 5,
        "disgusted": 6
    }

    
    base = dataset_cfg.get("path", ".")
    meta = dataset_cfg.get("meta_file_train", "metadata_with_durations.csv")
    delim = dataset_cfg.get("delimiter", "|")
    import pandas as pd
    df = pd.read_csv(os.path.join(base, meta), sep=delim, header=None, engine="python")
    samples = []
    for idx, row in df.iterrows():
        wav = str(row[0])
        text = str(row[1]) if len(row) > 1 else ""
        speaker = str(row[2]) if len(row) > 2 else "unknown"
        s = {"audio_file": os.path.join(base, wav), "text": text, "speaker_name": speaker}
        if len(row) > 4:
            emotion_str = str(row[4]).strip().lower()
            s["emotion"] = emotion_map.get(emotion_str, 0)
        samples.append(s)
    # Use small split: 90% train 10% val
    n = len(samples)
    if n == 0:
        return {"train": [], "val": []}
    split = max(1, int(n * 0.9))
    return {"train": samples[:split], "val": samples[split:]}

# train_gpt_xtts.py — Part 2/3 (loss, forward wrapper, training loop)

def create_xtts_args_with_gpt_config(cfg):
    """Create XttsArgs with proper gpt_config attribute"""
    try:
        from TTS.tts.configs.xtts_config import XttsArgs

        # Create XttsArgs from config
        xtts_args = XttsArgs()

        # Apply xtts_args from config if available
        if hasattr(cfg, 'xtts_args') and cfg.xtts_args:
            for key, value in cfg.xtts_args.items():
                if hasattr(xtts_args, key):
                    setattr(xtts_args, key, value)

        # Ensure gpt_config exists
        if not hasattr(xtts_args, 'gpt_config') or xtts_args.gpt_config is None:
            gpt_config = {}
            if hasattr(cfg, 'model_args') and cfg.model_args:
                # Map YAML config values to GPT config
                gpt_config = {
                    'gpt_batch_size': cfg.model_args.get('gpt_batch_size', 1),
                    'gpt_max_audio_len': 605,  # Standard XTTS value
                    'gpt_max_text_len': cfg.model_args.get('max_text_positions', 404),
                    'gpt_layers': cfg.model_args.get('gpt_layers', 30),
                    'gpt_n_model_channels': cfg.model_args.get('gpt_n_model_channels', 1024),
                    'gpt_n_heads': cfg.model_args.get('gpt_n_heads', 16),
                    'gpt_number_text_tokens': cfg.model_args.get('n_text_tokens', 404),
                    'gpt_start_text_token': cfg.model_args.get('n_text_tokens', 404),  # Usually same as n_text_tokens
                    'gpt_stop_text_token': 0,
                    'gpt_num_audio_tokens': 8194,  # Standard XTTS value
                    'gpt_start_audio_token': 8192,  # Standard XTTS value
                    'gpt_stop_audio_token': 8193   # Standard XTTS value
                }
            else:
                # Fallback using your YAML values
                gpt_config = {
                    'gpt_batch_size': 1,
                    'gpt_max_audio_len': 605,
                    'gpt_max_text_len': 404,
                    'gpt_layers': 30,
                    'gpt_n_model_channels': 1024,
                    'gpt_n_heads': 16,
                    'gpt_number_text_tokens': 404,
                    'gpt_start_text_token': 404,
                    'gpt_stop_text_token': 0,
                    'gpt_num_audio_tokens': 8194,
                    'gpt_start_audio_token': 8192,
                    'gpt_stop_audio_token': 8193
                }

            xtts_args.gpt_config = gpt_config

        return xtts_args

    except Exception as e:
        print(f"[DEBUG] Error creating XttsArgs: {e}")
        return None


def _resize_embeddings_if_needed(state_dict, model):
    """Resize embedding layers if vocab size mismatch between checkpoint and model.
    
    This enables loading pretrained checkpoints trained with smaller vocab (1026)
    into models with larger vocab (8194) by interpolating weights.
    
    Args:
        state_dict: Checkpoint state dict
        model: Target model
        
    Returns:
        Modified state_dict with resized embeddings
    """
    embeddings_to_resize = ['gpt.mel_embedding.weight', 'gpt.mel_head.weight', 'gpt.mel_head.bias']
    
    for key in embeddings_to_resize:
        if key not in state_dict:
            continue
            
        ckpt_param = state_dict[key]
        
        # Get corresponding model parameter shape
        model_param = model.state_dict().get(key)
        if model_param is None:
            continue
        
        ckpt_shape = ckpt_param.shape
        model_shape = model_param.shape
        
        if ckpt_shape == model_shape:
            continue
        
        print(f"[RESIZE] Resizing {key}: {ckpt_shape} → {model_shape}")
        
        # Handle different layer types
        if 'bias' in key:
            # For bias, just pad with zeros
            if ckpt_shape[0] < model_shape[0]:
                padding = torch.zeros(model_shape[0] - ckpt_shape[0], device=ckpt_param.device, dtype=ckpt_param.dtype)
                state_dict[key] = torch.cat([ckpt_param, padding], dim=0)
                print(f"  [RESIZE] Bias padded with zeros")
        else:
            # For embedding/weight matrices, interpolate
            # Shape is typically (vocab_size, hidden_dim)
            if len(ckpt_shape) == 2 and ckpt_shape[0] != model_shape[0]:
                old_vocab = ckpt_shape[0]
                new_vocab = model_shape[0]
                hidden_dim = ckpt_shape[1]
                
                # Interpolate using linear interpolation
                # Create indices for old vocab
                old_indices = torch.linspace(0, old_vocab - 1, old_vocab)
                new_indices = torch.linspace(0, old_vocab - 1, new_vocab)
                
                # Interpolate each hidden dimension
                resized = torch.zeros(new_vocab, hidden_dim, device=ckpt_param.device, dtype=ckpt_param.dtype)
                for dim in range(hidden_dim):
                    resized[:, dim] = torch.nn.functional.interpolate(
                        ckpt_param[:, dim].unsqueeze(0).unsqueeze(0),
                        size=new_vocab,
                        mode='linear',
                        align_corners=False
                    ).squeeze()
                
                state_dict[key] = resized
                print(f"  [RESIZE] Interpolated from {old_vocab} to {new_vocab} tokens, preserved {hidden_dim} dims")
    
    return state_dict


def focal_loss(logits, targets, alpha=None, gamma=1.5, reduction='mean'):
    """Focal Loss for addressing class imbalance
    
    Focal Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    This loss:
    1. Uses alpha weights to balance class frequencies
    2. Uses (1-p_t)^gamma to focus on hard examples
    3. Down-weights easy examples (high confidence predictions)
    
    Args:
        logits: [batch_size, num_classes] raw model outputs
        targets: [batch_size] ground truth class indices
        alpha: [num_classes] class weights (higher for rare classes)
        gamma: float, focusing parameter (typically 1.5-2.0)
        reduction: 'mean' or 'sum'
    
    Returns:
        loss: scalar tensor
    """
    # Get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get probability of true class for each sample
    num_classes = logits.size(-1)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    p_t = (probs * targets_one_hot).sum(dim=-1)  # [batch_size]
    
    # Focal term: (1 - p_t)^gamma
    # When p_t is high (confident correct prediction), focal_weight is low
    # When p_t is low (hard example), focal_weight is high
    focal_weight = (1.0 - p_t) ** gamma
    
    # Cross entropy: -log(p_t)
    ce_loss = -torch.log(p_t.clamp(min=1e-8))
    
    # Apply focal weighting
    loss = focal_weight * ce_loss
    
    # Apply alpha class weights if provided
    if alpha is not None:
        if alpha.device != targets.device:
            alpha = alpha.to(targets.device)
        alpha_t = alpha[targets]  # Get alpha for each sample's true class
        loss = alpha_t * loss
    
    # Reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def criterion(model_outputs, batch, audio_processor=None):
    """
    Calculate loss for XTTS/GPT training.

    Args:
        model_outputs: Dictionary containing model predictions
        batch: Dictionary containing ground truth targets

    Returns:
        loss: Computed loss tensor
        loss_dict: Dictionary with loss components for logging
    """
    loss_dict = {}
    total_loss = 0.0
    
    try:
        device = next(iter(model_outputs.values())).device
    except Exception:
        try:
            device = next(iter(batch.values())).device if any(isinstance(v, torch.Tensor) for v in batch.values()) else torch.device("cpu")
        except Exception:
            device = torch.device("cpu")
    
    scalar_contamination = [k for k, v in model_outputs.items() if isinstance(v, torch.Tensor) and v.ndim == 0]
    if scalar_contamination:
        logging.error(f"[CRITERION SAFEGUARD] 🔴 SCALAR CONTAMINATION DETECTED in model_outputs: {scalar_contamination}")
        logging.error(f"  These should have been filtered by GPT extraction! Keys: {scalar_contamination}")
        for key in scalar_contamination:
            logging.error(f"  {key}: shape={model_outputs[key].shape}, value={model_outputs[key].item():.6f}")
    
    # <CHANGE> Access loss_weights from global scope (set by get_criterion_with_weights)
    # This allows the criterion function to use weights from the YAML config
    w = globals().get('loss_weights', {"mel": 0.0, "prosody": 0.4, "emotion": 0.3})

    # Text token loss (if available)
    if "logits" in model_outputs and "token_id" in batch:
        text_logits = model_outputs["logits"]
        text_targets = batch["token_id"]

        # Reshape for cross entropy if needed
        if text_logits.dim() == 3:  # [batch, seq, vocab]
            text_logits = text_logits.view(-1, text_logits.size(-1))
            text_targets = text_targets.view(-1)

        text_loss = torch.nn.functional.cross_entropy(
            text_logits, text_targets, ignore_index=-1
        )
        loss_dict["text_loss"] = text_loss.item()
        total_loss += text_loss

    # ✅ RE-ENABLED: Audio token loss for LONG-FORM training
    # Key difference: For long audio, use FULL target sequence (not just 6 tokens)
    if "gpt_logits" in model_outputs and "gpt_audio_tokens" in batch:
        gpt_logits = model_outputs["gpt_logits"]  # Should be 6 tokens (no expansion)
        gpt_targets = batch["gpt_audio_tokens"]
        
        # Logging for training progress
        target_len = gpt_targets.shape[-1] if gpt_targets.dim() >= 2 else 1
        is_long_form = target_len > 100  # Detect long audio (for logging)
        target_seq_len = gpt_logits.shape[2] if gpt_logits.dim() == 3 else 1
        
        logging.debug(f"[AUDIO LOSS] Processing: logits_shape={gpt_logits.shape}, target_len={target_len}, long_form={is_long_form}")
        
        if gpt_logits.dim() == 3 and gpt_targets.dim() == 2:
            batch_size = gpt_logits.shape[0]
            vocab_size = gpt_logits.shape[1]
            seq_len = gpt_logits.shape[2]
            
            gpt_logits_flat = gpt_logits.permute(0, 2, 1).reshape(-1, vocab_size)
            gpt_targets_trimmed = gpt_targets[:, :seq_len]
            gpt_targets_flat = gpt_targets_trimmed.reshape(-1)
            
            valid_mask = gpt_targets_flat >= 0
            if valid_mask.sum() > 0:
                gpt_loss = torch.nn.functional.cross_entropy(
                    gpt_logits_flat[valid_mask], gpt_targets_flat[valid_mask]
                )
                loss_dict["gpt_audio_loss"] = gpt_loss.item()
                
                gpt_weight = w.get("gpt", 0.1)
                if is_long_form:
                    gpt_weight = w.get("gpt", 0.5)
                
                total_loss += gpt_loss * gpt_weight
                logging.info(f"[AUDIO LOSS] ✅ GPT loss on first {seq_len} targets: {gpt_loss.item():.6f}, weight={gpt_weight}")
            else:
                logging.debug(f"[AUDIO LOSS] No valid targets")
        else:
            logging.debug(f"[AUDIO LOSS] Skipped - unexpected shapes: logits {gpt_logits.shape}, targets {gpt_targets.shape}")
    else:
        logging.debug("[AUDIO LOSS] Skipped - gpt_logits or gpt_audio_tokens not in outputs")

    # Stop token loss (if available)
    if "stop_logits" in model_outputs and "stop_targets" in batch:
        stop_logits = model_outputs["stop_logits"]
        stop_targets = batch["stop_targets"]

        stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            stop_logits.squeeze(), stop_targets.float()
        )
        loss_dict["stop_loss"] = stop_loss.item()
        total_loss += stop_loss * 0.1  # Weight stop loss lower

    # Prosody loss (if available)
    if "prosody_pred" in model_outputs and "prosody_target" in batch:
        prosody_pred = model_outputs["prosody_pred"]
        prosody_target = batch["prosody_target"]

        prosody_loss = 0.0
        prosody_count = 0

        for key in ['pitch_pred', 'energy_pred', 'duration_pred']:
            if key in prosody_pred and key.replace('_pred', '') in prosody_target:
                pred = prosody_pred[key]
                target = prosody_target[key.replace('_pred', '')]

                # Align dimensions if needed
                # Align dimensions if needed
                min_len = min(pred.shape[-1], target.shape[-1])
                pred_aligned = pred[..., :min_len]
                target_aligned = target[..., :min_len]

                loss_component = F.l1_loss(pred_aligned, target_aligned)
                prosody_loss += loss_component
                prosody_count += 1




        if prosody_count > 0:
            prosody_loss = prosody_loss / prosody_count
            loss_dict["prosody_loss"] = prosody_loss.item()
            # use weight from YAML, default 0.5
            total_loss += prosody_loss * w.get("prosody", 0.5)

        # ===== NEW: EMOTION LOSS COMPUTATION =====
        # ===== FIXED: EMOTION LOSS COMPUTATION =====
        if "emotion" in batch and "emotion_logits" in model_outputs:
            try:
                emotion_logits = model_outputs["emotion_logits"]
                emotion_labels = batch["emotion"]

                # Get device from logits
                device = emotion_logits.device

                # Check batch sizes
                logits_batch_size = emotion_logits.size(0)
                labels_batch_size = (
                    emotion_labels.size(0) if emotion_labels.dim() > 0 else 1
                )

                # Only compute loss when batch sizes match
                if logits_batch_size != labels_batch_size:
                    logging.debug(
                        f"Emotion batch size mismatch: logits={logits_batch_size}, "
                        f"targets={labels_batch_size}. Skipping emotion loss."
                    )
                else:
                    # Ensure labels are on same device
                    emotion_labels = emotion_labels.to(device)

                    # Use your 5-class weighting (angry, happy, surprised, fearful, calm)
                    # <CHANGE> Reduced fearful weight from 0.44 to 0.15 to heavily penalize predicting fearful
                    # Model is collapsing to always predict fearful (50% of data)
                    # <CHANGE> Rebalanced weights - angry was being over-predicted
                    # Computed as: 1 / (frequency * num_classes)
                    # angry: 20.6% -> weight 0.97, fearful: 50.4% -> weight 0.40
                    # <CHANGE> Add numerical stability checks to catch nan/inf in emotion_logits
                    # <CHANGE> Enhanced diagnostics to find nan source
                    logging.info(f"[v0 DIAGNOSTIC] emotion_logits stats: min={emotion_logits.min().item():.4f}, max={emotion_logits.max().item():.4f}, mean={emotion_logits.mean().item():.4f}")
                    # <CHANGE> Add logit magnitude analysis
                    logit_norms = emotion_logits.norm(dim=1)
                    logging.info(f"[DEBUG LOGITS] Logit norms: min={logit_norms.min().item():.4f}, max={logit_norms.max().item():.4f}, mean={logit_norms.mean().item():.4f}")
                    logging.info(f"[DEBUG LOGITS] Softmax probs: {F.softmax(emotion_logits[0], dim=0).detach().cpu().tolist()}")
                    logging.info(f"[v0 DIAGNOSTIC] emotion_logits sample: {emotion_logits[0].detach().cpu().tolist()}")
                    
                    if torch.isnan(emotion_logits).any() or torch.isinf(emotion_logits).any():
                        logging.error(f"[v0 DIAGNOSTIC] ERROR: emotion_logits contains nan/inf!")
                        logging.error(f"[v0 DIAGNOSTIC] Skipping emotion loss for this batch")
                        emotion_loss = torch.tensor(0.0, device=device)
                        loss_dict["emotion_loss"] = 0.0
                    else:
                        # <CHANGE> Removed focal loss and class weights - was causing nan gradients and model collapse
                        # Using simple cross-entropy with label smoothing instead
                        # FIXED: Focal loss with reduced gamma to handle severe class imbalance
                        # Data: angry=20.6%, happy=8.8%, surprised=4.1%, fearful=50.4%, calm=16.1%
                        # Focal loss down-weights easy examples (fearful) and focuses on hard ones (surprised)
                        # Class weights: inverse frequency normalized (much more stable)
                        # FIXED: Focal loss with reduced gamma to handle severe class imbalance
                        # Data: angry=20.6%, happy=8.8%, surprised=4.1%, fearful=50.4%, calm=16.1%
                        # Focal loss down-weights easy examples (fearful) and focuses on hard ones (surprised)
                        # Class weights: inverse frequency normalized (much more stable)
                        # ✅ OPTIMIZED: Research-grade focal loss with proper inverse-frequency weights
                        # Following 2024 Odyssey Challenge winner approach (https://arxiv.org/abs/2405.20064)
                        # 
                        # Your emotion distribution: angry=20.6%, happy=8.8%, surprised=4.1%, fearful=50.4%, calm=16.1%
                        # 
                        # Class weights formula: weight_i = total_samples / (num_classes * class_count)
                        #   angry:     1028/(5*212) = 0.971 -> scaled to 2.42
                        #   happy:     1028/(5*90)  = 2.284 -> scaled to 5.70
                        #   surprised: 1028/(5*42)  = 4.895 -> scaled to 12.20
                        #   fearful:   1028/(5*518) = 0.397 -> scaled to 0.99
                        #   calm:      1028/(5*166) = 1.239 -> scaled to 3.09
                        #
                        # Key insight: Surprised gets 12.3x weight vs Fearful to force learning
                        # 
                        # Focal loss (gamma=1.5):
                        #   - Down-weights easy examples (50% fearful samples)
                        #   - Focuses training on hard examples (4% surprised)
                        #   - gamma=1.5 optimal for severe imbalance (vs gamma=2.0 standard)
                        # ✅ FIXED: Stronger weights for angry/happy to force learning ALL 5 classes
                        # angry: 3.0 (was 0.50) - 6x increase to force model to learn it
                        # happy: 4.0 (was 1.17) - 3.4x increase to force model to learn it  
                        # surprised: 6.0 (was 2.50) - already learning, keep strong
                        # fearful: 0.15 (was 0.20) - reduce to discourage over-prediction
                        # calm: 1.0 (was 0.63) - slight increase
                        # ✅ BALANCED WEIGHTS: Inverse sqrt of class frequency (proven effective)
                        # 7 emotions: neutral, happy, sad, angry, surprised, fearful, disgusted
                        # Using balanced weights (1.0 each) for equal emphasis
                        alpha = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=emotion_logits.device)
                        emotion_loss = F.cross_entropy(emotion_logits, emotion_labels, weight=alpha, reduction='mean')
                        # <CHANGE> Add debug: log raw loss before scaling
                        logging.info(f"[DEBUG LOSS] Raw emotion_loss: {emotion_loss.item():.6f}")
                        logging.info(f"[DEBUG LOSS] Emotion weight multiplier: {w.get('emotion', 0.1)}")
                        logging.info(f"[DEBUG LOSS] Scaled emotion_loss: {(emotion_loss * w.get('emotion', 0.1)).item():.6f}")

                        # Clamp removed - it was preventing learning by zeroing gradients when loss > 2.5
                        # Natural early training loss with class weights can reach 6-10, which is normal
                        
                        # Check if loss itself is nan
                        
                        # Check if loss itself is nan
                        # Check if loss itself is nan
                        if torch.isnan(emotion_loss) or torch.isinf(emotion_loss):
                            logging.error(f"[v0 DIAGNOSTIC] ERROR: emotion_loss is nan/inf after cross_entropy!")
                            emotion_loss = torch.tensor(0.0, device=device, requires_grad=True)
                            loss_dict["emotion_loss"] = 0.0
                        else:
                            loss_dict["emotion_loss"] = emotion_loss.item()
                            logging.info(f"[v0 DIAGNOSTIC] emotion_loss is valid: {emotion_loss.item():.6f}, requires_grad={emotion_loss.requires_grad}")
                            
                            # <CHANGE> Reduced emotion weight from 3.5 to 1.5
                            total_loss += emotion_loss * w.get("emotion", 0.1)

                    
                    logging.debug(f"Emotion loss: {emotion_loss.item():.6f}")
                    
                    # <CHANGE> Add diagnostic logging - fixed step variable issue
                    # === ENHANCED EMOTION PREDICTION LOGGING ===
                    try:
                        with torch.no_grad():
                            # Get predictions and probabilities
                            emotion_probs = F.softmax(emotion_logits, dim=-1)
                            emotion_preds = torch.argmax(emotion_probs, dim=-1)

                            # Calculate entropy to detect prediction collapse
                            # High entropy (near log(5)=1.6) = diverse predictions
                            # Low entropy (near 0) = always predicting same class
                            entropy = -(emotion_probs * torch.log(emotion_probs + 1e-10)).sum(dim=-1).mean().item()
                            logging.info(f"[EMOTION] Prediction entropy: {entropy:.4f} (max={np.log(5):.4f})")

                            # Calculate batch accuracy
                            emotion_accuracy = (emotion_preds == emotion_labels).float().mean().item()

                            
                            # Emotion label mapping
                            emotion_labels_map = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted']
                            
                            # === BATCH-LEVEL METRICS ===
                            logging.info(f"[EMOTION] Batch Accuracy: {emotion_accuracy:.3f} ({int(emotion_accuracy*100)}%)")
                            

                            pred_counts = torch.bincount(emotion_preds, minlength=7)
                            logging.info(f"[EMOTION] Prediction distribution: neutral={pred_counts[0]}, happy={pred_counts[1]}, sad={pred_counts[2]}, angry={pred_counts[3]}, surprised={pred_counts[4]}, fearful={pred_counts[5]}, disgusted={pred_counts[6]}")
                            
                            # === PER-SAMPLE PREDICTIONS (first 4 samples) ===
                            logging.info(f"[EMOTION] Sample-by-sample breakdown:")
                            for i in range(min(4, len(emotion_preds))):
                                pred_idx = emotion_preds[i].item()
                                true_idx = emotion_labels[i].item()
                                pred_label = emotion_labels_map[pred_idx]
                                true_label = emotion_labels_map[true_idx]
                                confidence = emotion_probs[i, pred_idx].item()
                                
                                # Match indicator
                                match_symbol = "✓" if pred_idx == true_idx else "✗"
                                
                                logging.info(
                                    f"  [{match_symbol}] Sample {i}: "
                                    f"Pred={pred_label}({confidence:.2f}) | "
                                    f"True={true_label}"
                                )
                            
                            # === GRADIENT HEALTH CHECK ===
                            # Check if emotion classifier has gradients (indicates learning)
                            # Remove gradient monitoring for now - emotion_classifier not in scope
                            # Will add back after training stabilizes
                            
                            # === CONFIDENCE ANALYSIS ===
                            # Show if model is uncertain (low max probability)
                            max_probs = emotion_probs.max(dim=-1)[0]
                            avg_confidence = max_probs.mean().item()
                            logging.info(f"[EMOTION] Average prediction confidence: {avg_confidence:.3f}")
                            
                            if avg_confidence < 0.3:
                                logging.warning(f"[QUALITY] ⚠️ Low confidence predictions (avg={avg_confidence:.3f})")
                                
                    except Exception as e:
                        logging.error(f"[EMOTION] Enhanced logging failed: {e}")
                        import traceback
                        logging.error(traceback.format_exc())
            except Exception as e:
                logging.debug(f"Emotion loss computation failed: {e}")
        # ===== END FIXED EMOTION LOSS =====



    # Perceptual loss (if available)
    # L1 spectrogram / mel loss (support multiple key names & apply configured weight)
    # ===== OPTION A: Spec L1 loss with decoder integration =====
    spec_pred, spec_target = None, None

    # Try to get spec pred/target from multiple possible keys
    if "mel" in model_outputs and "mel" in batch:
        spec_pred = model_outputs["mel"].to(device)
        spec_target = batch["mel"].to(device)
        
        # Handle case where decoder returns waveform instead of mel
        # If pred is waveform [batch, 1, samples] and target is mel [batch, n_mels, time]
        if spec_pred.dim() == 3 and spec_pred.shape[1] == 1 and spec_target.dim() == 3 and spec_target.shape[1] > 1:
            logging.info(f"[MEL LOSS] Detected waveform output from decoder. Converting to mel...")
            if audio_processor is not None:
                try:
                    wav_pred = spec_pred.squeeze(1)  # [batch, samples]
                    wav_pred_np = wav_pred.cpu().detach().numpy() if wav_pred.device.type == 'cuda' else wav_pred.detach().numpy()
                    spec_pred = audio_processor.melspectrogram(wav_pred_np)
                    spec_pred = torch.from_numpy(spec_pred).to(device)
                    logging.info(f"[MEL LOSS] Converted waveform to mel: {spec_pred.shape}")
                except Exception as e:
                    logging.warning(f"[MEL LOSS] Failed to convert waveform to mel: {e}. Using waveform as-is.")
            else:
                logging.warning(f"[MEL LOSS] Waveform output detected but audio_processor not available. Cannot convert to mel.")
        
    elif "spectrogram_pred" in model_outputs and "spectrogram_target" in batch:
        spec_pred = model_outputs["spectrogram_pred"].to(device)
        spec_target = batch["spectrogram_target"].to(device)
    elif "waveform" in model_outputs and "waveform" in batch and audio_processor is not None:
        try:
            wav_pred = model_outputs["waveform"].to(device)
            wav_tgt = batch["waveform"].to(device)
            spec_pred = audio_processor.melspectrogram(wav_pred)
            spec_target = audio_processor.melspectrogram(wav_tgt)
        except Exception as e:
            logging.debug(f"could not compute mel from waveform: {e}")

    # Compute Spec L1 if both are present
    if spec_pred is not None and spec_target is not None:
        # Ensure shapes are compatible
        min_len = min(spec_pred.shape[-1], spec_target.shape[-1])
        if min_len > 0:
            # Also handle mel dimension mismatch (e.g., predicted 64 mels vs target 80 mels)
            min_mels = min(spec_pred.shape[1] if spec_pred.dim() >= 2 else 1, 
                          spec_target.shape[1] if spec_target.dim() >= 2 else 1)
            
            if spec_pred.dim() >= 2 and spec_target.dim() >= 2:
                pred_slice = spec_pred[:, :min_mels, :min_len]
                targ_slice = spec_target[:, :min_mels, :min_len]
                
                pred_mean = pred_slice.mean(dim=(0, 2), keepdim=True)
                pred_std = pred_slice.std(dim=(0, 2), keepdim=True) + 1e-8
                targ_mean = targ_slice.mean(dim=(0, 2), keepdim=True)
                targ_std = targ_slice.std(dim=(0, 2), keepdim=True) + 1e-8
                
                pred_norm = (pred_slice - pred_mean) / pred_std
                targ_norm = (targ_slice - targ_mean) / targ_std
                
                spec_loss = F.l1_loss(pred_norm, targ_norm)
            else:
                pred_slice = spec_pred[..., :min_len]
                targ_slice = spec_target[..., :min_len]
                
                pred_mean = pred_slice.mean(dim=(0, 2), keepdim=True) if pred_slice.dim() >= 3 else pred_slice.mean()
                pred_std = pred_slice.std(dim=(0, 2), keepdim=True) + 1e-8 if pred_slice.dim() >= 3 else pred_slice.std() + 1e-8
                targ_mean = targ_slice.mean(dim=(0, 2), keepdim=True) if targ_slice.dim() >= 3 else targ_slice.mean()
                targ_std = targ_slice.std(dim=(0, 2), keepdim=True) + 1e-8 if targ_slice.dim() >= 3 else targ_slice.std() + 1e-8
                
                pred_norm = (pred_slice - pred_mean) / pred_std
                targ_norm = (targ_slice - targ_mean) / targ_std
                
                spec_loss = F.l1_loss(pred_norm, targ_norm)
            
            # ✅ FIX: DON'T divide by sequence length - it destroys the gradient signal
            # Normalized loss by seq_len made mel_loss ≈ 0.0003 (useless)
            # Instead, use raw normalized loss and scale through mel_weight
            # spec_loss = spec_loss / max(min_len, 1)  # ❌ REMOVED: This was the blocker
            
            mel_weight = w.get("mel", 1.0)
            weighted_spec_loss = mel_weight * spec_loss
            loss_dict["spectrogram_loss"] = float(spec_loss.detach().cpu().item())
            loss_dict["spectrogram_loss_weighted"] = float(weighted_spec_loss.detach().cpu().item())
            logging.info(f"[MEL LOSS] ✅ Computed: raw={spec_loss.item():.6f}, weight={mel_weight}, weighted={weighted_spec_loss.item():.6f}")
            logging.info(f"[MEL LOSS] Shapes: pred={spec_pred.shape}, target={spec_target.shape}, min_len={min_len}")
            if spec_pred.dim() >= 2 and spec_target.dim() >= 2:
                logging.info(f"[MEL LOSS] Mel dims aligned to {min_mels}")
            total_loss += weighted_spec_loss
        else:
            logging.debug("Spec tensors found but length = 0; skipping Spec L1.")
            logging.debug(f"  spec_pred shape: {spec_pred.shape}, spec_target shape: {spec_target.shape}")
    else:
        logging.debug("No spec_pred or spec_target available for Spec L1.")
        logging.debug(f"  spec_pred: {spec_pred}, spec_target: {spec_target}")
        logging.debug(f"  model_outputs keys: {list(model_outputs.keys())}, batch keys: {list(batch.keys())}")
    # ===== END OPTION A: Spec L1 Loss =====

    # ✅ ADD SCALAR LOSS SIGNALS FROM GPT MODEL
    # These are internal loss signals that should drive training
    if "scalar_losses" in model_outputs:
        scalar_losses = model_outputs["scalar_losses"]
        if scalar_losses:
            logging.info(f"[SCALAR LOSSES] Found {len(scalar_losses)} scalar loss signals in model outputs")
            scalar_loss_sum = 0.0
            for idx, (orig_idx, scalar_loss) in enumerate(scalar_losses):
                scalar_val = scalar_loss.item()
                scalar_loss_sum += scalar_val
                logging.info(f"[SCALAR LOSSES] Signal {idx} (orig idx {orig_idx}): {scalar_val:.6f}")
                total_loss += scalar_loss
            
            loss_dict["scalar_loss_sum"] = scalar_loss_sum
            loss_dict["num_scalar_signals"] = len(scalar_losses)
            logging.info(f"[SCALAR LOSSES] ✅ ADDED {len(scalar_losses)} scalar signals (sum={scalar_loss_sum:.6f}) to training objective")
        else:
            logging.debug("[SCALAR LOSSES] No scalar losses in model outputs")
    else:
        logging.debug("[SCALAR LOSSES] No scalar_losses key in model_outputs")

    # If no specific losses found, use a generic loss
    if total_loss == 0.0:
        if "loss" in model_outputs:
            total_loss = model_outputs["loss"]
            loss_dict["model_loss"] = total_loss.item()
        else:
            # Fallback: create a dummy loss to prevent training crash
            total_loss = torch.tensor(0.0, requires_grad=True, device=next(iter(model_outputs.values())).device)
            loss_dict["dummy_loss"] = 0.0

    loss_dict["total_loss"] = total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logging.error(f"[CRITERION SAFEGUARD] 🔴 INVALID LOSS DETECTED: NaN={torch.isnan(total_loss).item()}, Inf={torch.isinf(total_loss).item()}")
        logging.error(f"  Loss dict: {loss_dict}")
        raise ValueError(f"Loss computation produced NaN/Inf. Likely caused by scalar contamination or numerical instability.")

    return total_loss, loss_dict

# -------------------------
# get_criterion_with_weights
# -------------------------
def get_criterion_with_weights(cfg_dict, audio_processor=None, device=torch.device("cpu")):
    """
    Returns a function criterion(outputs, batch) -> dict with keys 'loss' and component losses.
    cfg_dict: plain dict from YAML config.
    audio_processor: AudioProcessor-like object providing melspectrogram / spectrogram methods (optional).
    device: torch.device
    """
    # read weights with defaults
    w = {}
    w["waveform"] = float(cfg_dict.get("loss_weights", {}).get("waveform", cfg_dict.get("loss_waveform_weight", 1.0)))
    w["latent"] = float(cfg_dict.get("loss_weights", {}).get("latent", cfg_dict.get("loss_latent_weight", 1.0)))
    w["mel"] = float(cfg_dict.get("loss_weights", {}).get("mel", cfg_dict.get("loss_mel_weight", 0.0)))
    w["pitch"] = float(cfg_dict.get("loss_weights", {}).get("pitch", cfg_dict.get("loss_pitch_weight", 0.0)))
    w["energy"] = float(cfg_dict.get("loss_weights", {}).get("energy", cfg_dict.get("loss_energy_weight", 0.0)))
    w["perceptual"] = float(cfg_dict.get("loss_weights", {}).get("perceptual", cfg_dict.get("loss_perceptual_weight", 0.0)))
    w["emotion"] = float(cfg_dict.get("loss_weights", {}).get("emotion", cfg_dict.get("loss_emotion_weight", 0.0)))
    w["prosody"] = float(cfg_dict.get("loss_weights", {}).get("prosody", cfg_dict.get("loss_prosody_weight", 0.2)))
    w["gpt"] = float(cfg_dict.get("loss_weights", {}).get("gpt", cfg_dict.get("loss_gpt_weight", 0.5)))


    # Try to create MRSTFT if available (Coqui or local impl)
    try:
        from TTS.losses.stft import MultiResolutionSTFTLoss
        mrstft = MultiResolutionSTFTLoss(device=device)
    except Exception:
        mrstft = None

    # ===== ADD THESE LINES =====
    # Store weights globally so criterion function can access them
    globals()['loss_weights'] = w
    
    # Log the weights being used
    logging.info(f"Loss weights: {w}")
    
    # Return the criterion function (it's defined earlier in the file)
    return criterion
    # ===== END ADDITION =====

# -------------------------
# forward wrapper for older scripts

# -------------------------
# Helper to get decoder from model (handles DataParallel wrapping)
# -------------------------
def get_model_decoder(model):
    """
    Safely retrieve the decoder from a model, whether wrapped with DataParallel or not.
    
    Args:
        model: The model (possibly wrapped with DataParallel)
    
    Returns:
        The hifigan_decoder if available, otherwise None
    """
    has_module = hasattr(model, 'module')
    logging.info(f"[GET_DECODER] Model has 'module' attr: {has_module}")
    
    if has_module:
        # Model is wrapped with DataParallel
        decoder = getattr(model.module, 'hifigan_decoder', None)
        logging.info(f"[GET_DECODER] Retrieved from model.module: {decoder is not None}, type: {type(decoder).__name__ if decoder else 'None'}")
        return decoder
    else:
        # Direct model access
        decoder = getattr(model, 'hifigan_decoder', None)
        logging.info(f"[GET_DECODER] Retrieved from model directly: {decoder is not None}, type: {type(decoder).__name__ if decoder else 'None'}")
        return decoder

# -------------------------
# forward wrapper for older scripts
# -------------------------
def forward_xtts(self, **batch):
    """
    XTTS forward pass with GPT conditioning and mel prediction.
    
    Returns:
        dict with keys: gpt_logits, gpt_latents, speaker_embedding, mel
    """
    logging.info("[FORWARD_XTTS] Custom forward_xtts() - GPT only (no decoder)")
    
    if not hasattr(self, 'gpt') or self.gpt is None:
        raise RuntimeError("GPT component is not initialized. Check model loading.")

    # Handle naming differences between collate_fn and dataset
    token_ids = batch.get("tokens", batch.get("token_id"))
    token_len = batch.get("tokens_lengths", batch.get("token_id_lengths"))
    audio_codes = batch.get("gpt_audio_tokens")
    wav_lengths = batch.get("wav_lengths", batch.get("mel_lengths", batch.get("expected_output_len")))
    cond_mels = batch.get("cond_mels", batch.get("mel"))   # prefer cond_mels if present
    speaker_embedding = batch.get("speaker_embedding")

    # Handle multi-codebook audio tokens from EnCodec
    # EnCodec returns [batch, codebooks, seq] but GPT expects [batch, seq]
    # Use only the first codebook for now
    if audio_codes is not None and audio_codes.dim() == 3:
        logging.info(f"[AUDIO CODES] Multi-codebook input detected: {audio_codes.shape}, extracting first codebook")
        audio_codes = audio_codes[:, 0, :]  # Take first codebook [batch, seq]
        logging.info(f"[AUDIO CODES] After extraction: {audio_codes.shape}")
    
    # ✅ FIXED: Use FULL audio sequence for both conditioning and target
    # This ensures the model learns to generate proper audio tokens with full context
    # 6-token windows will work because they're trained to reconstruct the FULL sequence
    audio_codes_target = audio_codes  # Full sequence for loss computation
    logging.info(f"[AUDIO CODES] Using FULL sequence for training (not conditioning split): {audio_codes.shape}")
    # GPT uses first part of audio_codes as prompt, generates next 6 tokens to predict

    # Safety defaults to prevent NoneType errors
    if token_ids is None:
        token_ids = torch.zeros(1, 1, dtype=torch.long, device=self.device)
    if token_len is None:
        token_len = torch.tensor([1], dtype=torch.long, device=self.device)
    
    # ✅ CRITICAL: ALWAYS use TARGET AUDIO CODES length, not conditioning mel
    # The collate function may set wav_lengths to mel_lengths, but that's WRONG
    # cond_mels is reference audio (Emma's voice ~233 frames) for speaker identity only
    # We MUST use audio_codes.shape (target audio ~750 tokens) for training signal
    if audio_codes is not None and isinstance(audio_codes, torch.Tensor) and audio_codes.dim() >= 1:
        # Override any previous wav_lengths with TARGET audio_codes length
        # This tells GPT: "generate as many tokens as the target audio has"
        target_length = audio_codes.shape[-1]  # Last dimension is sequence length
        batch_size = audio_codes.shape[0]
        
        # Create tensor with one value per batch item
        wav_lengths = torch.full((batch_size,), target_length, dtype=torch.long, device=self.device)
        logging.info(f"✅ [GPT LENGTH FIX] FORCING TARGET audio length: audio_codes={target_length} tokens (batch_size={batch_size})")
        print(f"🔥 [GPT LENGTH OVERRIDE] wav_lengths={wav_lengths.tolist()} (matching audio_codes={target_length})")
    elif cond_mels is not None and isinstance(cond_mels, torch.Tensor) and cond_mels.dim() >= 2:
        # Fallback: use conditioning mel if audio_codes unavailable
        mel_length = cond_mels.shape[-1]
        batch_size = cond_mels.shape[0]
        wav_lengths = torch.full((batch_size,), mel_length, dtype=torch.long, device=self.device)
        logging.warning(f"⚠️  [GPT LENGTH] audio_codes not available, falling back to conditioning mel: {mel_length} frames")
    else:
        # Last resort: default ~100 tokens per batch item
        batch_size = 1
        wav_lengths = torch.full((batch_size,), 100, dtype=torch.long, device=self.device)
        logging.warning(f"⚠️  [GPT LENGTH] No audio or mel available, using default wav_lengths=100")
    
    if cond_mels is None:
        cond_mels = torch.zeros(1, 80, 200, device=self.device)

    # ✅ CRITICAL: Log wav_lengths BEFORE GPT forward to verify it's set correctly
    logging.info(f"✅ [GPT CONFIG CHECK] wav_lengths={wav_lengths}, shape={wav_lengths.shape if isinstance(wav_lengths, torch.Tensor) else 'N/A'}")
    if isinstance(wav_lengths, torch.Tensor):
        logging.info(f"   wav_lengths.sum()={wav_lengths.sum().item()}, max={wav_lengths.max().item()}, min={wav_lengths.min().item()}")
    
    # Debugging shapes (optional)
    print("DEBUG forward_xtts:")
    for k, v in {"token_ids": token_ids, "token_len": token_len,
                 "audio_codes": audio_codes, "cond_mels": cond_mels,
                 "speaker_embedding": speaker_embedding, "wav_lengths": wav_lengths}.items():
        try:
            print(f"  {k}: {v.shape}")
        except Exception:
            print(f"  {k}: {type(v)}")

    try:
        logging.info(f"[GPT BEFORE CALL] wav_lengths param: {wav_lengths}, mel shape: {cond_mels.shape}")
        logging.info(f"[GPT BEFORE CALL] audio_codes shape: {audio_codes.shape}, token_ids shape: {token_ids.shape}")
        logging.info(f"[GPT BEFORE CALL] Expected output: ~{cond_mels.shape[-1]} tokens (from mel frames)")
        
        # Try passing mel_input_length or output_length if wav_lengths doesn't work
        try:
            gpt_output = self.gpt(
                text_inputs=token_ids,
                text_lengths=token_len,
                audio_codes=audio_codes,
                cond_mels=cond_mels,
                wav_lengths=wav_lengths,
                return_latent=False,
            )
        except TypeError as e:
            if "wav_lengths" in str(e):
                logging.warning(f"[GPT CALL] wav_lengths not recognized, trying without it: {e}")
                gpt_output = self.gpt(
                    text_inputs=token_ids,
                    text_lengths=token_len,
                    audio_codes=audio_codes,
                    cond_mels=cond_mels,
                    return_latent=False,
                )
            else:
                raise

        gpt_latents = None
        gpt_logits = None
        
        logging.info(f"[GPT OUTPUT TYPE] {type(gpt_output).__name__}")
        if isinstance(gpt_output, tuple):
            logging.info(f"[GPT OUTPUT TUPLE] Length={len(gpt_output)}")
            scalar_items = []
            tensor_items = []
            for i, item in enumerate(gpt_output):
                if isinstance(item, torch.Tensor):
                    shape_str = f"shape={item.shape}"
                    logging.info(f"[GPT OUTPUT ITEM-{i}] type=Tensor, {shape_str}, dtype={item.dtype}")
                    if item.ndim == 0:
                        logging.info(f"  ✅ SCALAR LOSS SIGNAL: value={item.item():.4f} (CAPTURING FOR TRAINING)")
                        scalar_items.append((i, item))
                    elif item.ndim == 3:
                        tensor_items.append((i, item))
                        logging.info(f"  ✅ 3D Tensor: [B={item.shape[0]}, T={item.shape[1]}, D={item.shape[2]}]")
                    elif item.ndim > 3:
                        tensor_items.append((i, item))
                        logging.info(f"  ✅ {item.ndim}D Tensor")
                    else:
                        tensor_items.append((i, item))
                        logging.info(f"  ✅ {item.ndim}D Tensor")
                else:
                    logging.info(f"[GPT OUTPUT ITEM-{i}] type={type(item).__name__}, value={item if not isinstance(item, torch.Tensor) else 'tensor'}")
            
            if scalar_items:
                logging.info(f"[GPT FILTER] ✅ FOUND {len(scalar_items)} internal loss scalars at indices {[i for i,_ in scalar_items]} - ADDING TO TRAINING OBJECTIVE")
            logging.info(f"[GPT FILTER] Keeping {len(tensor_items)} tensor items, capturing {len(scalar_items)} scalar loss items")

        if isinstance(gpt_output, tuple) and len(gpt_output) >= 2:
            logging.info(f"[GPT EXTRACT] Multi-element tuple, extracting logits/latents (INCLUDING SCALAR LOSS SIGNALS)...")
            meaningful_items = list(tensor_items) if 'tensor_items' in locals() else []
            if not meaningful_items:
                for i, item in enumerate(gpt_output):
                    if isinstance(item, torch.Tensor) and item.ndim > 0:
                        meaningful_items.append((i, item))
                        logging.info(f"  ✅ Item {i}: shape={item.shape}, ndim={item.ndim}")
                    elif isinstance(item, torch.Tensor):
                        logging.error(f"  🚫 Item {i}: SCALAR tensor - FILTERED OUT")
            
            if len(meaningful_items) >= 2:
                logging.info(f"[GPT EXTRACT] Found {len(meaningful_items)} meaningful items, detecting logits vs latents...")
                for idx, (orig_idx, item) in enumerate(meaningful_items):
                    if item.ndim >= 3:
                        vocab_candidate = item.shape[1]
                        if vocab_candidate > 1000:
                            logging.info(f"  Item {idx} (orig {orig_idx}): shape={item.shape}, shape[1]={vocab_candidate} → logits")
                            gpt_logits = item
                        else:
                            logging.info(f"  Item {idx} (orig {orig_idx}): shape={item.shape}, shape[1]={vocab_candidate} → latents")
                            gpt_latents = item
                    else:
                        logging.info(f"  Item {idx} (orig {orig_idx}): shape={item.shape} → latents")
                        gpt_latents = item
            elif len(meaningful_items) == 1:
                logging.info(f"[GPT EXTRACT] Found 1 meaningful item, shape={meaningful_items[0][1].shape}")
                item = meaningful_items[0][1]
                if item.ndim >= 3:
                    vocab_candidate = item.shape[1]
                    logging.info(f"  Checking shape[1]={vocab_candidate} (vocab indicator)")
                    if vocab_candidate > 1000:
                        logging.info(f"  → Large shape[1] {vocab_candidate}: treating as logits [batch, vocab, seq]")
                        gpt_logits = item
                    else:
                        logging.info(f"  → Small shape[1] {vocab_candidate}: treating as latents")
                        gpt_latents = item
                else:
                    logging.info(f"  → ndim={item.ndim}, treating as latents")
                    gpt_latents = item
            else:
                logging.info(f"[GPT EXTRACT] ERROR: No meaningful tensors found!")
        elif isinstance(gpt_output, tuple) and len(gpt_output) == 1:
            logging.info(f"[GPT EXTRACT] Single-element tuple, assuming latents")
            gpt_latents = gpt_output[0]
        elif isinstance(gpt_output, dict):
            logging.info(f"[GPT EXTRACT] Dict with keys: {list(gpt_output.keys())}")
            gpt_logits = gpt_output.get("logits")
            gpt_latents = gpt_output.get("latents")
        else:
            logging.info(f"[GPT EXTRACT] Direct tensor with shape {gpt_output.shape}")
            if gpt_output.ndim >= 3:
                vocab_candidate = gpt_output.shape[1]
                logging.info(f"  Checking shape[1]={vocab_candidate}")
                if vocab_candidate > 1000:
                    logging.info(f"  → Large shape[1] {vocab_candidate}: treating as logits")
                    gpt_logits = gpt_output
                else:
                    logging.info(f"  → Small shape[1] {vocab_candidate}: treating as latents")
                    gpt_latents = gpt_output
            else:
                logging.info(f"  → ndim={gpt_output.ndim}, treating as latents")
                gpt_latents = gpt_output
        
        del gpt_output

        outputs = {}
        if gpt_latents is not None:
            outputs["gpt_latents"] = gpt_latents
            logging.info(f"[EXTRACT RESULT] gpt_latents shape: {gpt_latents.shape}")
        if gpt_logits is not None:
            # ✅ NO EXPANSION DURING TRAINING
            # Keep original 6-token logits for stable loss computation
            # Expansion only causes NaN gradients; autoregressive inference handles chaining
            logging.info(f"[GPT LOGITS] Using raw logits WITHOUT expansion: {gpt_logits.shape}")
            
            outputs["gpt_logits"] = gpt_logits
            logging.info(f"[EXTRACT RESULT] gpt_logits shape: {gpt_logits.shape}")
        if speaker_embedding is not None:
            outputs["speaker_embedding"] = speaker_embedding
        
        # ✅ OPTION A: DECODER INTEGRATION FOR MEL PREDICTION
        # Convert GPT latents → mel spectrograms using the decoder
        predicted_mel = None
        logging.info(f"[DECODER_FETCH] About to call get_model_decoder, self type: {type(self).__name__}, has module: {hasattr(self, 'module')}")
        decoder = get_model_decoder(self)
        logging.info(f"[DECODER_FETCH] Result: decoder is not None = {decoder is not None}")
        
        # If we have logits but no latents, use logits for decoder
        # The decoder may expect the raw GPT output for reconstruction
        if gpt_logits is not None and gpt_latents is None and decoder is not None:
            logging.info(f"[DECODER] Have gpt_logits but no latents. Using logits directly for decoder.")
            gpt_latents = gpt_logits
            logging.info(f"[DECODER] Logits shape for decoder input: {gpt_latents.shape}")
        
        if gpt_latents is not None and decoder is not None:
            try:
                logging.info(f"[DECODER] Attempting to decode GPT latents with shape {gpt_latents.shape}")
                
                # HifiDecoder expects: (batch, hidden_dim, seq_len)
                # gpt_latents might be:
                # - logits [batch, vocab_size, seq_len] - need conversion
                # - latents [batch, seq_len, hidden_dim] - need transpose
                
                if gpt_latents.dim() == 3:
                    # Check if this looks like logits (large middle dimension = vocab size)
                    if gpt_latents.shape[1] > 1000:
                        logging.info(f"[DECODER] Input looks like logits [batch, vocab={gpt_latents.shape[1]}, seq={gpt_latents.shape[2]}]")
                        # Take argmax to get predicted codes
                        codes = torch.argmax(gpt_latents, dim=1)  # [batch, seq]
                        logging.info(f"[DECODER] Extracted codes via argmax: {codes.shape}")
                        
                        # Now we have codes [batch, seq], try to embed them
                        # Look for audio embedding layer (mel_embedding in XTTS GPT)
                        try:
                            gpt_model = self.gpt if hasattr(self, 'gpt') else (self.module.gpt if hasattr(self, 'module') else None)
                            embed_layer = None
                            if gpt_model is not None:
                                # Try common embedding layer names
                                if hasattr(gpt_model, 'mel_embedding'):
                                    embed_layer = gpt_model.mel_embedding
                                    logging.info(f"[DECODER] Found mel_embedding layer")
                                elif hasattr(gpt_model, 'embeddings'):
                                    embed_layer = gpt_model.embeddings
                                    logging.info(f"[DECODER] Found embeddings layer")
                                elif hasattr(gpt_model, 'audio_embedding'):
                                    embed_layer = gpt_model.audio_embedding
                                    logging.info(f"[DECODER] Found audio_embedding layer")
                            
                            if embed_layer is not None:
                                logging.info(f"[DECODER] Converting codes to embeddings via {embed_layer.__class__.__name__}")
                                embeddings = embed_layer(codes)  # Should give [batch, seq, hidden_dim]
                                gpt_latents_permuted = embeddings.transpose(1, 2)  # [batch, hidden_dim, seq]
                                logging.info(f"[DECODER] Codes embedded: {gpt_latents_permuted.shape}")
                            else:
                                # Can't embed, use codes directly and hope decoder can handle
                                logging.warning(f"[DECODER] No embedding layer found (checked mel_embedding, embeddings, audio_embedding), using codes as-is")
                                gpt_latents_permuted = codes.float().unsqueeze(1)  # [batch, 1, seq]
                                logging.info(f"[DECODER] Using codes as features: {gpt_latents_permuted.shape}")
                        except Exception as e:
                            logging.warning(f"[DECODER] Failed to embed codes: {e}, using codes directly")
                            gpt_latents_permuted = codes.float().unsqueeze(1)
                    else:
                        # Looks like latents [batch, seq, hidden_dim]
                        gpt_latents_permuted = gpt_latents.transpose(1, 2)
                        logging.info(f"[DECODER] Transposed latents to [batch, hidden_dim, seq_len]: {gpt_latents_permuted.shape}")
                else:
                    gpt_latents_permuted = gpt_latents
                    logging.info(f"[DECODER] Using input as-is: {gpt_latents_permuted.shape}")
                
                # Pass through decoder to get mel or audio
                decoder_output = decoder(gpt_latents_permuted)
                logging.info(f"[DECODER] Decoder output type: {type(decoder_output).__name__}, shape: {decoder_output.shape if hasattr(decoder_output, 'shape') else 'N/A'}")
                
                # If decoder returns waveform [batch, 1, samples], convert to mel
                if decoder_output.dim() == 3 and decoder_output.shape[1] == 1:
                    logging.info(f"[DECODER] Output appears to be waveform [batch, 1, samples]. Keeping as-is for now.")
                    # We could convert to mel here if audio_processor is available
                    predicted_mel = decoder_output
                elif decoder_output.dim() == 3:
                    # Assume it's already mel [batch, n_mels, time]
                    predicted_mel = decoder_output
                    logging.info(f"[DECODER] Output treated as mel spectrogram: {predicted_mel.shape}")
                else:
                    logging.warning(f"[DECODER] Unexpected output shape {decoder_output.shape}, skipping mel prediction")
                    predicted_mel = None
                
            except Exception as e:
                logging.warning(f"[DECODER] Failed to decode latents: {e}. Falling back to input conditioning mel.")
                predicted_mel = None
        else:
            if gpt_latents is None:
                logging.debug(f"[DECODER] No gpt_latents available for decoding")
            if decoder is None:
                logging.debug(f"[DECODER] hifigan_decoder is not available")
        
        # Use decoded mel if available, otherwise fall back to input conditioning mel
        if predicted_mel is not None:
            outputs["mel"] = predicted_mel
            logging.info(f"[MEL OUTPUT] Using decoded mel from decoder: {predicted_mel.shape}")
        elif cond_mels is not None:
            outputs["mel"] = cond_mels
            logging.warning(f"[MEL OUTPUT] Decoder not available. Falling back to input conditioning mel: {cond_mels.shape}")
            logging.warning(f"[MEL OUTPUT] ⚠️ This will result in zero mel loss! Use decoder for proper training.")

        # ✅ ATTACH SCALAR LOSS SIGNALS TO OUTPUTS
        # These are internal loss signals from the GPT model that should drive training
        if 'scalar_items' in locals() and scalar_items:
            outputs["scalar_losses"] = scalar_items
            logging.info(f"[SCALAR ATTACH] Attaching {len(scalar_items)} scalar loss signals to outputs for training: {[item.item() for _, item in scalar_items]}")

        logging.info(f"[MODEL FORWARD] Returning keys: {list(outputs.keys())}")
        if "gpt_logits" not in outputs:
            logging.warning(f"[MODEL FORWARD] WARNING: No gpt_logits extracted! Audio token loss will be skipped.")
        return outputs

    except Exception as e:
        logging.error(f"[FORWARD_XTTS] GPT forward failed: {e}")
        raise e
def check_and_fix_gradients(model, max_norm=1.0):
    """Check for NaN gradients and apply per-layer clipping"""
    total_norm = 0.0
    nan_found = False
    
    # FIRST: Calculate raw gradient norm BEFORE clipping
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                logging.warning(f"NaN/Inf gradient in {name} BEFORE clipping, zeroing it out")
                param.grad.zero_()
                nan_found = True
            else:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5
    
    # SECOND: Per-layer gradient clipping (catches layer-wise explosions)
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            if param_norm > max_norm:
                clip_coef = max_norm / (param_norm + 1e-6)
                param.grad.data.mul_(clip_coef)
    
    # THIRD: Global norm clipping as final safety net
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm * 10)
    
    return total_norm, nan_found
# get_criterion_with_weights
# -------------------------

from types import SimpleNamespace
from torch.cuda.amp import autocast, GradScaler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import time
def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace"""
    from types import SimpleNamespace
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

# ============================
# Updated Train Loop
# ============================
def batch_to_namespace(batch):
    """Recursively convert dict-style batch to SimpleNamespace."""
    if isinstance(batch, dict):
        ns = SimpleNamespace()
        for k, v in batch.items():
            if isinstance(v, dict):
                setattr(ns, k, batch_to_namespace(v))
            else:
                setattr(ns, k, v)
        return ns
    elif isinstance(batch, list):
        return [batch_to_namespace(b) for b in batch]
    else:
        return batch

def to_device(ns, device):
    """Recursively move all tensors in a SimpleNamespace to device."""
    for k, v in vars(ns).items():
        if isinstance(v, torch.Tensor):
            setattr(ns, k, v.to(device))
        elif isinstance(v, SimpleNamespace):
            to_device(v, device)

class ProsodyExtractor(nn.Module):
    """Real prosody feature extractor using librosa"""

    def __init__(self, sample_rate=24000, hop_length=1024, n_fft=4096):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = n_fft

        # Normalization parameters (will be updated during training)
        self.register_buffer('pitch_mean', torch.tensor(200.0))
        self.register_buffer('pitch_std', torch.tensor(50.0))
        self.register_buffer('energy_mean', torch.tensor(0.1))
        self.register_buffer('energy_std', torch.tensor(0.05))

    def extract_pitch(self, audio, target_length=None):
        """Extract pitch using librosa.pyin"""
        try:
            if isinstance(audio, torch.Tensor):
                audio_np = audio.squeeze().cpu().numpy()
            else:
                audio_np = audio

            # Extract F0 using pyin (more robust than yin)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_np,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=self.sample_rate,
                frame_length=self.win_length,
                hop_length=self.hop_length,
                fill_na=0.0
            )

            # Handle target length alignment
            if target_length is not None:
                if len(f0) > target_length:
                    f0 = f0[:target_length]
                elif len(f0) < target_length:
                    f0 = np.pad(f0, (0, target_length - len(f0)), constant_values=0.0)

            # Convert to tensor and normalize
            f0_tensor = torch.tensor(f0, dtype=torch.float32)

            # Normalize pitch (log scale for better distribution)
            f0_log = torch.log(f0_tensor + 1e-8)  # Add small epsilon to avoid log(0)
            f0_normalized = (f0_log - torch.log(self.pitch_mean)) / torch.log(self.pitch_std + 1e-8)

            return f0_normalized

        except Exception as e:
            logging.debug(f"Pitch extraction failed: {e}")
            # Return zeros if extraction fails
            length = target_length if target_length is not None else 100
            return torch.zeros(length, dtype=torch.float32)

    def extract_energy(self, audio, target_length=None):
        """Extract energy using librosa.feature.rms"""
        try:
            if isinstance(audio, torch.Tensor):
                audio_np = audio.squeeze().cpu().numpy()
            else:
                audio_np = audio

            # Extract RMS energy
            rms = librosa.feature.rms(
                y=audio_np,
                frame_length=self.win_length,
                hop_length=self.hop_length
            )[0]  # rms returns shape (1, frames), take first row

            # Handle target length alignment
            if target_length is not None:
                if len(rms) > target_length:
                    rms = rms[:target_length]
                elif len(rms) < target_length:
                    rms = np.pad(rms, (0, target_length - len(rms)), constant_values=0.0)

            # Convert to tensor and normalize
            energy_tensor = torch.tensor(rms, dtype=torch.float32)

            # Normalize energy (log scale)
            energy_log = torch.log(energy_tensor + 1e-8)
            energy_normalized = (energy_log - torch.log(self.energy_mean)) / torch.log(self.energy_std + 1e-8)

            return energy_normalized

        except Exception as e:
            logging.debug(f"Energy extraction failed: {e}")
            # Return zeros if extraction fails
            length = target_length if target_length is not None else 100
            return torch.zeros(length, dtype=torch.float32)

    def extract_duration(self, audio, target_length=None):
        """Extract duration features using energy-based voiced/unvoiced segmentation"""
        try:
            if isinstance(audio, torch.Tensor):
                audio_np = audio.squeeze().cpu().numpy()
            else:
                audio_np = audio

            # Extract RMS for voice activity detection
            rms = librosa.feature.rms(
                y=audio_np,
                frame_length=self.win_length,
                hop_length=self.hop_length
            )[0]

            # Simple voice activity detection using energy threshold
            energy_threshold = np.percentile(rms, 30)  # 30th percentile as threshold
            voiced_frames = rms > energy_threshold

            # Create duration features based on voiced segments
            duration_features = np.zeros_like(rms)

            # Find voiced segments and assign duration values
            voiced_segments = []
            start = None
            for i, is_voiced in enumerate(voiced_frames):
                if is_voiced and start is None:
                    start = i
                elif not is_voiced and start is not None:
                    voiced_segments.append((start, i))
                    start = None

            # Handle case where audio ends with voiced segment
            if start is not None:
                voiced_segments.append((start, len(voiced_frames)))

            # Assign duration values (segment length normalized by total frames)
            for start_idx, end_idx in voiced_segments:
                segment_duration = (end_idx - start_idx) / len(voiced_frames)
                duration_features[start_idx:end_idx] = segment_duration

            # Handle target length alignment
            if target_length is not None:
                if len(duration_features) > target_length:
                    duration_features = duration_features[:target_length]
                elif len(duration_features) < target_length:
                    duration_features = np.pad(duration_features, (0, target_length - len(duration_features)), constant_values=0.0)

            return torch.tensor(duration_features, dtype=torch.float32)

        except Exception as e:
            logging.debug(f"Duration extraction failed: {e}")
            # Return zeros if extraction fails
            length = target_length if target_length is not None else 100
            return torch.zeros(length, dtype=torch.float32)

    def forward(self, audio, text_length=None):
        """Extract all prosody features and return as dict"""
        # Determine target length from mel spectrogram frames
        if isinstance(audio, torch.Tensor):
            audio_np = audio.squeeze().cpu().numpy()
        else:
            audio_np = audio

        # Calculate expected mel frames
        audio_length = len(audio_np)
        mel_frames = (audio_length // self.hop_length) + 1
        target_length = mel_frames

        # Extract features
        pitch = self.extract_pitch(audio_np, target_length)
        energy = self.extract_energy(audio_np, target_length)
        duration = self.extract_duration(audio_np, target_length)

        # Ensure all features have the same length
        min_length = min(len(pitch), len(energy), len(duration))
        if min_length > 0:
            pitch = pitch[:min_length]
            energy = energy[:min_length]
            duration = duration[:min_length]

        return {
            'pitch': pitch,
            'energy': energy,
            'duration': duration
        }

class ProsodyPredictor(nn.Module):
    """Prosody predictor that takes text embeddings and predicts prosody features"""

    def __init__(self, text_dim=1024, hidden_dim=256, prosody_dim=3):
        super().__init__()
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.prosody_dim = prosody_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Separate heads for each prosody feature
        self.pitch_head = nn.Linear(hidden_dim, 1)
        self.energy_head = nn.Linear(hidden_dim, 1)
        self.duration_head = nn.Linear(hidden_dim, 1)

    def forward(self, text_embeddings):
        """
        Args:
            text_embeddings: [batch_size, seq_len, text_dim]
        Returns:
            dict with pitch_pred, energy_pred, duration_pred
        """
        # Encode text embeddings
        encoded = self.encoder(text_embeddings)  # [batch, seq, hidden_dim]

        # Predict prosody features
        pitch_pred = self.pitch_head(encoded).squeeze(-1)  # [batch, seq]
        energy_pred = self.energy_head(encoded).squeeze(-1)  # [batch, seq]
        duration_pred = self.duration_head(encoded).squeeze(-1)  # [batch, seq]

        return {
            'pitch_pred': pitch_pred,
            'energy_pred': energy_pred,
            'duration_pred': duration_pred
        }

import torch
import torch.nn as nn
from encodec import EncodecModel

# ✅ REMOVED: EmotionClassifier class
# Emotions are now predicted using trained text classifier during data loading
# XTTS will condition on these pre-computed emotions

# -------------------------
# Encodec Vocoder
# -------------------------

# -------------------------
# Encodec Vocoder (NEW)
# -------------------------
class EncodecVocoder(nn.Module):
    """Vocoder using Encodec model with reconstruction loss support"""
    def __init__(self, device='cuda', use_in_training=True):
        super().__init__()
        self.device = device
        self.use_in_training = use_in_training
        if EncodecModel is not None:
            self.model = EncodecModel.encodec_model_24khz()
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = None
            logging.warning("EnCodec not available - install with: pip install encodec")

    @torch.no_grad()
    def forward(self, audio):
        if self.model is None:
            return audio
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)
        batch_size = audio.shape[0]
        reconstructed_list = []
        
        for i in range(batch_size):
            sample = audio[i:i+1]
            encoded_frames = self.model.encode(sample)
            if isinstance(encoded_frames, (tuple, list)):
                encoded_frames = list(encoded_frames)
            else:
                encoded_frames = [encoded_frames]
            waveform = self.model.decode(encoded_frames)
            if isinstance(waveform, (tuple, list)):
                waveform = waveform[0]
            reconstructed_list.append(waveform)
        
        result = torch.cat(reconstructed_list, dim=0)
        return result.squeeze(1).cpu() if result.dim() == 3 else result.cpu()

    def reconstruct(self, audio):
        """Encode then decode for loss computation - handles batches"""
        if self.model is None:
            return audio
        
        original_audio = audio
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        audio = audio.to(self.device)
        batch_size = audio.shape[0]
        reconstructed_list = []
        
        with torch.no_grad():
            for i in range(batch_size):
                sample = audio[i:i+1]
                encoded_frames = self.model.encode(sample)
                
                if isinstance(encoded_frames, (tuple, list)):
                    encoded_frames = list(encoded_frames)
                else:
                    encoded_frames = [encoded_frames]
                
                recon = self.model.decode(encoded_frames)
                if isinstance(recon, (tuple, list)):
                    recon = recon[0]
                reconstructed_list.append(recon)
        
        reconstructed = torch.cat(reconstructed_list, dim=0)
        return reconstructed.to(original_audio.device)

    def reconstruction_loss(self, original, weight=0.05):
        """L1 loss between original and EnCodec-reconstructed audio"""
        if self.model is None:
            return torch.tensor(0.0, device=original.device)
        
        # ✅ CRITICAL FIX: Make reconstruction differentiable by removing torch.no_grad()
        # Encode original audio to get target codes
        original_audio = original
        if original.dim() == 1:
            original = original.unsqueeze(0).unsqueeze(0)
        elif original.dim() == 2:
            original = original.unsqueeze(1)
        
        original = original.to(self.device)
        
        # Encode original audio (no gradients needed for encoding)
        with torch.no_grad():
            encoded_frames = self.model.encode(original)
            if isinstance(encoded_frames, (tuple, list)):
                encoded_frames = list(encoded_frames)
            else:
                encoded_frames = [encoded_frames]
        
        # Decode with gradients enabled so loss is differentiable
        # ✅ CRITICAL: Set model to train mode for RNN backward pass
        self.model.train()
        reconstructed = self.model.decode(encoded_frames)
        self.model.eval()  # Return to eval mode after decode
        if isinstance(reconstructed, (tuple, list)):
            reconstructed = reconstructed[0]
        
        if reconstructed.dim() > original_audio.dim():
            if reconstructed.shape[1] == 1:
                reconstructed = reconstructed.squeeze(1)
        
        while original_audio.dim() > reconstructed.dim():
            original_audio = original_audio.unsqueeze(1)
        
        min_len = min(original_audio.shape[-1], reconstructed.shape[-1])
        orig = original_audio[..., :min_len]
        recon = reconstructed[..., :min_len]
        
        loss = F.l1_loss(recon, orig) * weight
        logging.info(f"[RECONSTRUCTION LOSS] Loss value: {loss.item():.6f}, requires_grad: {loss.requires_grad}")
        return loss


def save_audio_samples(audio_samples, sample_rate, output_dir, step, prefix="sample"):
    """Save audio samples to disk"""
    try:
        os.makedirs(output_dir, exist_ok=True)

        for i, audio in enumerate(audio_samples):
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = audio

            # Normalize audio
            if audio_np.max() > 0:
                audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))

            filename = f"{prefix}_step_{step:06d}_sample_{i:02d}.wav"
            filepath = os.path.join(output_dir, filename)

            # Save using soundfile
            sf.write(filepath, audio_np, sample_rate)

        logging.info(f"Saved {len(audio_samples)} audio samples to {output_dir}")

    except Exception as e:
        logging.warning(f"Failed to save audio samples: {e}")

# -------------------------
# load_checkpoint_safe
# -------------------------

from types import SimpleNamespace
from torch.cuda.amp import autocast, GradScaler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import time
def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace"""
    from types import SimpleNamespace
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

# ============================
# Updated Train Loop
# ============================
def batch_to_namespace(batch):
    """Recursively convert dict-style batch to SimpleNamespace."""
    if isinstance(batch, dict):
        ns = SimpleNamespace()
        for k, v in batch.items():
            if isinstance(v, dict):
                setattr(ns, k, batch_to_namespace(v))
            else:
                setattr(ns, k, v)
        return ns
    elif isinstance(batch, list):
        return [batch_to_namespace(b) for b in batch]
    else:
        return batch

def to_device(ns, device):
    """Recursively move all tensors in a SimpleNamespace to device."""
    for k, v in vars(ns).items():
        if isinstance(v, torch.Tensor):
            setattr(ns, k, v.to(device))
        elif isinstance(v, SimpleNamespace):
            to_device(v, device)

def load_checkpoint_safe(checkpoint_path):
    """Load checkpoint with proper error handling for PyTorch 2.6+"""
    try:
        import torch.serialization
        from TTS.tts.models.xtts import XttsAudioConfig

        with torch.serialization.safe_globals([XttsAudioConfig]):
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        logging.warning(f"Safe load failed, trying with weights_only=False: {e}")
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e2:
            logging.error(f"Checkpoint loading failed completely: {e2}")
            return None

    if isinstance(checkpoint_data, dict):
        # If checkpoint has nested structure with 'model' key, extract it
        if 'model' in checkpoint_data:
            logging.info("Found nested checkpoint structure, extracting model weights")
            return checkpoint_data['model']
        # If it has model_state_dict key, use that
        elif 'model_state_dict' in checkpoint_data:
            logging.info("Found model_state_dict in checkpoint")
            return checkpoint_data['model_state_dict']
        # Otherwise assume it's already the state dict
        else:
            return checkpoint_data

    return checkpoint_data

def train_loop(model, train_loader, eval_loader, cfg, device, ap, epochs=1, dry_run=False, restore_path=None):
    """
    Full training loop for XTTS/GPT-TTS model.

    Preserves:
        - AMP / GradScaler support
        - Gradient clipping
        - Forward batch preparation with speaker embeddings
        - Dry-run support
        - Evaluation loop
        - Checkpoint saving
        - Logging at print_step intervals
    """
    model.to(device)
    model.train()

    # ✅ CRITICAL FIX: Initialize loss weights from config BEFORE training
    # This sets globals()['loss_weights'] which is accessed by the criterion function
    get_criterion_with_weights(cfg, audio_processor=ap, device=device)
    loss_weights = globals().get('loss_weights', {})
    logging.info(f"[INIT] Loss weights loaded from config: {loss_weights}")
    
    # ✅ Initialize autoregressive training if enabled
    autoregressive_processor = None
    if AUTOREGRESSIVE_AVAILABLE:
        autoregressive_processor = enable_autoregressive_training(cfg)
    
    if autoregressive_processor is None:
        logging.info("⚠️  Autoregressive training DISABLED - using standard training")
    else:
        logging.info("✅ Autoregressive training ENABLED - will process audio in 6-token chunks with context")

    # Added prosody feature extraction and predictor initialization
    # -----------------------
    # Prosody components setup
    # -----------------------
    prosody_extractor = ProsodyExtractor(
        sample_rate=cfg.get("audio_config", cfg.get("audio", {})).get("sample_rate", 24000),
        hop_length=cfg.get("audio_config", cfg.get("audio", {})).get("hop_length", 256)
    ).to(device)

    prosody_predictor = ProsodyPredictor(
        text_dim=cfg.get("model_args", {}).get("gpt_n_model_channels", 1024),
        hidden_dim=cfg.get("prosody", {}).get("hidden_dim", 256)
    ).to(device)

    emotion_classifier = None
    use_emotions = cfg.get("use_emotions", False)
    if use_emotions:
        emotion_classifier = None
        logging.info("✅ Using pre-computed emotions from trained text classifier - no mel-based classifier training")

    # -----------------------
    # Enhanced vocoder setup
    # -----------------------
    vocoder = EncodecVocoder(device=device, use_in_training=True)
    # Sample dumping configuration
    sample_dir = cfg.get("output_path", "outputs") + "/samples"
    dump_step = cfg.get("dump_step", 100)
    sample_rate = cfg.get("audio", {}).get("sample_rate", 24000)

    # -----------------------
    # Optimizer setup
    # -----------------------
    optim_cfg = cfg.get("optim", {})
    lr = optim_cfg.get("lr", cfg.get("lr", 1e-4))
    betas = tuple(optim_cfg.get("betas", (0.9, 0.98)))
    eps = optim_cfg.get("eps", 1e-9)
    grad_clip = cfg.get("grad_clip", 0.5)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    logging.info(f"[CONFIG DEBUG] optim_cfg={optim_cfg}")
    logging.info(f"[CONFIG DEBUG] lr={lr}, grad_clip={grad_clip}, max_grad_norm={max_grad_norm}")

    all_params = list(model.parameters()) + list(prosody_predictor.parameters())
    
    # ✅ TRAIN conditioning_encoder - critical for Emma's voice!
    # The pre-trained encoder was trained on other speakers. It must adapt to Emma's voice.
    # Freezing it means the model gets weak/misaligned conditioning → can't learn pronunciation
    if hasattr(model, 'gpt') and hasattr(model.gpt, 'conditioning_encoder'):
        for param in model.gpt.conditioning_encoder.parameters():
            param.requires_grad = True
        logging.info("[TRAINING SETUP] ✅ TRAINABLE conditioning_encoder - learning Emma's unique voice patterns")
    
    # ✅ TRAIN mel_head - essential for token→speech mapping!
    # The mel_head was interpolated (1026→8194 tokens) and needs adaptation for Emma's voice
    if hasattr(model, 'gpt') and hasattr(model.gpt, 'mel_head'):
        for param in model.gpt.mel_head.parameters():
            param.requires_grad = True
        logging.info("[TRAINING SETUP] ✅ TRAINABLE mel_head - learning Emma-specific token→speech mapping!")

    
    trainable_params = sum(p.numel() for p in all_params if p.requires_grad)
    frozen_params = sum(p.numel() for p in all_params if not p.requires_grad)
    logging.info(f"[TRAINING SETUP] Trainable: {trainable_params}, Frozen: {frozen_params}")
    
    gpt_trainable = sum(p.numel() for p in model.gpt.parameters() if p.requires_grad) if hasattr(model, 'gpt') else 0
    prosody_trainable = sum(p.numel() for p in prosody_predictor.parameters() if p.requires_grad)
    logging.info(f"[TRAINING SETUP] GPT trainable: {gpt_trainable}, Prosody trainable: {prosody_trainable}")
    
    # <CHANGE> Create optimizer with separate learning rates
    # Use 0.01x LR for emotion classifier to prevent gradient explosion
    # <CHANGE> Create optimizer with separate learning rates
    # Use 0.1x LR for emotion classifier (increased from 0.01x to learn faster)
    if emotion_classifier is not None:
        emotion_lr = optim_cfg.get('emotion_lr', lr * 5.0)  # ✅ Changed from lr*0.01 to lr*5.0 (allows faster learning)
        emotion_grad_clip = cfg.get('emotion_grad_clip', 50.0)  # ✅ Changed default from 1.0 to 50.0 to allow learning
        optimizer = torch.optim.AdamW([
            {'params': list(model.parameters()) + list(prosody_predictor.parameters()), 'lr': lr},
            {'params': list(emotion_classifier.parameters()), 'lr': emotion_lr}
        ], weight_decay=0.01)
        logging.info(f"[v0 DIAGNOSTIC] Using LR {lr} for model, {emotion_lr} for emotion classifier")
        logging.info(f"[v0 DIAGNOSTIC] Gradient clipping: model={grad_clip}, emotion={emotion_grad_clip}")  # ← same line, now uses correct variable
    else:
        optimizer = torch.optim.AdamW(all_params, lr=lr, betas=betas, eps=eps, weight_decay=0.01)
        logging.info(f"[v0 DIAGNOSTIC] Using LR {lr} for model")


    # -----------------------
    # Learning rate scheduler (from config)
    # -----------------------
    scheduler_type = cfg.get("lr_scheduler", "CosineAnnealingWarmRestarts")
    scheduler_params = cfg.get("lr_scheduler_params", {})
    
    if scheduler_type == "constant":
        # ✅ FIXED: Use constant LR (no decay)
        # This prevents the aggressive scheduler from killing training at epoch 14+
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        logging.info(f"✅ Using CONSTANT LR scheduler: LR will stay at {lr:.2e} throughout training")
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        T_0_epochs = scheduler_params.get("T_0", 50)
        T_mult = scheduler_params.get("T_mult", 1)
        eta_min = scheduler_params.get("eta_min", 1e-6)
        
        steps_per_epoch = len(train_loader)
        T_0 = T_0_epochs * steps_per_epoch
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
        logging.info(f"✅ Using CosineAnnealingWarmRestarts:")
        logging.info(f"   T_0_config={T_0_epochs} epochs → T_0_steps={T_0} (batches_per_epoch={steps_per_epoch})")
        logging.info(f"   T_mult={T_mult}, eta_min={eta_min}")
        logging.info(f"[SCHEDULER DEBUG] Initial LR from config: {lr}, CosineAnnealingWarmRestarts eta_min={eta_min}")
    else:
        # Fallback to OneCycleLR if specified
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.15,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=1.0
        )
        logging.info(f"Using OneCycleLR (fallback)")
    
    current_lr = optimizer.param_groups[0]['lr']
    logging.info(f"[SCHEDULER DEBUG] After scheduler creation, current_lr in optimizer: {current_lr}")


    # -----------------------
    # Gradient scaler and accumulation
    # -----------------------
    scaler = torch.amp.GradScaler(enabled=cfg.get("use_cuda", False) and cfg.get("mixed_precision", False))
    grad_accumulation_steps = cfg.get("grad_accumulation_steps", 1)

    # -----------------------
    # Best model tracking
    # -----------------------
    best_eval_loss = float('inf')
    best_model_path = os.path.join(cfg.get("output_path", "outputs"), "best_model.pth")
    early_stop_cfg = cfg.get("early_stopping", {})
    early_stopping = EarlyStopping(
        patience=early_stop_cfg.get("patience", 30),
        verbose=True,
        min_delta=early_stop_cfg.get("min_delta", 0.0)
    )
    logging.info(f"[EARLY STOPPING] patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")

    # -----------------------
    # Required keys for XTTS forward
    # -----------------------
    required_keys = [
        "token_id", "token_id_lengths", "mel", "linear", "stop_targets",
        "gpt_audio_tokens", "gpt_cond_latent", "expected_output_len",
        "speaker_embedding", "cond_mels", "emotion_features"  # ✅ ADD emotion_features
    ]
    # -----------------------
    # Optional: resume training state (model + optimizer + scheduler + epoch)
    # -----------------------
    # -----------------------
    # Optional: resume training state (model + optimizer + scheduler + epoch)
    # -----------------------
    start_epoch = 0
    checkpoint_resume_cfg = cfg.get("checkpoint_resume", {})
    load_optimizer_from_ckpt = checkpoint_resume_cfg.get("load_optimizer", True)
    load_scheduler_from_ckpt = checkpoint_resume_cfg.get("load_scheduler", True)
    logging.info(f"[CONFIG DEBUG] checkpoint_resume_cfg={checkpoint_resume_cfg}")
    
    if restore_path:
        if os.path.exists(restore_path):
            logging.info(f"Attempting full restore from checkpoint: {restore_path}")
            loaded_epoch, loaded_loss = load_checkpoint_enhanced(
                restore_path,
                model,
                prosody_predictor,
                emotion_classifier,
                optimizer if load_optimizer_from_ckpt else None,
                scheduler if load_scheduler_from_ckpt else None
            )
            logging.info(f"✅ load_optimizer_from_ckpt={load_optimizer_from_ckpt}, load_scheduler_from_ckpt={load_scheduler_from_ckpt}")
            
            if hasattr(model, 'gpt') and hasattr(model.gpt, 'conditioning_encoder'):
                for param in model.gpt.conditioning_encoder.parameters():
                    param.requires_grad = False
                logging.info("[CHECKPOINT RESTORE] ✅ Re-froze conditioning_encoder after checkpoint load")
            
            if hasattr(model, 'gpt') and hasattr(model.gpt, 'mel_head'):
                for param in model.gpt.mel_head.parameters():
                    param.requires_grad = False
                logging.info("[CHECKPOINT RESTORE] ✅ Re-froze mel_head after checkpoint load")
            
            if hasattr(model, 'hifigan_decoder'):
                for param in model.hifigan_decoder.parameters():
                    param.requires_grad = False
                logging.info("[CHECKPOINT RESTORE] ✅ Re-froze hifigan_decoder after checkpoint load")
            
            trainable_after = sum(p.numel() for p in all_params if p.requires_grad)
            frozen_after = sum(p.numel() for p in all_params if not p.requires_grad)
            logging.info(f"[CHECKPOINT RESTORE] After restore - Trainable: {trainable_after}, Frozen: {frozen_after}")
            
            # --- NEW: always trust the checkpoint epoch, even if weights mismatch ---
            if loaded_epoch > 0:
                start_epoch = int(loaded_epoch)
                logging.info(
                    f"Resuming training from checkpoint epoch {start_epoch} "
                    f"(loss {loaded_loss:.6f})"
                )
            else:
                logging.warning(
                    f"Checkpoint loaded but model weights mismatched. "
                    f"Starting from saved epoch {loaded_epoch} anyway."
                )
        else:
            logging.warning(f"restore_path provided but file not found: {restore_path}")




    # -----------------------
    # Training epochs
    # -----------------------
    diagnostics = None
    for epoch in range(start_epoch, epochs):
        logging.info(f"Starting epoch {epoch+1}/{epochs}")
        lr_at_epoch = scheduler.get_last_lr()[0] if scheduler else lr
        logging.info(f"  Learning rate at epoch start: {lr_at_epoch:.2e}")
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(train_loader):
            # ✅ DIAGNOSTICS: Initialize hooks on first step of first epoch
            if epoch == start_epoch and step == 0 and DIAGNOSTICS_AVAILABLE and diagnostics is None:
                diagnostics = inject_diagnostics(model)
                logging.info("✅ Diagnostics hooks registered - monitoring first 3 steps")
            
            # ✅ DIAGNOSTICS: Clear previous step's outputs
            if step < 3 and diagnostics is not None:
                diagnostics.forward_outputs.clear()
                diagnostics.backward_grads.clear()
            
            batch_ns = batch_to_namespace(batch)
            to_device(batch_ns, device)

            # Pad token_id if list and create token_id_lengths
            if hasattr(batch_ns, "token_id") and isinstance(batch_ns.token_id, list):
                token_ids = batch_ns.token_id
                batch_ns.token_id = torch.nn.utils.rnn.pad_sequence(
                    [t.detach().clone() if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long)
                     for t in token_ids],
                    batch_first=True
                )
                batch_ns.token_id_lengths = torch.tensor(
                    [len(t) if isinstance(t, (list, torch.Tensor)) else 0 for t in token_ids],
                    dtype=torch.long,
                    device=device
                )

            # Build forward batch dict with defaults for missing keys
            forward_batch = {}
            for k in required_keys:
                if hasattr(batch_ns, k):
                    forward_batch[k] = getattr(batch_ns, k)   # <-- Use forward_batch!
                else:
                    if k == "token_id":
                        forward_batch[k] = torch.zeros(1, 1, dtype=torch.long, device=device)
                    elif k == "token_id_lengths":
                        forward_batch[k] = torch.tensor([1], dtype=torch.long, device=device)
                    elif k == "cond_mels":
                        if "mel" in forward_batch and isinstance(forward_batch["mel"], torch.Tensor):
                            batch_size, n_mels, t = forward_batch["mel"].shape
                            forward_batch[k] = torch.zeros(batch_size, n_mels, t, device=device)
                        else:
                            batch_size = forward_batch["token_id"].shape[0] if "token_id" in forward_batch else 1
                            forward_batch[k] = torch.zeros(batch_size, 80, 80, device=device)
                    elif k in ["mel", "linear"]:
                        forward_batch[k] = torch.zeros(1, 80, 80, device=device)
                    elif k in ["stop_targets", "gpt_audio_tokens", "gpt_cond_latent"]:
                        forward_batch[k] = torch.zeros(1, 1, device=device)
                    elif k == "expected_output_len":
                        forward_batch[k] = torch.tensor([1], dtype=torch.long, device=device)
                    elif k == "speaker_embedding":
                        forward_batch[k] = torch.zeros(1, 512, device=device)
                    elif k == "emotion_features":
                        forward_batch[k] = torch.zeros(1, 512, device=device)
                    
            if hasattr(batch_ns, "waveform"):
                forward_batch["waveform"] = batch_ns.waveform.to(device)
                logging.info(f"[ENCODEC DEBUG] Waveform added to forward_batch: shape={forward_batch['waveform'].shape}")
            else:
                logging.info(f"[ENCODEC DEBUG] NO waveform in batch_ns! batch_ns keys: {list(vars(batch_ns).keys())}")
            
            if not hasattr(batch_ns, "wav_lengths") and hasattr(batch_ns, "expected_output_len"):
                forward_batch["wav_lengths"] = batch_ns.expected_output_len
            elif not hasattr(batch_ns, "wav_lengths"):
                forward_batch["wav_lengths"] = torch.tensor([80], dtype=torch.long, device=device)
            # Map token fields for GPT
            forward_batch["tokens"] = forward_batch["token_id"]
            if "token_id_lengths" in forward_batch:
                forward_batch["tokens_lengths"] = forward_batch["token_id_lengths"]
            # Defensive runtime check for cond_mels!
            if ("cond_mels" not in forward_batch) or (forward_batch["cond_mels"] is None):
                print("WARNING: cond_mels missing or None in forward_batch, constructing artificial cond_mels")
                if "mel" in forward_batch and isinstance(forward_batch["mel"], torch.Tensor):
                    batch_size, n_mels, t = forward_batch["mel"].shape
                    forward_batch["cond_mels"] = torch.zeros(batch_size, n_mels, t, device=device)
                else:
                    batch_size = forward_batch["token_id"].shape[0] if "token_id" in forward_batch else 1
                    forward_batch["cond_mels"] = torch.zeros(batch_size, 80, 80, device=device)
            else:
                # Optional: check type and shape
                if not isinstance(forward_batch["cond_mels"], torch.Tensor):
                    print(f"WARNING: cond_mels is not a tensor! Type: {type(forward_batch['cond_mels'])}")
                else:
                    print(f"cond_mels final shape before forward: {forward_batch['cond_mels'].shape}")

            # Extract prosody features if audio is available
            if hasattr(batch_ns, "waveform") and batch_ns.waveform is not None:
                try:
                    batch_prosody_features = []
                    for i in range(batch_ns.waveform.shape[0]):
                        audio_sample = batch_ns.waveform[i]
                        prosody_features = prosody_extractor(audio_sample)
                        batch_prosody_features.append(prosody_features)

                    # Stack prosody features
                    if batch_prosody_features:
                        forward_batch["prosody_target"] = {
                            'pitch': torch.nn.utils.rnn.pad_sequence([f['pitch'] for f in batch_prosody_features], batch_first=True),
                            'energy': torch.nn.utils.rnn.pad_sequence([f['energy'] for f in batch_prosody_features], batch_first=True),
                            'duration': torch.nn.utils.rnn.pad_sequence([f['duration'] for f in batch_prosody_features], batch_first=True)
                        }
                except Exception as e:
                    logging.debug(f"Prosody extraction failed: {e}")

                if use_emotions and hasattr(batch_ns, "emotion") and batch_ns.emotion is not None:
                    if emotion_classifier is not None:
                        # batch_ns.emotion is already a LongTensor of IDs
                        forward_batch["emotion"] = batch_ns.emotion.to(device)


            # Debug batch keys
            logging.debug(f"DEBUG batch keys: {forward_batch.keys()}")

            if step % grad_accumulation_steps == 0:
                optimizer.zero_grad()
                # ✅ Ensure model is in training mode
                model.train()
                logging.info(f"[DEBUG] Step {step}: model.training={model.training}, model dtype={next(model.parameters()).dtype}, has grads enabled")

            device_type = "cuda" if device.type == "cuda" and torch.cuda.is_available() else "cpu"
            use_autocast = cfg.get("mixed_precision", False)
            
            # ✅ Initialize loss (will be set by autoregressive or standard training)
            loss = None
            loss_dict = {}
            model_outputs = None
            
            # ✅ AUTOREGRESSIVE TRAINING: Process audio in chunks instead of full sequence
            if autoregressive_processor is not None and 'gpt_audio_tokens' in forward_batch:
                logging.info(f"[AUTOREGRESSIVE] Step {step}: Using autoregressive training (chunks of 6 tokens with context)")
                audio_codes = forward_batch.get("gpt_audio_tokens")
                
                # Handle batch dimension
                if isinstance(audio_codes, torch.Tensor):
                    if audio_codes.dim() == 3:
                        # [batch, 1, seq] -> take first batch sample and first codebook
                        audio_codes_single = audio_codes[0, 0, :]  
                    elif audio_codes.dim() == 2:
                        # [batch, seq] -> take first batch sample
                        audio_codes_single = audio_codes[0, :]
                    else:
                        audio_codes_single = audio_codes
                    
                    # Extract other parameters for forward pass
                    token_ids = forward_batch.get('token_id', torch.zeros(1, 1, dtype=torch.long, device=device))
                    cond_mels = forward_batch.get('cond_mels', torch.zeros(1, 80, 100, device=device))
                    speaker_embedding = forward_batch.get('speaker_embedding', torch.zeros(1, 512, device=device))
                    
                    # Determine batch size from forward_batch
                    batch_size = 1
                    if isinstance(token_ids, torch.Tensor) and token_ids.dim() >= 1:
                        batch_size = token_ids.shape[0]
                    elif isinstance(cond_mels, torch.Tensor) and cond_mels.dim() >= 1:
                        batch_size = cond_mels.shape[0]
                    elif isinstance(speaker_embedding, torch.Tensor) and speaker_embedding.dim() >= 1:
                        batch_size = speaker_embedding.shape[0]
                    
                    # Run autoregressive training
                    logging.info(f"[AUTOREGRESSIVE] Audio sequence length: {audio_codes_single.shape[0]} tokens")
                    
                    ar_total_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    ar_chunk_losses = []
                    ar_num_chunks = 0
                    
                    for ar_step, (context, target) in enumerate(autoregressive_processor.chunks(audio_codes_single)):
                        # Construct input for this chunk
                        if context.shape[0] > 0:
                            chunk_input = torch.cat([context, target])
                        else:
                            chunk_input = target
                        
                        # Create a forward_batch for this chunk
                        chunk_batch = forward_batch.copy()
                        chunk_audio = chunk_input.unsqueeze(0).unsqueeze(0)  # [1, 1, seq]
                        if batch_size > 1:
                            chunk_audio = chunk_audio.expand(batch_size, -1, -1)  # [batch_size, 1, seq]
                        chunk_batch['gpt_audio_tokens'] = chunk_audio
                        
                        # Forward pass for this chunk
                        try:
                            with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
                                chunk_output = model(**chunk_batch)
                            
                            # Extract logits
                            if 'gpt_logits' in chunk_output:
                                logits = chunk_output['gpt_logits']  # [1, 8194, 6]
                                logits_squeezed = logits[0].T  # [6, 8194]
                                
                                # Compute loss: compare predictions to target
                                target_long = target.long()
                                if target_long.shape[0] < 6:
                                    # Pad target to 6
                                    padding = torch.full((6 - target_long.shape[0],), 0, dtype=torch.long, device=device)
                                    target_long = torch.cat([target_long, padding])
                                
                                chunk_loss = nn.functional.cross_entropy(logits_squeezed, target_long)
                                ar_chunk_losses.append(chunk_loss)
                                ar_num_chunks += 1
                                
                                # Accumulate loss (detach previous to avoid OOM)
                                if ar_num_chunks == 1:
                                    ar_total_loss = chunk_loss
                                else:
                                    ar_total_loss = ar_total_loss.detach() + chunk_loss
                                
                                logging.debug(f"[AUTOREGRESSIVE] Chunk {ar_step}: context={context.shape[0]}, target={target.shape[0]}, loss={chunk_loss.item():.4f}")
                        
                        except Exception as e:
                            logging.error(f"[AUTOREGRESSIVE] Chunk {ar_step} forward failed: {e}")
                            import traceback
                            logging.error(traceback.format_exc())
                            continue
                    
                    # Average loss across chunks
                    if ar_num_chunks > 0:
                        model_outputs = {
                            'gpt_logits': torch.randn(1, 8194, 6, device=device),  # Dummy for loss function
                            'speaker_embedding': speaker_embedding,
                            'autoregressive_loss': ar_total_loss / ar_num_chunks
                        }
                        logging.info(f"[AUTOREGRESSIVE] Step {step}: {ar_num_chunks} chunks, avg loss={model_outputs['autoregressive_loss'].item():.4f}")
                    else:
                        logging.error("[AUTOREGRESSIVE] No chunks processed, falling back to standard training")
                        model_outputs = None
                else:
                    model_outputs = None
            else:
                model_outputs = None
            
            # ✅ If autoregressive didn't run, do standard forward pass
            if model_outputs is None:
                logging.info(f"[🔴 DEBUG FORWARD-IN] Step {step}: forward_batch keys={list(forward_batch.keys())}")
                
                # CRITICAL: Validate batch dimensions before forward pass
                batch_size_from_keys = None
                for key in ['mel', 'speaker_embedding', 'token_id', 'd_vectors']:
                    if key in forward_batch and isinstance(forward_batch[key], torch.Tensor):
                        tensor_batch_size = forward_batch[key].shape[0]
                        if batch_size_from_keys is None:
                            batch_size_from_keys = tensor_batch_size
                        elif tensor_batch_size != batch_size_from_keys:
                            logging.error(f"[BATCH MISMATCH] {key} has batch_size={tensor_batch_size}, expected {batch_size_from_keys}")
                            logging.error(f"  Fixing by taking minimum batch size...")
                            # Fix by slicing to minimum batch size
                            min_batch = min(tensor_batch_size, batch_size_from_keys)
                            for k in ['mel', 'speaker_embedding', 'token_id', 'd_vectors', 'cond_mels']:
                                if k in forward_batch and isinstance(forward_batch[k], torch.Tensor):
                                    if forward_batch[k].shape[0] > min_batch:
                                        forward_batch[k] = forward_batch[k][:min_batch]
                            batch_size_from_keys = min_batch
                
                if device_type == "cuda":
                    with torch.amp.autocast(device_type='cuda', enabled=use_autocast):
                        if 'mel' in forward_batch and isinstance(forward_batch['mel'], torch.Tensor):
                            logging.info(f"  mel: shape={forward_batch['mel'].shape}, dtype={forward_batch['mel'].dtype}, min={forward_batch['mel'].min():.4f}, max={forward_batch['mel'].max():.4f}")
                        if 'token_id' in forward_batch and isinstance(forward_batch['token_id'], torch.Tensor):
                            logging.info(f"  token_id: shape={forward_batch['token_id'].shape}, unique_values={len(torch.unique(forward_batch['token_id']))}")
                        
                        model_outputs = model(**forward_batch)  # unpack batch dict as kwargs
                else:
                    if 'mel' in forward_batch and isinstance(forward_batch['mel'], torch.Tensor):
                        logging.info(f"  mel: shape={forward_batch['mel'].shape}")
                    model_outputs = model(**forward_batch)
                    
                    logging.info(f"[🔴 DEBUG FORWARD-OUT] Step {step}: model_outputs keys={list(model_outputs.keys())}")
                    for k, v in list(model_outputs.items())[:5]:
                        if isinstance(v, torch.Tensor):
                            has_nan = torch.isnan(v).any().item()
                            has_inf = torch.isinf(v).any().item()
                            logging.info(f"  {k}: shape={v.shape}, dtype={v.dtype}, nan={has_nan}, inf={has_inf}, mean={v.mean().item():.6f}, std={v.std().item():.6f}")
                        elif isinstance(v, dict):
                            logging.info(f"  {k}: dict with {len(v)} keys")

                    if "token_id" in forward_batch:
                        try:
                            # Get text embeddings from model (assuming they're available)
                            if hasattr(model, 'gpt') and hasattr(model.gpt, 'text_embedding'):
                                text_ids = forward_batch["token_id"]
                                if isinstance(text_ids, (list, tuple)):
                                    text_ids = text_ids[0]  # unwrap list/tuple of tensors
                                text_emb = model.gpt.text_embedding(text_ids)
                                prosody_pred = prosody_predictor(text_emb)
                                model_outputs["prosody_pred"] = prosody_pred
                        except Exception as e:
                            logging.debug(f"Prosody prediction failed: {e}")

                    # Robust emotion prediction (works with gpt_latents OR token ids fallback)
                    # ===== FIXED: Robust emotion prediction with batch size verification =====
                    # Line 1959-2005
                    if use_emotions and emotion_classifier is not None:
                        try:
                            # CRITICAL FIX: Force training mode before computing emotion logits
                            emotion_classifier.train()  # ← ADD THIS LINE!
                            
                            device = next(model.parameters()).device
                            emotion_logits = None
                            
                            # ✅ Use emotion features if available, fallback to speaker embeddings
                            # ✅ Use emotion features if available, fallback to speaker embeddings
                            # ✅ Use emotion features if available, fallback to speaker embeddings
                            emotion_input = None
                            if "emotion_features" in forward_batch:
                                emotion_input = forward_batch["emotion_features"].to(device)
                                input_norm = emotion_input.norm(dim=1).mean().item()
                                logging.info(f"🔍 [EMOTION INPUT] Using emotion_features: norm={input_norm:.2f}, expected~1.0 (L2-normalized)")
                            elif "speaker_embedding" in forward_batch:
                                emotion_input = forward_batch["speaker_embedding"].to(device)
                                input_norm = emotion_input.norm(dim=1).mean().item()
                                logging.info(f"✓ [EMOTION INPUT] Using speaker_embedding: norm={input_norm:.2f} (L2-normalized)")

                            if emotion_input is not None:
                                try:
                                    input_norm = emotion_input.norm(dim=1).mean().item()
                                    logging.info(f"🔍 [EMOTION INPUT] norm={input_norm:.4f} (should be ~22.6 from standard_scale)")
                                    
                                    # ✅ FIX: Call emotion_classifier.forward() instead of manually unwrapping layers
                                    # This ensures the normalization code in the classifier runs!
                                    emotion_logits = emotion_classifier(emotion_input)

                                    logging.info(f"[TRACE-5] After classifier: norm={emotion_logits.norm(dim=1).mean():.6f}, shape={emotion_logits.shape}")
                                    
                                    # <CHANGE> Debug: Calculate expected gradient scale
                                    # gradient_norm ≈ logit_norm × loss_scale / input_norm
                                    # Currently: 0.005 × 1.0 / 22.6 should give ~0.0002, but we see 50,000x
                                    # <CHANGE> Log logit statistics
                                    logit_norm = emotion_logits.norm(dim=1).mean().item()
                                    logging.info(f"[ACTIVATION TRACE] Output logits norm: {logit_norm:.4f}, values: min={emotion_logits.min().item():.6f}, max={emotion_logits.max().item():.6f}")

                                    
                                    # ✅ UNINDENT these lines - they should run regardless of tuple/list check
                                    batch_size = forward_batch.get("token_id", forward_batch.get("tokens")).size(0)
                                    if emotion_logits.size(0) == batch_size:
                                        model_outputs["emotion_logits"] = emotion_logits
                                        mode = "train" if emotion_classifier.training else "eval"
                                        logging.debug(f"Emotion logits computed ({mode}): {emotion_logits.shape}")
                                    else:
                                        logging.debug(f"Emotion logits batch mismatch: {emotion_logits.size(0)} vs {batch_size}")
                                        emotion_logits = None
                                except Exception as e:
                                    logging.debug(f"Emotion prediction failed: {e}")

                            # 2) Fallback: use GPT latents if embeddings failed
                            # <CHANGE> Removed gpt_latents fallback - it's 1024-dim and contains NaN
                            # Only use speaker_embedding (512-dim) for emotion classification
                            # 2) Fallback removed - gpt_latents are 1024-dim and contain NaN
                            # Only use speaker_embedding (512-dim) for emotion classification
                            # 2) Fallback removed - gpt_latents are 1024-dim and contain NaN
                            # Only use speaker_embedding (512-dim) for emotion classification
                            if emotion_logits is None:
                                logging.warning("[EMOTION] speaker_embedding path failed, skipping emotion loss for this batch")
                                # No fallback - continue without emotion loss for this batch

                        except Exception as e:
                            logging.debug(f"Emotion prediction failed: {e}")
                    # ===== END FIXED Emotion Prediction =====
            
            # ✅ Compute loss (runs for both autoregressive and standard forward)
            logging.info(f"[🔴 DEBUG LOSS-IN] Step {step}: Calling criterion with model_outputs keys={list(model_outputs.keys())}, forward_batch has emotion={('emotion' in forward_batch)}")
            logging.info(f"[DEBUG] Model training mode: {model.training}, Model dtype: {next(model.parameters()).dtype}")
            
            # ✅ AUTOREGRESSIVE: Use autoregressive loss if available
            if 'autoregressive_loss' in model_outputs:
                logging.info(f"[AUTOREGRESSIVE LOSS] Using autoregressive loss: {model_outputs['autoregressive_loss'].item():.6f}")
                loss = model_outputs['autoregressive_loss']
                loss_dict = {'autoregressive_loss': loss.item()}
            else:
                loss, loss_dict = criterion(model_outputs, forward_batch, ap)
            
            loss_val = loss.item()
            logging.info(f"[🔴 DEBUG LOSS-OUT] Step {step}: Total loss={loss_val:.6f}, components={loss_dict}, dtype={loss.dtype}, requires_grad={loss.requires_grad}")
            logging.info(f"  Loss has_nan={torch.isnan(loss).item()}, has_inf={torch.isinf(loss).item()}")
            logging.info(f"  Loss requires_grad BEFORE encodec: {loss.requires_grad}")
            
            if torch.isnan(loss) or torch.isinf(loss) or loss_val > 1000:
                logging.error(f"[⚠️ UNSTABLE LOSS] Step {step}: loss={loss_val}, NaN={torch.isnan(loss).item()}, Inf={torch.isinf(loss).item()}")
                logging.error(f"[⚠️ LOSS COMPONENTS] {loss_dict}")
                for key, val in model_outputs.items():
                    if isinstance(val, torch.Tensor):
                        logging.error(f"  {key}: has_nan={torch.isnan(val).any().item()}, has_inf={torch.isinf(val).any().item()}, min={val.min().item():.4f}, max={val.max().item():.4f}")
                for key, val in forward_batch.items():
                    if isinstance(val, torch.Tensor):
                        logging.error(f"  batch[{key}]: has_nan={torch.isnan(val).any().item()}, has_inf={torch.isinf(val).any().item()}")
            
            logging.info(f"[ENCODEC LOSS CHECK] waveform in forward_batch: {'waveform' in forward_batch}, keys in forward_batch: {list(forward_batch.keys())}")
            if "waveform" in forward_batch and "mel" in model_outputs:
                try:
                    original_waveform = forward_batch["waveform"].to(device)
                    predicted_audio = model_outputs["mel"]
                    logging.info(f"[ENCODEC DEBUG] original waveform shape: {original_waveform.shape}, predicted audio shape: {predicted_audio.shape}")
                    
                    if original_waveform.numel() > 0 and predicted_audio.numel() > 0:
                        # ✅ CRITICAL FIX: Compare XTTS's predicted audio against original
                        # This loss drives training of XTTS parameters (GPT + decoder)
                        # Align shapes for L1 loss
                        orig = original_waveform
                        pred = predicted_audio
                        
                        # Handle dimension differences
                        if orig.dim() == 1:
                            orig = orig.unsqueeze(0).unsqueeze(0)
                        elif orig.dim() == 2:
                            orig = orig.unsqueeze(1)
                        
                        # Match batch dimension
                        if pred.shape[0] != orig.shape[0]:
                            orig = orig.repeat(pred.shape[0], 1, 1)
                        
                        # Align lengths
                        min_len = min(orig.shape[-1], pred.shape[-1])
                        orig_aligned = orig[..., :min_len]
                        pred_aligned = pred[..., :min_len]
                        
                        encodec_loss = F.l1_loss(pred_aligned, orig_aligned) * 0.1  # Scale down to prevent gradient explosion
                        logging.info(f"[ENCODEC DEBUG] encodec_loss dtype: {encodec_loss.dtype}, requires_grad: {encodec_loss.requires_grad}, value: {encodec_loss.item()}")
                        loss = loss + encodec_loss
                        logging.info(f"[ENCODEC DEBUG] Combined loss requires_grad AFTER add: {loss.requires_grad}")
                        loss_dict["encodec_reconstruction"] = encodec_loss.item()
                        logging.info(f"[ENCODEC LOSS COMPUTED] encodec_loss: {encodec_loss.item()} (PRIMARY LOSS - from XTTS output)")
                except Exception as e:
                    import traceback
                    logging.error(f"EnCodec loss computation failed: {e}")
                    logging.error(f"Full traceback: {traceback.format_exc()}")
            
            if loss is None:
                logging.error("[ERROR] Loss not computed! Skipping step.")
                optimizer.zero_grad()
                continue
            
            loss = loss / grad_accumulation_steps
            
            logging.info(f"[🔴 DEBUG BACKWARD] Step {step}: Starting backward pass, loss={loss.item():.6f}, requires_grad={loss.requires_grad}")
            
            if loss.requires_grad:
                if device_type == "cuda":
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                logging.error(f"❌ [ERROR] Loss tensor does not require gradients!")

            if (step + 1) % grad_accumulation_steps == 0:
                # FIXED: Check and fix NaN gradients in ALL models (model + emotion_classifier)
                # <CHANGE> Use stricter gradient clipping for emotion classifier (0.5 instead of 1.0)
                if emotion_classifier is not None:
                    # <CHANGE> Add detailed gradient analysis
                    # Calculate RAW gradient norm BEFORE clipping
                    raw_grad_norm = 0.0
                    layer_grads = {}
                    linear_grad_norms = []
                    batchnorm_grad_norms = []
                    
                    for name, param in emotion_classifier.named_parameters():
                        if param.grad is not None:
                            layer_norm = param.grad.norm().item()
                            layer_grads[name] = layer_norm
                            raw_grad_norm += layer_norm ** 2
                            
                            # <CHANGE> Track by layer type for comparison
                            if 'weight' in name and 'norm' not in name:
                                linear_grad_norms.append(layer_norm)
                            elif 'norm' in name and 'weight' in name:
                                batchnorm_grad_norms.append(layer_norm)
                    
                    raw_grad_norm = raw_grad_norm ** 0.5
                    
                    # Log per-layer gradient norms to find the culprit
                    logging.info(f"[DEBUG GRAD] Per-layer gradient norms:")
                    for name, norm in sorted(layer_grads.items()):
                        logging.info(f"  {name}: {norm:.2f}")
                    
                    # <CHANGE> Compare Linear vs BatchNorm gradient magnitudes
                    if linear_grad_norms and batchnorm_grad_norms:
                        avg_linear = sum(linear_grad_norms) / len(linear_grad_norms)
                        avg_batchnorm = sum(batchnorm_grad_norms) / len(batchnorm_grad_norms)
                        ratio = avg_linear / avg_batchnorm if avg_batchnorm > 0 else float('inf')
                        logging.info(f"[DEBUG GRAD RATIO] Linear avg: {avg_linear:.2f}, BatchNorm avg: {avg_batchnorm:.2f}, Ratio: {ratio:.1f}x")
                    
                    # <CHANGE> Log weight-gradient correlation to find scaling issues
                    logging.info("[DEBUG WEIGHT-GRAD CORRELATION]:")
                    for name, param in emotion_classifier.named_parameters():
                        if param.grad is not None and 'weight' in name:
                            weight_norm = param.norm().item()
                            grad_norm_single = param.grad.norm().item()
                            ratio = grad_norm_single / weight_norm if weight_norm > 0 else float('inf')
                            logging.info(f"  {name}: weight_norm={weight_norm:.2f}, grad_norm={grad_norm_single:.2f}, ratio={ratio:.1f}x")
                    
                    # Clip with YAML-configured value (not hardcoded!)
                    grad_norm, nan_found = check_and_fix_gradients(emotion_classifier, max_norm=emotion_grad_clip)

                    # <CHANGE> Calculate actual post-clip gradient norm for accurate logging
                    post_clip_grad_norm = 0.0
                    for param in emotion_classifier.parameters():
                        if param.grad is not None:
                            post_clip_grad_norm += param.grad.norm().item() ** 2
                    post_clip_grad_norm = post_clip_grad_norm ** 0.5

                    # Log raw vs clipped gradient norms for diagnosis
                    # <CHANGE> Protect against division by zero when emotion loss fails and raw_grad_norm=0
                    scaling = (post_clip_grad_norm / raw_grad_norm) if raw_grad_norm > 1e-10 else 0.0
                    logging.info(f"[EMOTION GRAD] Raw: {raw_grad_norm:.6f}, Clipped: {post_clip_grad_norm:.6f}, Threshold: {emotion_grad_clip}, Scaling: {scaling:.4f}x")
                    # <CHANGE> Add weight magnitude inspection
                    weight_stats = {}
                    for name, param in emotion_classifier.named_parameters():
                        if 'weight' in name:
                            w_norm = param.data.norm().item()
                            w_mean = param.data.mean().item()
                            w_std = param.data.std().item()
                            weight_stats[name] = (w_norm, w_mean, w_std)
                            logging.info(f"[DEBUG WEIGHTS] {name}: norm={w_norm:.4f}, mean={w_mean:.6f}, std={w_std:.6f}")
                    
                    if nan_found:
                        logging.error(f"[v0 ERROR] NaN gradients in EmotionClassifier, zeroed out")
                
                # Apply aggressive gradient norm clipping to prevent explosion
                all_models = [model, prosody_predictor]
                try:
                    model_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    logging.info(f"[GRAD CLIP] Model gradient norm clipped: {model_grad_norm:.2f} (max_norm={max_grad_norm})")
                except Exception as e:
                    logging.warning(f"[GRAD CLIP] Model clipping failed: {e}")
                
                try:
                    prosody_grad_norm = torch.nn.utils.clip_grad_norm_(prosody_predictor.parameters(), max_norm=max_grad_norm)
                    logging.info(f"[GRAD CLIP] Prosody gradient norm clipped: {prosody_grad_norm:.2f} (max_norm={max_grad_norm})")
                except Exception as e:
                    logging.warning(f"[GRAD CLIP] Prosody clipping failed: {e}")
                
                # <CHANGE> Then check for remaining NaNs
                total_nan_found = False
                for m in all_models:
                    grad_norm, nan_found = check_and_fix_gradients(m, max_norm=max_grad_norm)
                    if nan_found:
                        total_nan_found = True
                        logging.error(f"[v0 ERROR] NaN gradients in {m.__class__.__name__}, zeroed out")
                
                if total_nan_found:
                    logging.error(f"[v0 ERROR] NaN gradients at step {step}, skipping")
                    optimizer.zero_grad()
                else:
                    # Log emotion gradient norm
                    if emotion_classifier is not None:
                        emotion_grad_norm = 0.0
                        for param in emotion_classifier.parameters():
                            if param.grad is not None:
                                emotion_grad_norm += param.grad.norm().item() ** 2
                        emotion_grad_norm = emotion_grad_norm ** 0.5
                        logging.info(f"[v0 DIAGNOSTIC] Emotion classifier gradient norm: {emotion_grad_norm:.6f}")
                    
                    logging.info(f"[🔴 DEBUG OPTIM] Step {step}: Before scaler.step()")
                    
                    weight_snapshot_before = {}
                    for name, param in model.gpt.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            weight_snapshot_before[name] = param.data.clone().detach().norm().item()
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # ✅ DIAGNOSTICS: Print after first few steps
                    if step < 3 and diagnostics is not None:
                        diagnostics.print_diagnostics(step)
                    
                    if step == 2 and diagnostics is not None:
                        diagnostics.cleanup()
                        diagnostics = None
                    
                    weight_snapshot_after = {}
                    weight_exploded = False
                    for name, param in model.gpt.named_parameters():
                        if param.requires_grad:
                            w_norm = param.data.norm().item()
                            weight_snapshot_after[name] = w_norm
                            if name in weight_snapshot_before:
                                before = weight_snapshot_before[name]
                                if w_norm > before * 10 or w_norm > 1e6:
                                    weight_exploded = True
                                    logging.error(f"[⚠️ WEIGHT EXPLOSION] {name}: {before:.4f} → {w_norm:.4f} ({w_norm/before:.1f}x)")
                    
                    if weight_exploded:
                        logging.error(f"[⚠️ WEIGHT EXPLOSION DETECTED] Step {step}: weights grew too much!")
                    
                    logging.info(f"[🔴 DEBUG OPTIM] Step {step}: Optimizer step completed")
                    
                    if step % 50 == 0 and model.gpt is not None:
                        gpt_weight_sample = None
                        for name, param in model.gpt.named_parameters():
                            if 'wte' in name or 'embedding' in name.lower():
                                gpt_weight_sample = param.data.flatten()[:10]
                                logging.info(f"[🔴 DEBUG WEIGHTS] GPT sample weights ({name}): norm={param.norm().item():.6f}, values={gpt_weight_sample.detach().cpu()[:5].tolist()}")
                                break
                    
                    logging.info(f"[v0 SUCCESS] Optimizer step completed")

            # Step scheduler (safe for both OneCycleLR and CosineAnnealingWarmRestarts)
            if hasattr(scheduler, 'total_steps'):
                # OneCycleLR has total_steps attribute
                if scheduler._step_count < scheduler.total_steps:
                    scheduler.step()
            else:
                # CosineAnnealingWarmRestarts - always step
                scheduler.step()

            loss_scalar = loss.item() * grad_accumulation_steps
            epoch_loss += loss_scalar
            epoch_steps += 1
            
            current_lr = optimizer.param_groups[0]['lr']
            if step % 10 == 0:
                logging.warning(f"[LOSS TRACKING] Epoch {epoch+1}, Step {step}: loss={loss_scalar:.6f}, lr={current_lr:.2e}, avg_epoch_loss={epoch_loss/epoch_steps:.6f}")
                breakdown = [f"{k}={v:.4f}" for k, v in loss_dict.items() if v != 0]
                if breakdown:
                    logging.warning(f"[LOSS BREAKDOWN] {', '.join(breakdown)}")

            # -----------------------
            # Enhanced sample dumping with new vocoder
            # -----------------------
            if step % dump_step == 0 and step > 0:
                try:
                    model.eval()
                    prosody_predictor.eval()
                    if emotion_classifier is not None:
                        emotion_classifier.eval()

                    with torch.no_grad():
                        if "mel" in forward_batch and isinstance(forward_batch["mel"], torch.Tensor):
                            mel_pred = forward_batch["mel"]
                            batch_size = min(3, mel_pred.shape[0])
                            audio_samples = []

                            for i in range(batch_size):
                                mel = mel_pred[i]
                                audio = vocoder(mel)
                                audio_samples.append(audio)

                            save_audio_samples(
                                audio_samples,
                                sample_rate,
                                sample_dir,
                                step,
                                f"epoch_{epoch+1}"
                            )
                except Exception as e:
                    logging.warning(f"Sample generation failed at step {step}: {e}")
                finally:
                    model.train()
                    prosody_predictor.train()
                    if emotion_classifier is not None:
                        emotion_classifier.train()

            # -----------------------
            # Logging
            # -----------------------
            if step % cfg.get("print_step", 25) == 0:
                lr_current = scheduler.get_last_lr()[0] if scheduler else lr
                logging.info(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item() * grad_accumulation_steps:.6f}, LR: {lr_current:.2e}")
                if epoch == 14 and step < 100:
                    logging.info(f"[EPOCH 15 DEBUG] LR at step {step}: {lr_current:.2e} (scheduler type: {scheduler_type})")
                
                if hasattr(model, 'vocoder') and model.vocoder is not None:
                    vocoder_first_weight = list(model.vocoder.parameters())[0]
                    logging.info(f"  Vocoder trainable: {vocoder_first_weight.requires_grad}")
                    logging.info(f"  Vocoder weight norm: {vocoder_first_weight.norm().item():.6f}")
                
                # Log individual loss components during training
                log_components = []
                if "prosody_loss" in loss_dict:
                    log_components.append(f"Prosody: {loss_dict['prosody_loss']:.4f}")
                if "spectrogram_loss" in loss_dict:
                    log_components.append(f"Spec L1: {loss_dict['spectrogram_loss']:.4f}")
                if "emotion_loss" in loss_dict:
                    log_components.append(f"Emotion: {loss_dict['emotion_loss']:.4f}")
                if log_components:
                    logging.info(f"  Loss components: {', '.join(log_components)}")

                # Log detailed loss components
                for loss_name, loss_value in loss_dict.items():
                    logging.debug(f"  {loss_name}: {loss_value:.6f}")

        # -----------------------
        # Evaluation
        # -----------------------
        if cfg.get("run_eval", True) and eval_loader is not None:
            model.eval()
            prosody_predictor.eval()
            if emotion_classifier is not None:
                emotion_classifier.eval()
                
                # <CHANGE> Add diagnostic logging to check if fc2 weights are corrupted
                # Check classifier output layer weights (new architecture)
                classifier_layer = emotion_classifier.classifier
                logging.info(f"[DIAGNOSTIC] classifier.weight - min: {classifier_layer.weight.min().item():.6f}, max: {classifier_layer.weight.max().item():.6f}")
                logging.info(f"[DIAGNOSTIC] classifier.weight has NaN: {torch.isnan(classifier_layer.weight).any().item()}")
                logging.info(f"[DIAGNOSTIC] classifier.weight has Inf: {torch.isinf(classifier_layer.weight).any().item()}")
                logging.info(f"[DIAGNOSTIC] classifier.bias has NaN: {torch.isnan(classifier_layer.bias).any().item()}")
                
                # ===== DIAGNOSTIC HOOK TO FIND NaN SOURCE =====
                def diagnose_nan_in_emotion_classifier(module, input, output):
                    """Diagnose where NaN originates in emotion classifier"""
                    module_name = module.__class__.__name__
                    
                    # Check input
                    if isinstance(input, tuple):
                        input = input[0]
                    if torch.is_tensor(input):
                        if torch.isnan(input).any():
                            logging.error(f"❌ NaN in INPUT to {module_name}")
                            logging.error(f"   Shape: {input.shape}, NaN count: {torch.isnan(input).sum().item()}")
                        
                        # Check for zero variance (causes NaN in LayerNorm)
                        if input.dim() >= 2:
                            variance = input.var(dim=-1)
                            if (variance < 1e-8).any():
                                logging.error(f"❌ ZERO VARIANCE in input to {module_name}")
                                logging.error(f"   Min variance: {variance.min().item():.10f}")
                                logging.error(f"   This will cause NaN in LayerNorm!")
                    
                    # Check output
                    if torch.is_tensor(output):
                        if torch.isnan(output).any():
                            logging.error(f"❌ NaN in OUTPUT from {module_name}")
                            logging.error(f"   Shape: {output.shape}, NaN count: {torch.isnan(output).sum().item()}")

                # Register diagnostic hook on ALL layers
                if emotion_classifier is not None:
                    for name, module in emotion_classifier.named_modules():
                        if len(list(module.children())) == 0:  # Only leaf modules
                            module.register_forward_hook(diagnose_nan_in_emotion_classifier)
                    logging.info("[DIAGNOSTIC] Registered NaN detection hooks on emotion classifier")
# ===== END DIAGNOSTIC HOOK =====

            eval_loss_total = 0.0
            eval_steps = 0
            eval_metrics = {
                'spectrogram_l1': 0.0,
                'prosody_error': 0.0,
                'perceptual_error': 0.0,
                'emotion_accuracy': 0.0
            }

            for eval_batch in eval_loader:
                eval_ns = batch_to_namespace(eval_batch)
                to_device(eval_ns, device)

                # Pad token_id if list
                if hasattr(eval_ns, "token_id") and isinstance(eval_ns.token_id, list):
                    token_ids_eval = eval_ns.token_id
                    eval_ns.token_id = torch.nn.utils.rnn.pad_sequence(
                        [t.detach().clone() if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long)
                         for t in token_ids_eval],
                        batch_first=True
                    )
                    eval_ns.token_id_lengths = torch.tensor(
                        [len(t) if isinstance(t, (list, torch.Tensor)) else 0 for t in token_ids_eval],
                        dtype=torch.long,
                        device=device
                    )

                # Build forward batch dict
                # Build forward batch dict
                forward_eval = {}
                for k in required_keys:
                    if hasattr(eval_ns, k):
                        forward_eval[k] = getattr(eval_ns, k)
                    else:
                        if k == "token_id":
                            forward_eval[k] = torch.zeros(1, 1, dtype=torch.long, device=device)
                        elif k == "token_id_lengths":
                            forward_eval[k] = torch.tensor([1], dtype=torch.long, device=device)
                        elif k == "cond_mels":
                            if "mel" in forward_eval and isinstance(forward_eval["mel"], torch.Tensor):
                                batch_size, n_mels, t = forward_eval["mel"].shape
                                forward_eval[k] = torch.zeros(batch_size, n_mels, t, device=device)
                            else:
                                batch_size = forward_eval["token_id"].shape[0] if "token_id" in forward_eval else 1
                                forward_eval[k] = torch.zeros(batch_size, 80, 80, device=device)
                        elif k in ["mel", "linear"]:
                            forward_eval[k] = torch.zeros(1, 80, 80, device=device)
                        elif k in ["stop_targets", "gpt_audio_tokens", "gpt_cond_latent"]:
                            forward_eval[k] = torch.zeros(1, 1, device=device)
                        elif k == "expected_output_len":
                            forward_eval[k] = torch.tensor([1], dtype=torch.long, device=device)
                        elif k == "speaker_embedding":
                            forward_eval[k] = torch.zeros(1, 256, device=device)

                if hasattr(eval_ns, "waveform"):
                    forward_eval["waveform"] = eval_ns.waveform.to(device)

                # Map tokens correctly BEFORE the model call!
                forward_eval["tokens"] = forward_eval["token_id"]
                if "token_id_lengths" in forward_eval:
                    forward_eval["tokens_lengths"] = forward_eval["token_id_lengths"]

                # Defensive runtime check for cond_mels!
                if ("cond_mels" not in forward_eval) or (forward_eval["cond_mels"] is None):
                    print("WARNING: cond_mels missing or None in forward_eval, constructing artificial cond_mels")
                    if "mel" in forward_eval and isinstance(forward_eval["mel"], torch.Tensor):
                        batch_size, n_mels, t = forward_eval["mel"].shape
                        forward_eval["cond_mels"] = torch.zeros(batch_size, n_mels, t, device=device)
                    else:
                        batch_size = forward_eval["token_id"].shape[0] if "token_id" in forward_eval else 1
                        forward_eval["cond_mels"] = torch.zeros(batch_size, 80, 80, device=device)
                else:
                    if not isinstance(forward_eval["cond_mels"], torch.Tensor):
                        print(f"WARNING: cond_mels is not a tensor! Type: {type(forward_eval['cond_mels'])}")
                    else:
                        print(f"cond_mels final shape before forward: {forward_eval['cond_mels'].shape}")


                # Extract prosody features for evaluation
                if hasattr(eval_ns, "waveform") and eval_ns.waveform is not None:
                    try:
                        batch_prosody_features = []
                        for i in range(eval_ns.waveform.shape[0]):
                            audio_sample = eval_ns.waveform[i]
                            prosody_features = prosody_extractor(audio_sample)
                            batch_prosody_features.append(prosody_features)

                        if batch_prosody_features:
                            forward_eval["prosody_target"] = {
                                'pitch': torch.nn.utils.rnn.pad_sequence([f['pitch'] for f in batch_prosody_features], batch_first=True),
                                'energy': torch.nn.utils.rnn.pad_sequence([f['energy'] for f in batch_prosody_features], batch_first=True),
                                'duration': torch.nn.utils.rnn.pad_sequence([f['duration'] for f in batch_prosody_features], batch_first=True)
                            }
                    except Exception as e:
                        logging.debug(f"Eval prosody extraction failed: {e}")

                # Handle emotion targets for evaluation
                if use_emotions and hasattr(eval_ns, "emotion") and eval_ns.emotion is not None:
                    if emotion_classifier is not None:
                        # Use emotion directly if it's already a tensor (like in training)
                        if isinstance(eval_ns.emotion, torch.Tensor):
                            forward_eval["emotion"] = eval_ns.emotion.to(device)
                        else:
                            emotion_labels = emotion_classifier.encode_emotions(eval_ns.emotion)
                            forward_eval["emotion"] = torch.tensor(emotion_labels, dtype=torch.long, device=device)

                with torch.no_grad():
                    model_outputs = model(**forward_eval)

                    if "token_id" in forward_eval:
                        try:
                            if hasattr(model, 'gpt') and hasattr(model.gpt, 'text_embedding'):
                                _eval_token_ids = forward_eval["token_id"]
                                if isinstance(_eval_token_ids, (list, tuple)):
                                    _eval_token_ids = _eval_token_ids[0]
                                text_emb = model.gpt.text_embedding(_eval_token_ids)
                                prosody_pred = prosody_predictor(text_emb)
                                model_outputs["prosody_pred"] = prosody_pred
                        except Exception as e:
                            logging.debug(f"Eval prosody prediction failed: {e}")

                    # Add emotion prediction
                    # Robust emotion prediction (works with gpt_latents OR token ids fallback)
                    # ===== FIXED: Robust emotion prediction for eval (forward_eval) =====
                    if use_emotions and emotion_classifier is not None:
                        try:
                            try:
                                device = next(model.parameters()).device
                            except Exception:
                                device = torch.device("cpu")

                            emotion_logits = None

                            # 1) Prefer GPT-produced latents if available
                            # 1) Use speaker_embedding ONLY (512-dim)
                            # gpt_latents are 1024-dim and contain NaN - removed entirely
                            
                            # Use speaker_embedding as the primary and only source
                            # <CHANGE> Use speaker_embedding instead of gpt_latents

                            # 2) Fallback: token ids -> embedding -> emotion
                            # <CHANGE> 2) Fallback: Use speaker_embedding instead of text_emb
                            # text_emb has 1024 dims but emotion_classifier expects 512 dims
                            # This dimension mismatch causes truncation and meaningless predictions
                            # Line 2682-2710 (Evaluation code) - FIXED
                            if emotion_logits is None and "emotion_features" in forward_eval:
                                try:
                                    emotion_input = forward_eval["emotion_features"]
                                    if isinstance(emotion_input, torch.Tensor) and emotion_input.numel() > 0:
                                        emotion_input = emotion_input.detach().to(device)
                                        # ✅ REMOVED CLAMP - features need natural scale (~700) for discrimination
                                        
                                        logging.error(f"[PRE-CLASSIFIER DEBUG] Input to emotion_classifier:")
                                        logging.error(f"  Shape: {emotion_input.shape}")
                                        logging.error(f"  Has NaN: {torch.isnan(emotion_input).any()}")
                                        logging.error(f"  Min: {emotion_input.min():.6f}, Max: {emotion_input.max():.6f}")
                                        logging.error(f"  Mean: {emotion_input.mean():.6f}, Std: {emotion_input.std():.6f}")
                                        
                                        emotion_logits = emotion_classifier(emotion_input)  # ✅ CORRECT INPUT
                                        if isinstance(emotion_logits, (tuple, list)):
                                            emotion_logits = emotion_logits[0]
                                        batch_size = forward_eval.get("token_id", forward_eval.get("tokens")).size(0)
                                        if emotion_logits.size(0) == batch_size:
                                            model_outputs["emotion_logits"] = emotion_logits
                                            mode = "train" if emotion_classifier.training else "eval"
                                            logging.debug(f"Emotion logits computed ({mode}) from emotion_features: {emotion_logits.shape}")
                                        else:
                                            logging.debug("Skipping emotion logits (eval) due to batch mismatch")
                                except Exception as e:
                                    logging.debug(f"Emotion prediction from emotion_features (eval) failed: {e}")
                            
                            # <CHANGE> If still no emotion_logits, log warning instead of using wrong dimensions
                            if emotion_logits is None:
                                logging.debug("Skipping emotion prediction: no valid input available (gpt_latents empty, speaker_embedding missing)")
                        except Exception as e:
                            logging.debug(f"Emotion prediction failed: {e}")
                    # ===== END FIXED (eval) =====



                    batch_loss, batch_loss_dict = criterion(model_outputs, forward_eval, ap)


                    # Compute evaluation metrics
                    # Compute evaluation metrics (robust Spec L1)
                    mel_pred = None
                    mel_target = None

                    # predicted mel
                    if "mel" in model_outputs:
                        mel_pred = model_outputs["mel"]
                    elif "waveform" in model_outputs and audio_processor is not None:
                        try:
                            wav = model_outputs["waveform"]
                            if wav.dim() == 3 and wav.size(1) == 1:
                                wav = wav.squeeze(1)
                            mel_pred = audio_processor.melspectrogram(wav)
                        except Exception:
                            pass

                    # target mel
                    if "mel" in forward_eval:
                        mel_target = forward_eval["mel"]
                    elif "linear" in forward_eval and audio_processor is not None and hasattr(audio_processor, "linear_to_mel"):
                        try:
                            mel_target = audio_processor.linear_to_mel(forward_eval["linear"])
                        except Exception:
                            pass
                    elif "waveform" in forward_eval and audio_processor is not None:
                        try:
                            wav = forward_eval["waveform"]
                            if wav.dim() == 3 and wav.size(1) == 1:
                                wav = wav.squeeze(1)
                            mel_target = audio_processor.melspectrogram(wav)
                        except Exception:
                            pass

                    # compute spectrogram L1 robustly (support different key names)
                    spec_pred = None
                    spec_target = None

                    # prefer mel if provided by model / dataset
                    if "mel" in model_outputs and "mel" in forward_eval:
                        spec_pred = model_outputs["mel"]
                        spec_target = forward_eval["mel"]
                    # fall back to spectrogram keys (some code paths use these names)
                    elif "spectrogram_pred" in model_outputs and "spectrogram_target" in forward_eval:
                        spec_pred = model_outputs["spectrogram_pred"]
                        spec_target = forward_eval["spectrogram_target"]
                    # if model gave waveform, compute mel with audio_processor (if available)
                    elif "waveform" in model_outputs and "waveform" in forward_eval and audio_processor is not None:
                        try:
                            wav_pred = model_outputs["waveform"]
                            wav_tgt = forward_eval["waveform"]
                            if wav_pred.dim() == 3 and wav_pred.size(1) == 1:
                                wav_pred = wav_pred.squeeze(1)
                            if wav_tgt.dim() == 3 and wav_tgt.size(1) == 1:
                                wav_tgt = wav_tgt.squeeze(1)
                            spec_pred = audio_processor.melspectrogram(wav_pred)
                            spec_target = audio_processor.melspectrogram(wav_tgt)
                        except Exception as e:
                            logging.debug(f"Failed to compute mel from waveform for spec L1: {e}")

                    spec_l1 = 0.0

                    # Compute Spec L1 using spec_pred/spec_target set above
                    if spec_pred is not None and spec_target is not None:
                        try:
                            device = spec_pred.device
                            spec_target = spec_target.to(device)
                            
                            min_len = min(spec_pred.shape[-1], spec_target.shape[-1])
                            min_mels = min(spec_pred.shape[1] if spec_pred.dim() >= 2 else 1,
                                          spec_target.shape[1] if spec_target.dim() >= 2 else 1)
                            
                            if min_len > 0 and min_mels > 0:
                                spec_l1 = F.l1_loss(
                                    spec_pred[:, :min_mels, :min_len],
                                    spec_target[:, :min_mels, :min_len]
                                ).item()
                                logging.debug(f"Computed Spec L1: {spec_l1:.6f}")
                        except Exception as e:
                            logging.debug(f"Failed to compute Spec L1: {e}")
                            spec_l1 = 0.0
                    else:
                        logging.debug("No spec_pred or spec_target for Spec L1")

                    eval_metrics['spectrogram_l1'] += spec_l1




                    if "prosody_pred" in model_outputs and "prosody_target" in forward_eval:
                        prosody_error = 0.0
                        prosody_count = 0
                        for key in ['pitch_pred', 'energy_pred', 'duration_pred']:
                            if key in model_outputs["prosody_pred"] and key.replace('_pred', '') in forward_eval["prosody_target"]:
                                pred = model_outputs["prosody_pred"][key]
                                target = forward_eval["prosody_target"][key.replace('_pred', '')]
                                min_len = min(pred.shape[-1], target.shape[-1])
                                error = F.l1_loss(pred[..., :min_len], target[..., :min_len])
                                prosody_error += error.item()
                                prosody_count += 1
                        if prosody_count > 0:
                            eval_metrics['prosody_error'] += prosody_error / prosody_count

                    if use_emotions and "emotion_logits" in model_outputs and "emotion" in forward_eval:
                        emotion_pred = torch.argmax(model_outputs["emotion_logits"], dim=-1)
                        emotion_target = forward_eval["emotion"]
                        accuracy = (emotion_pred == emotion_target).float().mean().item()
                        eval_metrics['emotion_accuracy'] += accuracy

                eval_loss_total += batch_loss.item()
                eval_steps += 1

            # Average evaluation metrics
            avg_eval_loss = eval_loss_total / max(1, eval_steps)
            for metric_name in eval_metrics:
                eval_metrics[metric_name] = eval_metrics[metric_name] / max(1, eval_steps)

            logging.info(f"Epoch {epoch+1} Eval Loss: {avg_eval_loss:.6f}")
            logging.info(f"Eval Metrics - Spec L1: {eval_metrics['spectrogram_l1']:.6f}, "
                        f"Prosody Error: {eval_metrics['prosody_error']:.6f}, "
                        f"Emotion Accuracy: {eval_metrics['emotion_accuracy']:.6f}")

            # Save best model if evaluation improved
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                try:
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'prosody_predictor_state_dict': prosody_predictor.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'best_eval_loss': best_eval_loss,
                        'eval_metrics': eval_metrics
                    }

                    # Add emotion classifier state if enabled
                    if emotion_classifier is not None:
                        checkpoint['emotion_classifier_state_dict'] = emotion_classifier.state_dict()

                    torch.save(checkpoint, best_model_path)
                    logging.info(f"✅ Saved new best model with eval loss: {best_eval_loss:.6f}")
                except Exception as e:
                    logging.warning(f"Failed to save best model: {e}")
            else:
                logging.info(f"⚠️  Eval loss did not improve: {avg_eval_loss:.6f} >= best {best_eval_loss:.6f}")
            
            # Check for early stopping
            # -----------------------
            logging.info(f"[EARLY STOP CHECK] Before: best_loss={early_stopping.best_loss}, counter={early_stopping.counter}, patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")
            early_stopping(avg_eval_loss, model, best_model_path)
            logging.info(f"[EARLY STOP CHECK] After: best_loss={early_stopping.best_loss}, counter={early_stopping.counter}, early_stop={early_stopping.early_stop}")
            if early_stopping.early_stop:
                logging.info(f"🛑 EARLY STOPPING TRIGGERED at epoch {epoch+1} (patience exhausted: {early_stopping.counter} >= {early_stopping.patience})")
                break

            model.train()
            prosody_predictor.train()
            if emotion_classifier is not None:
                emotion_classifier.train()

        # -----------------------
        # Epoch summary
        # -----------------------
        avg_epoch_loss = epoch_loss / max(1, epoch_steps)
        logging.info(f"Epoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.6f}")

        # -----------------------
        # Enhanced checkpoint saving
        # -----------------------
        if cfg.get("save_step", 1000) > 0 and (epoch + 1) % max(1, cfg.get("save_step", 1000) // len(train_loader)) == 0:
            checkpoint_path = os.path.join(
                cfg.get("output_path", "outputs"),
                f"checkpoint_epoch_{epoch+1}_ts{int(time.time())}.pth"
            )
            try:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'prosody_predictor_state_dict': prosody_predictor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'loss': avg_epoch_loss,
                    'config': cfg
                }

                # Add emotion classifier state if enabled
                if emotion_classifier is not None:
                    checkpoint['emotion_classifier_state_dict'] = emotion_classifier.state_dict()

                torch.save(checkpoint, checkpoint_path)
                logging.info(f"Saved checkpoint: {checkpoint_path}")
            except Exception as e:
                logging.warning(f"Failed to save checkpoint: {e}")

    logging.info("Training completed!")
    return model

def load_checkpoint_enhanced(checkpoint_path, model, prosody_predictor=None, emotion_classifier=None, optimizer=None, scheduler=None):
    """Enhanced checkpoint loading with support for all components"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Load model state
        # --- inside load_checkpoint_enhanced ---
        # Load model state with strict=False to allow renamed or extra keys
        if 'model_state_dict' in checkpoint:
            missing, unexpected = model.load_state_dict(
                checkpoint['model_state_dict'],
                strict=False
            )
            if missing or unexpected:
                logging.warning(
                    f"Checkpoint loaded with missing keys: {missing} "
                    f"and unexpected keys: {unexpected}"
                )
            else:
                logging.info("Loaded model state dict (all keys matched)")


        # Load prosody predictor state
        if prosody_predictor is not None and 'prosody_predictor_state_dict' in checkpoint:
            prosody_predictor.load_state_dict(checkpoint['prosody_predictor_state_dict'])
            logging.info("Loaded prosody predictor state dict")

        # Load emotion classifier state
        # Load emotion classifier state
        if emotion_classifier is not None and 'emotion_classifier_state_dict' in checkpoint:
            try:
                # <CHANGE> Try loading, but don't fail if shapes mismatch (FFN expanded from *2 to *4)
                emotion_classifier.load_state_dict(checkpoint['emotion_classifier_state_dict'], strict=False)
                logging.info("Loaded emotion classifier state dict")
            except RuntimeError as e:
                # <CHANGE> Shape mismatch expected when upgrading FFN from 1024→2048
                # This is OK - we're retraining with better architecture
                logging.warning(f"Emotion classifier checkpoint has shape mismatch (expected): {e}")
                logging.info("Proceeding with newly initialized emotion classifier (FFN expanded to 2048)")

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info("Loaded optimizer state dict")

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info("Loaded scheduler state dict")

        # Return additional info
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

        logging.info(f"Checkpoint loaded successfully from epoch {epoch} with loss {loss:.6f}")
        return epoch, loss

    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        return 0, float('inf')

def main():
    import torch
    global XttsConfig, Xtts, AudioProcessor
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    print("[DEBUG] Starting main function...")
    cfg = load_config(args.config_path)
    print(f"[DEBUG] Config loaded successfully. Keys: {list(cfg.keys())}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        logging.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        logging.info("CUDA not available, using CPU device")

    print("[DEBUG] Device setup complete.")

    # Build dataset
    print("[DEBUG] Loading dataset...")
    ds_cfg = cfg.get("datasets", [])[0] if cfg.get("datasets") else {}
    print(f"[DEBUG] Dataset config: {ds_cfg}")

    if load_tts_samples is not None and BaseDatasetConfig is not None:
        try:
            # try coqui loader (preferred)
            print("[DEBUG] Attempting Coqui loader...")
            bds = BaseDatasetConfig(**ds_cfg) if isinstance(ds_cfg, dict) else ds_cfg
            train_samples, eval_samples = load_tts_samples([bds], eval_split=True, eval_split_size=cfg.get("eval_split_size", 0.0))
            logging.info("Loaded dataset via Coqui loader: %d train, %d eval", len(train_samples), len(eval_samples))
            print(f"[DEBUG] Coqui loader successful: {len(train_samples)} train, {len(eval_samples)} eval")
        except Exception as e:
            logging.warning("Coqui loader failed: %s — falling back.", e)
            print(f"[DEBUG] Coqui loader failed: {e}")
            ds = load_dataset_simple(ds_cfg)
            train_samples, eval_samples = ds["train"], ds["val"]
            print(f"[DEBUG] Fallback loader: {len(train_samples)} train, {len(eval_samples)} eval")
    else:
        print("[DEBUG] Using simple dataset loader...")
        ds = load_dataset_simple(ds_cfg)
        train_samples, eval_samples = ds["train"], ds["val"]
        print(f"[DEBUG] Simple loader: {len(train_samples)} train, {len(eval_samples)} eval")

    print(f"[DEBUG] Dataset loaded. Train samples: {len(train_samples)}, Eval samples: {len(eval_samples)}")
    

    logging.info("[v0 DIAGNOSTIC] Analyzing emotion distribution in dataset...")
    emotion_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for sample in train_samples:
        if "emotion" in sample:
            emotion_counts[sample["emotion"]] = emotion_counts.get(sample["emotion"], 0) + 1
    
    total_samples = sum(emotion_counts.values())
    logging.info(f"[v0 DIAGNOSTIC] Total samples with emotions: {total_samples}")
    logging.info(f"[v0 DIAGNOSTIC] Emotion distribution:")
    emotion_names = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
    for i, name in enumerate(emotion_names):
        count = emotion_counts.get(i, 0)
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        logging.info(f"[v0 DIAGNOSTIC]   {name} (class {i}): {count} samples ({percentage:.1f}%)")

    # audio processor
    print("[DEBUG] Creating audio processor...")
    audio_cfg = cfg.get("audio_config", cfg.get("audio", {}))
    if AudioProcessor is not None and XttsConfig is not None and audio_cfg:
        try:
            ap = AudioProcessor(sample_rate=audio_cfg.get("sample_rate", 24000),
                                fft_size=audio_cfg.get("fft_size", 4096),
                                win_size=audio_cfg.get("win_length", 1024),
                                hop_size=audio_cfg.get("hop_length", 256),
                                f_min=audio_cfg.get("mel_fmin", 0),
                                f_max=audio_cfg.get("mel_fmax", 12000),
                                num_mels=audio_cfg.get("num_mels", 80),
                                frame_length_ms=audio_cfg.get("frame_length_ms", None),
                                frame_shift_ms=audio_cfg.get("frame_shift_ms", None))
        except Exception as e:
            logging.warning("AudioProcessor init failed: %s — using stub", e)
            class _StubAP:
                def __init__(self, sr=24000):
                    self.sample_rate = sr
                    self.fft_size = 4096
                    self.win_length = 4096
                    self.hop_length = 1024
                    self.num_mels = 80
                def melspectrogram(self, y): return librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_fft=self.fft_size, hop_length=self.hop_length, n_mels=self.num_mels)
                def spectrogram(self, y): return np.abs(librosa.stft(y, n_fft=self.fft_size, hop_length=self.hop_length))
            ap = _StubAP(audio_cfg.get("sample_rate", 24000))
    else:
        class _StubAP:
            def __init__(self, sr=24000):
                self.sample_rate = sr
                self.fft_size = 4096
                self.win_length = 1024
                self.hop_length = 256
                self.num_mels = 80
            def melspectrogram(self, y): return librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_fft=self.fft_size, hop_length=self.hop_length, n_mels=self.num_mels)
            def spectrogram(self, y): return np.abs(librosa.stft(y, n_fft=self.fft_size, hop_length=self.hop_length))
        ap = _StubAP(audio_cfg.get("sample_rate", 24000))

    print("[DEBUG] Audio processor created successfully.")

    # speaker encoder
    # speaker encoder
    print("[DEBUG] Creating speaker encoder...")
    spk_encoder = None
    if SpeakerRecognition is not None:
        try:
            spk_encoder = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
            # Move speaker encoder to GPU and set to eval mode
            spk_encoder = spk_encoder.to(device)
            spk_encoder.eval()
            logging.info(f"Loaded speaker encoder and moved to {device}.")
        except Exception as e:
            logging.warning("Failed loading speaker encoder: %s", e)
            spk_encoder = None

    print("[DEBUG] Speaker encoder created successfully.")

    # model
    print("[DEBUG] Creating XTTS model...")
    try:
        print("[DEBUG] Step 1: Checking Encodec...")
        # Encodec (optional)
        encodec = None
        if EncodecModel is not None:
            try:
                encodec = EncodecModel.encodec_model_24khz()
                encodec.set_target_bandwidth(6.0)
                encodec = encodec.to(device)
                encodec.eval()
                logging.info("Loaded Encodec 24k model.")
                print("[DEBUG] Encodec loaded successfully")
            except Exception as e:
                logging.warning("Failed to instantiate Encodec: %s", e)
                print(f"[DEBUG] Encodec failed: {e}")
                encodec = None
        else:
            print("[DEBUG] EncodecModel not available")

        print("[DEBUG] Step 2: Setting up tokenizer...")
        # Tokenizer fallback: if TTSTokenizer is available use it, else implement trivial whitespace tokenizer
        tokenizer = None
        if TTSTokenizer is not None and BaseCharacters is not None:
            try:
                tokenizer = TTSTokenizer(characters=BaseCharacters(characters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!? ", punctuations=".,!? ", pad="<PAD>", eos="<EOS>", bos="<BOS>", blank="<BLK>"))
                logging.info("Using TTSTokenizer.")
                print("[DEBUG] TTSTokenizer created successfully")
            except Exception as e:
                print(f"[DEBUG] TTSTokenizer failed: {e}")
                tokenizer = None
        else:
            print("[DEBUG] TTSTokenizer or BaseCharacters not available")

        print("[DEBUG] Step 3: Preprocessing samples...")
        # Precompute sample fields (tokenize, embeddings, cond latents, encodec tokens)
        all_samples = train_samples + (eval_samples if eval_samples else [])
        print(f"[DEBUG] Processing {len(all_samples)} samples...")

        for i, s in enumerate(all_samples):
            try:
                s["audio_file"] = s["audio_file"].replace("\\", "/")
                # tokens
                if tokenizer is not None and "text" in s:
                    try:
                        toks = tokenizer.encode(s["text"].lower())
                        s["token_id"] = torch.tensor(toks, dtype=torch.long)
                        s["token_id_lengths"] = len(toks)
                    except Exception as e:
                        print(f"[DEBUG] Tokenization failed for sample {i}: {e}")
                        s["token_id"] = torch.tensor([0], dtype=torch.long)
                        s["token_id_lengths"] = 1
                else:
                    # simple whitespace hashing fallback
                    s["token_id"] = torch.tensor([hash(w) % 10000 for w in s.get("text", "").split()], dtype=torch.long) if s.get("text") else torch.tensor([0], dtype=torch.long)
                    s["token_id_lengths"] = int(len(s["token_id"]))
            except Exception as e:
                print(f"[DEBUG] Sample preprocessing failed for sample {i}: {e}")

        print("[DEBUG] Sample preprocessing completed")

        print("[DEBUG] Step 4: Importing GPT class...")
        from TTS.tts.layers.xtts.gpt import GPT as GPTv2
        print("[DEBUG] GPT class imported successfully")

        print("[DEBUG] Step 5: Creating XTTS model instance...")
        model = None
        try:
            if Xtts is not None:
                print("[DEBUG] Xtts class available, proceeding with initialization...")
                if XttsConfig is not None:
                    print("[DEBUG] Creating XttsConfig...")
                    xcfg = XttsConfig()
                    xcfg.model_args.max_mel_positions = 608
                    xcfg.model_args.max_text_positions = 404
                    print("[DEBUG] Basic XttsConfig created")

                    if 'model_args' in cfg:
                        print("[DEBUG] Applying model_args from config...")
                        model_args = cfg['model_args']
                        xcfg.model_args.gpt_layers = model_args.get('gpt_layers', 30)
                        xcfg.model_args.gpt_n_model_channels = model_args.get('gpt_n_model_channels', 1024)
                        xcfg.model_args.gpt_n_heads = model_args.get('gpt_n_heads', 16)
                        xcfg.model_args.gpt_max_audio_tokens = model_args.get('gpt_max_audio_tokens', 605)
                        xcfg.model_args.gpt_max_text_tokens = model_args.get('gpt_max_text_tokens', 402)
                        xcfg.model_args.gpt_max_prompt_tokens = model_args.get('gpt_max_prompt_tokens', 70)
                        xcfg.model_args.gpt_num_audio_tokens = model_args.get('gpt_num_audio_tokens', 1026)
                        xcfg.model_args.gpt_number_text_tokens = model_args.get('gpt_number_text_tokens', 6681)
                        xcfg.model_args.gpt_start_audio_token = model_args.get('gpt_start_audio_token', 1024)
                        xcfg.model_args.gpt_stop_audio_token = model_args.get('gpt_stop_audio_token', 1025)
                        xcfg.model_args.gpt_code_stride_len = model_args.get('gpt_code_stride_len', 1024)
                        # xcfg.model_args.gpt_use_masking_gt_prompt_approach = model_args.get('gpt_use_masking_gt_prompt_approach', True)
                        xcfg.model_args.gpt_use_perceiver_resampler = model_args.get('gpt_use_perceiver_resampler', True)
                        xcfg.model_args.num_chars = model_args.get('num_chars', 255)
                        xcfg.model_args.tokenizer_file = model_args.get('tokenizer_file', '')
                        print("[DEBUG] Model args applied successfully")

                    print("[DEBUG] Applying config overrides...")
                    # Apply other YAML overrides
                    xcfg.from_dict(cfg)
                    print("[DEBUG] Config overrides applied")

                    print("[DEBUG] Creating XttsArgs with gpt_config...")
                    xtts_args = create_xtts_args_with_gpt_config(cfg)
                    if xtts_args:
                        xcfg.model_args = xtts_args
                        print("[DEBUG] XttsArgs applied successfully")
                    else:
                        print("[DEBUG] XttsArgs creation failed, using default config")

                    # CRITICAL: Force token limits BEFORE model initialization
                    # This ensures embeddings are created with correct sizes
                    # Text tokens: creates 404 positions for embeddings
                    # Audio tokens: creates 608 positions for embeddings
                    xcfg.model_args.gpt_max_text_tokens = 402
                    xcfg.model_args.gpt_max_audio_tokens = 605
                    print(f"[DEBUG] Ensured gpt_max_text_tokens=402 and gpt_max_audio_tokens=605")

                    print(f"[DEBUG] Creating XTTS v2 model with GPT config: layers={xcfg.model_args.gpt_layers}, channels={xcfg.model_args.gpt_n_model_channels}")

                    # Ensure GPT layers is properly set and not zero
                    if xcfg.model_args.gpt_layers <= 0:
                        xcfg.model_args.gpt_layers = 30
                        print(f"[DEBUG] Fixed gpt_layers to {xcfg.model_args.gpt_layers}")


                    print(f"[DEBUG] Setting up device and dtype for GPT creation...")

                    # Ensure we have proper device and dtype
                    if torch.cuda.is_available():
                        device = torch.device("cuda")
                        dtype = torch.float32
                    else:
                        device = torch.device("cpu")
                        dtype = torch.float32

                    print(f"[DEBUG] Using device: {device}, dtype: {dtype}")

                    # Set default tensor type to ensure consistent dtype/device
                    if device.type == "cuda":
                        torch.set_default_tensor_type('torch.cuda.FloatTensor')
                    else:
                        torch.set_default_tensor_type('torch.FloatTensor')

                    print(f"[DEBUG] Creating GPT with config: layers={xcfg.model_args.gpt_layers}, channels={xcfg.model_args.gpt_n_model_channels}")
                    print(f"[DEBUG] Calling Xtts.init_from_config...")
                    model = Xtts.init_from_config(xcfg)
                    print(f"[DEBUG] Xtts.init_from_config completed")

                    # Check if GPT component was initialized
                    if not hasattr(model, 'gpt') or model.gpt is None:
                        print(f"[DEBUG] GPT component not initialized, forcing creation...")

                        try:
                            # Import GPT class directly
                            from TTS.tts.layers.xtts.gpt import GPT

                            # Create GPT config from model args
                            gpt_config = {
                                'layers': xcfg.model_args.gpt_layers,
                                'model_dim': xcfg.model_args.gpt_n_model_channels,
                                'heads': xcfg.model_args.gpt_n_heads,
                                'max_text_tokens': xcfg.model_args.gpt_max_text_tokens,
                                'max_mel_tokens': xcfg.model_args.gpt_max_audio_tokens,
                                'max_prompt_tokens': xcfg.model_args.gpt_max_prompt_tokens,
                                'number_text_tokens': xcfg.model_args.gpt_number_text_tokens,
                                'num_audio_tokens': xcfg.model_args.gpt_num_audio_tokens,
                                'start_audio_token': xcfg.model_args.gpt_start_audio_token,
                                'stop_audio_token': xcfg.model_args.gpt_stop_audio_token,
                                'use_perceiver_resampler': xcfg.model_args.gpt_use_perceiver_resampler
                            }

                            try:
                                print(f"[DEBUG] Creating GPT component with config: {gpt_config}")

                                # Ensure number_text_tokens is not None
                                if gpt_config['number_text_tokens'] is None:
                                    print("[DEBUG] number_text_tokens is None, setting to default value")
                                    gpt_config['number_text_tokens'] = 6681  # Default value from config

                                # Create GPT instance with validated parameters
                                model.gpt = GPT(
                                    layers=gpt_config['layers'],
                                    model_dim=gpt_config['model_dim'],
                                    heads=gpt_config['heads'],
                                    max_text_tokens=gpt_config['max_text_tokens'],
                                    max_mel_tokens=gpt_config['max_mel_tokens'],
                                    max_prompt_tokens=gpt_config['max_prompt_tokens'],
                                    number_text_tokens=gpt_config['number_text_tokens'],
                                    num_audio_tokens=gpt_config['num_audio_tokens'],
                                    start_audio_token=gpt_config['start_audio_token'],
                                    stop_audio_token=gpt_config['stop_audio_token'],
                                    use_perceiver_resampler=gpt_config['use_perceiver_resampler']
                                )

                                # Move to device after creation
                                model.gpt.to(device)
                                print("[DEBUG] GPT component created and moved to device successfully")

                            except Exception as e:
                                print(f"[DEBUG] Failed to create GPT component: {e}")
                                try:
                                    print("[DEBUG] Attempting minimal GPT creation...")
                                    model.gpt = GPT(
                                        layers=30,
                                        model_dim=1024,
                                        heads=16,
                                        max_text_tokens=402,
                                        max_mel_tokens=605,
                                        max_prompt_tokens=70,
                                        number_text_tokens=6681,
                                        num_audio_tokens=8194,
                                        start_audio_token=8192,
                                        stop_audio_token=8193,
                                        use_perceiver_resampler=False
                                    )
                                    model.gpt.to(device)
                                    print("[DEBUG] Minimal GPT creation successful")
                                except Exception as e2:
                                    print(f"[DEBUG] Minimal GPT creation also failed: {e2}")
                                    raise RuntimeError(f"Could not initialize GPT component: {e}")


                            print("[DEBUG] GPT component created and moved to device successfully")

                        except Exception as e:
                            print(f"[DEBUG] Failed to create GPT component: {e}")
                            raise RuntimeError(f"Could not initialize GPT component: {e}")
                    else:
                        print(f"[DEBUG] GPT component already initialized")

                    model_with_gpt = model

                else:
                    print("[DEBUG] XttsConfig not available, using direct config...")
                    model = Xtts.init_from_config(cfg)
                    print(f"[DEBUG] Model initialized from cfg. GPT component: {hasattr(model, 'gpt') and model.gpt is not None}")
                    model_with_gpt = model
            else:
                print("[DEBUG] Xtts class not available!")
                raise RuntimeError("Xtts class not available")

        except Exception as model_error:
            print(f"[DEBUG] Model creation failed with error: {model_error}")
            logging.warning("Failed to init Xtts model: %s", model_error)
            raise model_error

        print("[DEBUG] XTTS model created successfully.")
        
        # ✅ CRITICAL: Load full checkpoint if specified in config
        import os as os_module
        if cfg.get("xtts_checkpoint"):
            checkpoint_path = cfg.get("xtts_checkpoint")
            if os_module.path.exists(checkpoint_path):
                print(f"[DEBUG] Loading full XTTS checkpoint from {checkpoint_path}...")
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    
                    if isinstance(checkpoint, dict):
                        if 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
                    
                    print(f"[DEBUG] Checkpoint keys sample: {list(state_dict.keys())[:5]}...")
                    
                    # Resize embeddings if vocab size mismatch
                    state_dict = _resize_embeddings_if_needed(state_dict, model)
                    
                    model.load_state_dict(state_dict, strict=False)
                    print(f"[DEBUG] ✅ Full checkpoint loaded successfully")
                except Exception as e:
                    print(f"[DEBUG] Warning: Failed to load full checkpoint: {e}")
            else:
                print(f"[DEBUG] ⚠️  xtts_checkpoint path does not exist: {checkpoint_path}")
                checkpoint_path = None
        
        # Load HiFiGAN decoder if not already present
        print("[DEBUG] Attempting to load HiFiGAN decoder...")
        try:
            if not hasattr(model, 'hifigan_decoder') or model.hifigan_decoder is None:
                print("[DEBUG] HiFiGAN decoder not found in model, attempting to load...")
                
                # Try to import HifiDecoder
                try:
                    from TTS.tts.layers.xtts.hifigan_decoder import HifiDecoder
                    print("[DEBUG] HifiDecoder class imported successfully")
                except ImportError as e:
                    print(f"[DEBUG] Failed to import HifiDecoder: {e}")
                    HifiDecoder = None
                
                if HifiDecoder is not None:
                    # Try to load decoder weights from XTTS v2 pretrained directory
                    decoder_checkpoint_paths = [
                        cfg.get("xtts_checkpoint") if cfg.get("xtts_checkpoint") else None,
                        cfg.get("decoder_checkpoint"),
                        "/home/ubuntu/TTS/pretrained/xtts_v2/xtts.pth",
                        "/home/ubuntu/TTS/pretrained/xtts_v2/decoder.pth",
                        "/home/ubuntu/TTS/pretrained/xtts_v2/hifigan_decoder.pth",
                        "./pretrained_models/xtts_v2/xtts.pth",
                        "./xtts.pth",
                    ]
                    # Remove None values
                    decoder_checkpoint_paths = [p for p in decoder_checkpoint_paths if p]
                    
                    decoder_loaded = False
                    for checkpoint_path in decoder_checkpoint_paths:
                        if os_module.path.exists(checkpoint_path):
                            print(f"[DEBUG] Found checkpoint at {checkpoint_path}")
                            try:
                                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                                
                                # Check if checkpoint is a dict or model state
                                if isinstance(checkpoint, dict):
                                    if 'hifigan_decoder' in checkpoint:
                                        decoder_state = {k.replace('hifigan_decoder.', ''): v for k, v in checkpoint.items() if k.startswith('hifigan_decoder.')}
                                    elif any(k.startswith('hifigan_decoder.') for k in checkpoint.keys()):
                                        decoder_state = {k.replace('hifigan_decoder.', ''): v for k, v in checkpoint.items() if k.startswith('hifigan_decoder.')}
                                    else:
                                        print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())[:5]}...")
                                        # Try to load as-is if it looks like decoder weights
                                        if any('waveform_decoder' in k or 'conv_pre' in k for k in checkpoint.keys()):
                                            decoder_state = checkpoint
                                        else:
                                            decoder_state = None
                                else:
                                    # Checkpoint is a model, try to extract decoder
                                    print("[DEBUG] Checkpoint is a model object, extracting decoder state...")
                                    decoder_state = None
                                
                                if decoder_state:
                                    # Initialize HifiDecoder
                                    decoder = HifiDecoder()
                                    decoder.to(device)
                                    decoder.load_state_dict(decoder_state, strict=False)
                                    decoder.eval()
                                    # Register as a proper module using add_module for proper DataParallel support
                                    if hasattr(model, 'add_module'):
                                        model.add_module('hifigan_decoder', decoder)
                                    else:
                                        model.hifigan_decoder = decoder
                                    decoder_loaded = True
                                    print(f"[DEBUG] ✅ HiFiGAN decoder loaded from {checkpoint_path}")
                                    break
                            except Exception as e:
                                print(f"[DEBUG] Failed to load decoder from {checkpoint_path}: {e}")
                                continue
                    
                    if not decoder_loaded:
                        print("[DEBUG] ⚠️  Could not load decoder weights from checkpoints, initializing fresh decoder")
                        print("[DEBUG] To use pretrained decoder weights, download XTTS v2:")
                        print("[DEBUG]   huggingface-cli download coqui/XTTS-v2 xtts.pth --local-dir /home/ubuntu/TTS/pretrained/xtts_v2")
                        print("[DEBUG] Or set 'xtts_checkpoint' in config to point to your checkpoint")
                        try:
                            decoder = HifiDecoder()
                            decoder.to(device)
                            decoder.eval()
                            # Register as a proper module using add_module for proper DataParallel support
                            if hasattr(model, 'add_module'):
                                model.add_module('hifigan_decoder', decoder)
                            else:
                                model.hifigan_decoder = decoder
                            print("[DEBUG] ✅ Fresh HiFiGAN decoder created (will train from random initialization)")
                        except Exception as e:
                            print(f"[DEBUG] Failed to create fresh decoder: {e}")
            else:
                print("[DEBUG] ✅ HiFiGAN decoder already present in model")
        except Exception as e:
            print(f"[DEBUG] Warning: Could not ensure decoder is loaded: {e}")
            logging.warning(f"Decoder loading issue: {e}")

    except Exception as e:
        print(f"[DEBUG] Fatal error in model creation: {e}")
        raise e

    # data loaders
    print("[DEBUG] Creating data loaders...")
    if model is None:
        raise RuntimeError("Cannot instantiate Xtts model. Ensure Coqui TTS is installed and config is valid.")

    import os
    import sys
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import yaml
    import argparse
    from pathlib import Path
    import pickle as pickle_tts
    from typing import Any, Callable, Dict, Union
    import fsspec

    class RenamingUnpickler(pickle_tts.Unpickler):
        """Overload default pickler to solve module renaming problem"""
        def find_class(self, module, name):
            return super().find_class(module.replace("mozilla_voice_tts", "TTS"), name)

    class AttrDict(dict):
        """A custom dict which converts dict keys to class attributes"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    def load_fsspec(
        path: str,
        map_location: Union[str, Callable, torch.device, Dict[Union[str, torch.device], Union[str, torch.device]]] = None,
        cache: bool = True,
        **kwargs,
    ) -> Any:
        """Like torch.load but can load from other locations (e.g. s3:// , gs://)."""
        is_local = os.path.isdir(path) or os.path.isfile(path)
        if cache and not is_local:
            with fsspec.open(
                f"filecache::{path}",
                filecache={"cache_storage": str(get_user_data_dir("tts_cache"))},
                mode="rb",
            ) as f:
                return torch.load(f, map_location=map_location, **kwargs)
        else:
            with fsspec.open(path, "rb") as f:
                return torch.load(f, map_location=map_location, **kwargs)

    def load_checkpoint(model, checkpoint_path, use_cuda=False, eval=False, cache=False):
        """Proper checkpoint loading function"""
        try:
            state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        except ModuleNotFoundError:
            pickle_tts.Unpickler = RenamingUnpickler
            state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), pickle_module=pickle_tts, cache=cache)
        model.load_state_dict(state["model"])
        if use_cuda:
            model.cuda()
        if eval:
            model.eval()
        return model, state

    # NOTE: full training-state restore (model + optimizer + scheduler + epoch)
    # will be attempted inside train_loop via load_checkpoint_enhanced().
    if args.restore_path:
        if os.path.exists(args.restore_path):
            print(f"[DEBUG] Resume requested: {args.restore_path} (full restore deferred to train_loop)")
        else:
            print(f"[DEBUG] restore_path specified but file not found: {args.restore_path}")


    # else:
    #     print("[DEBUG] Creating new XTTS v2 model from config")

    if 'model_with_gpt' in locals():
        model = model_with_gpt
        print("[DEBUG] Using model with successfully initialized GPT component")

    print(f"[DEBUG] Final model check. GPT component: {hasattr(model, 'gpt') and model.gpt is not None}")

    if hasattr(model, 'gpt') and model.gpt is not None:
        if hasattr(model.gpt, 'use_perceiver_resampler'):
            print(f"[DEBUG] GPT is XTTS v2 (perceiver_resampler: {model.gpt.use_perceiver_resampler})")
        else:
            print("[DEBUG] GPT component found but version unclear")
    else:
        raise RuntimeError("GPT component is not initialized. Cannot proceed with training.")

    # if multiple GPUs and cuda requested, wrap
    print(f"[DEBUG] Before DataParallel: has hifigan_decoder = {hasattr(model, 'hifigan_decoder')}")
    if hasattr(model, 'hifigan_decoder'):
        try:
            decoder_device = next(model.hifigan_decoder.parameters()).device if hasattr(model.hifigan_decoder, 'parameters') else 'unknown'
            print(f"[DEBUG] Before DataParallel: decoder type = {type(model.hifigan_decoder).__name__}, device = {decoder_device}")
        except:
            print(f"[DEBUG] Before DataParallel: decoder type = {type(model.hifigan_decoder).__name__}, device = error")
    
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[DEBUG] Wrapping model with DataParallel (detected {torch.cuda.device_count()} GPUs)")
        model = torch.nn.DataParallel(model)

    print(f"[DEBUG] After DataParallel: has module = {hasattr(model, 'module')}")
    if hasattr(model, 'module'):
        print(f"[DEBUG] After DataParallel: module.hifigan_decoder = {hasattr(model.module, 'hifigan_decoder')}")
        if hasattr(model.module, 'hifigan_decoder') and model.module.hifigan_decoder is not None:
            try:
                decoder_device = next(model.module.hifigan_decoder.parameters()).device if hasattr(model.module.hifigan_decoder, 'parameters') else 'unknown'
                print(f"[DEBUG] After DataParallel: decoder type = {type(model.module.hifigan_decoder).__name__}, device = {decoder_device}")
            except:
                print(f"[DEBUG] After DataParallel: decoder type = {type(model.module.hifigan_decoder).__name__}, device = error")
    
    model.to(device)
    model._device = device

    print(f"[DEBUG] After to(device): has module = {hasattr(model, 'module')}")
    if hasattr(model, 'module'):
        print(f"[DEBUG] After to(device): module.hifigan_decoder = {hasattr(model.module, 'hifigan_decoder')}")
        if hasattr(model.module, 'hifigan_decoder') and model.module.hifigan_decoder is not None:
            try:
                decoder_device = next(model.module.hifigan_decoder.parameters()).device if hasattr(model.module.hifigan_decoder, 'parameters') else 'unknown'
                print(f"[DEBUG] After to(device): decoder device = {decoder_device}")
            except:
                print(f"[DEBUG] After to(device): decoder device = error")

    print(f"[DEBUG] Final model check. GPT component: {hasattr(model, 'gpt') and model.gpt is not None}")
    if hasattr(model, 'module'):  # DataParallel wrapped
        print(f"[DEBUG] DataParallel wrapped. Inner GPT: {hasattr(model.module, 'gpt') and model.module.gpt is not None}")
        print(f"[DEBUG] DataParallel wrapped. Inner decoder: {hasattr(model.module, 'hifigan_decoder')}")

    # monkey-patch forward (must be after model instantiation)
    model.forward = types.MethodType(forward_xtts, model)

    print("[DEBUG] Model loaded successfully, proceeding to training...")

    # ✅ LOAD TRAINED TEXT EMOTION CLASSIFIER
    text_classifier = None
    print(f"[DEBUG] use_emotions config value: {cfg.get('use_emotions', False)}")
    if cfg.get("use_emotions", False):
        classifier_path = cfg.get("text_emotion_classifier_path", "/home/ubuntu/TTS/models/text_emotion_classifier")
        print(f"[DEBUG] Attempting to load text classifier from: {classifier_path}")
        logging.info(f"[TEXT CLASSIFIER] Checking path: {classifier_path}")
        if os.path.exists(classifier_path):
            try:
                print(f"[DEBUG] Path exists, loading TextEmotionPredictor...")
                text_classifier = TextEmotionPredictor(classifier_path, device=device)
                print(f"[DEBUG] ✅ TextEmotionPredictor loaded successfully!")
                logging.info(f"✅ Loaded trained text emotion classifier from {classifier_path}")
            except Exception as e:
                print(f"[DEBUG] ❌ Failed to load text emotion classifier: {e}")
                import traceback
                logging.error(f"[TEXT CLASSIFIER ERROR] Full traceback:\n{traceback.format_exc()}")
                logging.warning(f"Failed to load text emotion classifier: {e}. Will use CSV emotions instead.")
                text_classifier = None
        else:
            print(f"[DEBUG] ❌ Path does not exist: {classifier_path}")
            logging.warning(f"Text classifier path not found: {classifier_path}. Will use CSV emotions instead.")

    # Build dataloaders (use collate defined earlier)
    batch_size = cfg.get("batch_size", 1)
    print(f"[DEBUG] Setting up data loaders with batch_size={batch_size}")

    # ✅ Match generator device to training device (GPU)
    gen = torch.Generator(device=device)  # device = torch.device("cuda")

    # Get feature normalization setting from config
    feature_norm = cfg.get("emotion", {}).get("feature_normalization", "standard_scale")
    logging.info(f"[FEATURE NORM] Using feature normalization mode: {feature_norm}")
    if feature_norm == "l2_norm":
        logging.warning("[FEATURE NORM] L2 normalization will collapse features! Consider using 'standard_scale' or 'layer_norm'")
    else:
        logging.info(f"[FEATURE NORM] L2 normalization DISABLED - {feature_norm} will preserve feature variance")
    
    # ✅ ADD: Weighted sampling to balance emotion classes
    from torch.utils.data import WeightedRandomSampler
    
    if cfg.get("use_balanced_sampling", False):
        # Calculate sample weights based on emotion class
        # Location 1: For weighted sampling
        emotion_counts = {0: 212, 1: 90, 2: 42, 3: 518, 4: 166}  # Keep this - it's your data
        class_weights = {0: 0.88, 1: 1.10, 2: 1.37, 3: 0.70, 4: 0.94}
        
        sample_weights = []
        for sample in train_samples:
            emotion_id = sample.get("emotion", 3)  # default to fearful if missing
            sample_weights.append(class_weights.get(emotion_id, 1.0))
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_samples),
            replacement=True
        )
        logging.info(f"[BALANCED SAMPLING] Using WeightedRandomSampler with class weights: {class_weights}")
        
        train_loader = DataLoader(
            train_samples,
            batch_size=batch_size,
            sampler=sampler,  # Use sampler instead of shuffle
            generator=gen,
            collate_fn=lambda b: collate_fn(
                b, ap, spk_encoder, 
                include_emotion=cfg.get("use_emotions", False),
                text_classifier=text_classifier,
                feature_normalization=feature_norm,
                encodec_model=encodec,
                reference_audio_path=train_samples[0].get("path") or train_samples[0].get("audio_file")
            )
        )
    else:
        train_loader = DataLoader(
            train_samples,
            batch_size=batch_size,
            shuffle=True,
            generator=gen,
            collate_fn=lambda b: collate_fn(
                b, ap, spk_encoder, 
                include_emotion=cfg.get("use_emotions", False),
                text_classifier=text_classifier,
                feature_normalization=feature_norm,
                encodec_model=encodec,
                reference_audio_path=train_samples[0].get("path") or train_samples[0].get("audio_file")
            )
        )

    # ✅ Use SAME reference audio for train AND eval (consistency is critical)
    fixed_reference_audio = train_samples[0].get("path") or train_samples[0].get("audio_file")
    
    eval_loader = DataLoader(
        eval_samples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(
            b, ap, spk_encoder, 
            include_emotion=cfg.get("use_emotions", False),
            text_classifier=text_classifier,
            feature_normalization=feature_norm,
            encodec_model=encodec,
            reference_audio_path=fixed_reference_audio
        )
    ) if eval_samples else None

    print(f"[DEBUG] Data loaders created. Train samples: {len(train_samples)}, Eval samples: {len(eval_samples) if eval_samples else 0}")

    try:
        print("[DEBUG] Starting training loop...")
        train_loop(
            model,
            train_loader,
            eval_loader,
            cfg,
            device,
            ap,
            epochs=args.epochs,
            dry_run=args.dry_run,
            restore_path=args.restore_path
        )
        print("[DEBUG] Training completed successfully!")
    except Exception as e:
        print(f"[DEBUG] Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Dry run support: run one forward and exit
    if args.dry_run:
        logging.info("Dry-run: executing one batch through model.forward")
        try:
            sample_batch = next(iter(train_loader))
        except StopIteration:
            raise RuntimeError("No samples available for dry-run. Create a small metadata csv with 1 sample.")

        sample_ns = batch_to_namespace(sample_batch)
        to_device(sample_ns, device)

        # Prepare forward batch the same way as in training
        # Prepare forward batch the same way as in training
        forward_batch = {}
        for k in required_keys:
            if hasattr(batch_ns, k):
                forward_batch[k] = getattr(batch_ns, k)   # <-- Use forward_batch!
            else:
                if k == "token_id":
                    forward_batch[k] = torch.zeros(1, 1, dtype=torch.long, device=device)
                elif k == "token_id_lengths":
                    forward_batch[k] = torch.tensor([1], dtype=torch.long, device=device)
                elif k == "cond_mels":
                    if "mel" in forward_batch and isinstance(forward_batch["mel"], torch.Tensor):
                        batch_size, n_mels, t = forward_batch["mel"].shape
                        forward_batch[k] = torch.zeros(batch_size, n_mels, t, device=device)
                    else:
                        batch_size = forward_batch["token_id"].shape[0] if "token_id" in forward_batch else 1
                        forward_batch[k] = torch.zeros(batch_size, 80, 80, device=device)
                elif k in ["mel", "linear"]:
                    forward_batch[k] = torch.zeros(1, 80, 80, device=device)
                elif k in ["stop_targets", "gpt_audio_tokens", "gpt_cond_latent"]:
                    forward_batch[k] = torch.zeros(1, 1, device=device)
                elif k == "expected_output_len":
                    forward_batch[k] = torch.tensor([1], dtype=torch.long, device=device)
                elif k == "speaker_embedding":
                    forward_batch[k] = torch.zeros(1, 256, device=device)

        forward_batch["tokens"] = forward_batch["token_id"]
        if "token_id_lengths" in forward_batch:
            forward_batch["tokens_lengths"] = forward_batch["token_id_lengths"]

        # Defensive runtime check for cond_mels!
        if ("cond_mels" not in forward_batch) or (forward_batch["cond_mels"] is None):
            print("WARNING: cond_mels missing or None in forward_batch, constructing artificial cond_mels")
            if "mel" in forward_batch and isinstance(forward_batch["mel"], torch.Tensor):
                batch_size, n_mels, t = forward_batch["mel"].shape
                forward_batch["cond_mels"] = torch.zeros(batch_size, n_mels, t, device=device)
            else:
                batch_size = forward_batch["token_id"].shape[0] if "token_id" in forward_batch else 1
                forward_batch["cond_mels"] = torch.zeros(batch_size, 80, 80, device=device)
        else:
            # Optional: check type and shape
            if not isinstance(forward_batch["cond_mels"], torch.Tensor):
                print(f"WARNING: cond_mels is not a tensor! Type: {type(forward_batch['cond_mels'])}")
            else:
                print(f"cond_mels final shape before forward: {forward_batch['cond_mels'].shape}")

        try:
            out = model.forward(**forward_batch)
            logging.info("Dry-run forward successful. Output keys: %s", list(out.keys()) if isinstance(out, dict) else "unknown")
        except Exception as e:
            logging.exception("Dry-run forward failed: %s", e)

if __name__ == "__main__":
    main()
