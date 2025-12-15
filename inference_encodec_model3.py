"""
Inference using XTTS model with EnCodec quality enhancement
"""
import os
import torch
import torchaudio
import soundfile as sf
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from TTS.tts.configs.xtts_config import XttsConfig, XttsArgs
    from TTS.tts.models.xtts import Xtts
    from encodec import EncodecModel
except Exception as e:
    logger.error(f"Failed to import: {e}")
    exit(1)

original_torch_cat = torch.cat

def patched_torch_cat(tensors, dim=0, **kwargs):
    """Patch torch.cat to handle batch dimension mismatches"""
    try:
        return original_torch_cat(tensors, dim=dim, **kwargs)
    except RuntimeError as e:
        if "Sizes of tensors must match except in dimension" in str(e) and dim == 1:
            logger.warning(f"[torch.cat] Batch mismatch at dim={dim} - auto-fixing...")
            tensors = list(tensors)
            batch_sizes = [t.shape[0] if t.dim() > 0 else 1 for t in tensors]
            
            if len(set(batch_sizes)) > 1:
                target_batch = min(batch_sizes)
                logger.warning(f"  Batch sizes: {batch_sizes} → target: {target_batch}")
                
                fixed_tensors = []
                for i, t in enumerate(tensors):
                    if t.shape[0] != target_batch:
                        t_fixed = t[:target_batch]
                        logger.warning(f"    Tensor {i}: {t.shape} → {t_fixed.shape}")
                        fixed_tensors.append(t_fixed)
                    else:
                        fixed_tensors.append(t)
                
                try:
                    result = original_torch_cat(fixed_tensors, dim=dim, **kwargs)
                    logger.warning(f"  ✓ Fixed concatenation succeeded")
                    return result
                except Exception as e2:
                    logger.error(f"  ✗ Failed after fixing: {e2}")
                    raise
        raise

torch.cat = patched_torch_cat
logger.info("✓ torch.cat globally patched")


def load_model(checkpoint_path, device="cuda"):
    """Load EnCodec-trained model with torch.cat patch active"""
    logger.info(f"Loading EnCodec-trained checkpoint")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint.get("config", {})
    
    logger.info(f"Loading config from checkpoint...")
    config = XttsConfig()
    
    # CRITICAL: Set text tokens BEFORE any model initialization to avoid embedding size mismatch
    # Default XttsConfig has gpt_max_text_tokens=401 which creates embedding size 403
    # Training uses 402 which creates embedding size 404
    config.model_args.gpt_max_text_tokens = 402
    
    # Try to load full config object if it exists
    if isinstance(config_dict, XttsConfig):
        config = config_dict
        logger.info(f"✓ Loaded XttsConfig object from checkpoint")
        config.model_args.gpt_max_text_tokens = 402
    # Otherwise try to reconstruct from model_args dict
    elif isinstance(config_dict.get("model_args"), dict):
        logger.info(f"Reconstructing config from model_args dict...")
        config.model_args = XttsArgs(**config_dict["model_args"])
        # CRITICAL: Force gpt_max_text_tokens=402 for consistency with trained checkpoint
        config.model_args.gpt_max_text_tokens = 402
        logger.info(f"  gpt_max_audio_tokens: {config.model_args.gpt_max_audio_tokens}")
        logger.info(f"  gpt_layers: {config.model_args.gpt_layers}")
        logger.info(f"  gpt_max_text_tokens: {config.model_args.gpt_max_text_tokens} (forced to 402 for embedding compatibility)")
    else:
        logger.warning(f"Could not extract config from checkpoint, using defaults with gpt_max_text_tokens=402")
        logger.warning(f"This ensures embeddings match the trained checkpoint size")
    
    logger.info(f"Creating model with config: gpt_max_audio_tokens={config.model_args.gpt_max_audio_tokens}")
    model = Xtts.init_from_config(config)
    
    if model.gpt is None:
        from TTS.tts.layers.xtts.gpt import GPT
        model.gpt = GPT(
            layers=int(config.model_args.gpt_layers),
            model_dim=int(config.model_args.gpt_n_model_channels),
            heads=int(config.model_args.gpt_n_heads),
            max_text_tokens=int(getattr(config.model_args, 'gpt_max_text_tokens', 402) or 402),
            max_mel_tokens=int(getattr(config.model_args, 'gpt_max_audio_tokens', 605) or 605),
            max_prompt_tokens=int(getattr(config.model_args, 'gpt_max_prompt_tokens', 70) or 70),
            number_text_tokens=int(getattr(config.model_args, 'gpt_number_text_tokens', 6681) or 6681),
            num_audio_tokens=int(getattr(config.model_args, 'gpt_num_audio_tokens', 8194) or 8194),
            start_audio_token=int(getattr(config.model_args, 'gpt_start_audio_token', 8192) or 8192),
            stop_audio_token=int(getattr(config.model_args, 'gpt_stop_audio_token', 8193) or 8193),
            use_perceiver_resampler=bool(getattr(config.model_args, 'gpt_use_perceiver_resampler', True))
        )
        model.gpt.to(device)
    
    # Handle shape mismatches for embeddings (mel_pos_embedding, text_pos_embedding)
    state_dict = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    
    # Handle embedding layers with shape mismatches
    for key in list(state_dict.keys()):
        if "pos_embedding" in key and key in model_state:
            ckpt_shape = state_dict[key].shape
            model_shape = model_state[key].shape
            if ckpt_shape != model_shape:
                logger.warning(f"Shape mismatch for {key}: {ckpt_shape} → {model_shape}")
                
                # If checkpoint is smaller, pad it instead of skipping
                if ckpt_shape[0] < model_shape[0]:
                    logger.info(f"  Padding {key} from {ckpt_shape[0]} to {model_shape[0]} positions")
                    try:
                        # Load the trained weights and pad with them repeated/interpolated
                        trained_emb = state_dict[key]
                        padded_emb = torch.zeros(model_shape, dtype=trained_emb.dtype, device=trained_emb.device)
                        padded_emb[:trained_emb.shape[0]] = trained_emb
                        
                        # Initialize padded positions with same distribution as trained ones
                        std = trained_emb.std().item()
                        torch.nn.init.normal_(padded_emb[trained_emb.shape[0]:], mean=0, std=std)
                        
                        state_dict[key] = padded_emb
                        logger.info(f"  ✓ Padded and initialized with std={std:.6f}")
                    except Exception as e:
                        logger.warning(f"  Failed to pad, skipping: {e}")
                        del state_dict[key]
                else:
                    # Checkpoint is larger, skip
                    logger.warning(f"  Skipping (checkpoint larger)")
                    del state_dict[key]
    
    model.load_state_dict(state_dict, strict=False)
    
    # Load tokenizer
    tokenizer_file = "/home/ubuntu/TTS/pretrained/xtts_v2/vocab.json"
    try:
        from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
        model.tokenizer = VoiceBpeTokenizer(vocab_file=tokenizer_file)
    except TypeError:
        from tokenizers import Tokenizer as HFTokenizer
        raw_tokenizer = HFTokenizer.from_file(tokenizer_file)
        from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
        model.tokenizer = VoiceBpeTokenizer()
        model.tokenizer.tokenizer = raw_tokenizer
    
    model.to(device)
    model.eval()
    logger.info("✓ Model ready")
    return model


def apply_encodec_enhancement(waveform, device, bandwidth=6.0):
    """Apply EnCodec encode/decode for quality enhancement"""
    try:
        logger.info(f"[EnCodec] Applying quality enhancement to waveform shape {waveform.shape}")
        
        encodec = EncodecModel.encodec_model_24khz().to(device)
        encodec.set_target_bandwidth(bandwidth)
        encodec.eval()
        
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float().to(device)
        else:
            waveform = waveform.to(device)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        
        logger.info(f"[EnCodec] Input shape before encode: {waveform.shape}")
        
        with torch.no_grad():
            encoded_frames = encodec.encode(waveform)
            if isinstance(encoded_frames, (tuple, list)):
                encoded_frames = list(encoded_frames)
            else:
                encoded_frames = [encoded_frames]
            
            decoded = encodec.decode(encoded_frames)
            if isinstance(decoded, (tuple, list)):
                decoded = decoded[0]
        
        logger.info(f"[EnCodec] Output shape: {decoded.shape}")
        
        decoded = decoded.squeeze()
        if decoded.dim() > 1:
            decoded = decoded[0]
        
        result = decoded.cpu().numpy()
        logger.info(f"[EnCodec] Final output shape: {result.shape}")
        return result
    
    except Exception as e:
        logger.error(f"[EnCodec] Enhancement failed: {e}, returning original")
        import traceback
        logger.error(traceback.format_exc())
        if isinstance(waveform, torch.Tensor):
            return waveform.squeeze().cpu().numpy()
        return np.squeeze(waveform)


def estimate_audio_tokens(text, base_rate=10.0):
    """Estimate required audio tokens from text length
    
    Args:
        text: Input text string
        base_rate: tokens per word (INCREASED from 3.5 to 10.0 - audio tokens are different from words!)
    
    Returns:
        Estimated token count (min 200 for full training-length generation)
    """
    word_count = len(text.split())
    # Increase minimum to 200 tokens to match training distribution
    # Model was trained with audio_codes up to 779 tokens
    estimated_tokens = max(int(word_count * base_rate), 200)
    logger.info(f"[Length] Text: {word_count} words → ~{estimated_tokens} audio tokens (min 200)")
    return estimated_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="This sounds amazing!")
    parser.add_argument("--checkpoint", help="Path to trained checkpoint")
    parser.add_argument("--speaker_audio", default="/home/ubuntu/emma_dataset/wavs/newpodcastemmaclip_269.wav")
    parser.add_argument("--output", default="generated_audio.wav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encodec_bandwidth", type=float, default=6.0, help="EnCodec bandwidth in kbps")
    parser.add_argument("--skip_encodec", action="store_true", help="Skip EnCodec enhancement")
    parser.add_argument("--max_length", type=int, default=400, help="Max audio tokens (~256 samples/token at 24kHz). Default 400 = ~30+ seconds")
    parser.add_argument("--token_rate", type=float, default=10.0, help="Tokens per word for length estimation (default 10)")
    
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Device: {device}")
    logger.info(f"Using checkpoint: {args.checkpoint}")
    
    if not args.checkpoint:
        logger.error("--checkpoint required")
        return
    
    model = load_model(args.checkpoint, device=device)
    
    if not os.path.exists(args.speaker_audio):
        logger.error(f"Speaker audio not found: {args.speaker_audio}")
        return
    
    logger.info(f"Synthesizing: '{args.text}'")
    
    original_hifigan_forward = model.hifigan_decoder.forward
    
    def fixed_hifigan_forward(gpt_latents, g=None, *args, **kwargs):
        """Fix latent shape from [B, T, C] to [B, C, T] for HiFiGAN"""
        logger.info(f"[HiFiGAN] Input shape: {gpt_latents.shape}")
        
        if gpt_latents.dim() == 3:
            if gpt_latents.shape[1] != 1024 and gpt_latents.shape[2] == 1024:
                logger.info(f"[HiFiGAN] Permuting from [B, T, C] to [B, C, T]")
                gpt_latents = gpt_latents.permute(0, 2, 1)
                logger.info(f"[HiFiGAN] Permuted shape: {gpt_latents.shape}")
        
        return original_hifigan_forward(gpt_latents, g=g, *args, **kwargs)
    
    model.hifigan_decoder.forward = fixed_hifigan_forward
    
    try:
        gpt_cond, speaker_emb = model.get_conditioning_latents(audio_path=args.speaker_audio)
        
        if speaker_emb.ndim == 3 and speaker_emb.shape[-1] == 1:
            speaker_emb = speaker_emb.squeeze(-1)
        
        logger.info(f"Conditioning: gpt_cond={gpt_cond.shape}, speaker_emb={speaker_emb.shape}")
        
        max_length = args.max_length or estimate_audio_tokens(args.text, args.token_rate)
        logger.info(f"[Length Control] Setting max_length={max_length} (was unlimited 605)")
        
        with torch.no_grad():
            logger.info("Running inference with HiFiGAN vocoder...")
            inference_kwargs = {
                "text": args.text,
                "language": "en",
                "gpt_cond_latent": gpt_cond,
                "speaker_embedding": speaker_emb,
                "temperature": 0.75,
                "length_penalty": 1.0,
                "repetition_penalty": 5.0,
                "top_k": 50,
                "top_p": 0.85
            }
            
            try:
                outputs = model.inference(
                    **inference_kwargs,
                    max_length=max_length
                )
            except (TypeError, ValueError):
                logger.warning("max_length not supported in inference, trying without it")
                outputs = model.inference(**inference_kwargs)
            
            if "gpt_latents" in outputs:
                gpt_latents = outputs["gpt_latents"]
                if isinstance(gpt_latents, np.ndarray):
                    gpt_latents = torch.from_numpy(gpt_latents)
                logger.info(f"[DEBUG] GPT latents stats - shape: {gpt_latents.shape}, min: {gpt_latents.min():.4f}, max: {gpt_latents.max():.4f}, mean: {gpt_latents.mean():.4f}, std: {gpt_latents.std():.4f}")
                logger.info(f"[DEBUG] GPT latents first 5 timesteps:\n{gpt_latents[0, :5, :5]}")
                logger.info(f"[DEBUG] Checking if latents are just noise or structured...")
                sample_var = gpt_latents[0, :10, :].var(dim=0).mean().item()
                logger.info(f"[DEBUG] Sample variance across first 10 timesteps: {sample_var:.4f} (should be >0.1 if structured, <0.01 if repetitive)")
        
        audio_output = outputs["wav"]
        if hasattr(audio_output, 'cpu'):
            audio_np = audio_output.cpu().numpy()
        else:
            audio_np = audio_output
        
        logger.info(f"✓ HiFiGAN output shape: {audio_np.shape}")
        
        max_samples = max_length * 1114
        if audio_np.shape[0] > max_samples:
            logger.warning(f"[Truncate] Output {audio_np.shape[0]} samples exceeds expected {max_samples} → truncating")
            audio_np = audio_np[:max_samples]
            logger.info(f"[Truncate] Truncated to {audio_np.shape[0]} samples")
        
        if not args.skip_encodec:
            logger.info(f"Applying EnCodec enhancement (bandwidth={args.encodec_bandwidth} kbps)...")
            audio_np = apply_encodec_enhancement(audio_np, device, args.encodec_bandwidth)
            logger.info(f"✓ EnCodec output shape: {audio_np.shape}")
        
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        sf.write(args.output, audio_np, 24000)
        logger.info(f"✓ Audio saved: {args.output}")
        logger.info(f"✓ SUCCESS: Generated {audio_np.shape[0]} samples at 24kHz")
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
