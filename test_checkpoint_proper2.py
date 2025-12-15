#!/usr/bin/env python3
"""
Properly test checkpoint by reconstructing config from it.
Uses same initialization as training script.
"""

import torch
import torchaudio
import argparse
import logging
import sys
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

try:
    from TTS.tts.configs.xtts_config import XttsConfig, XttsArgs
    from TTS.tts.models.xtts import Xtts
    from TTS.tts.layers.xtts.gpt import GPT
except Exception as e:
    logger.error(f"Failed to import TTS: {e}")
    sys.exit(1)

def test_checkpoint(checkpoint_path, speaker_audio, text, device='cuda'):
    """Test checkpoint by properly loading it like training script does"""
    
    logger.info("=" * 80)
    logger.info("PROPER CHECKPOINT TEST")
    logger.info("=" * 80)
    
    # Step 1: Load checkpoint
    logger.info(f"\n⏳ Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    logger.info("✅ Checkpoint loaded\n")
    
    # Step 2: Reconstruct config from checkpoint
    logger.info("⏳ Reconstructing config from checkpoint...")
    ckpt_config = ckpt.get('config', {})
    
    if not ckpt_config:
        logger.error("❌ No config in checkpoint")
        return False
    
    # Create XttsConfig
    config = XttsConfig()
    
    # CRITICAL: Set text tokens BEFORE initializing model to avoid embedding size mismatch
    # Default XttsConfig has gpt_max_text_tokens=401 which creates embedding size 403
    # Training used 402 which creates embedding size 404
    # Must match before model.init_from_config() is called
    config.model_args.gpt_max_text_tokens = 402
    
    # Load model_args if present
    if 'model_args' in ckpt_config:
        model_args_dict = ckpt_config['model_args']
        if isinstance(model_args_dict, dict):
            # Get the actual trained values from checkpoint
            trained_args = XttsArgs(**model_args_dict)
            
            # CRITICAL: Use checkpoint values, with text_tokens=402 override
            config.model_args = trained_args
            config.model_args.gpt_max_text_tokens = 402
            
            logger.info(f"✅ Loaded model_args from checkpoint")
            logger.info(f"   gpt_layers: {config.model_args.gpt_layers}")
            logger.info(f"   gpt_n_model_channels: {config.model_args.gpt_n_model_channels}")
            logger.info(f"   gpt_max_audio_tokens: {config.model_args.gpt_max_audio_tokens}")
            logger.info(f"   gpt_max_text_tokens: {config.model_args.gpt_max_text_tokens} (forced to 402 for embedding compatibility)")
    
    # Step 3: Initialize model with reconstructed config
    logger.info(f"\n⏳ Creating model with config...")
    try:
        model = Xtts.init_from_config(config)
        logger.info("✅ Model initialized\n")
    except Exception as e:
        logger.error(f"❌ Failed to init model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Step 3.5: Create custom GPT component (required for checkpoint compatibility)
    logger.info("⏳ Creating custom GPT component...")
    if not hasattr(model, 'gpt') or model.gpt is None:
        try:
            def _get_or_default(obj, attr, default):
                val = getattr(obj, attr, default)
                return val if val is not None else default
            
            gpt_config = {
                'layers': _get_or_default(config.model_args, 'gpt_layers', 30),
                'model_dim': _get_or_default(config.model_args, 'gpt_n_model_channels', 1024),
                'heads': _get_or_default(config.model_args, 'gpt_n_heads', 16),
                'max_text_tokens': _get_or_default(config.model_args, 'gpt_max_text_tokens', 402),
                'max_mel_tokens': _get_or_default(config.model_args, 'gpt_max_audio_tokens', 605),
                'max_prompt_tokens': _get_or_default(config.model_args, 'gpt_max_prompt_tokens', 70),
                'number_text_tokens': _get_or_default(config.model_args, 'gpt_number_text_tokens', 6681),
                'num_audio_tokens': _get_or_default(config.model_args, 'gpt_num_audio_tokens', 8194),
                'start_audio_token': _get_or_default(config.model_args, 'gpt_start_audio_token', 8192),
                'stop_audio_token': _get_or_default(config.model_args, 'gpt_stop_audio_token', 8193),
                'use_perceiver_resampler': _get_or_default(config.model_args, 'gpt_use_perceiver_resampler', False)
            }
            
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
            logger.info("✅ Custom GPT component created\n")
        except Exception as e:
            logger.error(f"❌ Failed to create GPT component: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    else:
        logger.info("✅ GPT component already initialized\n")
    
    # Step 4: Move to device BEFORE loading state dict
    logger.info(f"⏳ Moving model to {device}...")
    model = model.to(device)
    if hasattr(model, 'gpt') and model.gpt is not None:
        model.gpt = model.gpt.to(device)
    logger.info(f"✅ Model on {device}\n")
    
    # Step 5: Load state dict with size mismatch handling
    logger.info("⏳ Loading state dict from checkpoint...")
    model_state_dict = ckpt.get('model_state_dict', {})
    
    # Check for embedding size mismatches and warn user
    if 'gpt.mel_pos_embedding.emb.weight' in model_state_dict:
        ckpt_shape = model_state_dict['gpt.mel_pos_embedding.emb.weight'].shape
        model_shape = model.gpt.mel_pos_embedding.emb.weight.shape
        if ckpt_shape != model_shape:
            logger.warning(f"⚠️  Embedding size mismatch:")
            logger.warning(f"    Checkpoint: gpt_max_audio_tokens ≈ {ckpt_shape[0]-3}")
            logger.warning(f"    Model:      gpt_max_audio_tokens ≈ {model_shape[0]-3}")
            logger.warning(f"    This checkpoint may be incomplete or partially trained")
    
    try:
        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
        if missing or unexpected:
            logger.warning(f"  Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
            if len(missing) > 0:
                logger.warning(f"  First missing keys: {list(missing)[:3]}")
        else:
            logger.info("✅ State dict loaded successfully\n")
    except RuntimeError as e:
        if "size mismatch" in str(e):
            logger.error(f"❌ Cannot load checkpoint - embedding size mismatch")
            logger.error(f"   Error: {e}")
            logger.error(f"\n   The checkpoint appears to be incomplete or corrupted.")
            logger.error(f"   It was saved with inconsistent configuration values.")
            return False
        else:
            raise
    
    # Step 5.5: Load tokenizer
    logger.info("⏳ Loading tokenizer...")
    try:
        tokenizer_file = "/home/ubuntu/TTS/pretrained/xtts_v2/vocab.json"
        if os.path.exists(tokenizer_file):
            from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
            model.tokenizer = VoiceBpeTokenizer(vocab_file=tokenizer_file)
            logger.info("✅ Tokenizer loaded\n")
        else:
            logger.warning(f"⚠️  Tokenizer file not found: {tokenizer_file}")
    except Exception as e:
        logger.error(f"❌ Failed to load tokenizer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Step 6: Set to eval mode
    logger.info("⏳ Setting model to eval...")
    try:
        model.eval()
        logger.info("✅ Model in eval mode\n")
    except Exception as e:
        logger.error(f"❌ Failed to set eval: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Step 7: Test inference
    logger.info("=" * 80)
    logger.info("INFERENCE TEST")
    logger.info("=" * 80)
    
    try:
        logger.info(f"\n⏳ Loading speaker audio...")
        wav, sr = torchaudio.load(speaker_audio)
        
        if sr != 24000:
            wav = torchaudio.transforms.Resample(sr, 24000)(wav)
        
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        
        logger.info(f"✓ Speaker audio: {wav.shape}\n")
        
        # Get conditioning
        logger.info(f"⏳ Getting conditioning latents...")
        with torch.no_grad():
            gpt_cond, speaker_emb = model.get_conditioning_latents(
                audio_path=speaker_audio
            )
        
        # Fix speaker_emb shape if it has extra dimensions
        while speaker_emb.dim() > 2:
            speaker_emb = speaker_emb.squeeze(-1)
        if speaker_emb.dim() == 1:
            speaker_emb = speaker_emb.unsqueeze(0)
        
        logger.info(f"✓ gpt_cond: {gpt_cond.shape}, speaker_emb: {speaker_emb.shape}\n")
        
        # Generate
        logger.info(f"⏳ Generating: '{text}'")
        with torch.no_grad():
            outputs = model.inference(
                text=text,
                language="en",
                gpt_cond_latent=gpt_cond,
                speaker_embedding=speaker_emb,
                temperature=0.75,
                length_penalty=1.0,
                repetition_penalty=5.0,
                top_k=50,
                top_p=0.85
            )
        
        if 'wav' in outputs:
            output_wav = outputs['wav']
            duration = output_wav.shape[-1] / 24000
            rms = torch.sqrt(torch.mean(output_wav ** 2))
            
            logger.info(f"✅ Generation succeeded")
            logger.info(f"   Duration: {duration:.2f}s")
            logger.info(f"   RMS: {rms:.6f}")
            
            # Save
            output_path = "/tmp/test_output.wav"
            torchaudio.save(output_path, output_wav.cpu(), 24000)
            logger.info(f"\n✓ Saved to: {output_path}")
            
            return True
        else:
            logger.error(f"❌ No 'wav' in outputs")
            return False
    
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--speaker_audio', 
                       default='/home/ubuntu/emma_dataset/wavs/newpodcastemmaclip_269.wav')
    parser.add_argument('--text', default='This is a test of the checkpoint model.')
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    success = test_checkpoint(
        args.checkpoint,
        args.speaker_audio,
        args.text,
        args.device
    )
    
    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✅ MODEL IS WORKING - Ready for training")
    else:
        logger.error("❌ MODEL HAS ISSUES")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
