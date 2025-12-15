(xtts-env3109) (base) root@cv5552379:/home/ubuntu/TTS# python test_checkpoint_proper2.py \
>     --checkpoint /home/ubuntu/TTS/outputs/emma_xtts_with_encodec/checkpoint_epoch_1_ts1765731041.pth \
>     --speaker_audio /home/ubuntu/emma_dataset/wavs/newpodcastemmaclip_269.wav \
>     --text "This is a test."
================================================================================
PROPER CHECKPOINT TEST
================================================================================

⏳ Loading checkpoint: /home/ubuntu/TTS/outputs/emma_xtts_with_encodec/checkpoint_epoch_1_ts1765731041.pth
/home/ubuntu/TTS/test_checkpoint_proper2.py:35: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ckpt = torch.load(checkpoint_path, map_location='cpu')
✅ Checkpoint loaded

⏳ Reconstructing config from checkpoint...
✅ Loaded model_args from checkpoint
   gpt_layers: 30
   gpt_n_model_channels: 1024
   gpt_max_audio_tokens: 605
   gpt_max_text_tokens: 402 (forced to 402 for embedding compatibility)

⏳ Creating model with config...
✅ Model initialized

⏳ Creating custom GPT component...
✅ GPT component already initialized

⏳ Moving model to cuda...
✅ Model on cuda

⏳ Loading state dict from checkpoint...
  Missing keys: 0, Unexpected: 372
⏳ Loading tokenizer...
✅ Tokenizer loaded

⏳ Setting model to eval...
GPT2InferenceModel has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
  - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
  - If you are not the owner of the model architecture class, please contact the model code owner to update it.
✅ Model in eval mode

================================================================================
INFERENCE TEST
================================================================================

⏳ Loading speaker audio...
✓ Speaker audio: torch.Size([1, 240745])

⏳ Getting conditioning latents...
✓ gpt_cond: torch.Size([1, 130, 1024]), speaker_emb: torch.Size([1, 512])

⏳ Generating: 'This is a test.'
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
❌ Inference failed: Sizes of tensors must match except in dimension 1. Expected size 1 but got size 4 for tensor number 2 in the list.
Traceback (most recent call last):
  File "/home/ubuntu/TTS/test_checkpoint_proper2.py", line 228, in test_checkpoint
    outputs = model.inference(
  File "/root/xtts-env3109/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/home/ubuntu/TTS/TTS/tts/models/xtts.py", line 554, in inference
    gpt_latents = self.gpt(
  File "/root/xtts-env3109/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/xtts-env3109/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/TTS/TTS/tts/layers/xtts/gpt.py", line 518, in forward
    text_logits, mel_logits = self.get_logits(
  File "/home/ubuntu/TTS/TTS/tts/layers/xtts/gpt.py", line 274, in get_logits
    emb = torch.cat([prompt, first_inputs, second_inputs], dim=1)
RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 1 but got size 4 for tensor number 2 in the list.


================================================================================
❌ MODEL HAS ISSUES
================================================================================
(xtts-env3109) (base) root@cv5552379:/home/ubuntu/TTS# python test_checkpoint_proper3.py \
>     --checkpoint /home/ubuntu/TTS/outputs/emma_xtts_with_encodec/checkpoint_epoch_1_ts1765731041.pth \
>     --speaker_audio /home/ubuntu/emma_dataset/wavs/newpodcastemmaclip_269.wav \
>     --text "This is a test."
python: can't open file '/home/ubuntu/TTS/test_checkpoint_proper3.py': [Errno 2] No such file or directory
(xtts-env3109) (base) root@cv5552379:/home/ubuntu/TTS# python inference_encodec_model3.py \
>   --checkpoint /home/ubuntu/TTS/outputs/emma_xtts_with_encodec/checkpoint_epoch_1_ts1765731041.pth \
>   --text "Hi I am Emma, how are u?" \
>   --speaker_audio /home/ubuntu/emma_dataset/wavs/newpodcastemmaclip_269.wav
INFO:__main__:✓ torch.cat globally patched
INFO:__main__:Device: cuda
INFO:__main__:Using checkpoint: /home/ubuntu/TTS/outputs/emma_xtts_with_encodec/checkpoint_epoch_1_ts1765731041.pth
INFO:__main__:Loading EnCodec-trained checkpoint
INFO:__main__:Loading config from checkpoint...
INFO:__main__:Reconstructing config from model_args dict...
INFO:__main__:  gpt_max_audio_tokens: 605
INFO:__main__:  gpt_layers: 30
INFO:__main__:  gpt_max_text_tokens: 402 (forced to 402 for embedding compatibility)
INFO:__main__:Creating model with config: gpt_max_audio_tokens=605
GPT2InferenceModel has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
  - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
  - If you are not the owner of the model architecture class, please contact the model code owner to update it.
INFO:__main__:✓ Model ready
INFO:__main__:Synthesizing: 'Hi I am Emma, how are u?'
INFO:__main__:Conditioning: gpt_cond=torch.Size([1, 130, 1024]), speaker_emb=torch.Size([1, 512])
INFO:__main__:[Length Control] Setting max_length=400 (was unlimited 605)
INFO:__main__:Running inference with HiFiGAN vocoder...
WARNING:__main__:max_length not supported in inference, trying without it
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
WARNING:__main__:[torch.cat] Batch mismatch at dim=1 - auto-fixing...
WARNING:__main__:  Batch sizes: [1, 1, 4] → target: 1
WARNING:__main__:    Tensor 2: torch.Size([4, 207, 1024]) → torch.Size([1, 207, 1024])
WARNING:__main__:  ✓ Fixed concatenation succeeded
INFO:__main__:[HiFiGAN] Input shape: torch.Size([1, 202, 1024])
INFO:__main__:[HiFiGAN] Permuting from [B, T, C] to [B, C, T]
INFO:__main__:[HiFiGAN] Permuted shape: torch.Size([1, 1024, 202])
INFO:__main__:[DEBUG] GPT latents stats - shape: torch.Size([1, 202, 1024]), min: -38.4508, max: 49.9259, mean: -0.0589, std: 1.9232
INFO:__main__:[DEBUG] GPT latents first 5 timesteps:
tensor([[-0.0266, -0.1142,  0.0922, -0.0196, -0.2209],
        [-1.7008, -0.3852,  0.0689,  1.0442, -0.0914],
        [ 0.0465, -0.0921, -0.4164,  0.3373,  0.0478],
        [-0.4211, -0.4579, -0.6048, -0.1384, -0.2553],
        [-1.1711, -0.9923, -1.0235, -0.2038,  0.2172]])
INFO:__main__:[DEBUG] Checking if latents are just noise or structured...
INFO:__main__:[DEBUG] Sample variance across first 10 timesteps: 1.4691 (should be >0.1 if structured, <0.01 if repetitive)
INFO:__main__:✓ HiFiGAN output shape: (206848,)
INFO:__main__:Applying EnCodec enhancement (bandwidth=6.0 kbps)...
INFO:__main__:[EnCodec] Applying quality enhancement to waveform shape (206848,)
/root/xtts-env3109/lib/python3.10/site-packages/torch/nn/utils/weight_norm.py:134: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
INFO:__main__:[EnCodec] Input shape before encode: torch.Size([1, 1, 206848])
INFO:__main__:[EnCodec] Output shape: torch.Size([1, 1, 207040])
INFO:__main__:[EnCodec] Final output shape: (207040,)
INFO:__main__:✓ EnCodec output shape: (207040,)
INFO:__main__:✓ Audio saved: generated_audio.wav
INFO:__main__:✓ SUCCESS: Generated 207040 samples at 24kHz
(xtts-env3109) (base) root@cv5552379:/home/ubuntu/TTS#