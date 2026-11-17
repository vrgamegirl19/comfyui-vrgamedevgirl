"""Split a song into vocals and a vocal-free copy with Demucs (htdemucs_ft), saved as 16-bit WAVs at the song's rate.

usage: split_stems.py SONG.wav OUT_VOCALS.wav OUT_NO_VOCALS.wav [--cpu]

Run it with ComfyUI's python_embeded (it has demucs, torch and soundfile). The demucs command line cannot save WAVs
there because torchaudio's FFmpeg backend fails to load, so this script calls the model directly. About 15 s on a 5090.
Only run it when ComfyUI is idle: it uses the GPU.
"""
import sys

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    src, out_vocals, out_inst = sys.argv[1:4]
    device = "cpu" if "--cpu" in sys.argv or not torch.cuda.is_available() else "cuda"
    audio, sr = sf.read(src, always_2d=True, dtype="float32")
    model = get_model("htdemucs_ft")
    model.eval()
    model.to(device)
    wav = torch.from_numpy(audio.T.copy())
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    if sr != model.samplerate:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    with torch.no_grad():
        sources = apply_model(model, ((wav - mean) / (std + 1e-8))[None].to(device), device=device,
                              shifts=1, split=True, overlap=0.25, progress=False)[0]
    sources = sources * std + mean
    names = list(model.sources)
    vocals = sources[names.index("vocals")].cpu()
    inst = sum(sources[i] for i, n in enumerate(names) if n != "vocals").cpu()
    if sr != model.samplerate:
        import torchaudio
        vocals = torchaudio.functional.resample(vocals, model.samplerate, sr)
        inst = torchaudio.functional.resample(inst, model.samplerate, sr)
    n = audio.shape[0]
    sf.write(out_vocals, np.clip(vocals.numpy().T[:n], -1, 1), sr, subtype="PCM_16")
    sf.write(out_inst, np.clip(inst.numpy().T[:n], -1, 1), sr, subtype="PCM_16")
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"ok: {n / sr:.2f} s at {sr} Hz -> {out_vocals}, {out_inst}")


if __name__ == "__main__":
    main()
