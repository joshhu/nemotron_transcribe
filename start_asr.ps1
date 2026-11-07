# Starts the Nemotron 3.5 Transcribe server (NeMo-Speech.cpp, native Windows ARM64 + CUDA) on 127.0.0.1:8090.
# Settings sized for short push-to-talk clips, leaving GPU memory for the Hermes chat model.
$root = $PSScriptRoot
& "$root\bin\nemo-speech\bin\nemo-speech.exe" serve `
    --asr-model "$root\models\NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-NVFP4.gguf" `
    --host 127.0.0.1 --port 8090 `
    --asr.salm.max_sequences 2 --asr.salm.n_ctx_per_sequence 8192 --asr.salm.max_context 8192
