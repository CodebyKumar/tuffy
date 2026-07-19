# Voice Model Weights

This directory caches local weights and voice files for the speech-to-text (STT) and text-to-speech (TTS) engines.

## Directory Structure

- **`whisper/`**: Caches Whisper model files (e.g., `ggml-small.en.bin`) for Speech-to-Text transcription.
- **`piper/`**: Caches Piper voice files (e.g., `.onnx` and `.onnx.json` formats) for Text-to-Speech synthesis.

## Version Control

These subdirectories contain large binary model files and are gitignored:
- `src/voice/weights/whisper/`
- `src/voice/weights/piper/`

The weights directory structure and this `README.md` file are checked into version control for documentation.
