# Jarvis
Jarvis is a low-latency, multi-modal AI voice assistant that enables natural conversations through an optimized streaming pipeline combining speech recognition, RAG-powered LLM inference, and text-to-speech synthesis. 

## Technical Highlights:
- Speech-to-Text: Integrated Whisper (whisper.cpp) for high accuracy real-time transcription.
- Language Model: Deployed Qwen3-VL-4B and Llama 3.2 3B with streaming token generation and custom ChatML prompt engineering for natural, real-time responses.
- Dual TTS Engine: Implemented support for both Piper TTS and Kokoro TTS using sherpa-onnx with user-selectable voices at runtime.
- Concurrent Audio Pipeline: Designed a parallel generation/playback architecture using persistent audio devices and pre-buffering to achieve seamless sentence-to-sentence transitions with near-zero latency.
- Smart Text Chunking: Implemented phrase-boundary detection for faster initial response times while maintaining natural speech flow.
- RAG: Designed retreival-augmented generation capabilities with hnswlib for context-aware responses with domain-specific knowledge.

#### Stack: C++ sherpa-onnx, llama.cpp, miniaudio, whisper.cpp, hnswlib
