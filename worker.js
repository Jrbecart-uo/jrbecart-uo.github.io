// worker.js
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

let transcriber = null;
let busy = false;

self.addEventListener('message', async (event) => {
    const { type, data } = event.data;

    if (type === 'load') {
        const model = data?.model || 'onnx-community/whisper-base';
        const isTurbo = model.includes('large-v3-turbo');
        transcriber = null;

        const progress_callback = (p) => {
            if (p.status === 'progress') {
                self.postMessage({ type: 'progress', data: p.progress });
            }
        };

        // Prefer WebGPU (much faster, enables live transcription); fall back to WASM
        const hasWebGPU = !!(self.navigator?.gpu && await self.navigator.gpu.requestAdapter().catch(() => null));
        const attempts = [];
        if (hasWebGPU) {
            attempts.push({
                device: 'webgpu',
                dtype: isTurbo ? 'q4f16' : { encoder_model: 'fp32', decoder_model_merged: 'q4' }
            });
        }
        attempts.push({ device: 'wasm', dtype: 'q8' });

        let lastError = null;
        for (const opts of attempts) {
            try {
                transcriber = await pipeline('automatic-speech-recognition', model, { ...opts, progress_callback });
                self.postMessage({ type: 'ready', data: { device: opts.device } });
                return;
            } catch (error) {
                console.warn(`[worker] load failed on ${opts.device}:`, error);
                lastError = error;
            }
        }
        self.postMessage({ type: 'error', data: 'Model load failed: ' + (lastError?.message || lastError) });
    }

    if (type === 'transcribe') {
        if (!transcriber) {
            self.postMessage({ type: 'error', data: 'Model not initialized.' });
            return;
        }
        // One inference at a time; drop overlapping live passes
        if (busy) {
            self.postMessage({ type: 'result', data: { skipped: true, interim: !!data.interim } });
            return;
        }
        busy = true;

        try {
            const { audio, language, interim } = data;

            if (!interim) {
                let peak = 0;
                for (let i = 0; i < audio.length; i++) {
                    const v = Math.abs(audio[i]);
                    if (v > peak) peak = v;
                }
                console.log(`[worker] final transcribe: ${(audio.length / 16000).toFixed(1)}s, peak ${peak.toFixed(4)}, lang ${language}`);
                if (peak < 0.001) {
                    self.postMessage({ type: 'error', data: 'Audio is silent (peak amplitude ~0). Check mic input device.' });
                    return;
                }
            }

            const result = await transcriber(audio, {
                language: language,       // forced 'en' or 'fr' — no auto-detect
                task: 'transcribe',
                chunk_length_s: 30,
                stride_length_s: 5
            });

            self.postMessage({ type: 'result', data: { text: result.text, interim: !!interim } });
        } catch (error) {
            self.postMessage({ type: 'error', data: error.message });
        } finally {
            busy = false;
        }
    }
});
