// worker.js
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

// On dual-GPU laptops, ask for the discrete GPU instead of the low-power one
if (env.backends?.onnx?.webgpu) {
    env.backends.onnx.webgpu.powerPreference = 'high-performance';
}

// Whisper pads every input to a 30s window, so each pass has a fixed cost
// regardless of audio length. Live drafts therefore use the cheapest model;
// the final pass on Stop uses the user-selected one.
const LIVE_MODEL = 'onnx-community/whisper-tiny';

let transcriber = null;      // user-selected model — final passes
let liveTranscriber = null;  // whisper-tiny — live draft passes
let busy = false;

async function tryLoad(model, progress_callback) {
    const isTurbo = model.includes('large-v3-turbo');

    let adapter = null;
    if (self.navigator?.gpu) {
        adapter = await self.navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null);
        if (adapter?.info) {
            console.log(`[worker] GPU adapter: vendor=${adapter.info.vendor} arch=${adapter.info.architecture} ${adapter.info.description || ''}`);
        }
    }

    const attempts = [];
    if (adapter) {
        attempts.push({
            device: 'webgpu',
            dtype: isTurbo ? 'q4f16' : { encoder_model: 'fp32', decoder_model_merged: 'q4' }
        });
    }
    attempts.push({ device: 'wasm', dtype: 'q8' });

    let lastError = null;
    for (const opts of attempts) {
        try {
            const pipe = await pipeline('automatic-speech-recognition', model, { ...opts, progress_callback });
            return { pipe, device: opts.device };
        } catch (error) {
            console.warn(`[worker] ${model} load failed on ${opts.device}:`, error);
            lastError = error;
        }
    }
    throw lastError;
}

self.addEventListener('message', async (event) => {
    const { type, data } = event.data;

    if (type === 'load') {
        const model = data?.model || 'onnx-community/whisper-base';
        transcriber = null;
        liveTranscriber = null;

        const progress_callback = (p) => {
            if (p.status === 'progress') {
                self.postMessage({ type: 'progress', data: p.progress });
            }
        };

        try {
            const main = await tryLoad(model, progress_callback);
            transcriber = main.pipe;

            if (model === LIVE_MODEL) {
                liveTranscriber = transcriber;
            } else {
                // Small (~40 MB) and quick; loaded before 'ready' so live works immediately
                const live = await tryLoad(LIVE_MODEL, progress_callback);
                liveTranscriber = live.pipe;
            }

            self.postMessage({ type: 'ready', data: { device: main.device } });
        } catch (error) {
            self.postMessage({ type: 'error', data: 'Model load failed: ' + (error?.message || error) });
        }
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

            const pipe = (interim && liveTranscriber) ? liveTranscriber : transcriber;

            const t0 = performance.now();
            const result = await pipe(audio, {
                language: language,       // forced 'en' or 'fr' — no auto-detect
                task: 'transcribe',
                chunk_length_s: 30,
                stride_length_s: 5
            });
            const durationMs = performance.now() - t0;
            console.log(`[worker] ${interim ? 'live(tiny)' : 'final'} pass: ${(audio.length / 16000).toFixed(1)}s audio in ${(durationMs / 1000).toFixed(1)}s`);

            self.postMessage({ type: 'result', data: { text: result.text, interim: !!interim, durationMs } });
        } catch (error) {
            self.postMessage({ type: 'error', data: error.message });
        } finally {
            busy = false;
        }
    }
});
