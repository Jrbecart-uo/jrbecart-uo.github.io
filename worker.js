// worker.js
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Configure environment for strict caching
env.allowLocalModels = false;
env.useBrowserCache = true;

let transcriber = null;

// Listen for messages from the main UI thread
self.addEventListener('message', async (event) => {
    const { type, data } = event.data;

    if (type === 'load') {
        try {
            transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny', {
                progress_callback: (progressData) => {
                    if (progressData.status === 'progress') {
                        // Send download progress back to the main thread
                        self.postMessage({ type: 'progress', data: progressData.progress });
                    }
                }
            });
            self.postMessage({ type: 'ready' });
        } catch (error) {
            self.postMessage({ type: 'error', data: error.message });
        }
    }

    if (type === 'transcribe') {
        if (!transcriber) {
            self.postMessage({ type: 'error', data: 'Model not initialized.' });
            return;
        }

        try {
            // Diagnostics: check the audio actually contains signal
            let peak = 0;
            for (let i = 0; i < data.length; i++) {
                const v = Math.abs(data[i]);
                if (v > peak) peak = v;
            }
            console.log(`[worker] transcribe: ${data.length} samples (${(data.length / 16000).toFixed(1)}s), peak amplitude ${peak.toFixed(4)}`);
            if (peak < 0.001) {
                self.postMessage({ type: 'error', data: 'Audio is silent (peak amplitude ~0). Check mic input device.' });
                return;
            }

            const result = await transcriber(data, {
                language: null, // Auto-detect English or French
                task: 'transcribe',
                chunk_length_s: 30,
                stride_length_s: 5,
                return_timestamps: true
            });

            console.log('[worker] result:', result);
            self.postMessage({ type: 'result', data: result });
        } catch (error) {
            self.postMessage({ type: 'error', data: error.message });
        }
    }
});
