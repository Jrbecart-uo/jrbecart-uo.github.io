// worker.js
import { pipeline, env } from 'https://jsdelivr.net';

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
            // data is the Float32Array passed from the main thread
            const result = await transcriber(data, {
                language: null, // Auto-detect English or French
                task: 'transcribe',
                return_timestamps: true
            });

            self.postMessage({ type: 'result', data: result });
        } catch (error) {
            self.postMessage({ type: 'error', data: error.message });
        }
    }
});
