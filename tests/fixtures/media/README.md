# Media fixtures

These short audio fixtures are generated locally with the Debian-packaged
eSpeak NG 1.52.0 text-to-speech engine and then converted to the fixed WAV
format with FFmpeg. The generated speech is original synthetic output; no
third-party recording is redistributed.

| File | Spoken text / content | Language | Approx. duration | Tests |
| --- | --- | --- | ---: | --- |
| `english-short.wav` | “This is a deterministic Captioner test. The short English sample validates subtitle generation.” | English | 6.6 seconds | default FFmpeg integration and slow ASR |
| `chinese-short.wav` | “这是一个用于 Captioner 的确定性测试。请生成字幕。” | Simplified Chinese | 8.5 seconds | explicit local/slow ASR validation |
| `silence.wav` | Generated digital silence | n/a | 5 seconds | unit/integration edge cases |
| `no-audio.mp4` | Black video stream with no audio stream | n/a | 1 second | no-audio FFprobe failure |

The default quality gate uses the files only with Fake ASR. The model-loading
test is marked `slow` and is not part of normal PR CI.
