# Whisper GGML Plus Example App

This example demonstrates a complete Flutter app flow built on top of `whisper_ggml_plus`.

It covers:

- loading a GGML Whisper model into app storage
- recording 16 kHz WAV audio with `record`
- transcribing a bundled `jfk.wav` sample file
- running the demo on Android, iOS, macOS, and Windows
- optionally enabling CoreML acceleration on iOS and macOS

The example is intentionally more detailed than the main package README. Use this document when you want to understand the full demo flow, model setup, and platform-specific caveats.

## Quick Start

### 1. Install dependencies

```bash
flutter pub get
```

### 2. Prepare a model

This example uses `WhisperModel.base` by default.

You can get a model from:
- [Official Whisper GGML Models](https://huggingface.co/ggerganov/whisper.cpp/tree/main)
- [Quantized Whisper.cpp models](https://huggingface.co/ggerganov/whisper.cpp)

Recommended starting points:

- `ggml-tiny.bin` — fastest for quick testing
- `ggml-base.bin` or `ggml-base-q5_0.bin` — good default balance
- `ggml-small-q5_0.bin` — better accuracy with higher memory use
- `ggml-large-v3-turbo-q3_k.bin` — best quality, heavier runtime and memory cost

### 3. Understand how model loading works in this example

At startup, `example/lib/main.dart` tries to:

1. load `assets/ggml-${model.modelName}.bin`
2. copy that `.bin` file into app storage
3. fall back to `WhisperController.downloadModel(model)` if the asset is unavailable

The example also includes `assets/jfk.wav`, so you can validate transcription without recording your own audio first.

### 4. Run the example

```bash
# Android
flutter run -d android

# iOS
flutter run -d ios

# macOS
flutter run -d macos

# Windows
flutter run -d windows
```

## What the Example App Actually Does

The current demo app has two main flows:

### Record and transcribe

When you press the microphone button, the app records audio to a temporary WAV file using:

```dart
await audioRecorder.start(
  const RecordConfig(sampleRate: 16000, encoder: AudioEncoder.wav),
  path: '${appDirectory.path}/test.wav',
);
```

When you stop recording, the app transcribes that recorded file with `WhisperController.transcribe(...)`.

### Transcribe the bundled sample file

When you press the folder button, the app copies `assets/jfk.wav` into a temporary file and transcribes it.

This is useful for validating that your model, native setup, and transcription flow are working even before microphone testing.

### Important limitation

This example demonstrates **file-based batch transcription**.

- It does **not** stream partial tokens while recording.
- It records a file first, then transcribes the completed file.
- The UI shows results after inference completes.

## Package and Example Dependencies

The example app currently depends on:

- `whisper_ggml_plus`
- `record`
- `path_provider`

That means:

- audio recording in this example is handled by `record`
- model and temporary file paths are resolved with `path_provider`
- transcription is handled by the package itself

## Basic Usage Example

This is the core package-level flow the example is using:

```dart
final controller = WhisperController();

final result = await controller.transcribe(
  model: WhisperModel.base,
  audioPath: audioPath,
  lang: 'auto',
);

if (result != null) {
  print(result.transcription.text);
}
```

## Loading or Downloading a Model

If you want to mirror the example app behavior in your own project, this is the important part:

```dart
final controller = WhisperController();

await controller.downloadModel(WhisperModel.base);
final path = await controller.getPath(WhisperModel.base);
print(path);
```

Notes:

- `downloadModel()` downloads the official GGML model file for the selected `WhisperModel`.
- `getPath()` returns the path where the model should live in app-writable storage.
- If you bundle a `.bin` model as a Flutter asset, you still need to copy it to a writable path before inference.

## Using Different Models

The example defaults to:

```dart
final model = WhisperModel.base;
```

You can switch that to another built-in enum such as:

- `WhisperModel.tiny`
- `WhisperModel.base`
- `WhisperModel.small`
- `WhisperModel.medium`
- `WhisperModel.large`
- `WhisperModel.largeV3Turbo`

When you change the model, make sure the corresponding GGML `.bin` file is available.

## Audio Format Notes

### What this example uses

This demo records WAV directly, so it avoids an additional conversion step during the microphone flow.

### If your real app uses MP3, M4A, MP4, or other formats

The core package does not bundle FFmpeg. For automatic conversion in a real application, follow the main package README and register `whisper_ggml_plus_ffmpeg`.

This example does **not** demonstrate that FFmpeg companion flow directly.

## Optional CoreML Setup (iOS/macOS Only)

For 3x+ faster transcription on Apple Silicon devices, you can optionally place a CoreML encoder next to the GGML `.bin` model.

This section is only relevant for iOS and macOS. It is not required for Android or Windows.

### What is `.mlmodelc`?

`.mlmodelc` is a **compiled CoreML model directory**, not a single file.

Typical contents:

- `model.mil`
- `coremldata.bin`
- `metadata.json`

Important rules:

1. `.mlmodelc` must remain a directory.
2. It must live in the same directory as the GGML `.bin` model.
3. The base names must match.
4. You cannot bundle `.mlmodelc` correctly with normal Flutter asset packaging.

### 1. Generate a CoreML encoder

```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp

python3.11 -m venv venv
source venv/bin/activate
pip install torch==2.5.0 "numpy<2.0" coremltools==8.1 openai-whisper ane_transformers

# Generate for base
./models/generate-coreml-model.sh base

# Or generate for Large-v3-Turbo
./models/generate-coreml-model.sh large-v3-turbo
```

Expected output:

```text
models/ggml-base-encoder.mlmodelc/
models/ggml-large-v3-turbo-encoder.mlmodelc/
```

### 2. Deploy the CoreML directory

#### Option A: download at runtime

```dart
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

Future<void> setupCoreMLModel() async {
  final appSupport = await getApplicationSupportDirectory();
  final mlmodelcDir =
      Directory('${appSupport.path}/models/ggml-base-encoder.mlmodelc');

  if (!await mlmodelcDir.exists()) {
    await mlmodelcDir.create(recursive: true);

    final files = ['model.mil', 'coremldata.bin', 'metadata.json'];
    for (final file in files) {
      final response = await http.get(
        Uri.parse('https://your-cdn.com/ggml-base-encoder.mlmodelc/$file'),
      );
      await File('${mlmodelcDir.path}/$file').writeAsBytes(response.bodyBytes);
    }
  }
}
```

#### Option B: add it in Xcode

1. Open the iOS or macOS runner project in Xcode.
2. Drag the `.mlmodelc` folder into the project navigator.
3. Choose **Create folder references**, not **Create groups**.
4. Add it to the correct target.

### 3. File structure example

#### Without CoreML

```text
/app/support/models/
└── ggml-base-q5_0.bin
```

#### With CoreML

```text
/app/support/models/
├── ggml-base-q5_0.bin
└── ggml-base-encoder.mlmodelc/
    ├── model.mil
    ├── coremldata.bin
    └── metadata.json
```

### 4. How detection works

When Whisper loads a GGML model, whisper.cpp automatically looks for the matching `-encoder.mlmodelc` directory next to that `.bin` file.

Example:

- `ggml-base-q5_0.bin` → `ggml-base-encoder.mlmodelc/`
- `ggml-large-v3-turbo-q3_k.bin` → `ggml-large-v3-turbo-encoder.mlmodelc/`

If the CoreML directory is present and valid, Apple devices can use it automatically.

## Performance Tips

### Model selection

| Model | Speed | Accuracy | Best for |
| --- | --- | --- | --- |
| `tiny` | Fastest | Lowest | smoke tests and quick validation |
| `base` | Fast | Good | default example usage |
| `small` | Medium | Better | better quality with moderate cost |
| `large-v3-turbo` | Slowest | Best | Apple Silicon or high-memory devices |

### Practical guidance

- Test in `--release` mode for realistic native performance.
- Prefer `base` or quantized models for mobile testing.
- Use `large-v3-turbo` only when you can afford the memory and runtime cost.
- CoreML is optional, but can significantly improve Apple-device performance.

## Troubleshooting

### The model file is not found

Possible causes:

- the `.bin` model was never copied into app storage
- the asset name does not match `ggml-${model.modelName}.bin`
- the fallback download has not completed successfully

Check that:

- `await controller.getPath(model)` points to a real file
- your selected `WhisperModel` matches the model you prepared
- the device has permission to write to the app directory

### Microphone recording works, but transcription fails

Check that:

- the app actually produced `test.wav`
- the model file exists in app storage
- the device has enough memory for the selected model
- you are not expecting streaming output from a batch API

### `jfk.wav` works but the microphone flow does not

That usually means the package itself is working, but the issue is in recording or file handling.

Check:

- microphone permission
- whether `record` can create the WAV file on that platform
- whether the recorded file path exists before transcription

### CoreML is not detected

Check that:

- `.mlmodelc` is a directory, not a file
- it is next to the `.bin` model
- the base names match
- you used runtime storage or Xcode folder references instead of Flutter assets

### Transcription is slow

That is often expected with larger models.

Try:

- `WhisperModel.base`
- a quantized model such as `q5_0` or `q3_k`
- `--release` mode
- optional CoreML on Apple devices

## Resources

- [Main package README](../README.md)
- [Whisper.cpp Repository](https://github.com/ggerganov/whisper.cpp)
- [Pre-trained Models](https://huggingface.co/ggerganov/whisper.cpp/tree/main)
- [GGML Quantization Guide](https://github.com/ggerganov/ggml#quantization)

## License

MIT License - See [LICENSE](../LICENSE) file for details.
