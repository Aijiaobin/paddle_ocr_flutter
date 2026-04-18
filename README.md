# paddle_ocr_flutter

Flutter plugin for on-device Chinese/English OCR using PaddlePaddle PP-OCRv5 mobile models with ONNX Runtime inference.

## Features

- PP-OCRv5 mobile models — latest generation, optimized for Chinese text
- On-device inference via ONNX Runtime (no network required)
- Detection + Recognition + Optional Direction Classification
- 18,383 character dictionary (Chinese, English, symbols)
- Reading-order sorted results (top-to-bottom, left-to-right)
- Android arm64-v8a support

## Model Sizes

| Model | Size | Purpose |
|-------|------|---------|
| det_v5.onnx | 4.6 MB | Text detection |
| rec_v5.onnx | 16 MB | Text recognition |
| cls_v2.onnx | 571 KB | Direction classification (optional) |

Total: ~21 MB bundled in the plugin.

## Installation

```yaml
dependencies:
  paddle_ocr_flutter: ^0.0.3
```

## Usage

```dart
import 'package:paddle_ocr_flutter/paddle_ocr_flutter.dart';

final ocr = PaddleOcrFlutter();

// Initialize (loads models, ~1-2s on first call)
await ocr.init();

// Run OCR on an image file
final results = await ocr.recognize('/path/to/image.jpg');

for (final r in results) {
  print('${r.text} (${r.confidence.toStringAsFixed(2)})');
}

// Release native resources when done
await ocr.dispose();
```

## API

### `PaddleOcrFlutter`

| Method | Description |
|--------|-------------|
| `init({int threadNum, String modelDir, String labelPath})` | Load models and initialize the OCR engine |
| `recognize(String imagePath, {int maxSizeLen})` | Run OCR on an image file, returns `List<OcrResult>` |
| `dispose()` | Release native resources |

### `OcrResult`

| Field | Type | Description |
|-------|------|-------------|
| `text` | `String` | Recognized text |
| `confidence` | `double` | Recognition confidence (0-1) |
| `points` | `List<OcrPoint>` | Bounding polygon vertices |
| `clsLabel` | `int` | Direction class (-1 if cls disabled) |
| `clsScore` | `double` | Direction confidence |

## Requirements

- Android: minSdk 24, arm64-v8a
- iOS: not yet supported

## Credits

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — PP-OCRv5 models
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — Mobile inference engine
- [OpenCV](https://opencv.org/) — Image preprocessing

## License

MIT License. See [LICENSE](LICENSE) for details.
