import 'paddle_ocr_flutter_platform_interface.dart';

export 'paddle_ocr_flutter_platform_interface.dart' show OcrResult, OcrPoint;

class PaddleOcrFlutter {
  bool _initialized = false;

  /// Initialize PaddleOCR engine with PP-OCRv5 mobile models.
  ///
  /// [threadNum] CPU threads for inference (default 4).
  /// [modelDir] Asset directory containing det_db.nb, rec_crnn.nb, cls.nb.
  /// [labelPath] Asset path to the dictionary file.
  Future<void> init({
    int threadNum = 4,
    String modelDir = 'models',
    String labelPath = 'labels/ppocr_keys_v1.txt',
  }) async {
    await PaddleOcrFlutterPlatform.instance.init(
      threadNum: threadNum,
      modelDir: modelDir,
      labelPath: labelPath,
    );
    _initialized = true;
  }

  /// Run OCR on an image file. Returns list of detected text regions.
  ///
  /// [imagePath] Absolute path to the image file.
  /// [maxSizeLen] Max side length for detection resize (default 960).
  Future<List<OcrResult>> recognize(String imagePath, {int maxSizeLen = 960}) {
    if (!_initialized) {
      throw StateError('PaddleOcrFlutter not initialized. Call init() first.');
    }
    return PaddleOcrFlutterPlatform.instance.recognize(imagePath, maxSizeLen: maxSizeLen);
  }

  /// Release native resources.
  Future<void> dispose() async {
    if (_initialized) {
      await PaddleOcrFlutterPlatform.instance.release();
      _initialized = false;
    }
  }
}
