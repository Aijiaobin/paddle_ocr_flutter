import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'paddle_ocr_flutter_method_channel.dart';

class OcrResult {
  final String text;
  final double confidence;
  final List<OcrPoint> points;
  final int clsLabel;
  final double clsScore;

  OcrResult({
    required this.text,
    required this.confidence,
    required this.points,
    this.clsLabel = -1,
    this.clsScore = 0.0,
  });

  factory OcrResult.fromMap(Map<String, dynamic> map) {
    final pointsList = (map['points'] as List?)?.map((p) {
      final pm = Map<String, dynamic>.from(p);
      return OcrPoint(x: pm['x'] as int, y: pm['y'] as int);
    }).toList() ?? [];

    return OcrResult(
      text: map['text'] as String? ?? '',
      confidence: (map['confidence'] as num?)?.toDouble() ?? 0.0,
      points: pointsList,
      clsLabel: map['clsLabel'] as int? ?? -1,
      clsScore: (map['clsScore'] as num?)?.toDouble() ?? 0.0,
    );
  }

  @override
  String toString() => 'OcrResult(text: $text, confidence: ${confidence.toStringAsFixed(3)})';
}

class OcrPoint {
  final int x;
  final int y;
  OcrPoint({required this.x, required this.y});

  @override
  String toString() => '($x, $y)';
}

abstract class PaddleOcrFlutterPlatform extends PlatformInterface {
  PaddleOcrFlutterPlatform() : super(token: _token);

  static final Object _token = Object();
  static PaddleOcrFlutterPlatform _instance = MethodChannelPaddleOcrFlutter();

  static PaddleOcrFlutterPlatform get instance => _instance;

  static set instance(PaddleOcrFlutterPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<Map<String, dynamic>> init({
    int threadNum = 4,
    String modelDir = 'models',
    String labelPath = 'labels/ppocr_keys_v1.txt',
  }) {
    throw UnimplementedError('init() has not been implemented.');
  }

  Future<List<OcrResult>> recognize(String imagePath, {int maxSizeLen = 960}) {
    throw UnimplementedError('recognize() has not been implemented.');
  }

  Future<void> release() {
    throw UnimplementedError('release() has not been implemented.');
  }
}
