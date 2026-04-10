import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'paddle_ocr_flutter_platform_interface.dart';

class MethodChannelPaddleOcrFlutter extends PaddleOcrFlutterPlatform {
  @visibleForTesting
  final methodChannel = const MethodChannel('paddle_ocr_flutter');

  @override
  Future<Map<String, dynamic>> init({
    int threadNum = 4,
    String modelDir = 'models',
    String labelPath = 'labels/ppocr_keys_v1.txt',
  }) async {
    final result = await methodChannel.invokeMethod<Map>('init', {
      'threadNum': threadNum,
      'modelDir': modelDir,
      'labelPath': labelPath,
    });
    return Map<String, dynamic>.from(result ?? {});
  }

  @override
  Future<List<OcrResult>> recognize(String imagePath, {int maxSizeLen = 960}) async {
    final results = await methodChannel.invokeMethod<List>('recognize', {
      'imagePath': imagePath,
      'maxSizeLen': maxSizeLen,
    });
    if (results == null) return [];
    return results.map((r) => OcrResult.fromMap(Map<String, dynamic>.from(r))).toList();
  }

  @override
  Future<void> release() async {
    await methodChannel.invokeMethod('release');
  }
}
