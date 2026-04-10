import 'package:flutter_test/flutter_test.dart';
import 'package:paddle_ocr_flutter/paddle_ocr_flutter.dart';
import 'package:paddle_ocr_flutter/paddle_ocr_flutter_platform_interface.dart';
import 'package:paddle_ocr_flutter/paddle_ocr_flutter_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockPaddleOcrFlutterPlatform
    with MockPlatformInterfaceMixin
    implements PaddleOcrFlutterPlatform {

  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final PaddleOcrFlutterPlatform initialPlatform = PaddleOcrFlutterPlatform.instance;

  test('$MethodChannelPaddleOcrFlutter is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelPaddleOcrFlutter>());
  });

  test('getPlatformVersion', () async {
    PaddleOcrFlutter paddleOcrFlutterPlugin = PaddleOcrFlutter();
    MockPaddleOcrFlutterPlatform fakePlatform = MockPaddleOcrFlutterPlatform();
    PaddleOcrFlutterPlatform.instance = fakePlatform;

    expect(await paddleOcrFlutterPlugin.getPlatformVersion(), '42');
  });
}
