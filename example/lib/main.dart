import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:paddle_ocr_flutter/paddle_ocr_flutter.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PP-OCRv5 Demo',
      theme: ThemeData(colorSchemeSeed: Colors.blue, useMaterial3: true),
      home: const OcrDemoPage(),
    );
  }
}

class OcrDemoPage extends StatefulWidget {
  const OcrDemoPage({super.key});

  @override
  State<OcrDemoPage> createState() => _OcrDemoPageState();
}

class _OcrDemoPageState extends State<OcrDemoPage> {
  final _ocr = PaddleOcrFlutter();
  final _picker = ImagePicker();
  String _status = 'Not initialized';
  List<OcrResult> _results = [];
  bool _loading = false;
  bool _initialized = false;
  File? _imageFile;

  @override
  void initState() {
    super.initState();
    _initOcr();
  }

  @override
  void dispose() {
    _ocr.dispose();
    super.dispose();
  }

  Future<void> _initOcr() async {
    setState(() { _loading = true; _status = 'Loading models...'; });
    try {
      await _ocr.init();
      setState(() { _status = 'Ready'; _initialized = true; });
    } catch (e) {
      setState(() { _status = 'Init failed: $e'; });
    } finally {
      setState(() { _loading = false; });
    }
  }

  Future<void> _pickAndRecognize(ImageSource source) async {
    if (!_initialized) {
      setState(() { _status = 'Not initialized yet'; });
      return;
    }
    try {
      final picked = await _picker.pickImage(
        source: source,
        maxWidth: 2048,
        imageQuality: 90,
      );
      if (picked == null) return;

      setState(() {
        _loading = true;
        _status = 'Recognizing...';
        _imageFile = File(picked.path);
        _results = [];
      });

      final results = await _ocr.recognize(picked.path);
      setState(() {
        _results = results;
        _status = '${results.length} text regions found';
      });
    } catch (e) {
      setState(() { _status = 'OCR failed: $e'; });
    } finally {
      setState(() { _loading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('PP-OCRv5 Demo')),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(12),
            child: Row(
              children: [
                Expanded(
                  child: Text(_status, style: const TextStyle(fontSize: 15)),
                ),
                if (_loading)
                  const SizedBox(
                    width: 20, height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _loading ? null : () => _pickAndRecognize(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.tonalIcon(
                    onPressed: _loading ? null : () => _pickAndRecognize(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                  ),
                ),
              ],
            ),
          ),
          if (_imageFile != null)
            Container(
              margin: const EdgeInsets.all(12),
              height: 200,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.grey.shade300),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.file(_imageFile!, fit: BoxFit.contain, width: double.infinity),
              ),
            ),
          Expanded(
            child: _results.isEmpty
                ? const Center(child: Text('No results yet', style: TextStyle(color: Colors.grey)))
                : ListView.separated(
                    padding: const EdgeInsets.all(12),
                    itemCount: _results.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 4),
                    itemBuilder: (ctx, i) {
                      final r = _results[i];
                      return Card(
                        child: ListTile(
                          dense: true,
                          title: Text(r.text, style: const TextStyle(fontSize: 15)),
                          trailing: Text(
                            r.confidence.toStringAsFixed(2),
                            style: TextStyle(color: Colors.grey.shade600, fontSize: 13),
                          ),
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}
