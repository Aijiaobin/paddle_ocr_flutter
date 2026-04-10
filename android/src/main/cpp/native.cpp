#include "native.h"
#include "ocr_ppredictor.h"
#include "preprocess.h"
#include <string>

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_paddle_1ocr_1flutter_PaddleOcrFlutterPlugin_nativeInit(
    JNIEnv *env, jobject thiz, jstring j_det_model_path,
    jstring j_rec_model_path, jstring j_cls_model_path,
    jint j_thread_num) {
  std::string det_model_path = jstring_to_cpp_string(env, j_det_model_path);
  std::string rec_model_path = jstring_to_cpp_string(env, j_rec_model_path);
  std::string cls_model_path = jstring_to_cpp_string(env, j_cls_model_path);

  ppredictor::OCR_Config conf;
  conf.thread_num = j_thread_num;

  auto *predictor = new ppredictor::OCR_PPredictor(conf);
  int ret = predictor->init_from_file(det_model_path, rec_model_path, cls_model_path);
  if (ret != 0) {
    LOGE("OCR init failed: %d", ret);
    delete predictor;
    return 0;
  }
  return reinterpret_cast<jlong>(predictor);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_paddle_1ocr_1flutter_PaddleOcrFlutterPlugin_nativeForward(
    JNIEnv *env, jobject thiz, jlong java_pointer, jobject original_image,
    jint j_max_size_len, jint j_run_det, jint j_run_cls, jint j_run_rec) {
  LOGI("native forward start");
  if (java_pointer == 0) {
    LOGE("native pointer is NULL");
    return cpp_array_to_jfloatarray(env, nullptr, 0);
  }

  cv::Mat origin = bitmap_to_cv_mat(env, original_image);
  if (origin.empty()) {
    LOGE("bitmap to cv::Mat failed");
    return cpp_array_to_jfloatarray(env, nullptr, 0);
  }

  auto *predictor = reinterpret_cast<ppredictor::OCR_PPredictor *>(java_pointer);
  std::vector<ppredictor::OCRPredictResult> results =
      predictor->infer_ocr(origin, j_max_size_len, j_run_det, j_run_cls, j_run_rec);
  LOGI("infer done, results: %zu", results.size());

  // Serialize results to float array for JNI transport
  std::vector<float> float_arr;
  for (const auto &r : results) {
    float_arr.push_back(r.points.size());
    float_arr.push_back(r.word_index.size());
    float_arr.push_back(r.score);
    for (const auto &point : r.points) {
      float_arr.push_back(point.at(0));
      float_arr.push_back(point.at(1));
    }
    for (int index : r.word_index) {
      float_arr.push_back(index);
    }
    float_arr.push_back(r.cls_label);
    float_arr.push_back(r.cls_score);
  }
  return cpp_array_to_jfloatarray(env, float_arr.data(), float_arr.size());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_paddle_1ocr_1flutter_PaddleOcrFlutterPlugin_nativeRelease(
    JNIEnv *env, jobject thiz, jlong java_pointer) {
  if (java_pointer == 0) return;
  delete reinterpret_cast<ppredictor::OCR_PPredictor *>(java_pointer);
}
