#include "ocr_ppredictor.h"
#include "common.h"
#include "ocr_cls_process.h"
#include "ocr_crnn_process.h"
#include "ocr_db_post_process.h"
#include "preprocess.h"
#include <numeric>

namespace ppredictor {

OCR_PPredictor::OCR_PPredictor(const OCR_Config &config)
    : _env(ORT_LOGGING_LEVEL_WARNING, "ppocr"), _config(config) {
  _session_opts.SetIntraOpNumThreads(config.thread_num);
  _session_opts.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
}

OCR_PPredictor::~OCR_PPredictor() = default;

int OCR_PPredictor::init_from_file(const std::string &det_model_path,
                                   const std::string &rec_model_path,
                                   const std::string &cls_model_path) {
  try {
    _det_session =
        std::make_unique<Ort::Session>(_env, det_model_path.c_str(), _session_opts);
    LOGI("det model loaded: %s", det_model_path.c_str());

    _rec_session =
        std::make_unique<Ort::Session>(_env, rec_model_path.c_str(), _session_opts);
    LOGI("rec model loaded: %s", rec_model_path.c_str());

    try {
      _cls_session =
          std::make_unique<Ort::Session>(_env, cls_model_path.c_str(), _session_opts);
      LOGI("cls model loaded: %s", cls_model_path.c_str());
    } catch (const Ort::Exception &cls_e) {
      LOGW("cls model load failed (non-fatal): %s", cls_e.what());
      _cls_session = nullptr;
    }
  } catch (const Ort::Exception &e) {
    LOGE("ORT init failed: %s", e.what());
    return -1;
  }
  return RETURN_OK;
}

// ---- ORT inference helpers ----

std::vector<float> OCR_PPredictor::run_det_model(const float *input_data,
                                                  int batch, int ch, int h,
                                                  int w) {
  Ort::AllocatorWithDefaultOptions alloc;
  auto in_name = _det_session->GetInputNameAllocated(0, alloc);
  auto out_name = _det_session->GetOutputNameAllocated(0, alloc);
  const char *input_names[] = {in_name.get()};
  const char *output_names[] = {out_name.get()};

  std::vector<int64_t> input_shape = {batch, ch, h, w};
  size_t input_count = batch * ch * h * w;
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float *>(input_data), input_count,
      input_shape.data(), input_shape.size());

  auto outputs = _det_session->Run(Ort::RunOptions{nullptr}, input_names,
                                   &input_tensor, 1, output_names, 1);
  float *out_data = outputs[0].GetTensorMutableData<float>();
  auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
  size_t out_count = out_info.GetElementCount();
  return std::vector<float>(out_data, out_data + out_count);
}

std::vector<float> OCR_PPredictor::run_rec_model(const float *input_data,
                                                  int batch, int ch, int h,
                                                  int w,
                                                  std::vector<int64_t> &out_shape) {
  Ort::AllocatorWithDefaultOptions alloc;
  auto in_name = _rec_session->GetInputNameAllocated(0, alloc);
  auto out_name = _rec_session->GetOutputNameAllocated(0, alloc);
  const char *input_names[] = {in_name.get()};
  const char *output_names[] = {out_name.get()};

  std::vector<int64_t> input_shape = {batch, ch, h, w};
  size_t input_count = batch * ch * h * w;
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float *>(input_data), input_count,
      input_shape.data(), input_shape.size());

  auto outputs = _rec_session->Run(Ort::RunOptions{nullptr}, input_names,
                                   &input_tensor, 1, output_names, 1);
  float *out_data = outputs[0].GetTensorMutableData<float>();
  auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
  out_shape = out_info.GetShape();
  size_t out_count = out_info.GetElementCount();
  return std::vector<float>(out_data, out_data + out_count);
}

std::vector<float> OCR_PPredictor::run_cls_model(const float *input_data,
                                                  int batch, int ch, int h,
                                                  int w, int64_t &out_size) {
  Ort::AllocatorWithDefaultOptions alloc;
  auto in_name = _cls_session->GetInputNameAllocated(0, alloc);
  auto out_name = _cls_session->GetOutputNameAllocated(0, alloc);
  const char *input_names[] = {in_name.get()};
  const char *output_names[] = {out_name.get()};

  std::vector<int64_t> input_shape = {batch, ch, h, w};
  size_t input_count = batch * ch * h * w;
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float *>(input_data), input_count,
      input_shape.data(), input_shape.size());

  auto outputs = _cls_session->Run(Ort::RunOptions{nullptr}, input_names,
                                   &input_tensor, 1, output_names, 1);
  float *out_data = outputs[0].GetTensorMutableData<float>();
  auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
  out_size = out_info.GetElementCount();
  return std::vector<float>(out_data, out_data + out_size);
}

// ---- OCR pipeline ----

std::vector<OCRPredictResult>
OCR_PPredictor::infer_ocr(cv::Mat &origin, int max_size_len, int run_det,
                           int run_cls, int run_rec) {
  LOGI("ocr infer start");
  std::vector<OCRPredictResult> ocr_results;

  if (run_det) {
    infer_det(origin, max_size_len, ocr_results);
    // Sort by reading order: top-to-bottom, left-to-right
    std::sort(ocr_results.begin(), ocr_results.end(),
              [](const OCRPredictResult &a, const OCRPredictResult &b) {
                if (a.points.empty() || b.points.empty()) return false;
                int ay = a.points[0][1], by = b.points[0][1];
                int ah = a.points[2][1] - a.points[0][1];
                int bh = b.points[2][1] - b.points[0][1];
                int thresh = std::min(ah, bh) / 2;
                if (std::abs(ay - by) < std::max(thresh, 10))
                  return a.points[0][0] < b.points[0][0]; // same row: sort by X
                return ay < by; // different row: sort by Y
              });
  }
  if (run_rec) {
    if (ocr_results.empty()) {
      ocr_results.emplace_back();
    }
    for (auto &r : ocr_results) {
      infer_rec(origin, run_cls, r);
    }
  } else if (run_cls && _cls_session) {
    auto cls_res = infer_cls(origin);
    OCRPredictResult res;
    res.cls_score = cls_res.cls_score;
    res.cls_label = cls_res.cls_label;
    ocr_results.push_back(res);
  }

  LOGI("ocr infer done, results: %zu", ocr_results.size());
  return ocr_results;
}

static cv::Mat DetResizeImg(const cv::Mat &img, int max_size_len,
                            std::vector<float> &ratio_hw) {
  int w = img.cols;
  int h = img.rows;
  float ratio = 1.f;
  int max_wh = std::max(w, h);
  if (max_wh > max_size_len) {
    ratio = static_cast<float>(max_size_len) / static_cast<float>(max_wh);
  }
  int resize_h = static_cast<int>(h * ratio);
  int resize_w = static_cast<int>(w * ratio);
  resize_h = std::max((resize_h / 32) * 32, 32);
  resize_w = std::max((resize_w / 32) * 32, 32);

  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
  ratio_hw.push_back(static_cast<float>(resize_h) / h);
  ratio_hw.push_back(static_cast<float>(resize_w) / w);
  return resize_img;
}

void OCR_PPredictor::infer_det(cv::Mat &origin, int max_size_len,
                               std::vector<OCRPredictResult> &ocr_results) {
  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};

  std::vector<float> ratio_hw;
  cv::Mat input_image = DetResizeImg(origin, max_size_len, ratio_hw);
  input_image.convertTo(input_image, CV_32FC3, 1 / 255.0f);
  const float *dimg = reinterpret_cast<const float *>(input_image.data);
  int input_size = input_image.rows * input_image.cols;

  std::vector<float> input_data(3 * input_size);
  neon_mean_scale(dimg, input_data.data(), input_size, mean, scale);

  LOGI("det input shape: 1x3x%dx%d", input_image.rows, input_image.cols);
  std::vector<float> output =
      run_det_model(input_data.data(), 1, 3, input_image.rows, input_image.cols);

  auto filtered_box = calc_filtered_boxes(output.data(), output.size(),
                                          input_image.rows, input_image.cols, origin);
  LOGI("det boxes: %zu", filtered_box.size());

  for (auto &box : filtered_box) {
    OCRPredictResult res;
    res.points = box;
    ocr_results.push_back(res);
  }
}

void OCR_PPredictor::infer_rec(const cv::Mat &origin_img, int run_cls,
                               OCRPredictResult &ocr_result) {
  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

  const auto &box = ocr_result.points;
  cv::Mat crop_img;
  if (!box.empty()) {
    crop_img = get_rotate_crop_image(origin_img, box);
  } else {
    crop_img = origin_img;
  }

  if (run_cls && _cls_session) {
    auto cls_res = infer_cls(crop_img);
    crop_img = cls_res.img;
    ocr_result.cls_score = cls_res.cls_score;
    ocr_result.cls_label = cls_res.cls_label;
  }

  float wh_ratio = static_cast<float>(crop_img.cols) / crop_img.rows;
  cv::Mat input_image = crnn_resize_img(crop_img, wh_ratio);
  input_image.convertTo(input_image, CV_32FC3, 1 / 255.0f);
  const float *dimg = reinterpret_cast<const float *>(input_image.data);
  int input_size = input_image.rows * input_image.cols;

  std::vector<float> input_data(3 * input_size);
  neon_mean_scale(dimg, input_data.data(), input_size, mean, scale);

  std::vector<int64_t> predict_shape;
  std::vector<float> predict_batch =
      run_rec_model(input_data.data(), 1, 3, input_image.rows, input_image.cols,
                    predict_shape);

  // CTC decode
  int last_index = 0;
  float score = 0.f;
  int count = 0;
  for (int n = 0; n < predict_shape[1]; n++) {
    int argmax_idx = static_cast<int>(
        argmax(&predict_batch[n * predict_shape[2]],
               &predict_batch[(n + 1) * predict_shape[2]]));
    float max_value = *std::max_element(
        &predict_batch[n * predict_shape[2]],
        &predict_batch[(n + 1) * predict_shape[2]]);
    if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
      score += max_value;
      count += 1;
      ocr_result.word_index.push_back(argmax_idx);
    }
    last_index = argmax_idx;
  }
  if (count > 0) score /= count;
  ocr_result.score = score;
  LOGI("rec words: %d, score: %.3f", count, score);
}

OCR_PPredictor::ClsPredictResult
OCR_PPredictor::infer_cls(const cv::Mat &img, float thresh) {
  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

  cv::Mat input_image = cls_resize_img(img);
  input_image.convertTo(input_image, CV_32FC3, 1 / 255.0f);
  const float *dimg = reinterpret_cast<const float *>(input_image.data);
  int input_size = input_image.rows * input_image.cols;

  std::vector<float> input_data(3 * input_size);
  neon_mean_scale(dimg, input_data.data(), input_size, mean, scale);

  int64_t out_size = 0;
  std::vector<float> scores =
      run_cls_model(input_data.data(), 1, 3, input_image.rows, input_image.cols,
                    out_size);

  float best_score = 0;
  int label = 0;
  for (int64_t i = 0; i < out_size; i++) {
    if (scores[i] > best_score) {
      best_score = scores[i];
      label = i;
    }
  }

  cv::Mat srcimg;
  img.copyTo(srcimg);
  if (label % 2 == 1 && best_score > thresh) {
    cv::rotate(srcimg, srcimg, 1);
  }

  ClsPredictResult res;
  res.cls_label = label;
  res.cls_score = best_score;
  res.img = srcimg;
  LOGI("cls label: %d, score: %.3f", label, best_score);
  return res;
}

std::vector<std::vector<std::vector<int>>>
OCR_PPredictor::calc_filtered_boxes(const float *pred, int pred_size,
                                    int output_height, int output_width,
                                    const cv::Mat &origin) {
  const double threshold = 0.3;
  const double maxvalue = 1;

  cv::Mat pred_map = cv::Mat::zeros(output_height, output_width, CV_32F);
  memcpy(pred_map.data, pred, output_height * output_width * sizeof(float));
  cv::Mat cbuf_map;
  pred_map.convertTo(cbuf_map, CV_8UC1);

  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);

  auto boxes = boxes_from_bitmap(pred_map, bit_map);
  float ratio_h = output_height * 1.0f / origin.rows;
  float ratio_w = output_width * 1.0f / origin.cols;
  return filter_tag_det_res(boxes, ratio_h, ratio_w, origin);
}

} // namespace ppredictor
