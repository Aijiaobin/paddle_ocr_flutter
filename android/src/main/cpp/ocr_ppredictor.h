#pragma once

#include "common.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ppredictor {

struct OCR_Config {
  int thread_num = 4;
};

struct OCRPredictResult {
  std::vector<int> word_index;
  std::vector<std::vector<int>> points;
  float score;
  float cls_score;
  int cls_label = -1;
};

class OCR_PPredictor {
public:
  OCR_PPredictor(const OCR_Config &config);
  ~OCR_PPredictor();

  int init_from_file(const std::string &det_model_path,
                     const std::string &rec_model_path,
                     const std::string &cls_model_path);

  std::vector<OCRPredictResult> infer_ocr(cv::Mat &origin, int max_size_len,
                                          int run_det, int run_cls,
                                          int run_rec);

private:
  // ORT inference helpers
  std::vector<float> run_det_model(const float *input_data, int batch, int ch,
                                   int h, int w);
  std::vector<float> run_rec_model(const float *input_data, int batch, int ch,
                                   int h, int w,
                                   std::vector<int64_t> &out_shape);
  std::vector<float> run_cls_model(const float *input_data, int batch, int ch,
                                   int h, int w, int64_t &out_size);

  void infer_det(cv::Mat &origin, int max_side_len,
                 std::vector<OCRPredictResult> &ocr_results);
  void infer_rec(const cv::Mat &origin, int run_cls,
                 OCRPredictResult &ocr_result);

  struct ClsPredictResult {
    float cls_score;
    int cls_label = -1;
    cv::Mat img;
  };
  ClsPredictResult infer_cls(const cv::Mat &origin, float thresh = 0.9);

  std::vector<std::vector<std::vector<int>>>
  calc_filtered_boxes(const float *pred, int pred_size, int output_height,
                      int output_width, const cv::Mat &origin);

  Ort::Env _env;
  std::unique_ptr<Ort::Session> _det_session;
  std::unique_ptr<Ort::Session> _rec_session;
  std::unique_ptr<Ort::Session> _cls_session;
  Ort::SessionOptions _session_opts;
  OCR_Config _config;
};

} // namespace ppredictor
