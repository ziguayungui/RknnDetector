#pragma once

#include <jni.h>
#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include <chrono>
#include <android/log.h>
#include "rknn_api.h"

#define LOG_TAG "RKNN_NPU"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

struct DetectionResult {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

static rknn_context g_ctx = 0;
static bool is_model_loaded = false;
static rknn_input_output_num g_io_num;
static rknn_tensor_attr *g_input_attrs = nullptr;
static rknn_tensor_attr *g_output_attrs = nullptr;

int post_process_yolo(const rknn_output *outputs,
                      const rknn_tensor_attr *out_attrs,
                      int out_count,
                      float conf_threshold,
                      float nms_threshold,
                      std::vector<DetectionResult> &results);
