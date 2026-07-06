#include "native-lib.h"
#include <algorithm>
#include <cmath>
#include <set>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM     2
#define NMS_THRESH        0.45
#define BOX_THRESH        0.25
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

const char* labels[OBJ_CLASS_NUM] = {"head", "mask"};

const int anchor0[6] = {10, 13, 16, 30, 33, 23};
const int anchor1[6] = {30, 61, 62, 45, 59, 119};
const int anchor2[6] = {116, 90, 156, 198, 373, 326};

inline static float clamp_val(float val, float min, float max) { 
    return val > min ? (val < max ? val : max) : min; 
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, 
                              float xmin1, float ymin1, float xmax1, float ymax1) {
    float w = std::max(0.f, std::min(xmax0, xmax1) - std::max(xmin0, xmin1) + 1.0f);
    float h = std::max(0.f, std::min(ymax0, ymax1) - std::max(ymin0, ymin1) + 1.0f);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0f) * (ymax0 - ymin0 + 1.0f) + 
              (xmax1 - xmin1 + 1.0f) * (ymax1 - ymin1 + 1.0f) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, 
               std::vector<int>& order, int filterId, float threshold) {
    for (int i = 0; i < validCount; ++i) {
        if (order[i] == -1 || classIds[i] != filterId) {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1 || classIds[i] != filterId) {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices) {
    float key;
    int key_index;
    int low  = left;
    int high = right;
    if (left < right) {
        key_index = indices[left];
        key       = input[left];
        while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low]   = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) {
                low++;
            }
            input[high]   = input[low];
            indices[high] = indices[low];
        }
        input[low]   = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

static float sigmoid(float x) { 
    return 1.0f / (1.0f + std::exp(-x)); 
}

static float unsigmoid(float y) { 
    return -1.0f * std::log((1.0f / y) - 1.0f); 
}

inline static float __clip(float val, float min, float max) {
    return val <= min ? min : (val >= max ? max : val);
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    return (int8_t)__clip(dst_val, -128.f, 127.f);
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { 
    return ((float)qnt - (float)zp) * scale; 
}

static int process(int8_t* input, int* anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, 
                   float threshold, int32_t zp, float scale) {
    int validCount = 0;
    int grid_len   = grid_h * grid_w;
    float thres    = unsigmoid(threshold);
    int8_t thres_i8 = qnt_f32_to_affine(thres, zp, scale);
    
    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_i8) {
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    int8_t* in_ptr = input + offset;
                    
                    float box_x = sigmoid(deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0f - 0.5f;
                    float box_y = sigmoid(deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0f - 0.5f;
                    float box_w = sigmoid(deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0f;
                    float box_h = sigmoid(deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0f;
                    
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0f);
                    box_y -= (box_h / 2.0f);

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > thres_i8) {
                        objProbs.push_back(sigmoid(deqnt_affine_to_f32(maxClassProbs, zp, scale)) * 
                                          sigmoid(deqnt_affine_to_f32(box_confidence, zp, scale)));
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;
}

int post_process_yolo(const rknn_output* outputs,
                      const rknn_tensor_attr* out_attrs,
                      int out_count,
                      float conf_threshold,
                      float nms_threshold,
                      std::vector<DetectionResult>& results) {
    results.clear();

    if (out_count < 3) {
        LOGD("后处理：输出数量不足，需要3个输出，实际: %d", out_count);
        return 0;
    }

    int model_in_h = 640;
    int model_in_w = 640;

    std::vector<int32_t> qnt_zps;
    std::vector<float> qnt_scales;
    for (int i = 0; i < out_count; ++i) {
        qnt_zps.push_back(out_attrs[i].zp);
        qnt_scales.push_back(out_attrs[i].scale);
        LOGD("输出[%d]: zp=%d, scale=%f", i, out_attrs[i].zp, out_attrs[i].scale);
    }

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    int stride0 = 8;
    int grid_h0 = model_in_h / stride0;
    int grid_w0 = model_in_w / stride0;
    process((int8_t*)outputs[0].buf, (int*)anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0,
            filterBoxes, objProbs, classId, conf_threshold, qnt_zps[0], qnt_scales[0]);

    int stride1 = 16;
    int grid_h1 = model_in_h / stride1;
    int grid_w1 = model_in_w / stride1;
    process((int8_t*)outputs[1].buf, (int*)anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1,
            filterBoxes, objProbs, classId, conf_threshold, qnt_zps[1], qnt_scales[1]);

    int stride2 = 32;
    int grid_h2 = model_in_h / stride2;
    int grid_w2 = model_in_w / stride2;
    process((int8_t*)outputs[2].buf, (int*)anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2,
            filterBoxes, objProbs, classId, conf_threshold, qnt_zps[2], qnt_scales[2]);

    int validCount = (int)filterBoxes.size() / 4;
    if (validCount <= 0) {
        return 0;
    }

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }

    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(classId.begin(), classId.end());
    for (auto c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1) {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0];
        float y1 = filterBoxes[n * 4 + 1];
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];

        x1 = clamp_val(x1, 0.f, (float)model_in_w);
        y1 = clamp_val(y1, 0.f, (float)model_in_h);
        x2 = clamp_val(x2, 0.f, (float)model_in_w);
        y2 = clamp_val(y2, 0.f, (float)model_in_h);

        DetectionResult res;
        res.x1 = x1 / model_in_w;
        res.y1 = y1 / model_in_h;
        res.x2 = x2 / model_in_w;
        res.y2 = y2 / model_in_h;
        res.score = objProbs[i];
        res.class_id = classId[n];
        results.push_back(res);
    }

    LOGD("后处理完成，检出目标数: %zu", results.size());
    return (int)results.size();
}
