#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"
#include "../models/qwen2/model.hpp"

#include <memory>
#include <vector>

struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::Qwen2Model> model;
    LlaisysQwen2Weights weights{};
    std::vector<std::unique_ptr<LlaisysTensor>> handles;
};

namespace {
llaisysTensor_t handle_for(LlaisysQwen2Model *model, const llaisys::tensor_t &tensor) {
    model->handles.push_back(std::make_unique<LlaisysTensor>(LlaisysTensor{tensor}));
    return model->handles.back().get();
}

llaisysTensor_t *handle_array(LlaisysQwen2Model *model, const std::vector<llaisys::tensor_t> &tensors) {
    auto *handles = new llaisysTensor_t[tensors.size()];
    for (size_t i = 0; i < tensors.size(); ++i) handles[i] = handle_for(model, tensors[i]);
    return handles;
}
}

__C {
LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device,
                                            int *device_ids, int ndevice) {
    if (meta == nullptr || device_ids == nullptr || ndevice <= 0) return nullptr;
    auto *result = new LlaisysQwen2Model;
    result->model = std::make_unique<llaisys::models::Qwen2Model>(*meta, device, device_ids[0]);
    auto &model = *result->model;
    result->weights.in_embed = handle_for(result, model.in_embed);
    result->weights.out_embed = handle_for(result, model.out_embed);
    result->weights.out_norm_w = handle_for(result, model.out_norm_w);
    result->weights.attn_norm_w = handle_array(result, model.attn_norm_w);
    result->weights.attn_q_w = handle_array(result, model.attn_q_w);
    result->weights.attn_q_b = handle_array(result, model.attn_q_b);
    result->weights.attn_k_w = handle_array(result, model.attn_k_w);
    result->weights.attn_k_b = handle_array(result, model.attn_k_b);
    result->weights.attn_v_w = handle_array(result, model.attn_v_w);
    result->weights.attn_v_b = handle_array(result, model.attn_v_b);
    result->weights.attn_o_w = handle_array(result, model.attn_o_w);
    result->weights.mlp_norm_w = handle_array(result, model.mlp_norm_w);
    result->weights.mlp_gate_w = handle_array(result, model.mlp_gate_w);
    result->weights.mlp_up_w = handle_array(result, model.mlp_up_w);
    result->weights.mlp_down_w = handle_array(result, model.mlp_down_w);
    return result;
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    if (model == nullptr) return;
    delete[] model->weights.attn_norm_w;
    delete[] model->weights.attn_q_w;
    delete[] model->weights.attn_q_b;
    delete[] model->weights.attn_k_w;
    delete[] model->weights.attn_k_b;
    delete[] model->weights.attn_v_w;
    delete[] model->weights.attn_v_b;
    delete[] model->weights.attn_o_w;
    delete[] model->weights.mlp_norm_w;
    delete[] model->weights.mlp_gate_w;
    delete[] model->weights.mlp_up_w;
    delete[] model->weights.mlp_down_w;
    delete model;
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
    if (model == nullptr) return nullptr;
    return &model->weights;
}

void llaisysQwen2ModelReset(LlaisysQwen2Model *model) {
    if (model == nullptr) return;
    model->model->reset();
}

int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (model == nullptr) return -1;
    return model->model->infer(token_ids, ntoken);
}
}
