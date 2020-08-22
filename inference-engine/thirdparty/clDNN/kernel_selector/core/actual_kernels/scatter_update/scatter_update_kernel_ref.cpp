/*
// Copyright (c) 2019-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "scatter_update_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>
#include <iostream>

namespace kernel_selector {
static size_t GetScatterUpdateChannelIndex(const scatter_update_params& params) {
    Tensor::DataChannelName name = Tensor::DataChannelName::X;

    const size_t Dict_size = params.inputs[0].GetDims().size();

    switch (params.axis) {
        case ScatterUpdateAxis::X:
            return Dict_size - 1;
        case ScatterUpdateAxis::Y:
            return Dict_size - 2;
        case ScatterUpdateAxis::Z:
            return Dict_size - 3;
        case ScatterUpdateAxis::W:
            return 2;
        case ScatterUpdateAxis::FEATURE:
            return 1;
        case ScatterUpdateAxis::BATCH:
            return 0;
        default:
            break;
    }

    return DataTensor::Channelndex(params.output.GetLayout(), name);
}

ParamsKey ScatterUpdateKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

static size_t GetNonEmptyDimsNumber(const DataTensor& data_tensor) {
    if (data_tensor.LogicalSize() != 1) {
        // Count the number of "one size" dimensions starting with X to Batch
        size_t one_size_dims = 0;
        for (auto& i : data_tensor.GetDims()) {
            if (i.v == 1)
                one_size_dims++;
            else
                break;
        }
        return data_tensor.Dimentions() - one_size_dims;
    } else {
        return 1;
    }
}

static inline std::string GetOrderString(std::vector<std::string>& order) {
    std::string order_str = order[0];
    for (size_t i = 1; i < order.size(); i++)
        order_str += ", " + order[i];
    
    return order_str;
}

static inline std::vector<std::string> GetDefaultOrder(size_t size) {
    std::vector<std::string> default_order;
    if (size <= 4) {
        default_order = {"b", "f", "y", "x"};
    } else if (size == 5) {
        default_order = {"b", "f", "z", "y", "x"};
    } else if (size == 6) {
        default_order = {"b", "f", "w", "z", "y", "x"};
    }
    
    return default_order;
}

static std::string GetUpdatesIndexOrder(const scatter_update_params& params, size_t axis) {
    std::vector<std::string> default_order = GetDefaultOrder(params.inputs[2].GetDims().size());

    size_t indices_dims_num = GetNonEmptyDimsNumber(params.inputs[1]);
    std::string FYX_indices_size = "(INPUT1_FEATURE_NUM * INPUT1_SIZE_Y * INPUT1_SIZE_X)";
    std::string YX_indices_size = "(INPUT1_SIZE_Y * INPUT1_SIZE_X)";
    std::string X_indices_size = "(INPUT1_SIZE_X)";
    
    // Shift indices of ScatterUpdate updates input related to Indices dims
    for (size_t i = default_order.size()-1; i > (axis + indices_dims_num - 1); i--)
        default_order[i] = default_order[i - indices_dims_num + 1];

    // Insert Indices indexes in axis dimention in the Update index order
    for (size_t i = axis; i < (axis + indices_dims_num) && i < default_order.size(); i++){
        switch(i - axis){
            case 0:
                default_order[i] = "(OUTPUT_INDEX_ON_AXIS /" + FYX_indices_size + ")";
                break;
            case 1:
                default_order[i] = "((OUTPUT_INDEX_ON_AXIS %" + FYX_indices_size + ")/" + YX_indices_size + ")";
                break;
            case 2:
                default_order[i] = "(((OUTPUT_INDEX_ON_AXIS %" + FYX_indices_size + ")%" + YX_indices_size + ")/" + X_indices_size + ")";
                break;
            case 3:
                default_order[i] = "(((OUTPUT_INDEX_ON_AXIS %" + FYX_indices_size + ")%" + YX_indices_size + ")%" + X_indices_size + ")";
                break;
        }
    }

    return GetOrderString(default_order);
}

static std::string GetSecondIterOutputIndexOrder(const scatter_update_params& params, size_t axis){
    std::vector<std::string> default_order = GetDefaultOrder(params.output.GetDims().size());
    default_order[axis] = "indices[OUTPUT_INDEX_ON_AXIS]";
    return GetOrderString(default_order);
}

CommonDispatchData ScatterUpdateKernelRef::SetDefault(const scatter_update_params& params, const optional_params&, bool is_second/*, std::vector<std::string>& size_deviders_for_cl*/) const {
    CommonDispatchData runInfo;
    const auto& output = params.output;

    std::vector<size_t> global(3);

    const size_t INDICES_SIZE = params.inputs[1].LogicalSize();

    switch (params.inputs[2].GetLayout()){
    case DataLayout::bfyx:
        global = {output.X().v, output.Y().v, output.Feature().v * output.Batch().v};
        if (is_second){
            switch (params.axis){
            case ScatterUpdateAxis::BATCH:
                global[2] = INDICES_SIZE * output.Feature().v;
                break;
            case ScatterUpdateAxis::FEATURE:
                global[2] = INDICES_SIZE * output.Batch().v;
                break;
            case ScatterUpdateAxis::Y:
                global[1] = INDICES_SIZE;
                break;
            case ScatterUpdateAxis::X:
                global[0] = INDICES_SIZE;
                break;
            default: break;
            }
        }
        break;

    case DataLayout::bfzyx:
        global = {output.X().v * output.Y().v, output.Z().v, output.Feature().v * output.Batch().v};
        if (is_second){
            switch (params.axis){
            case ScatterUpdateAxis::BATCH:
                global[2] = INDICES_SIZE * output.Feature().v;
                break;
            case ScatterUpdateAxis::FEATURE:
                global[2] = INDICES_SIZE * output.Batch().v;
                break;
            case ScatterUpdateAxis::Z:
                global[1] = INDICES_SIZE;
                break;
            case ScatterUpdateAxis::Y:
                global[0] = INDICES_SIZE * output.X().v;
                break;
            case ScatterUpdateAxis::X:
                global[0] = INDICES_SIZE * output.Y().v;
                break;
            default: break;
            }
        }
        break;

    case DataLayout::bfwzyx:
        global = {output.X().v * output.Y().v, output.Z().v * output.W().v, output.Feature().v * output.Batch().v};
        std::cout << "Here we are, bfwzyx! output.B().v = " << params.inputs[0].Batch().v << "; output.F().v = " << params.inputs[0].Feature().v << "; output.W().v =" << params.inputs[0].W().v << "; output.Z().v =" << params.inputs[0].Z().v << "; output.Y().v =" << params.inputs[0].Y().v << "; output.X().v = " << params.inputs[0].X().v << std::endl;
        if (is_second){
            switch (params.axis){
            case ScatterUpdateAxis::BATCH:
                global[2] = INDICES_SIZE * output.Feature().v;
                break;
            case ScatterUpdateAxis::FEATURE:
                global[2] = INDICES_SIZE * output.Batch().v;
                break;
            case ScatterUpdateAxis::Z:
                global[1] = INDICES_SIZE * output.W().v;
                break;
            case ScatterUpdateAxis::W:
                global[1] = INDICES_SIZE * output.Z().v;
                break;
            case ScatterUpdateAxis::Y:
                global[0] = INDICES_SIZE * output.X().v;
                break;
            case ScatterUpdateAxis::X:
                global[0] = INDICES_SIZE * output.Y().v;
                break;
            }
        }
        break;
    default: break;
    }
    
    std::vector<size_t> local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = global[2];
    
    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    runInfo.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;
    runInfo.fp16UnitUsed = params.inputs[2].GetDType() == Datatype::F16;

    return runInfo;
}

static std::string GetOutputIndexOnAxis(const scatter_update_params& params, size_t axis){
    std::vector<std::string> default_order = GetDefaultOrder(params.output.GetDims().size());
    return std::string(default_order[axis]);
}

JitConstants ScatterUpdateKernelRef::GetJitConstants(const scatter_update_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("UPDATES_INDEX_ORDER", GetUpdatesIndexOrder(params, GetScatterUpdateChannelIndex(params))));
    jit.AddConstant(MakeJitConstant("SECOND_ITER_OUTPUT_INDEX_ORDER", GetSecondIterOutputIndexOrder(params, GetScatterUpdateChannelIndex(params))));
    jit.AddConstant(MakeJitConstant("OUTPUT_INDEX_ON_AXIS", GetOutputIndexOnAxis(params, GetScatterUpdateChannelIndex(params))));
    jit.AddConstant(MakeJitConstant("AXIS_VALUE", GetScatterUpdateChannelIndex(params)));
    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = { "", GetDefaultOrder(params.output.GetDims().size()), "val", params.inputs[0].GetDType() };
    
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

bool ScatterUpdateKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType:: SCATTER_UPDATE || o.GetType() != KernelType::SCATTER_UPDATE) {
        return false;
    }

    const scatter_update_params& params = static_cast<const scatter_update_params&>(p);

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

KernelsData ScatterUpdateKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    const scatter_update_params& orgParams = static_cast<const scatter_update_params&>(params);

    const size_t INDICES_SIZE = orgParams.inputs[1].LogicalSize();
    const uint axis = GetScatterUpdateChannelIndex(orgParams);
    uint start_with_iteration = 0;


    std::vector<size_t> output_dims;
    switch (orgParams.output.GetLayout()){
    case DataLayout::bfyx:
        output_dims = {orgParams.output.Batch().v, orgParams.output.Feature().v, orgParams.output.Y().v, orgParams.output.X().v};
        break;
    case DataLayout::bfzyx:
        output_dims = {orgParams.output.Batch().v, orgParams.output.Feature().v, orgParams.output.Z().v, orgParams.output.Y().v, orgParams.output.X().v};
        break;
    case DataLayout::bfwzyx:
        output_dims = {orgParams.output.Batch().v, orgParams.output.Feature().v, orgParams.output.W().v, orgParams.output.Z().v, orgParams.output.Y().v, orgParams.output.X().v};
        break;
    default: break;
    }

    if (output_dims[axis] == INDICES_SIZE){
        start_with_iteration = 1;
    }

    KernelData kd = KernelData::Default<scatter_update_params>(params, (2 - start_with_iteration));
    scatter_update_params& newParams = *static_cast<scatter_update_params*>(kd.params.get());
    auto cldnn_jit = GetJitConstants(newParams);

    for (int i = start_with_iteration; i < 2; i++){
        auto runInfo = SetDefault(newParams, options, (i == 1));
        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);

        if (i == 1){
            cldnn_jit.AddConstant(MakeJitConstant("IS_SECOND_ITER", "true"));
        }
        std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[i - start_with_iteration];

        FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point, "", false, false, 3, GetFusedPrimitiveInputsCount(params));
    }

    kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;
    
    return {kd};
}
}  // namespace kernel_selector
