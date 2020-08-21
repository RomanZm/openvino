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


#include "include/include_all.cl"

#define GET_UPDATES_INDEX(idx_order) INPUT2_GET_INDEX(idx_order)
#define GET_OUTPUT_INDEX(idx_order) OUTPUT_GET_INDEX(idx_order)

KERNEL(scatter_update_ref)(const __global INPUT0_TYPE* dictionary,
                   const __global INPUT1_TYPE* indices,
                   const __global INPUT2_TYPE* updates, 
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif

)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint bf = get_global_id(2);

#ifndef IS_SECOND_ITER
    const uint b = bf / OUTPUT_FEATURE_NUM;
    const uint f = bf % OUTPUT_FEATURE_NUM;

    const uint output_idx = OUTPUT_GET_INDEX(b, f, y, x);
    const INPUT0_TYPE val = dictionary[output_idx];
    #if HAS_FUSED_OPS
        FUSED_OPS_FIRST_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_FIRST_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif
    //printf("First time!!! b: %d f: %d y: %d x: %d || output_idx: %d; output: %f;\n", b, f, y, x, output_idx, output[output_idx]);
#else
    uint b, f;
    if (AXIS_VALUE == 1){
        b = bf % OUTPUT_BATCH_NUM;
        f = bf / OUTPUT_BATCH_NUM;
    }
    else{
        b = bf / OUTPUT_FEATURE_NUM;
        f = bf % OUTPUT_FEATURE_NUM;
    }

    const int Indices_el_under_axis_idx = indices[AXIS_IDX];
    const uint output_idx = GET_OUTPUT_INDEX(SECOND_ITER_OUTPUT_INDEX_ORDER);
    const uint updates_idx = GET_UPDATES_INDEX(UPDATES_INDEX_ORDER);
    const INPUT2_TYPE val = updates[updates_idx];
    #if HAS_FUSED_OPS
        FUSED_OPS_SECOND_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SECOND_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif
    //if (f%10 == 0 && x % 50 == 0){
    //printf("Second!!! b: %d f: %d y: %d x: %d;|| AXIS_IDX: %d; indices[AXIS_IDX]: %f || updates_idx: %d; updates: %f; output_idx: %d; output: %f \n",
    //        b, f, y, x, AXIS_IDX, indices[AXIS_IDX], updates_idx, updates[updates_idx], output_idx, output[output_idx]);
    //}
    
    //printf("First time!!! b: %d f: %d y: %d x: %d || output_idx: %d; output: %f\n", b, f, y, x, output_idx, output[output_idx]);
    
#endif
}

#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX