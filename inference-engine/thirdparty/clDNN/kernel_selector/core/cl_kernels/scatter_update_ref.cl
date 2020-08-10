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

#define UPDATE_GET_IDX(B, F, Y, X) (X + INPUT2_SIZE_X*(Y+INPUT2_SIZE_Y*(F+INPUT2_FEATURE_NUM*B)))
#define GET_UPDATES_INDEX(idx_order) UPDATE_GET_IDX(idx_order)
#define GET_OUTPUT_INDEX(idx_order) OUTPUT_GET_INDEX(idx_order)

KERNEL(scatter_update_ref)(const __global INPUT0_TYPE* dictionary,
                   const __global INPUT1_TYPE* indices,
                   const __global INPUT2_TYPE* updates, 
                   __global OUTPUT_TYPE* output
/*#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
*/
)
{
    const uint b = get_global_id(0);
    const uint f = get_global_id(1);
    const uint yx = get_global_id(2);
    uint x, y;
    if (AXIS_VALUE == 3){
        x = yx / OUTPUT_SIZE_Y;
        y = yx % OUTPUT_SIZE_Y;
        
    }
    else{
        y = yx / OUTPUT_SIZE_X;
        x = yx % OUTPUT_SIZE_X;
    }

    

    #ifndef IS_SECOND_ITER
    for (uint i = 0; i<REDUCE_NUMB; i++){
        const uint output_idx = OUTPUT_GET_INDEX(b, f, y*REDUCE_NUMB + i, x);
        output[output_idx] = dictionary[output_idx];
    }
    //printf("First time!!! b: %d f: %d y: %d x: %d || output_idx: %d; output: %f\n", b, f, y, x, output_idx, output[output_idx]);
    return;
    #endif
    
    const uint sec_output_idx = GET_OUTPUT_INDEX(SECOND_ITER_OUTPUT_INDEX_ORDER);
    const uint updates_idx = GET_UPDATES_INDEX(UPDATES_INDEX_ORDER);
    output[sec_output_idx] = updates[updates_idx];
    //printf("Second!!! b: %d f: %d y: %d x: %d || updates_idx: %d; updates: %f; output_idx: %d; output: %f \n",
               // b, f, y, x, updates_idx, updates[updates_idx], output_idx, output[sec_output_idx]);
    
    //printf("First time!!! b: %d f: %d y: %d x: %d || output_idx: %d; output: %f\n", b, f, y, x, output_idx, output[output_idx]);
    
//#if HAS_FUSED_OPS
    //FUSED_OPS;
    //output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
//#else
    //output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
//#endif

}

#undef GET_UPDATES_INDEX
#undef UPDATE_GET_IDX
#undef GET_OUTPUT_INDEX