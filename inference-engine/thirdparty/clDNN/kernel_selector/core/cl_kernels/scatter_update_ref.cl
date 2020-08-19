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

#define GET_UPDATES_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)
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
#ifndef IS_SECOND_ITER
    #if OUTPUT_DIMS == 6
        #define ORDER b,f,w,z,y,x
        const uint x = (uint)get_global_id(0) % OUTPUT_SIZE_X;
        const uint y = (uint)get_global_id(0) / OUTPUT_SIZE_X;
        const uint z = (uint)get_global_id(1) % OUTPUT_SIZE_Z;
        const uint w = (uint)get_global_id(1) / OUTPUT_SIZE_Z;
        const uint f = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
        const uint b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 5
        #define ORDER b,f,z,y,x
        const uint x = (uint)get_global_id(0);
        const uint y = (uint)get_global_id(1) % OUTPUT_SIZE_Y;
        const uint z = (uint)get_global_id(1) / OUTPUT_SIZE_Y;
        const uint f = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
        const uint b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 4
        #define ORDER b,f,y,x
        const uint x = (uint)get_global_id(0);
        const uint y = (uint)get_global_id(1);
        const uint f = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
        const uint b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
    #endif

    const uint output_idx = GET_OUTPUT_INDEX(ORDER);
    output[output_idx] = dictionary[output_idx];
#else
    #if OUTPUT_DIMS == 6
        #define ORDER b,f,w,z,y,x
        uint b, f, w, z, y, x;
        switch(AXIS_VALUE){
        case 0:
        case 2:
        case 4:
            f = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
            b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
            x = (uint)get_global_id(0) % OUTPUT_SIZE_X;
            y = (uint)get_global_id(0) / OUTPUT_SIZE_X;
            z = (uint)get_global_id(1) % OUTPUT_SIZE_Z;
            w = (uint)get_global_id(1) / OUTPUT_SIZE_Z;
            break;
        case 1:
            f = (uint)get_global_id(2) / OUTPUT_BATCH_NUM;
            b = (uint)get_global_id(2) % OUTPUT_BATCH_NUM;
            x = (uint)get_global_id(0) % OUTPUT_SIZE_X;
            y = (uint)get_global_id(0) / OUTPUT_SIZE_X;
            z = (uint)get_global_id(1) % OUTPUT_SIZE_Z;
            w = (uint)get_global_id(1) / OUTPUT_SIZE_Z;
            break;
        case 3:
            f = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
            b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
            x = (uint)get_global_id(0) % OUTPUT_SIZE_X;
            y = (uint)get_global_id(0) / OUTPUT_SIZE_X;
            z = (uint)get_global_id(1) / OUTPUT_SIZE_W;
            w = (uint)get_global_id(1) % OUTPUT_SIZE_W;
            break;
        case 5:
            f = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
            b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
            x = (uint)get_global_id(0) / OUTPUT_SIZE_Y;
            y = (uint)get_global_id(0) % OUTPUT_SIZE_Y;
            z = (uint)get_global_id(1) % OUTPUT_SIZE_Z;
            w = (uint)get_global_id(1) / OUTPUT_SIZE_Z;
            break;
        }
    #elif OUTPUT_DIMS == 5
        #define ORDER b,f,z,y,x
        const uint z = (uint)get_global_id(1);
        uint f;
        uint b;
        uint x;
        uint y;
        switch(AXIS_VALUE){
        case 0:
        case 3:
            f = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
            b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
            x = (uint)get_global_id(0) % OUTPUT_SIZE_X;
            y = (uint)get_global_id(0) / OUTPUT_SIZE_X;
            break;
        case 1:
            f = (uint)get_global_id(2) / OUTPUT_BATCH_NUM;
            b = (uint)get_global_id(2) % OUTPUT_BATCH_NUM;
            x = (uint)get_global_id(0) % OUTPUT_SIZE_X;
            y = (uint)get_global_id(0) / OUTPUT_SIZE_X;
            break;
        case 4:
            f = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
            b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
            x = (uint)get_global_id(0) / OUTPUT_SIZE_Y;
            y = (uint)get_global_id(0) % OUTPUT_SIZE_Y;
            break;
        }
    #elif OUTPUT_DIMS == 4
        #define ORDER b,f,y,x
        const uint x = (uint)get_global_id(0);
        const uint y = (uint)get_global_id(1);
        uint f;
        uint b;
        if (AXIS_VALUE == 0){
            f = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
            b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
        }
        else{
            f = (uint)get_global_id(2) / OUTPUT_BATCH_NUM;
            b = (uint)get_global_id(2) % OUTPUT_BATCH_NUM;
        }
    #endif
    
    const uint output_idx = GET_OUTPUT_INDEX(SECOND_ITER_OUTPUT_INDEX_ORDER);
    const uint updates_idx = GET_UPDATES_INDEX(INPUT2, , UPDATES_INDEX_ORDER);
    INPUT2_TYPE val = updates[updates_idx];
#if HAS_FUSED_OPS
    FUSED_OPS;
    output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
#else
    output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
#endif
#endif
}

#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX