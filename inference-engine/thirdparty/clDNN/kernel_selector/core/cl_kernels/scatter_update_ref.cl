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

#define GET_UPDATES_INDEX(prefix, idx_order) CAT(prefix, _GET_INDEX)(idx_order)
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
    /*#if OUTPUT_DIMS == 5
        printf("First time!!! b: %d f: %d z: %d y: %d x: %d || output_idx: %d; output: %f;\n", b, f,z, y, x, output_idx, output[output_idx]);
    #elif*/#if OUTPUT_DIMS == 6
        printf("First time!!! b: %d f: %d w: %d z: %d y: %d x: %d || output_idx: %d; output: %f;\n", b, f,w,z, y, x, output_idx, output[output_idx]);
    #endif
#else
    const uint chan0 = get_global_id(0);
    const uint chan1 = get_global_id(1);
    const uint chan2 = get_global_id(2);
    #if OUTPUT_DIMS == 6
        #define ORDER b,f,w,z,y,x
        uint b, f, w, z, y, x;
        switch(AXIS_VALUE){
        case 1:
            f = chan2 / OUTPUT_BATCH_NUM;
            b = chan2 % OUTPUT_BATCH_NUM;
            x = chan0 % OUTPUT_SIZE_X;
            y = chan0 / OUTPUT_SIZE_X;
            z = chan1 % OUTPUT_SIZE_Z;
            w = chan1 / OUTPUT_SIZE_Z;
            break;
        case 3:
            f = chan2 % OUTPUT_FEATURE_NUM;
            b = chan2 / OUTPUT_FEATURE_NUM;
            x = chan0 % OUTPUT_SIZE_X;
            y = chan0 / OUTPUT_SIZE_X;
            z = chan1 / OUTPUT_SIZE_W;
            w = chan1 % OUTPUT_SIZE_W;
            break;
        case 5:
            f = chan2 % OUTPUT_FEATURE_NUM;
            b = chan2 / OUTPUT_FEATURE_NUM;
            x = chan0 / OUTPUT_SIZE_Y;
            y = chan0 % OUTPUT_SIZE_Y;
            z = chan1 % OUTPUT_SIZE_Z;
            w = chan1 / OUTPUT_SIZE_Z;
            break;
        default:
            f = chan2 % OUTPUT_FEATURE_NUM;
            b = chan2 / OUTPUT_FEATURE_NUM;
            x = chan0 % OUTPUT_SIZE_X;
            y = chan0 / OUTPUT_SIZE_X;
            z = chan1 % OUTPUT_SIZE_Z;
            w = chan1 / OUTPUT_SIZE_Z;
            break;
        }
    #elif OUTPUT_DIMS == 5
        #define ORDER b,f,z,y,x
        const uint z = chan1;
        uint f;
        uint b;
        uint x;
        uint y;
        switch(AXIS_VALUE){
        case 0:
        case 2:
        case 3:
            f = chan2 % OUTPUT_FEATURE_NUM;
            b = chan2 / OUTPUT_FEATURE_NUM;
            x = chan0 % OUTPUT_SIZE_X;
            y = chan0 / OUTPUT_SIZE_X;
            break;
        case 1:
            f = chan2 / OUTPUT_BATCH_NUM;
            b = chan2 % OUTPUT_BATCH_NUM;
            x = chan0 % OUTPUT_SIZE_X;
            y = chan0 / OUTPUT_SIZE_X;
            break;
        case 4:
            f = chan2 % OUTPUT_FEATURE_NUM;
            b = chan2 / OUTPUT_FEATURE_NUM;
            x = chan0 / OUTPUT_SIZE_Y;
            y = chan0 % OUTPUT_SIZE_Y;
            break;
        default: break;
        }
    #elif OUTPUT_DIMS == 4
        #define ORDER b,f,y,x
        const uint x = chan0;
        const uint y = chan1;
        uint f;
        uint b;
        if (AXIS_VALUE == 0){
            f = chan2 % OUTPUT_FEATURE_NUM;
            b = chan2 / OUTPUT_FEATURE_NUM;
        }
        else{
            f = chan2 / OUTPUT_BATCH_NUM;
            b = chan2 % OUTPUT_BATCH_NUM;
        }
    #endif

    const uint output_idx = GET_OUTPUT_INDEX(SECOND_ITER_OUTPUT_INDEX_ORDER);
    const uint updates_idx = GET_UPDATES_INDEX(INPUT2, UPDATES_INDEX_ORDER);
    output[output_idx] = updates[updates_idx];

    /*#if OUTPUT_DIMS == 5
        printf("Second!!! b: %d f: %d z: %d y: %d x: %d;|| AXIS_IDX: %d; indices[AXIS_IDX]: %f || updates_idx: %d; updates: %f; output_idx: %d; output: %f \n",
            b, f,z, y, x, OUTPUT_INDEX_ON_AXIS, indices[OUTPUT_INDEX_ON_AXIS], updates_idx, updates[updates_idx], output_idx, output[output_idx]);
    
    #elif */#if OUTPUT_DIMS == 6
        printf("Second!!! b: %d f: %d w: %d z: %d y: %d x: %d;|| AXIS_IDX: %d; indices[AXIS_IDX]: %f || updates_idx: %d; updates: %f; output_idx: %d; output: %f \n",
            b, f,w,z, y, x, OUTPUT_INDEX_ON_AXIS, indices[OUTPUT_INDEX_ON_AXIS], updates_idx, updates[updates_idx], output_idx, output[output_idx]);
    #endif
/*#if HAS_FUSED_OPS
    FUSED_OPS;
    output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
#else
    output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
#endif*/
#endif
}

#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX
