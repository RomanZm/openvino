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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>

#include <api/input_layout.hpp>
#include <api/memory.hpp>
#include <api/scatter_update.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>

#include <cstddef>
#include <tests/test_utils/test_utils.h>

#include <exception>

using namespace cldnn;
using namespace ::tests;

TEST(scatter_update_gpu_fp16, d14_axisB) {
    //  Dictionary : 2x2x1x1
    //  Indexes : 1x4x1x1
    //  Axis : 0
    //  Output : 1x4x2x1
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 1.f, 0.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f

    //try{
    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 1, 1 } }); // Indexes
    auto input3 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 1, 1 } });// Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_b;

    set_values(input1, {
        FLOAT16(1.0f), FLOAT16(2.0f),
        FLOAT16(3.0f), FLOAT16(4.0f)
    });

    set_values(input2, {
        0.f, 1.f,
        1.f, 0.f
    });

    set_values(input3, {
        333.f, 8.f,
        8.f, 333.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(input_layout("InputUpdates", input3.get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis, tensor(1, 4, 2, 1))//add 3d input after "InputText"
    );
    
    network network(engine, topology); 
    
    
    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);
    
    auto outputs = network.execute();
    

    auto output = outputs.at("scatter_update").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {//Change it!!!
        1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f
    };

    
    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
    /*}
    catch(std::exception& e){
        std::cout << "Exception!: " << e.what() << std::endl;
    }*/

}

TEST(scatter_update_gpu_fp16, d222_axisB) {
    //  Dictionary : 3x2x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 0
    //  Output : 2x2x2x2
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 2, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_b;

    set_values(input1, {
        FLOAT16(1.f), FLOAT16(2.f), FLOAT16(3.f),
        FLOAT16(4.f), FLOAT16(5.f), FLOAT16(6.f),

        FLOAT16(7.f), FLOAT16(8.f), FLOAT16(9.f),
        FLOAT16(10.f), FLOAT16(11.f), FLOAT16(12.f)
    });

    set_values(input2, {
        0.f, 1.f,
        2.f, 1.f
    });

    set_values(input3, {
        3.f, 8.f,
        8.f, 8.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(input_layout("InputUpdates", input3.get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_update_gpu_fp16, d22_axisY) {
    //  Dictionary : 2x2x3x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 2, 1, 3 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_y;

    set_values(input1, {
        FLOAT16(1.f), FLOAT16(2.f), FLOAT16(3.f),
        FLOAT16(4.f), FLOAT16(5.f), FLOAT16(6.f),

        FLOAT16(7.f), FLOAT16(8.f), FLOAT16(9.f),
        FLOAT16(10.f), FLOAT16(11.f), FLOAT16(12.f)
    });

    set_values(input2, {
        0.f, 1.f, 2.f, 1.f
    });

    set_values(input3, {
        0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(input_layout("InputUpdates", input3.get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_update_gpu_fp16, d22_axisF) {
    //  Dictionary : 2x3x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 3, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::scatter_update::scatter_update_axis::along_f;

    set_values(input1, {
            FLOAT16(1.f), FLOAT16(2.f), FLOAT16(3.f),
            FLOAT16(4.f), FLOAT16(5.f), FLOAT16(6.f),

            FLOAT16(7.f), FLOAT16(8.f), FLOAT16(9.f),
            FLOAT16(10.f), FLOAT16(11.f), FLOAT16(12.f)
    });

    set_values(input2, {
            0.f, 1.f, 2.f, 1.f
    });

    set_values(input3, {
            0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(input_layout("InputUpdates", input3.get_layout()));
    topology.add(
            scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_update_gpu_fp32, d14_axisB) {
    //  Dictionary : 2x2x1x1
    //  Indexes : 1x4x1x1
    //  Axis : 0
    //  Output : 1x4x2x1
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 1.f, 0.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 1, 1 } }); // Indexes
    auto input3 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 1, 1 } }); // Indexes
    auto axis = cldnn::scatter_update::scatter_update_axis::along_b;

    set_values(input1, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    set_values(input2, {
        0.f, 1.f,
        1.f, 0.f
    });

    set_values(input3, {
        0.f, 1.f,
        1.f, 0.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(input_layout("InputUpdates", input3.get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis, tensor(1, 4, 1, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(scatter_update_gpu_fp32, d222_axisB) {
    //  Dictionary : 3x2x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 0
    //  Output : 2x2x2x2
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 2, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::scatter_update::scatter_update_axis::along_b;

    set_values(input1, {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,

        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    });

    set_values(input2, {
        0.f, 1.f, 2.f, 1.f
    });

    set_values(input3, {
        0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(input_layout("InputUpdates", input3.get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(scatter_update_gpu_fp32, d22_axisY) {
    //  Dictionary : 2x2x3x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 3 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::scatter_update::scatter_update_axis::along_y;

    set_values(input1, {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,

        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    });

    set_values(input2, {
        0.f, 1.f, 2.f, 1.f
    });

    set_values(input3, {
        0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(input_layout("InputUpdates", input3.get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(scatter_update_gpu_fp32, d22_axisF) {
    //  Dictionary : 2x3x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 1
    //  Output : 2x2x2x2
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 3, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::scatter_update::scatter_update_axis::along_f;

    set_values(input1, {
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,

            7.f, 8.f, 9.f,
            10.f, 11.f, 12.f
    });

    set_values(input2, {
            0.f, 1.f, 2.f, 1.f
    });

    set_values(input3, {
            0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(input_layout("InputUpdates", input3.get_layout()));
    topology.add(
            scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(scatter_update_gpu_int32, d22_axisF) {
    //  Dictionary : 2x3x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 1
    //  Output : 2x2x2x2
    //  Input values in i32

    //  Indexes:
    //  0, 1, 2, 1
    //
    //  Dictionary:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //
    //  Output:
    //  1, 2, 3, 4, 5, 6, 3, 4, 7, 8, 9, 10, 11, 12, 9, 10

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 3, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::scatter_update::scatter_update_axis::along_f;

    set_values(input1, {
            1, 2, 3,
            4, 5, 6,

            7, 8, 9,
            10, 11, 12
    });

    set_values(input2, {
            0, 1, 2, 1
    });

    set_values(input3, {
            0, 1, 2, 1
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(input_layout("InputUpdates", input3.get_layout()));
    topology.add(
            scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    auto output_ptr = output.pointer<int>();

    std::vector<int> expected_results = {
            1, 2, 3, 4, 5, 6, 3, 4, 7, 8, 9, 10, 11, 12, 9, 10
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(scatter_update_gpu_int32, d14_axisB) {
    //  Dictionary : 2x2x1x1
    //  Indexes : 1x4x1x1
    //  Axis : 0
    //  Output : 1x4x2x1
    //  Input values in i32

    //  Indexes:
    //  0, 1, 1, 0
    //
    //  Dictionary:
    //  1, 2, 3, 4
    //
    //  Output:
    //  1, 2, 3, 4, 3, 4, 1, 2

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 2, 1, 1 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::i32, format::bfyx, { 1, 4, 1, 1 } }); // Indexes
    auto input3 = memory::allocate(engine, { data_types::i32, format::bfyx, { 1, 4, 1, 1 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_b;

    set_values(input1, {
            1, 2,
            3, 4
    });

    set_values(input2, {
            0, 1,
            1, 0
    });

    set_values(input3, {
            0, 1,
            1, 0
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(input_layout("InputUpdates", input3.get_layout()));
    topology.add(
            scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis, tensor(1, 4, 1, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    auto output_ptr = output.pointer<int>();

    std::vector<int> expected_results = {
            1, 2, 3, 4, 3, 4, 1, 2
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(scatter_update_gpu_int32, d222_axisB) {
    //  Dictionary : 3x2x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 0
    //  Output : 2x2x2x2
    //  Input values in i32

    //  Indexes:
    //  0, 1, 2, 1
    //
    //  Dictionary:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //
    //  Output:
    //  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 5, 6, 7, 8

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::i32, format::bfyx, { 3, 2, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 2, 1, 1 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_b;

    set_values(input1, {
            1, 2, 3,
            4, 5, 6,

            7, 8, 9,
            10, 11, 12
    });

    set_values(input2, {
            0, 1, 2, 1
    });

    set_values(input3, {
            0, 1, 2, 1
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(input_layout("InputUpdates", input3.get_layout()));
    topology.add(
            scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    auto output_ptr = output.pointer<int>();

    std::vector<int> expected_results = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 5, 6, 7, 8
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(scatter_update_gpu_int32, d22_axisY) {
    //  Dictionary : 2x2x3x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in i32

    //  Indexes:
    //  0, 1, 2, 1
    //
    //  Dictionary:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //
    //  Output:
    //  1, 2, 3, 2, 4, 5, 6, 5, 7, 8, 9, 8, 10, 11, 12, 11

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 2, 1, 3 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 2, 1, 1 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_y;

    set_values(input1, {
            1, 2, 3,
            4, 5, 6,

            7, 8, 9,
            10, 11, 12
    });

    set_values(input2, {
            0, 1, 2, 1
    });

    set_values(input3, {
            0, 1, 2, 1
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(input_layout("InputUpdates", input3.get_layout()));
    topology.add(
            scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    auto output_ptr = output.pointer<int>();

    std::vector<int> expected_results = {
            1, 2, 3, 2, 4, 5, 6, 5, 7, 8, 9, 8, 10, 11, 12, 11
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(scatter_update_gpu_fp32, d41_axisB) {
    //  Dictionary : 2x2x3x1
    //  Indexes : 4x1x1x1
    //  Axis : 0
    //  Output : 4x1x2x3
    //  Input values in fp32, indices in i32

    //  Indexes:
    //  0, 1, 1, 0
    //
    //  Dictionary:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //
    //  Output:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //  7, 8, 9, 10, 11, 12
    //  1, 2, 3, 4, 5, 6,

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 3 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } }); // Indexes
    auto input3 = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_b;

    set_values(input1, {
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,

            7.f, 8.f, 9.f,
            10.f, 11.f, 12.f
               });

    set_values(input2, {
            0, 1, 1, 0
               });

    set_values(input3, {
            0, 1, 1, 0
               });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(input_layout("InputUpdates", input3.get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis, tensor(4, 1, 3, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
            7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
            7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f
    };

    ASSERT_EQ(expected_results.size(), output_ptr.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]) << " at i=" << i;
    }
}

TEST(scatter_update_gpu_fp32, d41_axisF) {
    //  Dictionary : 2x3x2x1
    //  Indexes : 4x1x1x1
    //  Axis : 0
    //  Output : 2x4x1x2
    //  Input values in fp32, indices in i32

    //  Indexes:
    //  1, 0, 1, 2
    //
    //  Dictionary:
    //  1, 2,   3, 4,   5, 6,
    //  7, 8,   9, 10,  11, 12
    //
    //  Output:
    //  3, 4,   1, 2,   3, 4,   5, 6,
    //  9, 10,  7, 8,   9, 10,  11, 12

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 3, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } }); // Indexes
    auto input3 = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_f;

    set_values(input1, {
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
            7.f, 8.f, 9.f, 10.f, 11.f, 12.f
               });

    set_values(input2, {
            1, 0, 1, 2
               });

    set_values(input3, {
            1, 0, 1, 2
               });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(input_layout("InputUpdates", input3.get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis, tensor(2, 4, 2, 1))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
            3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
            9.f, 10.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    };

    ASSERT_EQ(expected_results.size(), output_ptr.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]) << " at i=" << i;
    }
}

TEST(scatter_update_gpu_fp32, d2_axisX) {
    //  Dictionary : 2x2x1x1
    //  Indexes : 2x1x1x1
    //  Axis : 0
    //  Output : 2x2x1x2
    //  Input values in fp32, indices in i32

    //  Indexes:
    //  0, 0
    //
    //  Dictionary:
    //  1, 2, 3, 4
    //
    //  Output:
    //  1, 1, 2, 2, 3, 3, 4, 4

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 1, 1, 1 } }); // Indexes
    auto input3 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 1, 1, 1 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_x;

    set_values(input1, {
            1.f, 2.f,
            3.f, 4.f,
               });

    set_values(input2, {
            0, 0
               });

    set_values(input3, {
            0, 0
               });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(input_layout("InputUpdates", input3.get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis, tensor(2, 2, 2, 1))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
            1.f, 1.f, 2.f, 2.f,
            3.f, 3.f, 4.f, 4.f
    };

    ASSERT_EQ(expected_results.size(), output_ptr.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]) << " at i=" << i;
    }
}
