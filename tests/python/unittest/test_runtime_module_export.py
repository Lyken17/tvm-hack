# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te

import tvm.testing

from tvm.contrib import utils

header_file_dir_path = utils.tempdir()


def gen_engine_header():
    code = r"""
        #ifndef _ENGINE_H_
        #define _ENGINE_H_
        #include <cstdint>
        #include <string>
        #include <sstream>
        #include <vector>
        class Engine {
        };

        #endif
        """
    header_file = header_file_dir_path.relpath("gcc_engine.h")
    with open(header_file, "w") as f:
        f.write(code)


def generate_engine_module():
    code = r"""
        #include <tvm/runtime/c_runtime_api.h>
        #include <dlpack/dlpack.h>
        #include "gcc_engine.h"

        extern "C" void gcc_1_(float* gcc_input4, float* gcc_input5,
                float* gcc_input6, float* gcc_input7, float* out) {
            Engine engine;
        }
        """
    import tvm.runtime._ffi_api

    gen_engine_header()
    csource_module = tvm.runtime._ffi_api.CSourceModuleCreate(code, "cc", [], None)
    return csource_module


@tvm.testing.uses_gpu
def test_mod_export():
    def verify_gpu_mod_export(obj_format):
        for device in ["llvm", "cuda"]:
            if not tvm.testing.device_enabled(device):
                print("skip because %s is not enabled..." % device)
                return

        synthetic_mod, synthetic_params = relay.testing.synthetic.get_workload()
        synthetic_llvm_mod, synthetic_llvm_params = relay.testing.synthetic.get_workload()
        with tvm.transform.PassContext(opt_level=3):
            _, synthetic_gpu_lib, _ = relay.build_module.build(
                synthetic_mod, "cuda", params=synthetic_params, mod_name="cudalib"
            )
            _, synthetic_llvm_cpu_lib, _ = relay.build_module.build(
                synthetic_llvm_mod, "llvm", params=synthetic_llvm_params, mod_name="llvmlib"
            )

        from tvm.contrib import utils

        temp = utils.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        synthetic_gpu_lib.import_module(synthetic_llvm_cpu_lib)
        synthetic_gpu_lib.export_library(path_lib)
        loaded_lib = tvm.runtime.load_module(path_lib)
        assert loaded_lib.type_key == "library"
        assert loaded_lib.imported_modules[0].type_key == "cuda"
        #  dso modules are merged together
        assert len(loaded_lib.imported_modules) == 1

    def verify_multi_dso_mod_export(obj_format):
        for device in ["llvm"]:
            if not tvm.testing.device_enabled(device):
                print("skip because %s is not enabled..." % device)
                return

        A = te.placeholder((1024,), name="A")
        B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
        s = te.create_schedule(B.op)
        mod0 = tvm.build(s, [A, B], "llvm", name="myadd0")
        mod1 = tvm.build(s, [A, B], "llvm", name="myadd1")

        from tvm.contrib import utils

        temp = utils.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)

        mod0.import_module(mod1)
        mod0.export_library(path_lib)
        loaded_lib = tvm.runtime.load_module(path_lib)
        assert loaded_lib.type_key == "library"
        # dso modules are merged
        assert len(loaded_lib.imported_modules) == 0

    def verify_json_import_dso(obj_format):
        for device in ["llvm"]:
            if not tvm.testing.device_enabled(device):
                print("skip because %s is not enabled..." % device)
                return

        # Get subgraph Json.
        subgraph_json = (
            "json_rt_0\n"
            + "input 0 10 10\n"
            + "input 1 10 10\n"
            + "input 2 10 10\n"
            + "input 3 10 10\n"
            + "add 4 inputs: 0 1 shape: 10 10\n"
            + "sub 5 inputs: 4 2 shape: 10 10\n"
            + "mul 6 inputs: 5 3 shape: 10 10\n"
            + "json_rt_1\n"
            + "input 0 10 10\n"
            + "input 1 10 10\n"
            + "input 2 10 10\n"
            + "input 3 10 10\n"
            + "add 4 inputs: 0 1 shape: 10 10\n"
            + "sub 5 inputs: 4 2 shape: 10 10\n"
            + "mul 6 inputs: 5 3 shape: 10 10"
        )

        from tvm.contrib import utils

        temp = utils.tempdir()
        subgraph_path = temp.relpath("subgraph.examplejson")
        with open(subgraph_path, "w") as f:
            f.write(subgraph_json)

        # Get Json and module.
        A = te.placeholder((1024,), name="A")
        B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
        s = te.create_schedule(B.op)
        f = tvm.build(s, [A, B], "llvm", name="myadd")
        try:
            ext_lib = tvm.runtime.load_module(subgraph_path, "examplejson")
        except:
            print("skip because Loader of examplejson is not presented")
            return
        ext_lib.import_module(f)
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        ext_lib.export_library(path_lib)
        lib = tvm.runtime.load_module(path_lib)
        assert lib.type_key == "examplejson"
        assert lib.imported_modules[0].type_key == "library"

    def verify_multi_c_mod_export():
        from shutil import which

        if which("gcc") is None:
            print("Skip test because gcc is not available.")

        for device in ["llvm"]:
            if not tvm.testing.device_enabled(device):
                print("skip because %s is not enabled..." % device)
                return

        synthetic_mod, synthetic_params = relay.testing.synthetic.get_workload()
        with tvm.transform.PassContext(opt_level=3):
            _, synthetic_cpu_lib, _ = relay.build_module.build(
                synthetic_mod, "llvm", params=synthetic_params
            )

        A = te.placeholder((1024,), name="A")
        B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
        s = te.create_schedule(B.op)
        f = tvm.build(s, [A, B], "c", name="myadd")
        engine_module = generate_engine_module()
        from tvm.contrib import utils

        temp = utils.tempdir()
        file_name = "deploy_lib.so"
        path_lib = temp.relpath(file_name)
        synthetic_cpu_lib.import_module(f)
        synthetic_cpu_lib.import_module(engine_module)
        kwargs = {"options": ["-O2", "-std=c++14", "-I" + header_file_dir_path.relpath("")]}
        synthetic_cpu_lib.export_library(path_lib, fcompile=False, **kwargs)
        loaded_lib = tvm.runtime.load_module(path_lib)
        assert loaded_lib.type_key == "library"
        # dso modules are merged
        assert len(loaded_lib.imported_modules) == 0

    for obj_format in [".so", ".tar"]:
        verify_gpu_mod_export(obj_format)
        verify_multi_dso_mod_export(obj_format)
        verify_json_import_dso(obj_format)

    verify_multi_c_mod_export()


if __name__ == "__main__":
    test_mod_export()
