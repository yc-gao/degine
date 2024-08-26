package(default_visibility = ["//visibility:public"])

cc_library(
    name = "torch_headers",
    hdrs = glob(["include/**/*.h"]),
    includes = ["include", "include/torch/csrc/api/include"],
)

cc_import(
    name = "torch_c10",
    shared_library = "lib/libc10.so",
    deps = [":torch_headers"],
)

cc_import(
    name = "torch_cpu",
    shared_library = "lib/libtorch_cpu.so",
    deps = [":torch_headers"],
)

cc_library(
    name = "torch",
    deps = [
        ":torch_c10",
        ":torch_cpu",
    ]
)

