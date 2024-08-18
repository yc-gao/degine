package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "TorchToMLIR",
    srcs = glob(["*.cc", "*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@torch//:torch",
        "@llvm-project//mlir:CAPIIR",
        "@torch-mlir//:TorchCAPI",
    ],
)

