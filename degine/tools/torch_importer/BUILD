cc_binary(
    name = "torch_importer",
    srcs = glob(["*.cc", "*.cpp", "*.h"]),
    deps = [
        "@torch//:torch",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIR",
        "@torch-mlir//:TorchCAPI",
    ]
)

