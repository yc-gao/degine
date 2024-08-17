cc_binary(
    name = "degine-opt",
    srcs = glob(["*.cc", "*.h"]),
    deps = [
        "//degine/proto:onnx_cc_proto",
        "//degine/Conversion/TorchToMLIR",
        "@torch//:torch",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:AllToLLVMIRTranslations",
    ]
)
