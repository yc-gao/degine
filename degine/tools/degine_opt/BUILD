cc_binary(
    name = "degine_opt",
    srcs = glob(["*.cc", "*.h"]),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:AllToLLVMIRTranslations",
        "@stablehlo//:register",
        "@torch-mlir//:TorchMLIRInitAll",
        "//degine/proto:onnx_cc_proto",
        "@torch//:torch",
    ]
)
