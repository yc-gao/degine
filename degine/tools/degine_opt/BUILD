cc_binary(
    name = "degine_opt",
    srcs = glob(["*.cc", "*.h"]),
    deps = [
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:AllExtensions",
        "@llvm-project//mlir:AllToLLVMIRTranslations",
        "@llvm-project//mlir:MlirOptLib",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:linalg_passes",
        "@stablehlo//:tosa_passes",
        "@torch-mlir//:TorchMLIRInitAll",
    ]
)

