cc_binary(
    name = "degine_opt",
    srcs = glob(["*.cc", "*.h"]),
    deps = [
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:AllExtensions",
        "@llvm-project//mlir:AllToLLVMIRTranslations",
        "@llvm-project//mlir:MlirOptLib",
    ]
)

