package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nvbench_headers",
    hdrs = glob(["install/include/**"]),
    strip_include_prefix = "install/include",
)

cc_library(
    name = "nvbench",
    srcs = ["install/lib/libnvbench.so"],
    deps = [
        ":nvbench_headers",
    ],
)

cc_library(
    name = "nvbench_main",
    srcs = ["install/lib/objects-Release/nvbench.main/main.cu.o"],
    deps = [
        ":nvbench_headers",
        ":nvbench",
    ]
)

