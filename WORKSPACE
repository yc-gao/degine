workspace(name = "degine")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

# protobuf begin
http_archive(
    name = "rules_python",
    sha256 = "5868e73107a8e85d8f323806e60cad7283f34b32163ea6ff1020cf27abef6036",
    strip_prefix = "rules_python-0.25.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.25.0/rules_python-0.25.0.tar.gz",
)
http_archive(
    name = "com_google_protobuf",
    sha256 = "1ed0260a8509f43ee3aa66088f849c2b4655871676c053e80dbc4d129e2a643a",
    strip_prefix = "protobuf-3.27.3",
    url = "https://github.com/protocolbuffers/protobuf/archive/v3.27.3.tar.gz",
)
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()
# protobuf end

# llvm begin
http_archive(
    name = "llvm_zlib",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
    sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
    strip_prefix = "zlib-ng-2.0.7",
    url = "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
)
http_archive(
    name = "llvm_zstd",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
    sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
    strip_prefix = "zstd-1.5.2",
    url = "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
)
http_archive(
    name = "llvm-raw",
    build_file_content = "# empty",
    # sha256 = "7b252d54d93f583e32e601f2dcac3d54201231e140ffe65cc91bf089bb9eefa9",
    strip_prefix = "llvm-project-585523750e2bbe374d1cb3bf4ff9d53de29b9593",
    url = "https://github.com/llvm/llvm-project/archive/585523750e2bbe374d1cb3bf4ff9d53de29b9593.zip",
)
load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
llvm_configure(name = "llvm-project")
# llvm end

# torch-mlir begin
http_archive(
    name = "stablehlo",
    # sha256 = "a469ecc3d6747f9effdc1c7813568953dd1dc30070ca8f4f6f8a4d405e8c687e",
    strip_prefix = "stablehlo-c28d55e91b4a5daaff18a33ce7e9bbd0f171256a",
    url = "https://github.com/openxla/stablehlo/archive/c28d55e91b4a5daaff18a33ce7e9bbd0f171256a.zip",
)
http_archive(
    name = "com_github_bazelbuild_buildtools",
    sha256 = "ae34c344514e08c23e90da0e2d6cb700fcd28e80c02e23e4d5715dddcb42f7b3",
    strip_prefix = "buildtools-4.2.2",
    url = "https://github.com/bazelbuild/buildtools/archive/refs/tags/4.2.2.tar.gz",
)
http_archive(
    name = "torch-mlir-raw",
    build_file_content = "# empty",
    # sha256 = "3de03bf04fda933e22ff2b04d4f88194f175c846abc20856cd7d41d4e9fd342e",
    strip_prefix = "torch-mlir-56a663690ccd378182ea7dbf95b7b2a54463e3e9",
    url = "https://github.com/llvm/torch-mlir/archive/56a663690ccd378182ea7dbf95b7b2a54463e3e9.zip",
)
load("@torch-mlir-raw//utils/bazel:configure.bzl", "torch_mlir_configure")
torch_mlir_configure(name = "torch-mlir")
# torch-mlir end

http_archive(
    name = "torch",
    build_file_content = """
cc_library(
    name = "torch",
    srcs = glob(["lib/*.so", "lib/*.a"], exclude = ["lib/libnnapi_backend.so"]),
    hdrs = glob(["include/**/*.h"]),
    includes = ["include", "include/torch/csrc/api/include"],
    visibility = ["//visibility:public"],
)
    """,
    strip_prefix = "libtorch",
    sha256 = "f739db778882e8826b92ab9e140c9c66a05041c621121386aae718c0110679fc",
    url = "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu118.zip",
)
