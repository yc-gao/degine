load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def torch_configure(name = "torch"):
    maybe(
        http_archive,
        name = name,
        build_file = "//3rdparty/torch:torch.BUILD",
        strip_prefix = "libtorch",
        sha256 = "f739db778882e8826b92ab9e140c9c66a05041c621121386aae718c0110679fc",
        url = "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu118.zip",
    )
