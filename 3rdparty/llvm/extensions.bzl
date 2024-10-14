load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//3rdparty/llvm:workspace.bzl", "llvm_configure")

def impl_(ctx):
    http_archive(
        name = "llvm-raw",
        urls = [
            "https://github.com/llvm/llvm-project/archive/df0864e761107b07e38f5503e0cbee0cebb4c5e8.zip",
        ],
        sha256 = "cd31721d540164019c8470be1343d149feb27dcdb3ad59ff22a786391d0c89a1",
        strip_prefix = "llvm-project-df0864e761107b07e38f5503e0cbee0cebb4c5e8",
        build_file_content = "# empty",
    )
    http_archive(
        name = "llvm_zlib",
        urls = [
            "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
        ],
        sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
        strip_prefix = "zlib-ng-2.0.7",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
    )
    http_archive(
        name = "llvm_zstd",
        urls = [
            "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
        ],
        sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
        strip_prefix = "zstd-1.5.2",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
    )
    llvm_configure(name = "llvm-project")


llvm = module_extension(
    implementation = impl_,
)
