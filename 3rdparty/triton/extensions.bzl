load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def impl_(ctx):
    http_archive(
        name = "triton-raw",
        urls = [
            "https://github.com/triton-lang/triton/archive/refs/tags/v2.1.0.tar.gz",
        ],
        sha256 = "4338ca0e80a059aec2671f02bfc9320119b051f378449cf5f56a1273597a3d99",
        strip_prefix = "triton-2.1.0",
        build_file_content = "# empty",
    )


triton = module_extension(
    implementation = impl_,
)

