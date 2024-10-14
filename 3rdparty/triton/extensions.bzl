load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//3rdparty/triton:workspace.bzl", "triton_configure")

def impl_(ctx):
    http_archive(
        name = "triton-raw",
        urls = [
            "https://github.com/triton-lang/triton/archive/c120c4cc067d002c508f960d1b501d28342d417b.zip",
        ],
        sha256 = "bb119c472b7f2eb10b59d05fc061c4d014adfc38bb6f5e0b826d7da940412958",
        strip_prefix = "triton-c120c4cc067d002c508f960d1b501d28342d417b",
        build_file_content = "# empty",
    )
    triton_configure(name = "triton")


triton = module_extension(
    implementation = impl_,
)

