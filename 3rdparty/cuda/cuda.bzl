def check_cuda(cuda_home):
    # TODO: impl
    return True


def _impl(repository_ctx):
    cuda_toolkit_root = repository_ctx.attr.cuda_toolkit_root

    if not cuda_toolkit_root:
        root = repository_ctx.which("nvcc").dirname.dirname
        if check_cuda(root):
            cuda_toolkit_root = root

    if not cuda_toolkit_root:
        fail("Failed to find cuda toolkit root\n")

    repository_ctx.symlink(cuda_toolkit_root.get_child("include"), "include")
    repository_ctx.symlink(cuda_toolkit_root.get_child("bin"), "bin")
    repository_ctx.symlink(cuda_toolkit_root.get_child("lib64"), "lib64")
    # TODO:
    repository_ctx.symlink("/usr/lib/x86_64-linux-gnu/libcuda.so", "lib/libcuda.so")

    repository_ctx.symlink(repository_ctx.path(
        Label("//3rdparty/cuda:BUILD.template")), "BUILD")


cuda_configure = repository_rule(
    implementation=_impl,
    attrs={
        "cuda_toolkit_root": attr.string(default="")
    },
)
