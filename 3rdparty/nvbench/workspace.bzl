def _impl(repository_ctx):
    url = repository_ctx.attr.url
    strip_prefix = repository_ctx.attr.strip_prefix
    checksum = repository_ctx.attr.sha256
    repository_ctx.download_and_extract(
        url = url,
        output = 'repo',
        stripPrefix = strip_prefix,
    )

    result = repository_ctx.execute(["bash", "-c", "cmake -S repo -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=`pwd`/install"])
    print(result.stdout)
    if result.return_code:
        fail(result.stderr)

    result = repository_ctx.execute(["bash", "-c", "cmake --build build -j 8"])
    print(result.stdout)
    if result.return_code:
        fail(result.stderr)

    result = repository_ctx.execute(["bash", "-c", "cmake --install build"])
    print(result.stdout)
    if result.return_code:
        fail(result.stderr)

    repository_ctx.symlink(Label("//3rdparty/nvbench:nvbench.BUILD"), "BUILD.bazel")

nvbench_config = repository_rule(
    implementation=_impl,
    attrs={
        "url": attr.string(mandatory=True),
        "strip_prefix": attr.string(),
        "sha256": attr.string(),
    }
)
