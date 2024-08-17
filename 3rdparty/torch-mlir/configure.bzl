def _impl(repository_ctx):
    self_dir = repository_ctx.path(
        Label("//3rdparty/torch-mlir:configure.bzl")).dirname
    overlay_path = self_dir.get_child("torch-mlir-overlay")

    repository_ctx.download_and_extract(
        stripPrefix=repository_ctx.attr.strip_prefix,
        url=repository_ctx.attr.url,
        sha256=repository_ctx.attr.sha256
    )

    cmd = ["cp", "-af", "-T", overlay_path, "."]
    exec_result = repository_ctx.execute(cmd, timeout=20)
    if exec_result.return_code != 0:
        fail(("Failed to execute overlay script: '{cmd}'\n" +
              "Exited with code {return_code}\n" +
              "stdout:\n{stdout}\n" +
              "stderr:\n{stderr}\n").format(
            cmd=" ".join([str(arg) for arg in cmd]),
            return_code=exec_result.return_code,
            stdout=exec_result.stdout,
            stderr=exec_result.stderr,
        ))


torch_mlir_configure = repository_rule(
    implementation=_impl,
    attrs={
        "url": attr.string(mandatory=True),
        "sha256": attr.string(),
        "strip_prefix": attr.string(),
    }
)
