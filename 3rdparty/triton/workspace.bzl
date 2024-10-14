DEFAULT_OVERLAY_PATH = "triton-overlay"

def _overlay_directories(repository_ctx):
    script_path = repository_ctx.path(Label("//3rdparty/triton:overlay_directories.py"))

    src_path = repository_ctx.path(Label("@triton-raw//:WORKSPACE")).dirname
    overlay_path = repository_ctx.path(Label("//3rdparty/triton:" + DEFAULT_OVERLAY_PATH))

    python_bin = repository_ctx.which("python3")
    if not python_bin:
        # Windows typically just defines "python" as python3. The script itself
        # contains a check to ensure python3.
        python_bin = repository_ctx.which("python")

    if not python_bin:
        fail("Failed to find python3 binary")

    cmd = [
        python_bin,
        script_path,
        "--src",
        src_path,
        "--overlay",
        overlay_path,
        "--target",
        ".",
    ]
    exec_result = repository_ctx.execute(cmd, timeout = 20)

    if exec_result.return_code != 0:
        fail(("Failed to execute overlay script: '{cmd}'\n" +
              "Exited with code {return_code}\n" +
              "stdout:\n{stdout}\n" +
              "stderr:\n{stderr}\n").format(
            cmd = " ".join([str(arg) for arg in cmd]),
            return_code = exec_result.return_code,
            stdout = exec_result.stdout,
            stderr = exec_result.stderr,
        ))

def impl_(repository_ctx):
    _overlay_directories(repository_ctx)
    # repository_ctx.file(
    #     "BUILD.bazel",
    #     content = "# empty",
    # )

triton_configure = repository_rule(
    implementation = impl_,
    local = True,
    configure = True,
)

