DEFAULT_TARGETS = [
    "AArch64",
    "AMDGPU",
    "ARM",
    "AVR",
    "BPF",
    "Hexagon",
    "Lanai",
    "LoongArch",
    "Mips",
    "MSP430",
    "NVPTX",
    "PowerPC",
    "RISCV",
    "Sparc",
    "SystemZ",
    "VE",
    "WebAssembly",
    "X86",
    "XCore",
]


def _overlay_directories(repository_ctx):
    self_dir = repository_ctx.path(
        Label("//3rdparty/llvm-project:configure.bzl")).dirname
    overlay_path = self_dir.get_child("llvm-project-overlay")

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


def _extract_cmake_settings(repository_ctx, llvm_cmake):
    # The list to be written to vars.bzl
    # `CMAKE_CXX_STANDARD` may be used from WORKSPACE for the toolchain.
    c = {
        "CMAKE_CXX_STANDARD": None,
        "LLVM_VERSION_MAJOR": None,
        "LLVM_VERSION_MINOR": None,
        "LLVM_VERSION_PATCH": None,
        "LLVM_VERSION_SUFFIX": None,
    }

    # It would be easier to use external commands like sed(1) and python.
    # For portability, the parser should run on Starlark.
    llvm_cmake_path = llvm_cmake
    # llvm_cmake_path = repository_ctx.path(
    #     Label("//:" + llvm_cmake))
    for line in repository_ctx.read(llvm_cmake_path).splitlines():
        # Extract "set ( FOO bar ... "
        setfoo = line.partition("(")
        if setfoo[1] != "(":
            continue
        if setfoo[0].strip().lower() != "set":
            continue

        # `kv` is assumed as \s*KEY\s+VAL\s*\).*
        # Typical case is like
        #   LLVM_REQUIRED_CXX_STANDARD 17)
        # Possible case -- It should be ignored.
        #   CMAKE_CXX_STANDARD ${...} CACHE STRING "...")
        kv = setfoo[2].strip()
        i = kv.find(" ")
        if i < 0:
            continue
        k = kv[:i]

        # Prefer LLVM_REQUIRED_CXX_STANDARD instead of CMAKE_CXX_STANDARD
        if k == "LLVM_REQUIRED_CXX_STANDARD":
            k = "CMAKE_CXX_STANDARD"
            c[k] = None
        if k not in c:
            continue

        # Skip if `CMAKE_CXX_STANDARD` is set with
        # `LLVM_REQUIRED_CXX_STANDARD`.
        # Then `v` will not be desired form, like "${...} CACHE"
        if c[k] != None:
            continue

        # Pick up 1st word as the value.
        # Note: It assumes unquoted word.
        v = kv[i:].strip().partition(")")[0].partition(" ")[0]
        c[k] = v

    # Synthesize `LLVM_VERSION` for convenience.
    c["LLVM_VERSION"] = "{}.{}.{}".format(
        c["LLVM_VERSION_MAJOR"],
        c["LLVM_VERSION_MINOR"],
        c["LLVM_VERSION_PATCH"],
    )

    c["PACKAGE_VERSION"] = "{}.{}.{}{}".format(
        c["LLVM_VERSION_MAJOR"],
        c["LLVM_VERSION_MINOR"],
        c["LLVM_VERSION_PATCH"],
        c["LLVM_VERSION_SUFFIX"],
    )

    return c


def _write_dict_to_file(repository_ctx, filepath, header, vars):
    # (fci + individual vars) + (fcd + dict items) + (fct)
    fci = header
    fcd = "\nllvm_vars={\n"
    fct = "}\n"

    for k, v in vars.items():
        fci += '{} = "{}"\n'.format(k, v)
        fcd += '    "{}": "{}",\n'.format(k, v)

    repository_ctx.file(filepath, content=fci + fcd + fct)


def _impl(repository_ctx):
    _overlay_directories(repository_ctx)

    llvm_cmake = "llvm/CMakeLists.txt"
    vars = _extract_cmake_settings(
        repository_ctx,
        llvm_cmake,
    )

    # Grab version info and merge it with the other vars
    version = _extract_cmake_settings(
        repository_ctx,
        "cmake/Modules/LLVMVersion.cmake",
    )
    version = {k: v for k, v in version.items() if v != None}
    vars.update(version)

    _write_dict_to_file(
        repository_ctx,
        filepath="vars.bzl",
        header="# Generated from {}\n\n".format(llvm_cmake),
        vars=vars,
    )

    # Create a starlark file with the requested LLVM targets.
    targets = repository_ctx.attr.targets
    repository_ctx.file(
        "llvm/targets.bzl",
        content="llvm_targets = " + str(targets),
        executable=False,
    )


llvm_configure = repository_rule(
    implementation=_impl,
    attrs={
        "url": attr.string(mandatory=True),
        "sha256": attr.string(),
        "strip_prefix": attr.string(),
        "targets": attr.string_list(default=DEFAULT_TARGETS),
    }
)
