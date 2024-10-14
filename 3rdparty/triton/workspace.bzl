def impl_(repository_ctx):
    repository_ctx.file(
        "BUILD.bazel",
        content = "# empty",
    )

triton_configure = repository_rule(
    implementation = impl_,
    local = True,
    configure = True,
)

