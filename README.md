```shell
bazel --output_base=.cache/bazel \
    build --config=generic_gcc //degine/...
bazel --output_base=.cache/bazel \
    run --config=generic_gcc //degine:refresh_compile_commands
```
