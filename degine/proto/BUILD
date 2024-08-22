load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "onnx_proto",
    srcs = ["onnx.proto"],
)

cc_proto_library(
    name = "onnx_cc_proto",
    deps = [":onnx_proto"],
)

