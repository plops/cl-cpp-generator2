cc_library(
    name = "my_lib",
    srcs = ["my_lib/MyClass.cpp"],
    hdrs = ["my_lib/MyClass.h"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    visibility = ["//visibility:public"],
    deps = ["//cc:my_lib"],
)
