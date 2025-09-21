cc_library(
    name = "my_lib",
    srcs = ["cc/my_lib/MyClass.cpp"],
    hdrs = ["cc/my_lib/MyClass.h"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "main",
    srcs = ["cc/main.cpp"],
    visibility = ["//visibility:public"],
    deps = ["//:my_lib"],
)
