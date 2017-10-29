cc_library(
    name = "main",
    srcs = glob([
        "benchmark-1.2.0/src/*.cc",
        "benchmark-1.2.0/src/*.h",
    ]),
    hdrs = glob([
        "benchmark-1.2.0/include/**/*.h",
    ]),
    copts = [
        "-Iexternal/benchmark/benchmark-1.2.0/include",
        "-DHAVE_STD_REGEX",
    ],
    includes = ["benchmark-1.2.0/include"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)
