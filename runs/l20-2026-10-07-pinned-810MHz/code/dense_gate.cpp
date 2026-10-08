// Reuse the pinned backend test implementation outside the upstream test tree.
#define main upstream_backend_tests_main
#include BACKEND_TEST_SOURCE
#undef main

int main(int argc, char ** argv) {
    const bool smoke = argc == 2 && std::string(argv[1]) == "--cpu-smoke";
    ggml_backend_load_all();
    ggml_backend_ptr target(ggml_backend_init_by_name(smoke ? "CPU" : "CUDA0", nullptr));
    ggml_backend_ptr reference(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!target || !reference) {
        fprintf(stderr, "Required backend unavailable\n");
        return 2;
    }
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(reference.get()));
    using use_ref_fn = void (*)(ggml_backend_t, bool);
    auto use_ref = (use_ref_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_use_ref");
    if (!use_ref) {
        fprintf(stderr, "CPU reference implementation unavailable\n");
        return 3;
    }
    use_ref(reference.get(), true);
    auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_threads) {
        set_threads(reference.get(), 4);
    }
    const std::vector<std::pair<int64_t, int64_t>> shapes = smoke
        ? std::vector<std::pair<int64_t, int64_t>>{{256, 16}}
        : std::vector<std::pair<int64_t, int64_t>>{
            {4096, 4096}, {4096, 1024}, {4096, 14336}, {14336, 4096},
            {3072, 3072}, {3072, 1024}, {3072, 8192}, {8192, 3072},
            {2048, 2048}, {2048, 512}, {2048, 8192}, {8192, 2048}};
    auto printer = create_printer(CONSOLE);
    int passed = 0;
    int total = 0;
    for (auto shape : shapes) {
        for (auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
            for (int n = 2; n <= 8; ++n) {
                test_mul_mat test(type, GGML_TYPE_F32, shape.second, n, shape.first, {1, 1}, {1, 1});
                const auto status = test.eval(target.get(), reference.get(), "MUL_MAT", printer.get());
                ++total;
                if (status == test_status_t::OK) {
                    ++passed;
                } else {
                    fprintf(stderr, "DENSE_GATE_FAILURE k=%lld m=%lld n=%d type=%s\n",
                            (long long) shape.first, (long long) shape.second, n, ggml_type_name(type));
                }
            }
        }
    }
    printf("DENSE_GATE_RESULT passed=%d total=%d cpu_smoke=%d\n", passed, total, smoke);
    ggml_quantize_free();
    return passed == total ? 0 : 1;
}
