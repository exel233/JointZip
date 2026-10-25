// Thin process wrapper around the untouched SZ3 v3.3.2 public C++ API.
// The upstream CLI allocates 2*N*sizeof(T), which can be below the API bound.
// The allocating public overload handles that bound and emits the same format.
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "SZ3/api/sz.hpp"

template <typename T>
std::vector<T> read_file(const char* path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open input");
    const auto size = input.tellg();
    if (size <= 0 || static_cast<size_t>(size) % sizeof(T))
        throw std::runtime_error("invalid input byte length");
    std::vector<T> values(static_cast<size_t>(size) / sizeof(T));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(values.data()), size))
        throw std::runtime_error("incomplete input read");
    return values;
}

void write_file(const char* path, const char* data, size_t size) {
    std::ofstream output(path, std::ios::binary);
    if (!output || !output.write(data, size)) throw std::runtime_error("incomplete output write");
}

int main(int argc, char** argv) {
    try {
        if (argc == 6 && std::string(argv[1]) == "c") {
            const size_t n = std::stoull(argv[4]);
            const double epsilon = std::stod(argv[5]);
            auto input = read_file<float>(argv[2]);
            if (!n || n != input.size() || !std::isfinite(epsilon) || epsilon <= 0)
                throw std::runtime_error("invalid shape or epsilon");
            SZ3::Config config(n);
            config.errorBoundMode = SZ3::EB_ABS;
            config.absErrorBound = epsilon;
            size_t size = 0;
            std::unique_ptr<char[]> output(SZ_compress<float>(config, input.data(), size));
            write_file(argv[3], output.get(), size);
        } else if (argc == 4 && std::string(argv[1]) == "d") {
            auto input = read_file<char>(argv[2]);
            SZ3::Config config;
            std::unique_ptr<float[]> output(SZ_decompress<float>(config, input.data(), input.size()));
            write_file(argv[3], reinterpret_cast<const char*>(output.get()), config.num * sizeof(float));
        } else {
            throw std::runtime_error("usage: sz3_api c input.f32 output.sz3 N epsilon | d input.sz3 output.f32");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
