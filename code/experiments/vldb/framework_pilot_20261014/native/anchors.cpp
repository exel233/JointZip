// Thin pilot bridge: use the unchanged production QoI analysis and validator.
#include <bits/stdc++.h>
#include "../../../../tools/qpet_correlation_guided_20261013_v1/CorrelationQoI.hpp"

std::vector<float> read_f32(const char* path, size_t count) {
  std::ifstream input(path, std::ios::binary);
  std::vector<float> data(count);
  if (!input.read(reinterpret_cast<char*>(data.data()), count * sizeof(float)))
    throw std::runtime_error("truncated input");
  char extra;
  if (input.read(&extra, 1)) throw std::runtime_error("trailing input");
  return data;
}
int main(int argc, char** argv) {
  try {
    if (argc != 9) throw std::runtime_error("anchors original.f32 output.f64 W T epsilon tau rho max_lag");
    size_t w = std::stoull(argv[3]), t = std::stoull(argv[4]);
    double e = std::stod(argv[5]), a = std::stod(argv[6]), r = std::stod(argv[7]);
    size_t k = std::stoull(argv[8]);
    auto source = read_f32(argv[1], w * t);
    std::ofstream output(argv[2], std::ios::binary);
    for (size_t i = 0; i < w; ++i) {
      std::vector<float> row(source.begin() + i*t, source.begin() + (i+1)*t);
      double anchor = e;
      if (r >= 0 || (a > 0 && a < 2)) {
        qpet_correlation::CorrelationQoI qoi(row, 1, t, e,
          a > 0 && a < 2 ? a : -1, r, k, e,
          qpet_correlation::BoundEstimator::qpet_probabilistic,
          qpet_correlation::BoundAllocation::uniform,
          qpet_correlation::ComplianceMode::posthoc, 1.0, .999, 2.0);
        anchor = qoi.local_bounds().front();
        for (double b : qoi.local_bounds())
          if (b != anchor) throw std::runtime_error("nonuniform anchor");
      }
      output.write(reinterpret_cast<char*>(&anchor), sizeof(anchor));
    }
    if (!output) throw std::runtime_error("output failure");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
