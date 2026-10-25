#include "CorrelationQoI.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message)
{
  if (!condition)
    throw std::runtime_error(message);
}

std::vector<float> signal(std::size_t count)
{
  auto values = std::vector<float>(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto x = static_cast<double>(index);
    values[index] = static_cast<float>(
        0.7 * std::sin(0.13 * x) + 0.2 * std::cos(0.037 * x) + 0.002 * x);
  }
  return values;
}

void test_analytical_acf_gradient()
{
  auto values = signal(96);
  constexpr std::size_t lag = 7;
  const auto analytical = qpet_correlation::acf_gradient(values.data(), values.size(), lag);
  constexpr float step = 1e-3F;
  double largest_error = 0.0;
  for (std::size_t index = 0; index < values.size(); ++index) {
    auto plus = values;
    auto minus = values;
    plus[index] += step;
    minus[index] -= step;
    const auto numerical =
        (qpet_correlation::acf(plus.data(), plus.size(), lag) -
         qpet_correlation::acf(minus.data(), minus.size(), lag)) /
        (2.0 * step);
    largest_error = std::max(largest_error, std::abs(numerical - analytical[index]));
  }
  require(largest_error < 2e-5, "analytical ACF gradient failed finite-difference check");
  require(
      std::abs(std::accumulate(analytical.begin(), analytical.end(), 0.0)) < 1e-10,
      "ACF gradient is not invariant to a constant offset");
}

void test_constant_channel_becomes_lossless()
{
  auto values = std::vector<float>(128, 3.25F);
  qpet_correlation::CorrelationQoI qoi(values, 1, 128, 0.2, 0.01, 0.99, 8, 0.2);
  require(
      std::all_of(qoi.local_bounds().begin(), qoi.local_bounds().end(), [](double bound) {
        return bound == 0.0;
      }),
      "near-constant channels must be encoded losslessly");
}

void test_online_constant_pearson_semantics()
{
  auto values = std::vector<float>(32, 3.25F);
  qpet_correlation::CorrelationQoI qoi(
      values,
      1,
      values.size(),
      0.2,
      -1.0,
      0.99,
      std::numeric_limits<std::size_t>::max(),
      0.2);
  (void)qoi.interpret_eb(values.data() + 7, 7);
  require(
      qoi.check_compliance(values[7], values[7]),
      "online Pearson rejected an exact constant-channel value");
  (void)qoi.interpret_eb(values.data() + 8, 8);
  require(
      !qoi.check_compliance(values[8], values[8] + 0.01F),
      "online Pearson accepted a changed constant-channel value");
  require(
      qoi.accepted_count() == 1 && qoi.correction_count() == 1,
      "online constant-channel decisions were not counted consistently");
}

void test_pearson_only_repair_ignores_disabled_acf_lag()
{
  auto values = signal(32);
  auto decoded = values;
  for (std::size_t index = 0; index < decoded.size(); ++index)
    decoded[index] += static_cast<float>(
        0.05 * std::sin(0.71 * static_cast<double>(index)));
  const auto repair = qpet_correlation::repair_constraints(
      values,
      decoded,
      1,
      values.size(),
      -1.0,
      0.999,
      std::numeric_limits<std::size_t>::max(),
      true);
  require(repair.corrected_values > 0, "Pearson-only repair changed no values");
  const auto report = qpet_correlation::verify_constraints(
      values,
      decoded,
      1,
      values.size(),
      0.1,
      -1.0,
      0.999,
      std::numeric_limits<std::size_t>::max());
  require(report.all_ok(), "Pearson-only repair failed with disabled ACF");
}

void test_gradient_balanced_qpet_bounds()
{
  auto values = signal(128);
  constexpr double tolerance = 0.05;
  constexpr double confidence = 0.999;
  constexpr double k = 2.0;
  qpet_correlation::CorrelationQoI uniform(
      values,
      1,
      values.size(),
      0.5,
      tolerance,
      -1.0,
      8,
      0.5,
      qpet_correlation::BoundEstimator::qpet_probabilistic,
      qpet_correlation::BoundAllocation::uniform,
      qpet_correlation::ComplianceMode::posthoc,
      1.0,
      confidence,
      k);
  qpet_correlation::CorrelationQoI balanced(
      values,
      1,
      values.size(),
      0.5,
      tolerance,
      -1.0,
      8,
      0.5,
      qpet_correlation::BoundEstimator::qpet_probabilistic,
      qpet_correlation::BoundAllocation::gradient_balanced,
      qpet_correlation::ComplianceMode::posthoc,
      1.0,
      confidence,
      k);
  require(
      std::all_of(
          balanced.local_bounds().begin(),
          balanced.local_bounds().end(),
          [](double bound) { return std::isfinite(bound) && bound > 0.0; }),
      "gradient-balanced QPET produced an invalid local bound");
  bool observed_larger = false;
  bool observed_smaller = false;
  for (std::size_t index = 0; index < values.size(); ++index) {
    observed_larger = observed_larger ||
        balanced.local_bounds()[index] > uniform.local_bounds()[index];
    observed_smaller = observed_smaller ||
        balanced.local_bounds()[index] < uniform.local_bounds()[index];
  }
  require(
      observed_larger && observed_smaller,
      "gradient-balanced QPET did not redistribute the uniform error budget");

  const auto log_term = std::log(2.0 / (1.0 - confidence));
  for (std::size_t lag = 1; lag <= 8; ++lag) {
    const auto gradient = qpet_correlation::acf_gradient(
        values.data(), values.size(), lag);
    long double l1 = 0.0L;
    long double l2_squared = 0.0L;
    long double allocated = 0.0L;
    for (std::size_t index = 0; index < gradient.size(); ++index) {
      l1 += std::abs(gradient[index]);
      l2_squared += static_cast<long double>(gradient[index]) * gradient[index];
      allocated += std::abs(gradient[index]) * balanced.local_bounds()[index];
    }
    const auto deterministic = tolerance / static_cast<double>(l1);
    const auto probabilistic = k * tolerance * std::sqrt(
        1.0 / (2.0 * static_cast<double>(l2_squared) * log_term));
    const auto budget = std::max(deterministic, probabilistic) *
        static_cast<double>(l1);
    require(
        static_cast<double>(allocated) <= budget * (1.0 + 1e-12),
        "gradient-balanced QPET exceeded the matched first-order budget");
  }
}

void test_exact_validator_rejects_a_violation()
{
  auto values = signal(128);
  qpet_correlation::CorrelationQoI qoi(values, 1, 128, 0.05, 0.002, 0.999, 16, 0.05);
  (void)qoi.interpret_eb(values.data() + 31, 31);
  require(
      !qoi.check_compliance(values[31], values[31] + 0.051F),
      "exact validator accepted a pointwise-bound violation");
  require(qoi.correction_count() == 1, "rejected reconstruction was not counted");
  (void)qoi.interpret_eb(values.data() + 31, 31);
  require(qoi.check_compliance(values[31], values[31]), "exact reconstruction was rejected");
  require(qoi.accepted_count() == 1, "accepted reconstruction was not counted");
}

void test_independent_constraint_report()
{
  auto values = signal(128);
  auto decoded = values;
  for (std::size_t index = 0; index < decoded.size(); ++index)
    decoded[index] += static_cast<float>(2e-4 * std::sin(static_cast<double>(index)));
  const auto report = qpet_correlation::verify_constraints(
      values, decoded, 1, 128, 1e-3, 1e-3, 0.999, 16);
  require(report.all_ok(), "independent validator rejected a compliant reconstruction");
  decoded[17] += 0.01F;
  const auto failed = qpet_correlation::verify_constraints(
      values, decoded, 1, 128, 1e-3, 1e-3, 0.999, 16);
  require(!failed.pointwise_ok, "independent validator missed a pointwise-bound violation");
}

void test_independent_validator_rejects_nonfinite_values()
{
  auto values = signal(128);
  auto decoded = values;
  decoded[23] = std::numeric_limits<float>::quiet_NaN();
  const auto nan_report = qpet_correlation::verify_constraints(
      values, decoded, 1, 128, 0.1, 0.01, -1.0, 16);
  require(!nan_report.all_ok(), "independent validator accepted NaN in ACF-only mode");
  require(
      !nan_report.pointwise_ok && !nan_report.acf_ok && !nan_report.correlation_ok,
      "non-finite reconstruction did not invalidate every hard-constraint flag");
  decoded[23] = std::numeric_limits<float>::infinity();
  const auto infinity_report = qpet_correlation::verify_constraints(
      values, decoded, 1, 128, 0.1, -1.0, 0.99, 16);
  require(!infinity_report.all_ok(), "independent validator accepted infinity");
}

void test_canonical_degenerate_pearson_semantics()
{
  auto below = std::vector<float>{-1e-8F, 1e-8F, -1e-8F, 1e-8F};
  auto below_changed = below;
  below_changed[0] = 0.0F;
  const auto below_report = qpet_correlation::verify_constraints(
      below, below_changed, 1, below.size(), 1.0, -1.0, 0.99, 0);
  require(
      !below_report.correlation_ok && below_report.min_series_correlation == 0.0,
      "below-floor source accepted a numerically changed reconstruction");

  auto above = std::vector<float>{-2e-8F, 2e-8F, -2e-8F, 2e-8F};
  auto above_scaled = above;
  for (auto& value : above_scaled)
    value *= 2.0F;
  const auto above_report = qpet_correlation::verify_constraints(
      above, above_scaled, 1, above.size(), 1.0, -1.0, 0.99, 0);
  require(
      above_report.correlation_ok && above_report.min_series_correlation > 0.999999,
      "above-floor source did not use ordinary Pearson correlation");

  auto zeros = std::vector<float>(8, 0.0F);
  auto signed_zeros = zeros;
  for (std::size_t index = 0; index < signed_zeros.size(); ++index) {
    if (index % 2 != 0)
      signed_zeros[index] = -0.0F;
  }
  const auto signed_zero_report = qpet_correlation::verify_constraints(
      zeros, signed_zeros, 1, zeros.size(), 1.0, -1.0, 0.99, 0);
  require(
      signed_zero_report.correlation_ok &&
          signed_zero_report.min_series_correlation == 1.0,
      "degenerate Pearson treated signed zero as a numeric change");

  auto subnormal = zeros;
  subnormal[3] = std::numeric_limits<float>::denorm_min();
  const auto subnormal_report = qpet_correlation::verify_constraints(
      zeros, subnormal, 1, zeros.size(), 1.0, -1.0, 0.99, 0);
  require(
      !subnormal_report.correlation_ok &&
          subnormal_report.min_series_correlation == 0.0,
      "degenerate Pearson hid a subnormal float32 change");
}

void test_posthoc_greedy_repair()
{
  auto values = signal(128);
  auto decoded = values;
  for (std::size_t index = 0; index < decoded.size(); ++index)
    decoded[index] += static_cast<float>(
        0.08 * std::sin(0.71 * static_cast<double>(index)));
  const auto before = qpet_correlation::verify_constraints(
      values, decoded, 1, 128, 0.1, 0.002, 0.999, 16);
  require(!before.all_ok(), "repair test distortion unexpectedly satisfies all constraints");
  const auto distorted = decoded;
  const auto repair = qpet_correlation::repair_constraints(
      values, decoded, 1, 128, 0.002, 0.999, 16, true);
  require(repair.corrected_series == 1, "repair did not count its affected series");
  require(
      repair.corrected_values > 0 && repair.corrected_values < values.size(),
      "greedy repair did not produce a sparse correction");
  for (std::size_t index = 0; index < decoded.size(); ++index) {
    require(
        decoded[index] == distorted[index] || decoded[index] == values[index],
        "repair created a value outside the QPET post-hoc correction model");
  }
  const auto after = qpet_correlation::verify_constraints(
      values, decoded, 1, 128, 0.1, 0.002, 0.999, 16);
  require(after.all_ok(), "greedy post-hoc repair failed a hard constraint");
}

void test_quantized_target_repair()
{
  auto values = signal(128);
  auto decoded = values;
  auto targets = values;
  for (std::size_t index = 0; index < decoded.size(); ++index) {
    decoded[index] += static_cast<float>(
        0.08 * std::sin(0.71 * static_cast<double>(index)));
    targets[index] += static_cast<float>(
        1e-4 * std::sin(0.37 * static_cast<double>(index)));
  }
  const auto distorted = decoded;
  const auto repair = qpet_correlation::repair_constraints(
      values, decoded, 1, 128, 0.002, 0.999, 16, true, &targets);
  require(repair.corrected_values > 0, "quantized-target repair changed no values");
  bool used_nonexact_target = false;
  for (std::size_t index = 0; index < decoded.size(); ++index) {
    require(
        decoded[index] == distorted[index] || decoded[index] == targets[index],
        "repair created a value outside the supplied target model");
    used_nonexact_target = used_nonexact_target ||
        (decoded[index] == targets[index] && targets[index] != values[index]);
  }
  require(used_nonexact_target, "repair ignored all non-exact targets");
  const auto report = qpet_correlation::verify_constraints(
      values, decoded, 1, 128, 0.1, 0.002, 0.999, 16);
  require(report.all_ok(), "quantized-target repair failed a hard constraint");
}

void test_pointwise_pre_repair_for_transformed_base()
{
  auto values = signal(128);
  auto decoded = values;
  decoded[17] += 0.2F;
  const auto repair = qpet_correlation::repair_constraints(
      values, decoded, 1, 128, 1.0, 0.0, 16, true, nullptr, 0.1);
  require(repair.corrected_values == 1, "point-wise pre-repair changed the wrong count");
  require(decoded[17] == values[17], "point-wise pre-repair did not restore the violation");
  const auto report = qpet_correlation::verify_constraints(
      values, decoded, 1, 128, 0.1, 1.0, 0.0, 16);
  require(report.all_ok(), "point-wise pre-repair failed the final validator");
}

void test_expected_repair_exhaustion_status()
{
  auto values = signal(32);
  auto decoded = values;
  for (std::size_t index = 0; index < decoded.size(); ++index)
    decoded[index] += static_cast<float>(
        0.08 * std::sin(0.71 * static_cast<double>(index)));
  const auto targets = decoded;
  const auto before = qpet_correlation::verify_constraints(
      values, decoded, 1, values.size(), 0.2, 0.002, 0.999, 8);
  require(!before.all_ok(), "exhaustion fixture unexpectedly satisfies constraints");
  const auto repair = qpet_correlation::repair_constraints(
      values,
      decoded,
      1,
      values.size(),
      0.002,
      0.999,
      8,
      true,
      &targets,
      0.2,
      false);
  require(!repair.feasible, "repair exhaustion did not return expected infeasible status");
  require(decoded == targets, "infeasible repair state changed an unavailable target");

  auto fail_closed_decoded = targets;
  auto threw = false;
  try {
    (void)qpet_correlation::repair_constraints(
        values,
        fail_closed_decoded,
        1,
        values.size(),
        0.002,
        0.999,
        8,
        true,
        &targets,
        0.2);
  }
  catch (const std::runtime_error&) {
    threw = true;
  }
  require(threw, "legacy repair exhaustion no longer fails closed");
}

}  // namespace

int main()
{
  test_analytical_acf_gradient();
  test_constant_channel_becomes_lossless();
  test_online_constant_pearson_semantics();
  test_pearson_only_repair_ignores_disabled_acf_lag();
  test_gradient_balanced_qpet_bounds();
  test_exact_validator_rejects_a_violation();
  test_independent_constraint_report();
  test_independent_validator_rejects_nonfinite_values();
  test_canonical_degenerate_pearson_semantics();
  test_posthoc_greedy_repair();
  test_quantized_target_repair();
  test_pointwise_pre_repair_for_transformed_base();
  test_expected_repair_exhaustion_status();
  std::cout << "CorrelationQoI analytical and hard-constraint tests passed\n";
  return 0;
}
