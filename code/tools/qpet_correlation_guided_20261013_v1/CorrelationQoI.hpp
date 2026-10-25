#pragma once

#include "QoZ/qoi/QoI.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qpet_correlation {

constexpr double kDenominatorFloor = 1e-15;

inline double pearson(const float* left, const float* right, std::size_t count)
{
  if (count < 2)
    return 0.0;
  long double left_mean = 0.0L;
  long double right_mean = 0.0L;
  for (std::size_t index = 0; index < count; ++index) {
    left_mean += left[index];
    right_mean += right[index];
  }
  left_mean /= static_cast<long double>(count);
  right_mean /= static_cast<long double>(count);
  long double covariance = 0.0L;
  long double left_energy = 0.0L;
  long double right_energy = 0.0L;
  for (std::size_t index = 0; index < count; ++index) {
    const auto a = static_cast<long double>(left[index]) - left_mean;
    const auto b = static_cast<long double>(right[index]) - right_mean;
    covariance += a * b;
    left_energy += a * a;
    right_energy += b * b;
  }
  const auto denominator = std::sqrt(left_energy * right_energy);
  if (!(denominator > kDenominatorFloor))
    return 0.0;
  return static_cast<double>(covariance / denominator);
}

inline double acf(const float* series, std::size_t count, std::size_t lag)
{
  if (lag == 0 || lag >= count)
    throw std::invalid_argument("ACF lag must be in [1, count - 1]");
  return pearson(series, series + lag, count - lag);
}

inline std::vector<double> acf_gradient(
    const float* series, std::size_t count, std::size_t lag)
{
  if (lag == 0 || lag >= count)
    throw std::invalid_argument("ACF gradient lag must be in [1, count - 1]");
  const auto paired = count - lag;
  long double left_mean = 0.0L;
  long double right_mean = 0.0L;
  for (std::size_t index = 0; index < paired; ++index) {
    left_mean += series[index];
    right_mean += series[index + lag];
  }
  left_mean /= static_cast<long double>(paired);
  right_mean /= static_cast<long double>(paired);

  auto left_centered = std::vector<long double>(paired);
  auto right_centered = std::vector<long double>(paired);
  long double left_energy = 0.0L;
  long double right_energy = 0.0L;
  long double covariance = 0.0L;
  for (std::size_t index = 0; index < paired; ++index) {
    left_centered[index] = static_cast<long double>(series[index]) - left_mean;
    right_centered[index] = static_cast<long double>(series[index + lag]) - right_mean;
    left_energy += left_centered[index] * left_centered[index];
    right_energy += right_centered[index] * right_centered[index];
    covariance += left_centered[index] * right_centered[index];
  }
  const auto denominator = std::sqrt(left_energy * right_energy);
  if (!(denominator > kDenominatorFloor))
    return std::vector<double>(count, 0.0);

  const auto correlation = covariance / denominator;
  auto gradient = std::vector<double>(count, 0.0);
  for (std::size_t index = 0; index < paired; ++index) {
    const auto derivative_left =
        right_centered[index] / denominator - correlation * left_centered[index] / left_energy;
    const auto derivative_right =
        left_centered[index] / denominator - correlation * right_centered[index] / right_energy;
    gradient[index] += static_cast<double>(derivative_left);
    gradient[index + lag] += static_cast<double>(derivative_right);
  }
  return gradient;
}

struct ConstraintReport {
  double max_abs_error = 0.0;
  double max_acf_error = 0.0;
  double min_series_correlation = 1.0;
  bool pointwise_ok = true;
  bool acf_ok = true;
  bool correlation_ok = true;

  bool all_ok() const { return pointwise_ok && acf_ok && correlation_ok; }
};

enum class BoundEstimator : std::uint32_t {
  legacy_deterministic = 0,
  qpet_probabilistic = 1,
};

enum class BoundAllocation : std::uint32_t {
  uniform = 0,
  gradient_balanced = 1,
};

enum class ComplianceMode : std::uint32_t {
  online = 0,
  posthoc = 1,
};

inline ConstraintReport verify_constraints(
    const std::vector<float>& original,
    const std::vector<float>& reconstructed,
    std::size_t channels,
    std::size_t time,
    double pointwise_tolerance,
    double acf_tolerance,
    double minimum_correlation,
    std::size_t max_lag)
{
  if (original.size() != reconstructed.size() || original.size() != channels * time)
    throw std::invalid_argument("constraint verification shape mismatch");
  ConstraintReport report;
  report.min_series_correlation = 1.0;
  const auto numeric_tolerance = 64.0 * std::numeric_limits<double>::epsilon();
  for (std::size_t index = 0; index < original.size(); ++index) {
    if (!std::isfinite(original[index]) || !std::isfinite(reconstructed[index])) {
      report.max_abs_error = std::numeric_limits<double>::infinity();
      report.max_acf_error = std::numeric_limits<double>::infinity();
      report.min_series_correlation = -1.0;
      report.pointwise_ok = false;
      report.acf_ok = false;
      report.correlation_ok = false;
      return report;
    }
    report.max_abs_error = std::max(
        report.max_abs_error,
        std::abs(static_cast<double>(original[index]) - reconstructed[index]));
  }

  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto* source = original.data() + channel * time;
    const auto* decoded = reconstructed.data() + channel * time;
    if (acf_tolerance >= 0.0) {
      for (std::size_t lag = 1; lag <= max_lag; ++lag)
        report.max_acf_error = std::max(
            report.max_acf_error, std::abs(acf(source, time, lag) - acf(decoded, time, lag)));
    }
    if (minimum_correlation >= 0.0) {
      long double source_energy = 0.0L;
      long double mean = 0.0L;
      for (std::size_t index = 0; index < time; ++index)
        mean += source[index];
      mean /= static_cast<long double>(time);
      for (std::size_t index = 0; index < time; ++index) {
        const auto centered = static_cast<long double>(source[index]) - mean;
        source_energy += centered * centered;
      }
      double correlation = 0.0;
      if (source_energy <= kDenominatorFloor) {
        // Canonical degenerate-series semantics shared by the online check,
        // incremental repair state, and NumPy validation: exact float32
        // numeric equality gives correlation one, any changed value gives
        // zero. In particular +0.0F and -0.0F compare equal here even though
        // byte-stream replay is verified separately at the bit level.
        correlation = std::equal(
            source,
            source + time,
            decoded) ? 1.0 : 0.0;
      }
      else {
        correlation = pearson(source, decoded, time);
      }
      report.min_series_correlation = std::min(report.min_series_correlation, correlation);
    }
  }
  report.pointwise_ok = report.max_abs_error <= pointwise_tolerance + numeric_tolerance;
  report.acf_ok = acf_tolerance < 0.0 || report.max_acf_error <= acf_tolerance + numeric_tolerance;
  report.correlation_ok =
      minimum_correlation < 0.0 ||
      report.min_series_correlation + numeric_tolerance >= minimum_correlation;
  return report;
}

class CorrelationQoI final : public QoZ::concepts::QoIInterface<float, 2> {
 public:
  using Range = QoZ::multi_dimensional_range<float, 2>;
  using iterator = typename Range::iterator;

  CorrelationQoI(
      const std::vector<float>& original,
      std::size_t channels,
      std::size_t time,
      double pointwise_tolerance,
      double acf_tolerance,
      double minimum_correlation,
      std::size_t max_lag,
      double global_error_bound,
      BoundEstimator estimator = BoundEstimator::legacy_deterministic,
      BoundAllocation allocation = BoundAllocation::uniform,
      ComplianceMode compliance_mode = ComplianceMode::online,
      double estimator_scale = 1.0,
      double qpet_confidence = 0.999,
      double qpet_k = 2.0)
      : original_(original),
        candidate_(original),
        channels_(channels),
        time_(time),
        pointwise_tolerance_(pointwise_tolerance),
        acf_tolerance_(acf_tolerance),
        minimum_correlation_(minimum_correlation),
        max_lag_(acf_tolerance >= 0.0
            ? std::min(max_lag, time > 2 ? time - 2 : std::size_t{0})
            : std::size_t{0}),
        global_error_bound_(global_error_bound),
        estimator_(estimator),
        allocation_(allocation),
        compliance_mode_(compliance_mode),
        estimator_scale_(estimator_scale),
        qpet_confidence_(qpet_confidence),
        qpet_k_(qpet_k),
        source_constant_channels_(channels, false),
        local_bounds_(original.size(), pointwise_tolerance)
  {
    if (channels == 0 || time < 3 || original.size() != channels * time)
      throw std::invalid_argument("CorrelationQoI expects a non-empty [channel,time>=3] window");
    if (!(pointwise_tolerance > 0.0) || !(global_error_bound > 0.0))
      throw std::invalid_argument("pointwise and global error bounds must be positive");
    if (acf_tolerance < 0.0 && minimum_correlation < 0.0)
      throw std::invalid_argument("at least one correlation constraint must be enabled");
    if (acf_tolerance >= 0.0 && (!(acf_tolerance > 0.0) || max_lag_ == 0))
      throw std::invalid_argument("enabled ACF tolerance and max lag must be positive");
    if (minimum_correlation >= 0.0 &&
        (!(minimum_correlation > 0.0) || minimum_correlation > 1.0))
      throw std::invalid_argument("minimum Pearson correlation must be in (0,1]");
    if (!(estimator_scale_ > 0.0) || !std::isfinite(estimator_scale_))
      throw std::invalid_argument("QPET estimator scale must be finite and positive");
    if (!(qpet_confidence_ > 0.0) || !(qpet_confidence_ < 1.0) ||
        !std::isfinite(qpet_confidence_))
      throw std::invalid_argument("QPET estimator confidence must be in (0,1)");
    if (!(qpet_k_ > 0.0) || !std::isfinite(qpet_k_))
      throw std::invalid_argument("QPET estimator k must be finite and positive");
    QoZ::concepts::QoIInterface<float, 2>::id = 16;
    compute_local_bounds();
  }

  float interpret_eb(float) const override
  {
    return static_cast<float>(std::min(pointwise_tolerance_, global_error_bound_));
  }

  float interpret_eb(const iterator& iter) const override { return interpret_eb(*iter); }

  float interpret_eb(const float*, std::ptrdiff_t offset) override
  {
    if (offset < 0 || static_cast<std::size_t>(offset) >= local_bounds_.size())
      throw std::out_of_range("QPET supplied an invalid QoI offset");
    current_offset_ = static_cast<std::size_t>(offset);
    const auto bound = std::min(local_bounds_[current_offset_], global_error_bound_);
    return static_cast<float>(std::isfinite(bound) && bound > 0.0 ? bound : 0.0);
  }

  void update_tolerance(float, float) override {}

  bool check_compliance(float original_value, float decoded_value, bool = false) const override
  {
    if (current_offset_ >= candidate_.size())
      throw std::logic_error("check_compliance called before interpret_eb");
    const auto previous = candidate_[current_offset_];
    candidate_[current_offset_] = decoded_value;
    const auto channel = current_offset_ / time_;
    const auto begin = channel * time_;
    bool compliant =
        std::abs(static_cast<double>(original_value) - decoded_value) <= pointwise_tolerance_;

    if (compliance_mode_ == ComplianceMode::posthoc) {
      if (!compliant) {
        candidate_[current_offset_] = previous;
        ++correction_count_;
      }
      else {
        ++accepted_count_;
      }
      return compliant;
    }

    if (compliant && acf_tolerance_ >= 0.0) {
      for (std::size_t lag = 1; lag <= max_lag_; ++lag) {
        const auto deviation = std::abs(
            acf(original_.data() + begin, time_, lag) -
            acf(candidate_.data() + begin, time_, lag));
        if (!(deviation <= acf_tolerance_)) {
          compliant = false;
          break;
        }
      }
    }

    if (compliant && minimum_correlation_ >= 0.0) {
      auto correlation = 0.0;
      if (source_constant_channels_[channel]) {
        const auto exact = std::equal(
            original_.begin() + static_cast<std::ptrdiff_t>(begin),
            original_.begin() + static_cast<std::ptrdiff_t>(begin + time_),
            candidate_.begin() + static_cast<std::ptrdiff_t>(begin));
        correlation = exact ? 1.0 : 0.0;
      }
      else {
        correlation = pearson(
            original_.data() + begin, candidate_.data() + begin, time_);
      }
      if (!(correlation >= minimum_correlation_))
        compliant = false;
    }

    if (!compliant) {
      candidate_[current_offset_] = previous;
      ++correction_count_;
    }
    else {
      ++accepted_count_;
    }
    return compliant;
  }

  void precompress_block(const std::shared_ptr<Range>&) override {}
  void postcompress_block() override {}
  void pre_compute(const float*) override {}
  void print() override {}
  float get_global_eb() const override { return static_cast<float>(global_error_bound_); }
  void set_global_eb(float bound) override { global_error_bound_ = bound; }
  void set_qoi_tolerance(double tolerance) override { acf_tolerance_ = tolerance; }
  void init() override {}
  double eval(float value) const override { return value; }
  std::string get_expression(const std::string = "x") const override
  {
    return "ACF(x,lags) and Pearson(x,x_hat)";
  }
  void set_dims(const std::vector<std::size_t>& dimensions) override
  {
    if (dimensions.size() != 2 || dimensions[0] != channels_ || dimensions[1] != time_)
      throw std::invalid_argument("CorrelationQoI dimensions cannot change after analysis");
  }

  const std::vector<double>& local_bounds() const { return local_bounds_; }
  std::size_t correction_count() const { return correction_count_; }
  std::size_t accepted_count() const { return accepted_count_; }

 private:
  void make_channel_lossless(std::size_t channel)
  {
    std::fill_n(local_bounds_.begin() + channel * time_, time_, 0.0);
  }

  void compute_local_bounds()
  {
    for (std::size_t channel = 0; channel < channels_; ++channel) {
      const auto begin = channel * time_;
      const auto* series = original_.data() + begin;

      long double mean = 0.0L;
      for (std::size_t index = 0; index < time_; ++index)
        mean += series[index];
      mean /= static_cast<long double>(time_);
      long double centered_energy = 0.0L;
      for (std::size_t index = 0; index < time_; ++index) {
        const auto centered = static_cast<long double>(series[index]) - mean;
        centered_energy += centered * centered;
      }
      if (!(centered_energy > kDenominatorFloor)) {
        source_constant_channels_[channel] = true;
        make_channel_lossless(channel);
        continue;
      }

      if (minimum_correlation_ >= 0.0) {
        const auto radius = std::sqrt(static_cast<double>(centered_energy)) *
            std::sqrt(std::max(0.0, 1.0 - minimum_correlation_ * minimum_correlation_));
        const auto uniform_bound = estimator_scale_ * radius /
            std::sqrt(static_cast<double>(time_));
        for (std::size_t index = 0; index < time_; ++index)
          local_bounds_[begin + index] =
              std::min(local_bounds_[begin + index], uniform_bound);
      }

      if (acf_tolerance_ >= 0.0) {
        for (std::size_t lag = 1; lag <= max_lag_; ++lag) {
          const auto gradient = acf_gradient(series, time_, lag);
          long double gradient_l1 = 0.0L;
          long double gradient_l2_squared = 0.0L;
          for (const auto value : gradient) {
            gradient_l1 += std::abs(value);
            gradient_l2_squared += static_cast<long double>(value) * value;
          }
          if (!(gradient_l1 > 0.0L) || !(gradient_l2_squared > 0.0L)) {
            make_channel_lossless(channel);
            break;
          }

          if (estimator_ == BoundEstimator::legacy_deterministic) {
            for (std::size_t index = 0; index < time_; ++index) {
              const auto magnitude = std::abs(gradient[index]);
              if (magnitude > 0.0) {
                const auto bound = estimator_scale_ * acf_tolerance_ /
                    (static_cast<double>(time_) * magnitude);
                local_bounds_[begin + index] =
                    std::min(local_bounds_[begin + index], bound);
              }
            }
          }
          else {
            // QPET Algorithm 3, Theorems 5.3 and 5.4.  The deterministic
            // and sub-Gaussian estimates are both computed, and the larger
            // point-wise QoI tolerance is retained exactly as in the paper.
            // Post-hoc validation/correction provides the final hard guarantee.
            const auto deterministic =
                acf_tolerance_ / static_cast<double>(gradient_l1);
            const auto log_term = std::log(2.0 / (1.0 - qpet_confidence_));
            const auto probabilistic = qpet_k_ * acf_tolerance_ * std::sqrt(
                1.0 /
                (2.0 * static_cast<double>(gradient_l2_squared) * log_term));
            const auto qpet_bound = std::max(deterministic, probabilistic);
            if (allocation_ == BoundAllocation::gradient_balanced) {
              // Maximize the geometric mean of the per-point bounds under
              // the same first-order weighted budget as QPET's uniform
              // estimate.  For sum_i |g_i| b_i <= B, the uncapped optimum
              // of sum_i log(b_i) assigns an equal |g_i| b_i contribution
              // to every active point.  Exact post-hoc repair remains the
              // final guarantee for the nonlinear ACF constraint.
              const auto active = static_cast<double>(std::count_if(
                  gradient.begin(), gradient.end(), [](double value) {
                    return std::abs(value) > 0.0;
                  }));
              if (!(active > 0.0)) {
                make_channel_lossless(channel);
                break;
              }
              const auto weighted_budget = qpet_bound *
                  static_cast<double>(gradient_l1);
              for (std::size_t index = 0; index < time_; ++index) {
                const auto magnitude = std::abs(gradient[index]);
                if (magnitude > 0.0) {
                  const auto bound = estimator_scale_ * weighted_budget /
                      (active * magnitude);
                  local_bounds_[begin + index] =
                      std::min(local_bounds_[begin + index], bound);
                }
              }
            }
            else {
              const auto bound = estimator_scale_ * qpet_bound;
              for (std::size_t index = 0; index < time_; ++index) {
                local_bounds_[begin + index] =
                    std::min(local_bounds_[begin + index], bound);
              }
            }
          }
        }
      }
    }
  }

  std::vector<float> original_;
  mutable std::vector<float> candidate_;
  std::size_t channels_;
  std::size_t time_;
  double pointwise_tolerance_;
  mutable double acf_tolerance_;
  double minimum_correlation_;
  std::size_t max_lag_;
  double global_error_bound_;
  BoundEstimator estimator_;
  BoundAllocation allocation_;
  ComplianceMode compliance_mode_;
  double estimator_scale_;
  double qpet_confidence_;
  double qpet_k_;
  std::vector<bool> source_constant_channels_;
  std::vector<double> local_bounds_;
  mutable std::size_t current_offset_ = std::numeric_limits<std::size_t>::max();
  mutable std::size_t correction_count_ = 0;
  mutable std::size_t accepted_count_ = 0;
};

struct RepairReport {
  std::size_t corrected_values = 0;
  std::size_t corrected_series = 0;
  bool feasible = true;
};

namespace detail {

struct PairStatistics {
  std::size_t count = 0;
  long double left_sum = 0.0L;
  long double right_sum = 0.0L;
  long double left_square_sum = 0.0L;
  long double right_square_sum = 0.0L;
  long double cross_sum = 0.0L;

  double correlation() const
  {
    if (count < 2)
      return 0.0;
    const auto n = static_cast<long double>(count);
    const auto covariance = cross_sum - left_sum * right_sum / n;
    const auto left_energy = std::max(0.0L, left_square_sum - left_sum * left_sum / n);
    const auto right_energy = std::max(0.0L, right_square_sum - right_sum * right_sum / n);
    const auto denominator = std::sqrt(left_energy * right_energy);
    return denominator > kDenominatorFloor
        ? static_cast<double>(covariance / denominator)
        : 0.0;
  }
};

inline PairStatistics pair_statistics(
    const float* left, const float* right, std::size_t count)
{
  PairStatistics stats;
  stats.count = count;
  for (std::size_t index = 0; index < count; ++index) {
    const auto a = static_cast<long double>(left[index]);
    const auto b = static_cast<long double>(right[index]);
    stats.left_sum += a;
    stats.right_sum += b;
    stats.left_square_sum += a * a;
    stats.right_square_sum += b * b;
    stats.cross_sum += a * b;
  }
  return stats;
}

class SeriesRepairState {
 public:
  SeriesRepairState(
      const float* original,
      float* reconstructed,
      const float* repair_targets,
      std::size_t count,
      double acf_tolerance,
      double minimum_correlation,
      std::size_t max_lag)
      : original_(original),
        reconstructed_(reconstructed),
        repair_targets_(repair_targets),
        count_(count),
        acf_tolerance_(acf_tolerance),
        minimum_correlation_(minimum_correlation),
        max_lag_(acf_tolerance >= 0.0
            ? std::min(max_lag, count > 2 ? count - 2 : std::size_t{0})
            : std::size_t{0}),
        source_acf_(max_lag_),
        acf_stats_(max_lag_),
        repaired_(count, false),
        pearson_stats_(pair_statistics(original, reconstructed, count))
  {
    long double source_mean = 0.0L;
    for (std::size_t index = 0; index < count_; ++index) {
      source_mean += original_[index];
      nonexact_count_ += original_[index] != reconstructed_[index];
    }
    source_mean /= static_cast<long double>(count_);
    long double source_energy = 0.0L;
    for (std::size_t index = 0; index < count_; ++index) {
      const auto centered = static_cast<long double>(original_[index]) - source_mean;
      source_energy += centered * centered;
    }
    source_constant_ = source_energy <= kDenominatorFloor;
    for (std::size_t lag = 1; lag <= max_lag_; ++lag) {
      source_acf_[lag - 1] = acf(original_, count_, lag);
      acf_stats_[lag - 1] =
          pair_statistics(reconstructed_, reconstructed_ + lag, count_ - lag);
    }
  }

  bool compliant() const
  {
    if (acf_tolerance_ >= 0.0) {
      for (std::size_t lag = 1; lag <= max_lag_; ++lag) {
        if (!(std::abs(source_acf_[lag - 1] - acf(reconstructed_, count_, lag)) <=
              acf_tolerance_))
          return false;
      }
    }
    if (minimum_correlation_ < 0.0)
      return true;
    if (source_constant_)
      return nonexact_count_ == 0;
    return pearson(original_, reconstructed_, count_) >= minimum_correlation_;
  }

  bool fast_compliant() const
  {
    if (acf_tolerance_ >= 0.0) {
      for (std::size_t lag = 1; lag <= max_lag_; ++lag) {
        if (!(std::abs(
                  source_acf_[lag - 1] - acf_stats_[lag - 1].correlation()) <=
              acf_tolerance_))
          return false;
      }
    }
    if (minimum_correlation_ < 0.0)
      return true;
    if (source_constant_)
      return nonexact_count_ == 0;
    return pearson_stats_.correlation() >= minimum_correlation_;
  }

  bool can_repair(std::size_t index) const
  {
    return !repaired_[index] && reconstructed_[index] != repair_targets_[index];
  }

  double objective_if_repaired(std::size_t index) const
  {
    const auto old_value = static_cast<long double>(reconstructed_[index]);
    const auto new_value = static_cast<long double>(repair_targets_[index]);
    const auto delta = new_value - old_value;
    double objective = 0.0;
    if (acf_tolerance_ >= 0.0) {
      for (std::size_t lag = 1; lag <= max_lag_; ++lag) {
        auto stats = acf_stats_[lag - 1];
        update_acf_statistics(stats, index, lag, old_value, new_value, delta);
        objective = std::max(
            objective,
            std::abs(source_acf_[lag - 1] - stats.correlation()) / acf_tolerance_);
      }
    }
    if (minimum_correlation_ >= 0.0) {
      if (source_constant_)
        return std::max(objective, nonexact_count_ > 1 ? 2.0 : 0.0);
      auto stats = pearson_stats_;
      stats.right_sum += delta;
      stats.right_square_sum += new_value * new_value - old_value * old_value;
      stats.cross_sum += static_cast<long double>(original_[index]) * delta;
      const auto correlation = stats.correlation();
      if (minimum_correlation_ < 1.0) {
        objective = std::max(
            objective,
            (1.0 - correlation) / (1.0 - minimum_correlation_));
      }
      else if (correlation < 1.0) {
        objective = std::numeric_limits<double>::infinity();
      }
    }
    return objective;
  }

  void repair(std::size_t index)
  {
    const auto old_value = static_cast<long double>(reconstructed_[index]);
    const auto new_value = static_cast<long double>(repair_targets_[index]);
    const auto delta = new_value - old_value;
    for (std::size_t lag = 1; lag <= max_lag_; ++lag)
      update_acf_statistics(
          acf_stats_[lag - 1], index, lag, old_value, new_value, delta);
    pearson_stats_.right_sum += delta;
    pearson_stats_.right_square_sum += new_value * new_value - old_value * old_value;
    pearson_stats_.cross_sum += static_cast<long double>(original_[index]) * delta;
    const auto was_exact = reconstructed_[index] == original_[index];
    reconstructed_[index] = repair_targets_[index];
    const auto is_exact = reconstructed_[index] == original_[index];
    repaired_[index] = true;
    if (!was_exact && is_exact)
      --nonexact_count_;
    else if (was_exact && !is_exact)
      ++nonexact_count_;
  }

 private:
  void update_acf_statistics(
      PairStatistics& stats,
      std::size_t index,
      std::size_t lag,
      long double old_value,
      long double new_value,
      long double delta) const
  {
    if (index < count_ - lag) {
      stats.left_sum += delta;
      stats.left_square_sum += new_value * new_value - old_value * old_value;
      stats.cross_sum += delta * reconstructed_[index + lag];
    }
    if (index >= lag) {
      stats.right_sum += delta;
      stats.right_square_sum += new_value * new_value - old_value * old_value;
      stats.cross_sum += static_cast<long double>(reconstructed_[index - lag]) * delta;
    }
  }

  const float* original_;
  float* reconstructed_;
  const float* repair_targets_;
  std::size_t count_;
  double acf_tolerance_;
  double minimum_correlation_;
  std::size_t max_lag_;
  std::vector<double> source_acf_;
  std::vector<PairStatistics> acf_stats_;
  std::vector<bool> repaired_;
  PairStatistics pearson_stats_;
  bool source_constant_ = false;
  std::size_t nonexact_count_ = 0;
};

}  // namespace detail

inline RepairReport repair_constraints(
    const std::vector<float>& original,
    std::vector<float>& reconstructed,
    std::size_t channels,
    std::size_t time,
    double acf_tolerance,
    double minimum_correlation,
    std::size_t max_lag,
    bool greedy,
    const std::vector<float>* repair_targets = nullptr,
    double pointwise_tolerance = std::numeric_limits<double>::infinity(),
    bool fail_on_exhaustion = true)
{
  if (original.size() != reconstructed.size() || original.size() != channels * time)
    throw std::invalid_argument("constraint repair shape mismatch");
  if (repair_targets != nullptr && repair_targets->size() != original.size())
    throw std::invalid_argument("constraint repair target shape mismatch");
  const auto& targets = repair_targets == nullptr ? original : *repair_targets;
  RepairReport report;
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto begin = channel * time;
    bool changed_series = false;
    if (std::isfinite(pointwise_tolerance)) {
      if (!(pointwise_tolerance > 0.0))
        throw std::invalid_argument("point-wise repair tolerance must be positive");
      for (std::size_t index = 0; index < time; ++index) {
        const auto offset = begin + index;
        if (std::abs(
                static_cast<double>(original[offset]) - reconstructed[offset]) <=
            pointwise_tolerance)
          continue;
        if (std::abs(static_cast<double>(original[offset]) - targets[offset]) >
            pointwise_tolerance)
          throw std::runtime_error("repair target violates the point-wise tolerance");
        reconstructed[offset] = targets[offset];
        ++report.corrected_values;
        changed_series = true;
      }
    }
    detail::SeriesRepairState state(
        original.data() + begin,
        reconstructed.data() + begin,
        targets.data() + begin,
        time,
        acf_tolerance,
        minimum_correlation,
        max_lag);
    if (state.compliant()) {
      if (changed_series)
        ++report.corrected_series;
      continue;
    }
    ++report.corrected_series;
    // Incremental sufficient statistics make the common rejection path O(lags)
    // instead of recomputing every ACF/Pearson over the whole series.  Once
    // they report feasibility, compliant() performs the direct calculation;
    // if cancellation made the fast check optimistic, repair continues.
    while (!state.fast_compliant() || !state.compliant()) {
      auto selected = time;
      auto best_objective = std::numeric_limits<double>::infinity();
      auto largest_error = -1.0;
      for (std::size_t index = 0; index < time; ++index) {
        if (!state.can_repair(index))
          continue;
        if (!greedy) {
          selected = index;
          break;
        }
        const auto objective = state.objective_if_repaired(index);
        const auto error = std::abs(
            static_cast<double>(original[begin + index]) - reconstructed[begin + index]);
        if (objective < best_objective ||
            (objective == best_objective && error > largest_error)) {
          selected = index;
          best_objective = objective;
          largest_error = error;
        }
      }
      if (selected == time) {
        if (fail_on_exhaustion)
          throw std::runtime_error("post-hoc repair exhausted all non-exact values");
        report.feasible = false;
        return report;
      }
      state.repair(selected);
      ++report.corrected_values;
      changed_series = true;
    }
  }
  return report;
}

}  // namespace qpet_correlation
