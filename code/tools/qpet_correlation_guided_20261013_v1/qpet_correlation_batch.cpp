#include "CorrelationQoI.hpp"
#include "FeasibleValueTargets.hpp"
#include "SequenceBeamSearch.hpp"
#include "SubsetRankCodec.hpp"

#include "SPERR_C_API.h"
#include "compressor_32/serf_qt_compressor_32.h"
#include "compressor_32/serf_xor_compressor_32.h"
#include "decompressor_32/serf_qt_decompressor_32.h"
#include "decompressor_32/serf_xor_decompressor_32.h"

#include "QoZ/utils/Statistic.hpp"
#include "QoZ/api/impl/SZInterp.hpp"
#include "QoZ/compressor/SZInterpolationCompressor.hpp"
#include "QoZ/compressor/SZQoIInterpolationCompressor.hpp"
#include "QoZ/encoder/HuffmanEncoder.hpp"
#include "QoZ/encoder/QoIEncoder.hpp"
#include "QoZ/lossless/Lossless_zstd.hpp"
#include "QoZ/quantizer/IntegerQuantizer.hpp"
#include "QoZ/quantizer/QoIIntegerQuantizer.hpp"
#include "QoZ/utils/Config.hpp"

#include <zstd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using QPETCompressor = QoZ::SZQoIInterpolationCompressor<
    float,
    2,
    QoZ::VariableEBLinearQuantizer<float, float>,
    QoZ::EBLogQuantizer<float>,
    QoZ::QoIEncoder<int>,
    QoZ::Lossless_zstd>;

using HPEZCompressor = QoZ::SZInterpolationCompressor<
    float,
    2,
    QoZ::LinearQuantizer<float>,
    QoZ::HuffmanEncoder<int>,
    QoZ::Lossless_zstd>;

constexpr std::array<char, 4> kMagic = {'Q', 'P', 'C', '1'};
constexpr std::array<unsigned char, 4> kSerfSmoothMagic = {'S', 'Q', 'S', '1'};
constexpr std::array<unsigned char, 4> kSemanticBlockMagic = {'S', 'B', 'K', '1'};
constexpr std::uint32_t kVersion = 12;
constexpr std::uint32_t kAdaptivePackedVersion = 13;
constexpr std::uint32_t kFactorizedPackedVersion = 14;
constexpr std::uint32_t kSubsetRankPackedVersion = 15;
constexpr std::uint32_t kPredictivePackedVersion = 16;
constexpr std::uint8_t kSemanticSchemaLegacy = 1;
constexpr std::uint8_t kSemanticSchemaStrictConstants = 2;
constexpr std::uint8_t kSemanticConstantOnlyMode = 0xffU;
constexpr double kDisabledConstraint = -1.0;
constexpr std::uint8_t kPackedQuantizedCorrectionFormat = 6U;
constexpr std::uint8_t kAdaptivePackedCorrectionFormat = 7U;
constexpr std::uint8_t kFactorizedPackedCorrectionFormat = 8U;
constexpr std::uint8_t kSubsetRankPackedCorrectionFormat = 9U;
constexpr std::uint8_t kPredictivePackedCorrectionFormat = 10U;
constexpr std::uint32_t kHasAcf = 1U << 0U;
constexpr std::uint32_t kHasPearson = 1U << 1U;
constexpr std::size_t kZrcarSeedCount = 2;
constexpr std::size_t kFvroAdditionalZrcarSeedCount = 1;
constexpr std::size_t kFullFvroZrcarSeedCount =
    kZrcarSeedCount + kFvroAdditionalZrcarSeedCount;
constexpr std::size_t kZrcarBaseRecordCount = 4;
constexpr std::size_t kZrcarBudgetPerSeed = 16;
constexpr std::size_t kZrcarPrefixBudgetPerSeed = 10;
constexpr std::size_t kZrcarBeamWidth = 4;
constexpr std::size_t kZrcarInfeasibleBeamWidth = 2;
constexpr std::size_t kZrcarBranching = 4;
constexpr std::size_t kZrcarMaximumDepth = 4;
constexpr std::size_t kZrcarMaximumGeneratedPerSeed =
    kZrcarPrefixBudgetPerSeed +
    kZrcarMaximumDepth *
        (kZrcarBeamWidth + kZrcarInfeasibleBeamWidth) * kZrcarBranching;
static_assert(
    kZrcarPrefixBudgetPerSeed < kZrcarBudgetPerSeed,
    "ZRCAR must reserve a nonzero serializer budget for local beam search");
constexpr std::size_t kFvroGlobalSerializerBudget = 8;
constexpr std::size_t kFvroMaximumDifficultChannels = 4;
constexpr std::size_t kFvroChannelAlternativeCount = 3;
constexpr std::size_t kFvroChannelSerializerBudget =
    kFvroMaximumDifficultChannels * kFvroChannelAlternativeCount;
constexpr std::size_t kFvroFullSerializerBudgetPerBase =
    kFvroGlobalSerializerBudget + kFvroChannelSerializerBudget;
constexpr std::size_t kFvroLocalVariantsPerLattice = 3;
constexpr std::array<int, 5> kFvroLatticeDenominators = {
    7, 15, 31, 63, 127};
constexpr std::size_t kFvroLocalProposalBudgetPerBase =
    kFvroMaximumDifficultChannels * kFvroLatticeDenominators.size() *
    kFvroLocalVariantsPerLattice;
constexpr std::size_t kSsbbMaximumDifficultChannels = 8;
constexpr std::size_t kSsbbMaximumSerializedPerChannel = 3;
constexpr std::size_t kSsbbMaximumSerializedPerBase =
    kSsbbMaximumDifficultChannels * kSsbbMaximumSerializedPerChannel;
constexpr std::size_t kSsbbFullSearchStartsPerBase = 2;
constexpr std::size_t kSsbbMaximumCeilingPaths = 5;
constexpr std::size_t kSsbbCompositionFormatSlots =
    static_cast<std::size_t>(kPredictivePackedCorrectionFormat) + 1U;
static_assert(
    kSsbbCompositionFormatSlots == 11,
    "SSBB composition must retain one stratification slot for every tag 0--10");
constexpr std::size_t kSsbbCompositionRateSlots = 24;
constexpr std::size_t kSsbbCompositionBeamWidth =
    1 + kSsbbMaximumDifficultChannels * kSsbbMaximumSerializedPerChannel +
    kSsbbCompositionFormatSlots + kSsbbCompositionRateSlots;
constexpr std::size_t kSsbbCompositionMaximumTransitionsPerStart =
    kSsbbMaximumDifficultChannels * kSsbbCompositionBeamWidth *
    kSsbbMaximumSerializedPerChannel;
constexpr std::size_t kMaximumDecodedFloatBytes = 64U * 1024U * 1024U;
constexpr std::size_t kMaximumDecodedValues =
    kMaximumDecodedFloatBytes / sizeof(float);
constexpr std::size_t kCompactRecordHeaderBytes =
    sizeof(std::uint16_t) + 2U * sizeof(std::uint8_t) + sizeof(double) +
    3U * sizeof(std::uint32_t);
static_assert(kCompactRecordHeaderBytes == 24U);
static_assert(kMaximumDecodedValues > 0);
constexpr std::array<double, 22> kSperrBoundRatios = {
    1.0, 0.9375, 0.875, 0.8125, 0.75, 0.6875, 0.625, 0.5625,
    0.5, 0.4375, 0.375, 0.3125, 0.25, 0.1875, 0.125, 0.0625,
    0.046875, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125};
constexpr std::array<double, 8> kSerfBoundRatios = {
    1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625};
constexpr std::array<double, 3> kCjrfRelaxedBoundRatios = {
    1.25, 1.5, 2.0};
constexpr std::array<std::uint32_t, 4> kCjrfRelaxedBaseModes = {
    1U, 3U, 5U, 6U};
constexpr std::array<double, 3> kCfsrSurrogateBoundRatios = {
    0.25, 0.5, 1.0};
constexpr std::array<std::uint32_t, 4> kCfsrSurrogateBaseModes = {
    1U, 3U, 5U, 6U};
constexpr std::array<std::pair<std::size_t, std::size_t>, 4> kVehicleSignalGroups = {{
    {0, 8}, {8, 2}, {10, 3}, {13, 3}}};

enum class RepairOptimizer : std::uint32_t {
  legacy = 0,
  zrcar = 1,
  fvro_value = 2,
  fvro = 3,
  ssbb_value = 4,
  ssbb = 5,
};

enum class BaseFrontier : std::uint32_t {
  incumbent_only = 0,
  cjrf_v1 = 1,
};

enum class CjrfWinnerKind : std::uint8_t {
  none = 0,
  legacy = 1,
  fvro = 2,
  zrcar = 3,
  ssbb = 4,
};

enum class SurrogateFrontier : std::uint32_t {
  none = 0,
  cfsr_v1 = 1,
};

enum class CfsrWinnerKind : std::uint8_t {
  none = 0,
  legacy = 1,
  fvro = 2,
  zrcar = 3,
  ssbb = 4,
};

bool uses_cjrf(BaseFrontier frontier)
{
  return frontier == BaseFrontier::cjrf_v1;
}

bool uses_cfsr(SurrogateFrontier frontier)
{
  return frontier == SurrogateFrontier::cfsr_v1;
}

float cfsr_float_bound_not_above(double registered_bound)
{
  auto backend_bound = static_cast<float>(registered_bound);
  if (static_cast<double>(backend_bound) > registered_bound)
    backend_bound = std::nextafter(backend_bound, 0.0F);
  return backend_bound;
}

bool uses_full_ssbb(RepairOptimizer optimizer)
{
  return optimizer == RepairOptimizer::ssbb;
}

bool uses_fvro(RepairOptimizer optimizer)
{
  return optimizer == RepairOptimizer::fvro_value ||
      optimizer == RepairOptimizer::fvro ||
      optimizer == RepairOptimizer::ssbb_value ||
      optimizer == RepairOptimizer::ssbb;
}

bool uses_zrcar(RepairOptimizer optimizer)
{
  return optimizer == RepairOptimizer::zrcar ||
      optimizer == RepairOptimizer::fvro ||
      optimizer == RepairOptimizer::ssbb;
}

bool uses_ssbb(RepairOptimizer optimizer)
{
  return optimizer == RepairOptimizer::ssbb_value ||
      optimizer == RepairOptimizer::ssbb;
}

bool uses_second_stage(RepairOptimizer optimizer)
{
  return uses_fvro(optimizer) || uses_zrcar(optimizer);
}

struct PaperBudgetAudit {
  std::ofstream plans_output;
  std::ofstream candidates_output;
  std::ofstream winners_output;
  std::ofstream failures_output;
  std::size_t analyses = 0;
  std::size_t plans = 0;
  std::size_t base_attempts = 0;
  std::size_t base_failures = 0;
  std::size_t base_candidates = 0;
  std::size_t base_passes = 0;
  std::size_t base_bound_violations = 0;
  double analysis_seconds = 0.0;
};

struct Settings {
  // Encoder-only, absent from QPC framing and old generic behavior.
  // 0=frozen generic, 1=paper uniform, 2=prespecified four-bound control.
  int paper_budget_profile = 0;
  PaperBudgetAudit* paper_budget_audit = nullptr;
  std::size_t paper_budget_window = 0;
  std::size_t batch = 0;
  std::size_t channels = 0;
  std::size_t time = 0;
  double pointwise_tolerance = 0.0;
  double acf_tolerance = -1.0;
  double minimum_correlation = -1.0;
  std::size_t max_lag = 0;
  int qoz_level = 3;
  std::uint32_t correction_mode = 2;
  std::vector<double> estimator_scales = {1.0};
  double qpet_confidence = 0.999;
  double qpet_k = 2.0;
  std::size_t group_size = 1;
  std::vector<int> repair_denominators = {7, 15, 31, 63, 127};
  // Encoder-search policy only.  The generic profile never consults the
  // vehicle-specific channel partition.  Legacy semantic modes remain
  // decodable so previously produced QPC streams keep working.
  bool enable_vehicle_semantic = false;
  // Encoder-search policy only. It is intentionally absent from QPC v12.
  RepairOptimizer repair_optimizer = RepairOptimizer::legacy;
  // Encoder-search ablation policy only; it is deliberately not serialized.
  // A decoded stream therefore retains the decoder-neutral legacy ceiling 6,
  // while the compress CLI restores the historical omitted-argument behavior
  // (SSBB -> 8, every other optimizer -> 6).
  std::uint8_t correction_format_ceiling =
      kPackedQuantizedCorrectionFormat;
  // Orthogonal encoder-only base-frontier policy. It is not a repair
  // optimizer and is deliberately absent from QPC framing.
  BaseFrontier base_frontier = BaseFrontier::incumbent_only;
  // Bit 0/1/2 denotes the fixed 1.25/1.5/2.0 nested prefix respectively.
  std::uint8_t cjrf_ratio_mask = 0U;
  // Orthogonal encoder-only certified feasible-surrogate re-encoding policy.
  // Like CJRF, it is deliberately absent from QPC framing.
  SurrogateFrontier surrogate_frontier = SurrogateFrontier::none;
  // Bit 0/1/2 denotes the fixed 0.25/0.5/1.0 nested prefix respectively.
  std::uint8_t cfsr_ratio_mask = 0U;
};

struct EncodedWindow {
  qpet_correlation::ConstraintReport paper_base_report;
  std::uint32_t logical_windows = 1;
  std::uint32_t base_mode = 0;
  double selected_global_bound = 0.0;
  double selected_estimator_scale = 1.0;
  std::uint32_t corrections = 0;
  std::uint32_t online_corrections = 0;
  std::uint32_t candidates_tested = 0;
  std::uint32_t repair_candidates_tested = 1;
  std::uint32_t candidate_failures = 0;
  // Encoder-search diagnostics only.  Neither field is serialized into QPC1.
  // The group counter remains one only when the final selected window is the
  // base-only semantic shared candidate; a later portfolio winner naturally
  // replaces it with zero.
  std::uint32_t shared_base_search_groups = 0;
  std::size_t shared_local_reference_bytes = 0;
  // Encoder-search diagnostics only. These fields are not serialized into
  // QPC v12 and therefore cannot affect legacy decoding.
  std::uint32_t zrcar_candidates_tested = 0;
  std::uint32_t zrcar_prefix_candidates_tested = 0;
  std::uint32_t zrcar_local_candidates_tested = 0;
  std::uint32_t zrcar_base_records_tested = 0;
  std::uint32_t zrcar_legacy_seeds_selected = 0;
  std::uint32_t zrcar_fvro_seeds_selected = 0;
  std::uint32_t zrcar_states_generated = 0;
  std::uint32_t zrcar_infeasible_states = 0;
  std::uint32_t zrcar_infeasible_frontier_expansions = 0;
  std::uint32_t zrcar_infeasible_frontier_recoveries = 0;
  bool zrcar_selected = false;
  bool zrcar_from_fvro_seed = false;
  std::size_t legacy_repair_bytes = 0;
  std::size_t zrcar_legacy_reference_bytes = 0;
  std::uint32_t fvro_global_candidates_generated = 0;
  std::uint32_t fvro_global_candidates_serialized = 0;
  std::uint32_t fvro_global_candidates_infeasible = 0;
  std::uint32_t fvro_channel_candidates_generated = 0;
  std::uint32_t fvro_channel_candidates_serialized = 0;
  std::uint32_t fvro_channel_candidates_infeasible = 0;
  std::uint32_t fvro_local_proposals_generated = 0;
  std::uint32_t fvro_local_proposals_infeasible = 0;
  std::uint32_t fvro_selected_channels = 0;
  std::uint32_t fvro_strict_channel_steps = 0;
  std::uint32_t fvro_seeds_retained = 0;
  std::uint32_t fvro_base_records_tested = 0;
  std::uint32_t fvro_reference_exact_escapes = 0;
  std::uint32_t fvro_final_exact_escapes = 0;
  std::uint8_t fvro_reference_quantized_format = 0;
  std::uint8_t fvro_final_quantized_format = 0;
  bool fvro_reference_quantized = false;
  bool fvro_final_quantized = false;
  bool fvro_selected = false;
  std::size_t fvro_legacy_reference_bytes = 0;
  std::uint32_t ssbb_base_records_tested = 0;
  std::uint32_t ssbb_channels_searched = 0;
  std::uint32_t ssbb_positions_processed = 0;
  std::uint32_t ssbb_actions_generated = 0;
  std::uint32_t ssbb_states_generated = 0;
  std::uint32_t ssbb_states_deduplicated = 0;
  std::uint32_t ssbb_statistics_feasible_states = 0;
  std::uint32_t ssbb_statistics_infeasible_states = 0;
  std::uint32_t ssbb_feasible_states_retained = 0;
  std::uint32_t ssbb_infeasible_states_retained = 0;
  std::uint32_t ssbb_states_pruned_by_beam = 0;
  std::uint32_t ssbb_terminal_states = 0;
  std::uint32_t ssbb_terminal_direct_feasible = 0;
  std::uint32_t ssbb_terminal_direct_infeasible = 0;
  std::uint32_t ssbb_outputs_returned = 0;
  std::uint32_t ssbb_candidates_serialized = 0;
  std::uint32_t ssbb_seeds_retained = 0;
  std::uint32_t ssbb_selected_channels = 0;
  // One canonical reserialization is evaluated at every certified SSBB
  // start before any reconstruction-changing search.  The aggregate fields
  // describe all work; the two byte references below describe only the
  // start that produced this EncodedWindow's final winner.
  std::uint32_t ssbb_representation_candidates_serialized = 0;
  std::uint32_t ssbb_representation_improved_starts = 0;
  std::size_t ssbb_representation_gross_saved_bytes = 0;
  std::uint32_t ssbb_composition_starts = 0;
  std::uint32_t ssbb_composition_local_options = 0;
  std::uint32_t ssbb_composition_transitions_generated = 0;
  std::uint32_t ssbb_composition_transitions_deduplicated = 0;
  std::uint32_t ssbb_composition_direct_feasible = 0;
  std::uint32_t ssbb_composition_direct_infeasible = 0;
  std::uint32_t ssbb_composition_candidates_serialized = 0;
  std::uint32_t ssbb_composition_singletons_serialized = 0;
  std::uint32_t ssbb_composition_pairs_serialized = 0;
  std::uint32_t ssbb_composition_frontier_carries = 0;
  std::uint32_t ssbb_composition_anchor_retention_events = 0;
  std::uint32_t ssbb_composition_format_retention_events = 0;
  std::uint32_t ssbb_composition_rate_retention_events = 0;
  std::uint32_t ssbb_composition_states_pruned = 0;
  std::uint32_t ssbb_composition_max_frontier = 0;
  std::uint32_t ssbb_composition_selected_channels = 0;
  bool ssbb_composition_selected = false;
  std::size_t ssbb_composition_reference_bytes = 0;
  bool ssbb_selected = false;
  std::size_t ssbb_reference_bytes = 0;
  std::size_t ssbb_search_reference_bytes = 0;
  // CJRF provenance and aggregate work diagnostics. These fields are never
  // serialized into QPC; final feasibility comes only from decoder replay.
  bool cjrf_relaxed_base = false;
  double cjrf_relaxed_ratio = 1.0;
  bool cjrf_selected = false;
  CjrfWinnerKind cjrf_winner_kind = CjrfWinnerKind::none;
  std::uint32_t cjrf_relaxed_base_attempts = 0;
  std::uint32_t cjrf_relaxed_base_failures = 0;
  std::uint32_t cjrf_relaxed_base_pruned = 0;
  std::uint32_t cjrf_relaxed_base_duplicates = 0;
  std::uint32_t cjrf_relaxed_base_records = 0;
  std::uint32_t cjrf_relaxed_base_outside_pointwise = 0;
  std::uint32_t cjrf_relaxed_legacy_repaired = 0;
  std::uint32_t cjrf_relaxed_legacy_repair_candidates = 0;
  std::size_t cjrf_reference_bytes = 0;
  // CFSR base origin, final provenance, and aggregate encoder work.  The
  // surrogate itself is never serialized; only its official base bytes and
  // the independently replayed correction payload can reach a QPC record.
  bool cfsr_surrogate_base = false;
  double cfsr_surrogate_ratio = 0.0;
  bool cfsr_selected = false;
  CfsrWinnerKind cfsr_winner_kind = CfsrWinnerKind::none;
  std::uint32_t cfsr_surrogate_sources = 0;
  std::uint32_t cfsr_surrogate_base_attempts = 0;
  std::uint32_t cfsr_surrogate_base_failures = 0;
  std::uint32_t cfsr_surrogate_base_pruned = 0;
  std::uint32_t cfsr_surrogate_base_duplicates = 0;
  std::uint32_t cfsr_surrogate_base_records = 0;
  std::uint32_t cfsr_surrogate_base_outside_original_pointwise = 0;
  std::uint32_t cfsr_surrogate_legacy_repaired = 0;
  std::uint32_t cfsr_surrogate_legacy_repair_candidates = 0;
  std::size_t cfsr_reference_bytes = 0;
  // Encoder-only state used to continue a verified reconstruction search.
  // It is never serialized into QPC v12.
  std::vector<float> correction_channel_scales;
  std::vector<unsigned char> payload;
  std::vector<unsigned char> correction_payload;
  std::vector<float> reconstruction;
  qpet_correlation::ConstraintReport report;

  std::size_t encoded_size() const
  {
    if (payload.size() >
        std::numeric_limits<std::size_t>::max() - correction_payload.size())
      throw std::overflow_error("QPET encoded candidate size overflow");
    return payload.size() + correction_payload.size();
  }
};

struct DecodedWindowRecord {
  std::uint32_t logical_windows = 0;
  std::uint32_t base_mode = 0;
  double selected_global_bound = 0.0;
  std::uint32_t corrections = 0;
  std::vector<unsigned char> payload;
  std::vector<unsigned char> correction_payload;
  std::vector<float> base_reconstruction;
  std::vector<float> reconstruction;
};

class BackendCandidateFailure final : public std::runtime_error {
 public:
  explicit BackendCandidateFailure(const std::string& message)
      : std::runtime_error(message)
  {}
};

std::size_t checked_add_size(
    std::size_t left,
    std::size_t right,
    const char* message)
{
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(message);
  return left + right;
}

std::size_t compact_record_bytes(const EncodedWindow& encoded)
{
  return checked_add_size(
      kCompactRecordHeaderBytes,
      encoded.encoded_size(),
      "QPET compact record size overflow");
}

std::size_t compact_base_record_lower_bound_bytes(
    const EncodedWindow& encoded)
{
  return checked_add_size(
      kCompactRecordHeaderBytes,
      encoded.payload.size(),
      "QPET compact base-record lower bound overflow");
}

std::uint32_t checked_add_u32(
    std::uint32_t left,
    std::uint32_t right,
    const char* message)
{
  if (right > std::numeric_limits<std::uint32_t>::max() - left)
    throw std::overflow_error(message);
  return left + right;
}

template <typename T>
void write_scalar(std::ostream& stream, const T& value)
{
  stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!stream)
    throw std::runtime_error("failed while writing QPET correlation stream");
}

template <typename T>
T read_scalar(std::istream& stream)
{
  T value{};
  stream.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!stream)
    throw std::runtime_error("truncated QPET correlation stream");
  return value;
}

std::vector<float> read_floats(const std::string& path, std::size_t count)
{
  auto input = std::ifstream(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open input file: " + path);
  const auto bytes = static_cast<std::size_t>(input.tellg());
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      bytes != count * sizeof(float))
    throw std::runtime_error("input file size does not match the requested tensor shape");
  input.seekg(0);
  auto values = std::vector<float>(count);
  input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
  if (!input)
    throw std::runtime_error("failed to read input data");
  for (const auto value : values) {
    if (!std::isfinite(value))
      throw std::runtime_error("input contains NaN or infinity");
  }
  return values;
}

void write_floats(const std::string& path, const std::vector<float>& values)
{
  auto output = std::ofstream(path, std::ios::binary);
  if (!output)
    throw std::runtime_error("cannot open reconstruction file: " + path);
  output.write(
      reinterpret_cast<const char*>(values.data()),
      static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!output)
    throw std::runtime_error("failed while writing reconstruction");
}

std::vector<unsigned char> read_bytes(const std::string& path)
{
  auto input = std::ifstream(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open byte stream: " + path);
  const auto end = input.tellg();
  if (end < 0)
    throw std::runtime_error("cannot determine byte stream size: " + path);
  auto bytes = std::vector<unsigned char>(static_cast<std::size_t>(end));
  input.seekg(0);
  input.read(
      reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!input && !bytes.empty())
    throw std::runtime_error("failed to read byte stream: " + path);
  return bytes;
}

void write_bytes(const std::string& path, const std::vector<unsigned char>& bytes)
{
  auto output = std::ofstream(path, std::ios::binary);
  if (!output)
    throw std::runtime_error("cannot open byte stream for writing: " + path);
  output.write(
      reinterpret_cast<const char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
  if (!output)
    throw std::runtime_error("failed while writing byte stream: " + path);
}

std::uint32_t parse_correction_mode(const std::string& value)
{
  if (value == "online")
    return 0;
  if (value == "posthoc_sequential")
    return 1;
  if (value == "posthoc_greedy")
    return 2;
  throw std::invalid_argument(
      "correction mode must be online, posthoc_sequential, or posthoc_greedy");
}

std::vector<double> parse_estimator_scales(const std::string& value)
{
  auto scales = std::vector<double>();
  auto input = std::stringstream(value);
  std::string item;
  while (std::getline(input, item, ',')) {
    if (item.empty())
      throw std::invalid_argument("QPET estimator scale list contains an empty item");
    scales.push_back(std::stod(item));
  }
  return scales;
}

std::vector<int> parse_repair_denominators(const std::string& value)
{
  if (value == "none")
    return {};
  auto denominators = std::vector<int>();
  auto input = std::stringstream(value);
  std::string item;
  while (std::getline(input, item, ',')) {
    if (item.empty())
      throw std::invalid_argument("repair denominator list contains an empty item");
    denominators.push_back(std::stoi(item));
  }
  return denominators;
}

bool parse_candidate_profile(const std::string& value)
{
  if (value == "generic")
    return false;
  if (value == "vehicle_semantic")
    return true;
  throw std::invalid_argument(
      "candidate profile must be generic or vehicle_semantic");
}

RepairOptimizer parse_repair_optimizer(const std::string& value)
{
  if (value == "legacy")
    return RepairOptimizer::legacy;
  if (value == "zrcar")
    return RepairOptimizer::zrcar;
  if (value == "fvro_value")
    return RepairOptimizer::fvro_value;
  if (value == "fvro")
    return RepairOptimizer::fvro;
  if (value == "ssbb_value")
    return RepairOptimizer::ssbb_value;
  if (value == "ssbb")
    return RepairOptimizer::ssbb;
  throw std::invalid_argument(
      "repair optimizer must be legacy, zrcar, fvro_value, fvro, "
      "ssbb_value, or ssbb");
}

const char* repair_optimizer_name(RepairOptimizer optimizer)
{
  switch (optimizer) {
    case RepairOptimizer::legacy:
      return "legacy";
    case RepairOptimizer::zrcar:
      return "zrcar";
    case RepairOptimizer::fvro_value:
      return "fvro_value";
    case RepairOptimizer::fvro:
      return "fvro";
    case RepairOptimizer::ssbb_value:
      return "ssbb_value";
    case RepairOptimizer::ssbb:
      return "ssbb";
  }
  throw std::logic_error("unknown QPET repair optimizer");
}

std::uint8_t parse_correction_format_ceiling(const std::string& value)
{
  // Match complete canonical tokens rather than relying on stoull, which
  // accepts prefixes such as "7junk" unless its parsed length is inspected.
  if (value == "6")
    return kPackedQuantizedCorrectionFormat;
  if (value == "7")
    return kAdaptivePackedCorrectionFormat;
  if (value == "8")
    return kFactorizedPackedCorrectionFormat;
  if (value == "9")
    return kSubsetRankPackedCorrectionFormat;
  if (value == "10")
    return kPredictivePackedCorrectionFormat;
  throw std::invalid_argument(
      "correction format ceiling must be exactly 6, 7, 8, 9, or 10");
}

std::uint8_t omitted_correction_format_ceiling(RepairOptimizer optimizer)
{
  return uses_ssbb(optimizer)
      ? kFactorizedPackedCorrectionFormat
      : kPackedQuantizedCorrectionFormat;
}

BaseFrontier parse_base_frontier(const std::string& value)
{
  if (value == "incumbent_only")
    return BaseFrontier::incumbent_only;
  if (value == "cjrf_v1")
    return BaseFrontier::cjrf_v1;
  throw std::invalid_argument(
      "base frontier must be incumbent_only or cjrf_v1");
}

const char* base_frontier_name(BaseFrontier frontier)
{
  switch (frontier) {
    case BaseFrontier::incumbent_only:
      return "incumbent_only";
    case BaseFrontier::cjrf_v1:
      return "cjrf_v1";
  }
  throw std::logic_error("unknown QPET base frontier");
}

std::uint8_t parse_cjrf_ratio_mask(const std::string& value)
{
  if (value == "none")
    return 0U;
  if (value == "1.25")
    return 1U;
  if (value == "1.5")
    return 3U;
  if (value == "2" || value == "2.0")
    return 7U;
  throw std::invalid_argument(
      "CJRF relaxed ratio ceiling must be none, 1.25, 1.5, or 2");
}

SurrogateFrontier parse_surrogate_frontier(const std::string& value)
{
  if (value == "none")
    return SurrogateFrontier::none;
  if (value == "cfsr_v1")
    return SurrogateFrontier::cfsr_v1;
  throw std::invalid_argument(
      "surrogate frontier must be none or cfsr_v1");
}

const char* surrogate_frontier_name(SurrogateFrontier frontier)
{
  switch (frontier) {
    case SurrogateFrontier::none:
      return "none";
    case SurrogateFrontier::cfsr_v1:
      return "cfsr_v1";
  }
  throw std::logic_error("unknown QPET surrogate frontier");
}

std::uint8_t parse_cfsr_ratio_mask(const std::string& value)
{
  if (value == "none")
    return 0U;
  if (value == "0.25")
    return 1U;
  if (value == "0.5")
    return 3U;
  if (value == "1" || value == "1.0")
    return 7U;
  throw std::invalid_argument(
      "CFSR surrogate ratio ceiling must be none, 0.25, 0.5, or 1");
}

std::size_t correction_format_ceiling_path_count(std::uint8_t ceiling)
{
  if (ceiling < kPackedQuantizedCorrectionFormat ||
      ceiling > kPredictivePackedCorrectionFormat)
    throw std::invalid_argument(
        "correction format ceiling path count requires 6, 7, 8, 9, or 10");
  const auto count = static_cast<std::size_t>(
      ceiling - kPackedQuantizedCorrectionFormat + 1U);
  if (count > kSsbbMaximumCeilingPaths)
    throw std::logic_error("correction ceiling path count exceeded its cap");
  return count;
}

std::uint32_t correction_format_required_stream_version(std::uint8_t format)
{
  if (format == kPredictivePackedCorrectionFormat)
    return kPredictivePackedVersion;
  if (format == kSubsetRankPackedCorrectionFormat)
    return kSubsetRankPackedVersion;
  if (format == kFactorizedPackedCorrectionFormat)
    return kFactorizedPackedVersion;
  if (format == kAdaptivePackedCorrectionFormat)
    return kAdaptivePackedVersion;
  if (format <= kPackedQuantizedCorrectionFormat)
    return kVersion;
  throw std::invalid_argument("unknown QPET correction format");
}

std::uint32_t correction_payload_required_stream_version(
    const std::vector<unsigned char>& payload)
{
  return payload.empty()
      ? kVersion
      : correction_format_required_stream_version(payload.front());
}

void validate_settings(
    const Settings& settings,
    bool compact_group_framing = true)
{
  if (settings.batch == 0 || settings.channels == 0 || settings.time < 3)
    throw std::invalid_argument("QPET correlation expects [batch>0,channels>0,time>=3]");
  if (settings.channels > std::numeric_limits<std::uint32_t>::max() ||
      settings.time > std::numeric_limits<std::uint32_t>::max() ||
      settings.batch > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("tensor dimensions exceed the stream format");
  // Reject NaN/Inf before any sentinel or range comparison.  In particular,
  // NaN must not accidentally behave like a disabled correlation constraint.
  if (!std::isfinite(settings.pointwise_tolerance) ||
      !std::isfinite(settings.acf_tolerance) ||
      !std::isfinite(settings.minimum_correlation) ||
      !std::isfinite(settings.qpet_confidence) ||
      !std::isfinite(settings.qpet_k))
    throw std::invalid_argument("all QPET floating-point settings must be finite");
  for (const auto scale : settings.estimator_scales) {
    if (!std::isfinite(scale))
      throw std::invalid_argument("all QPET estimator scales must be finite");
  }

  if (!(settings.pointwise_tolerance > 0.0))
    throw std::invalid_argument("pointwise tolerance must be finite and positive");
  const auto acf_enabled = settings.acf_tolerance != kDisabledConstraint;
  const auto pearson_enabled =
      settings.minimum_correlation != kDisabledConstraint;
  if (acf_enabled &&
      (!(settings.acf_tolerance > 0.0) || settings.acf_tolerance > 2.0))
    throw std::invalid_argument(
        "ACF tolerance must be the disabled sentinel -1 or in (0,2]");
  if (pearson_enabled &&
      (!(settings.minimum_correlation > 0.0) ||
       settings.minimum_correlation > 1.0))
    throw std::invalid_argument("enabled minimum Pearson correlation must be in (0,1]");
  if (!acf_enabled && !pearson_enabled)
    throw std::invalid_argument("at least one correlation constraint must be enabled");
  if (acf_enabled &&
      (settings.max_lag == 0 || settings.max_lag > settings.time - 2))
    throw std::invalid_argument("max_lag must be in [1,time-2] when ACF is enabled");
  if (settings.qoz_level < 0 || settings.qoz_level > 4)
    throw std::invalid_argument("official HPEZ QoZ level must be in [0,4]");
  if (settings.correction_mode > 2)
    throw std::invalid_argument("invalid QPET correction mode");
  if (settings.channels < 2U && settings.correction_mode == 0)
    throw std::invalid_argument(
        "online QPET uses an official two-dimensional HPEZ path and requires "
        "at least two channels");
  if (uses_second_stage(settings.repair_optimizer) &&
      settings.correction_mode == 0)
    throw std::invalid_argument(
        "ZRCAR/FVRO/SSBB/CJRF require a post-hoc correction mode");
  if (uses_second_stage(settings.repair_optimizer) &&
      settings.enable_vehicle_semantic)
    throw std::invalid_argument(
        "ZRCAR/FVRO/SSBB/CJRF are defined only for the generic candidate profile");
  if (uses_second_stage(settings.repair_optimizer) &&
      settings.group_size != 1)
    throw std::invalid_argument(
        "ZRCAR/FVRO/SSBB/CJRF require group_size=1 so each logical window is independent");
  if (settings.correction_format_ceiling !=
          kPackedQuantizedCorrectionFormat &&
      settings.correction_format_ceiling !=
          kAdaptivePackedCorrectionFormat &&
      settings.correction_format_ceiling !=
          kFactorizedPackedCorrectionFormat &&
      settings.correction_format_ceiling !=
          kSubsetRankPackedCorrectionFormat &&
      settings.correction_format_ceiling !=
          kPredictivePackedCorrectionFormat)
    throw std::invalid_argument(
        "correction format ceiling must be exactly 6, 7, 8, 9, or 10");
  if (!uses_ssbb(settings.repair_optimizer) &&
      settings.correction_format_ceiling !=
          kPackedQuantizedCorrectionFormat)
    throw std::invalid_argument(
        "correction formats 7, 8, 9, and 10 are SSBB-only encoder ablations");
  if (settings.base_frontier != BaseFrontier::incumbent_only &&
      settings.base_frontier != BaseFrontier::cjrf_v1)
    throw std::invalid_argument(
        "unknown QPET base-frontier policy");
  if (!uses_cjrf(settings.base_frontier) && settings.cjrf_ratio_mask != 0U)
    throw std::invalid_argument(
        "incumbent-only base frontier cannot carry CJRF ratios");
  if (uses_cjrf(settings.base_frontier) &&
      (settings.repair_optimizer != RepairOptimizer::ssbb ||
       settings.correction_mode != 2U ||
       settings.enable_vehicle_semantic || settings.group_size != 1U ||
       settings.correction_format_ceiling !=
           kSubsetRankPackedCorrectionFormat ||
       !(settings.pointwise_tolerance *
             kCjrfRelaxedBoundRatios.front() >
         settings.pointwise_tolerance) ||
       !std::isfinite(
           settings.pointwise_tolerance *
           kCjrfRelaxedBoundRatios.back()) ||
       settings.pointwise_tolerance *
               kCjrfRelaxedBoundRatios.back() >
           static_cast<double>(std::numeric_limits<float>::max()) ||
       (settings.cjrf_ratio_mask != 1U &&
        settings.cjrf_ratio_mask != 3U &&
        settings.cjrf_ratio_mask != 7U)))
    throw std::invalid_argument(
        "CJRF v1 requires generic full SSBB, posthoc_greedy, group_size=1, "
        "correction_format_ceiling=9, a finite backend-representable "
        "relaxed-bound range, and a nonempty nested ratio prefix");
  if (settings.surrogate_frontier != SurrogateFrontier::none &&
      settings.surrogate_frontier != SurrogateFrontier::cfsr_v1)
    throw std::invalid_argument(
        "unknown QPET surrogate-frontier policy");
  if (!uses_cfsr(settings.surrogate_frontier) &&
      settings.cfsr_ratio_mask != 0U)
    throw std::invalid_argument(
        "disabled surrogate frontier cannot carry CFSR ratios");
  if (uses_cfsr(settings.surrogate_frontier)) {
    if (uses_cjrf(settings.base_frontier) ||
        settings.base_frontier != BaseFrontier::incumbent_only ||
        settings.repair_optimizer != RepairOptimizer::ssbb ||
        settings.correction_mode != 2U ||
        settings.enable_vehicle_semantic || settings.group_size != 1U ||
        settings.correction_format_ceiling !=
            kSubsetRankPackedCorrectionFormat ||
        (settings.cfsr_ratio_mask != 1U &&
         settings.cfsr_ratio_mask != 3U &&
         settings.cfsr_ratio_mask != 7U))
      throw std::invalid_argument(
          "CFSR v1 requires incumbent-only generic full SSBB, "
          "posthoc_greedy, group_size=1, correction_format_ceiling=9, "
          "and a nonempty nested surrogate-ratio prefix");
    auto previous_bound = 0.0;
    auto previous_backend_bound = 0.0F;
    for (std::size_t ratio_index = 0;
         ratio_index < kCfsrSurrogateBoundRatios.size();
         ++ratio_index) {
      if ((settings.cfsr_ratio_mask & (1U << ratio_index)) == 0U)
        continue;
      const auto bound = settings.pointwise_tolerance *
          kCfsrSurrogateBoundRatios[ratio_index];
      const auto backend_bound = cfsr_float_bound_not_above(bound);
      if (!(bound > previous_bound) || !std::isfinite(bound) ||
          bound > static_cast<double>(std::numeric_limits<float>::max()) ||
          !(backend_bound > previous_backend_bound) ||
          !std::isfinite(backend_bound))
        throw std::invalid_argument(
            "CFSR v1 requires positive, distinct, finite, "
            "float-representable surrogate bounds");
      previous_bound = bound;
      previous_backend_bound = backend_bound;
    }
  }
  if (settings.estimator_scales.empty() || settings.estimator_scales.size() > 16)
    throw std::invalid_argument("QPET requires between 1 and 16 estimator scales");
  for (const auto scale : settings.estimator_scales) {
    if (!(scale > 0.0))
      throw std::invalid_argument("QPET estimator scales must be finite and positive");
  }
  if (!(settings.qpet_confidence > 0.0) || !(settings.qpet_confidence < 1.0))
    throw std::invalid_argument("QPET confidence must be finite and in (0,1)");
  if (!(settings.qpet_k > 0.0))
    throw std::invalid_argument("QPET k must be finite and positive");
  if (settings.group_size == 0 || settings.group_size > settings.batch ||
      (compact_group_framing &&
       settings.group_size > std::numeric_limits<std::uint16_t>::max()))
    throw std::invalid_argument(
        "QPET compact group size must be in [1,min(batch,65535)]");
  if (settings.repair_denominators.size() > 16)
    throw std::invalid_argument("QPET supports at most 16 repair denominators");
  auto denominators = settings.repair_denominators;
  std::sort(denominators.begin(), denominators.end());
  if (std::adjacent_find(denominators.begin(), denominators.end()) != denominators.end())
    throw std::invalid_argument("QPET repair denominators must be unique");
  for (const auto denominator : denominators) {
    if (denominator < 1 || denominator > 127)
      throw std::invalid_argument("QPET repair denominators must be in [1,127]");
  }
  if (uses_fvro(settings.repair_optimizer) &&
      (settings.repair_denominators.size() !=
           kFvroLatticeDenominators.size() ||
       !std::equal(
           settings.repair_denominators.begin(),
           settings.repair_denominators.end(),
           kFvroLatticeDenominators.begin())))
    throw std::invalid_argument(
        "FVRO v1 and SSBB v2 require repair_denominators exactly "
        "7,15,31,63,127");
}

void canonicalize_official_interp_metadata(QoZ::Config& config)
{
  static_assert(
      std::is_trivially_copyable_v<QoZ::Interp_Meta> &&
          std::is_standard_layout_v<QoZ::Interp_Meta>,
      "CFSR mode-1 metadata canonicalization requires an object-image ABI");
  const auto clear_padding = [](QoZ::Interp_Meta& metadata) {
    const auto interp_algo = metadata.interpAlgo;
    const auto interp_paradigm = metadata.interpParadigm;
    const auto cubic_spline_type = metadata.cubicSplineType;
    const auto interp_direction = metadata.interpDirection;
    const auto adjacent_interp = metadata.adjInterp;
    const auto dimension_coefficients = metadata.dimCoeffs;
    // The pinned official encoder writes Interp_Meta as an object image.  Its
    // ABI has padding between the five uint8 fields and float coefficient
    // array.  Zero only those otherwise indeterminate bytes, then restore the
    // exact official semantic fields.  This is enabled only for CFSR mode 1;
    // the frozen depth-0 encoder policy and all decoder behavior stay intact.
    std::memset(&metadata, 0, sizeof(metadata));
    metadata.interpAlgo = interp_algo;
    metadata.interpParadigm = interp_paradigm;
    metadata.cubicSplineType = cubic_spline_type;
    metadata.interpDirection = interp_direction;
    metadata.adjInterp = adjacent_interp;
    metadata.dimCoeffs = dimension_coefficients;
  };
  clear_padding(config.interpMeta);
  for (auto& metadata : config.interpMeta_list)
    clear_padding(metadata);
}

QoZ::Config tune_official_hpez(
    const std::vector<float>& original,
    const Settings& settings,
    double global_bound,
    bool canonicalize_interp_metadata = false)
{
  auto config = QoZ::Config(settings.channels, settings.time);
  config.cmprAlgo = QoZ::ALGO_INTERP;
  config.errorBoundMode = QoZ::EB_ABS;
  config.absErrorBound = global_bound;
  config.QoZ = settings.qoz_level;
  config.qoi = 0;
  config.verbose = false;
  config.rng = QoZ::data_range(original.data(), original.size());

  if (config.rng > std::numeric_limits<float>::epsilon()) {
    config.relErrorBound = config.absErrorBound / config.rng;
    auto tuning_data = original;
    // This is the official HPEZ predictor/anchor/interpolation tuning routine.
    // We retain its selected interpolation metadata even if its optional
    // Lorenzo comparison wins, because the QPET variable-EB path is the
    // official SZQoIInterpolationCompressor.
    Tuning<float, 2>(config, tuning_data.data());
    config.cmprAlgo = QoZ::ALGO_INTERP;
  }
  else {
    config.maxStep = 64;
    config.alpha = 1.5;
    config.beta = 2.0;
  }

  config.qoi = 16;
  config.qoi_tuned = true;
  config.use_global_eb = false;
  config.absErrorBound = global_bound;
  config.qoiEBBase = global_bound / 1030.0;
  config.qoiEBLogBase = 2.0;
  config.qoiQuantbinCnt = 128;
  if (canonicalize_interp_metadata)
    canonicalize_official_interp_metadata(config);
  return config;
}

std::shared_ptr<qpet_correlation::CorrelationQoI> make_qoi(
    const std::vector<float>& original,
    const Settings& settings,
    double global_bound,
    double estimator_scale = 1.0,
    qpet_correlation::BoundAllocation allocation =
        qpet_correlation::BoundAllocation::uniform)
{
  const auto posthoc = settings.correction_mode != 0;
  return std::make_shared<qpet_correlation::CorrelationQoI>(
      original,
      settings.channels,
      settings.time,
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag,
      global_bound,
       posthoc ? qpet_correlation::BoundEstimator::qpet_probabilistic
               : qpet_correlation::BoundEstimator::legacy_deterministic,
      allocation,
      posthoc ? qpet_correlation::ComplianceMode::posthoc
              : qpet_correlation::ComplianceMode::online,
      estimator_scale,
      settings.qpet_confidence,
      settings.qpet_k);
}

QPETCompressor make_compressor(
    const std::shared_ptr<QoZ::concepts::QoIInterface<float, 2>>& qoi,
    double global_bound)
{
  return QPETCompressor(
      QoZ::VariableEBLinearQuantizer<float, float>(65536 / 2),
      QoZ::EBLogQuantizer<float>(
          global_bound / 1030.0, 2.0, 128 / 2, global_bound),
      qoi,
      QoZ::QoIEncoder<int>(),
      QoZ::Lossless_zstd());
}

HPEZCompressor make_global_compressor(double global_bound)
{
  return HPEZCompressor(
      QoZ::LinearQuantizer<float>(global_bound, 65536 / 2),
      QoZ::HuffmanEncoder<int>(),
      QoZ::Lossless_zstd());
}

template <typename T>
void append_pod(std::vector<unsigned char>& output, const T& value)
{
  const auto* begin = reinterpret_cast<const unsigned char*>(&value);
  output.insert(output.end(), begin, begin + sizeof(T));
}

template <typename T>
T read_pod(const std::vector<unsigned char>& input, std::size_t& offset)
{
  if (offset > input.size() || input.size() - offset < sizeof(T))
    throw std::runtime_error("truncated embedded QPET payload");
  T value{};
  std::memcpy(&value, input.data() + offset, sizeof(T));
  offset += sizeof(T);
  return value;
}

void append_varint(std::vector<unsigned char>& output, std::uint64_t value)
{
  do {
    auto byte = static_cast<unsigned char>(value & 0x7fU);
    value >>= 7U;
    if (value != 0)
      byte |= 0x80U;
    output.push_back(byte);
  } while (value != 0);
}

std::uint64_t read_varint(
    const std::vector<unsigned char>& input, std::size_t& offset)
{
  std::uint64_t value = 0;
  for (unsigned int shift = 0; shift < 64; shift += 7) {
    if (offset >= input.size())
      throw std::runtime_error("truncated embedded QPET varint");
    const auto byte = input[offset++];
    if (shift == 63 && (byte & 0xfeU) != 0)
      throw std::runtime_error("embedded QPET varint overflow");
    value |= static_cast<std::uint64_t>(byte & 0x7fU) << shift;
    if ((byte & 0x80U) == 0)
      return value;
  }
  throw std::runtime_error("invalid embedded QPET varint");
}

std::size_t varint_size(std::uint64_t value)
{
  std::size_t bytes = 1;
  while (value >= 0x80U) {
    value >>= 7U;
    ++bytes;
  }
  return bytes;
}

std::uint32_t float32_bits(float value)
{
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "float32 bit comparison requires four bytes");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct StrictConstantLayout {
  std::vector<unsigned char> active_mask;
  std::vector<float> constant_values;
  std::vector<float> active_values;
  std::size_t active_count = 0;
};

StrictConstantLayout split_strict_constant_channels(
    const std::vector<float>& original,
    std::size_t channels,
    std::size_t time)
{
  if (channels == 0 || time == 0 ||
      channels > std::numeric_limits<std::size_t>::max() / time ||
      original.size() != channels * time)
    throw std::invalid_argument("strict-constant split shape mismatch");
  StrictConstantLayout layout;
  layout.active_mask.assign((channels + 7) / 8, 0);
  layout.active_values.reserve(original.size());
  layout.constant_values.reserve(channels);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto begin = channel * time;
    const auto first_bits = float32_bits(original[begin]);
    bool is_constant = true;
    for (std::size_t index = 1; index < time; ++index)
      is_constant = is_constant && float32_bits(original[begin + index]) == first_bits;
    if (is_constant) {
      layout.constant_values.push_back(original[begin]);
      continue;
    }
    layout.active_mask[channel / 8] |=
        static_cast<unsigned char>(1U << (channel % 8));
    ++layout.active_count;
    layout.active_values.insert(
        layout.active_values.end(),
        original.begin() + static_cast<std::ptrdiff_t>(begin),
        original.begin() + static_cast<std::ptrdiff_t>(begin + time));
  }
  if (layout.active_count + layout.constant_values.size() != channels ||
      layout.active_values.size() != layout.active_count * time)
    throw std::logic_error("strict-constant split accounting mismatch");
  return layout;
}

std::vector<unsigned char> make_strict_constant_block_payload(
    const StrictConstantLayout& layout,
    const std::vector<unsigned char>& official_payload)
{
  if (layout.constant_values.size() >
      (std::numeric_limits<std::size_t>::max() - layout.active_mask.size()) / sizeof(float) ||
      official_payload.size() >
          std::numeric_limits<std::size_t>::max() - layout.active_mask.size() -
              layout.constant_values.size() * sizeof(float))
    throw std::overflow_error("strict-constant block payload size overflow");
  auto payload = layout.active_mask;
  if (!layout.constant_values.empty()) {
    const auto* constant_bytes =
        reinterpret_cast<const unsigned char*>(layout.constant_values.data());
    payload.insert(
        payload.end(),
        constant_bytes,
        constant_bytes + layout.constant_values.size() * sizeof(float));
  }
  payload.insert(payload.end(), official_payload.begin(), official_payload.end());
  return payload;
}

struct ParsedStrictConstantBlock {
  std::vector<unsigned char> active_mask;
  std::vector<float> constant_values;
  std::vector<unsigned char> official_payload;
  std::size_t active_count = 0;
};

ParsedStrictConstantBlock parse_strict_constant_block_payload(
    const std::vector<unsigned char>& payload,
    std::size_t channels,
    std::uint32_t block_mode)
{
  if (channels == 0)
    throw std::runtime_error("strict-constant block has zero channels");
  const auto mask_bytes = (channels + 7) / 8;
  if (payload.size() < mask_bytes)
    throw std::runtime_error("truncated strict-constant active mask");
  ParsedStrictConstantBlock parsed;
  parsed.active_mask.assign(payload.begin(), payload.begin() + mask_bytes);
  if (channels % 8 != 0) {
    const auto valid_bits = static_cast<unsigned char>((1U << (channels % 8)) - 1U);
    if ((parsed.active_mask.back() & static_cast<unsigned char>(~valid_bits)) != 0)
      throw std::runtime_error("strict-constant active mask has out-of-range bits");
  }
  for (std::size_t channel = 0; channel < channels; ++channel)
    parsed.active_count +=
        (parsed.active_mask[channel / 8] &
         static_cast<unsigned char>(1U << (channel % 8))) != 0;
  const auto constant_count = channels - parsed.active_count;
  if (constant_count > (payload.size() - mask_bytes) / sizeof(float))
    throw std::runtime_error("truncated strict-constant float32 table");
  parsed.constant_values.resize(constant_count);
  if (constant_count != 0) {
    std::memcpy(
        parsed.constant_values.data(),
        payload.data() + mask_bytes,
        constant_count * sizeof(float));
  }
  for (const auto value : parsed.constant_values) {
    if (!std::isfinite(value))
      throw std::runtime_error("strict-constant table contains a non-finite value");
  }
  const auto official_offset = mask_bytes + constant_count * sizeof(float);
  parsed.official_payload.assign(
      payload.begin() + static_cast<std::ptrdiff_t>(official_offset), payload.end());
  if (block_mode == kSemanticConstantOnlyMode) {
    if (parsed.active_count != 0 || !parsed.official_payload.empty())
      throw std::runtime_error("constant-only block contains active channels or codec bytes");
  }
  else {
    if (block_mode != 3 && block_mode != 4 && block_mode != 6)
      throw std::runtime_error("schema-2 block uses an unsupported official codec");
    if (parsed.active_count == 0 || parsed.official_payload.empty())
      throw std::runtime_error("schema-2 codec block has no active data");
    if (block_mode == 3 && parsed.active_count < 2)
      throw std::runtime_error("schema-2 SPERR block requires at least two active channels");
  }
  return parsed;
}

std::vector<float> scatter_strict_constant_block(
    const ParsedStrictConstantBlock& parsed,
    const std::vector<float>& active_values,
    std::size_t channels,
    std::size_t time)
{
  if (channels == 0 || time == 0 ||
      channels > std::numeric_limits<std::size_t>::max() / time ||
      parsed.active_count > std::numeric_limits<std::size_t>::max() / time ||
      active_values.size() != parsed.active_count * time ||
      parsed.constant_values.size() != channels - parsed.active_count)
    throw std::runtime_error("strict-constant scatter dimensions are inconsistent");
  auto reconstructed = std::vector<float>(channels * time);
  std::size_t active_channel = 0;
  std::size_t constant_channel = 0;
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto begin = channel * time;
    const auto active =
        (parsed.active_mask[channel / 8] &
         static_cast<unsigned char>(1U << (channel % 8))) != 0;
    if (active) {
      const auto active_begin = active_channel * time;
      std::copy(
          active_values.begin() + static_cast<std::ptrdiff_t>(active_begin),
          active_values.begin() + static_cast<std::ptrdiff_t>(active_begin + time),
          reconstructed.begin() + static_cast<std::ptrdiff_t>(begin));
      ++active_channel;
    }
    else {
      const auto value = parsed.constant_values[constant_channel++];
      std::fill(
          reconstructed.begin() + static_cast<std::ptrdiff_t>(begin),
          reconstructed.begin() + static_cast<std::ptrdiff_t>(begin + time),
          value);
    }
  }
  if (active_channel != parsed.active_count ||
      constant_channel != parsed.constant_values.size())
    throw std::runtime_error("strict-constant scatter did not consume its channel tables");
  for (const auto value : reconstructed) {
    if (!std::isfinite(value))
      throw std::runtime_error("strict-constant scatter produced a non-finite value");
  }
  return reconstructed;
}

std::vector<float> apply_serf_neighbor_smoothing(
    const std::vector<float>& base,
    std::size_t channels,
    std::size_t time,
    const std::vector<float>& coefficients)
{
  if (base.size() != channels * time || coefficients.size() != channels || time < 3)
    throw std::invalid_argument("invalid Serf smoothing shape");
  auto reconstructed = base;
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto begin = channel * time;
    const auto coefficient = static_cast<double>(coefficients[channel]);
    for (std::size_t index = 1; index + 1 < time; ++index) {
      const auto center = static_cast<double>(base[begin + index]);
      const auto neighbor_mean = 0.5 * (
          static_cast<double>(base[begin + index - 1]) +
          static_cast<double>(base[begin + index + 1]));
      const auto value = center + coefficient * (neighbor_mean - center);
      if (!std::isfinite(value) ||
          std::abs(value) > static_cast<double>(std::numeric_limits<float>::max()))
        throw std::runtime_error("Serf smoothing produced a non-finite float32 value");
      reconstructed[begin + index] = static_cast<float>(value);
    }
  }
  return reconstructed;
}

void apply_correction_payload(
    std::vector<float>& reconstruction,
    const std::vector<unsigned char>& payload,
    std::uint32_t expected_corrections,
    std::size_t channels,
    std::size_t time,
    std::uint32_t stream_version = kVersion);

std::vector<float> decode_strict_constant_semantic_block(
    const std::vector<unsigned char>& payload,
    const Settings& settings,
    double global_bound,
    std::uint32_t block_mode);

std::vector<float> decode_payload(
    const std::vector<unsigned char>& payload,
    const Settings& settings,
    double global_bound,
    std::uint32_t base_mode = 0,
    std::uint32_t* semantic_correction_count = nullptr,
    std::uint32_t stream_version = kVersion)
{
  if (semantic_correction_count != nullptr)
    *semantic_correction_count = 0;
  if (payload.empty())
    throw std::runtime_error("empty official QPET payload");
  if (base_mode <= 2U && settings.channels < 2U)
    throw QoZ::CodecFormatError(
        "official two-dimensional HPEZ modes 0/1/2 are undefined for one channel");
  auto dummy = std::vector<float>(settings.channels * settings.time, 0.0F);
  auto reconstructed = std::vector<float>(dummy.size());
  if (base_mode == 0 || base_mode == 2) {
    const auto allocation = base_mode == 2
        ? qpet_correlation::BoundAllocation::gradient_balanced
        : qpet_correlation::BoundAllocation::uniform;
    auto qoi = make_qoi(dummy, settings, global_bound, 1.0, allocation);
    auto compressor = make_compressor(qoi, global_bound);
    compressor.decompress(payload.data(), payload.size(), reconstructed.data());
  }
  else if (base_mode == 1) {
    if (settings.channels < 2U)
      throw QoZ::CodecFormatError(
          "official two-dimensional HPEZ mode is undefined for one channel");
    auto compressor = make_global_compressor(global_bound);
    compressor.decompress(payload.data(), payload.size(), reconstructed.data());
  }
  else if (base_mode == 3) {
    if (payload.size() <= 10)
      throw std::runtime_error("truncated official SPERR payload");
    std::size_t dimx = 0;
    std::size_t dimy = 0;
    std::size_t dimz = 0;
    int is_float = 0;
    C_API::sperr_parse_header(
        const_cast<unsigned char*>(payload.data()),
        &dimx,
        &dimy,
        &dimz,
        &is_float);
    if (dimx != settings.time || dimy != settings.channels ||
        dimz != 1 || is_float != 1)
      throw std::runtime_error("official SPERR payload shape/type mismatch");
    void* decoded = nullptr;
    const auto status = C_API::sperr_decomp_2d(
        payload.data() + 10,
        payload.size() - 10,
        1,
        settings.time,
        settings.channels,
        &decoded);
    if (status != 0 || decoded == nullptr) {
      std::free(decoded);
      throw std::runtime_error(
          "official sperr_decomp_2d failed with code " + std::to_string(status));
    }
    const auto* values = static_cast<const float*>(decoded);
    std::copy(values, values + reconstructed.size(), reconstructed.begin());
    std::free(decoded);
  }
  else if (base_mode == 4) {
    auto zstd = QoZ::Lossless_zstd();
    auto decoded_size = payload.size();
    auto* decoded = zstd.decompress(payload.data(), decoded_size);
    const auto expected_bytes = reconstructed.size() * sizeof(float);
    if (decoded == nullptr || decoded_size != expected_bytes) {
      zstd.postdecompress_data(decoded);
      throw std::runtime_error("official Zstd lossless base payload has an invalid size");
    }
    std::memcpy(reconstructed.data(), decoded, expected_bytes);
    zstd.postdecompress_data(decoded);
  }
  else if (base_mode == 5 || base_mode == 6) {
    const auto bytes = Array<std::uint8_t>(
        std::vector<std::uint8_t>(payload.begin(), payload.end()));
    const auto values = base_mode == 5
        ? SerfXORDecompressor32().Decompress(bytes)
        : SerfQtDecompressor32().Decompress(bytes);
    if (values.size() != reconstructed.size())
      throw std::runtime_error("official Serf payload decoded an invalid value count");
    std::copy(values.begin(), values.end(), reconstructed.begin());
  }
  else if (base_mode == 7) {
    constexpr auto header_bytes = kSerfSmoothMagic.size() + 3 * sizeof(std::uint32_t);
    if (payload.size() < header_bytes)
      throw std::runtime_error("truncated Serf smoothing payload");
    if (!std::equal(
            kSerfSmoothMagic.begin(), kSerfSmoothMagic.end(), payload.begin()))
      throw std::runtime_error("invalid Serf smoothing payload magic");
    std::size_t offset = kSerfSmoothMagic.size();
    const auto channels = read_pod<std::uint32_t>(payload, offset);
    const auto time = read_pod<std::uint32_t>(payload, offset);
    const auto serf_size = read_pod<std::uint32_t>(payload, offset);
    if (channels != settings.channels || time != settings.time ||
        serf_size == 0 ||
        offset > payload.size() ||
        payload.size() - offset < channels * sizeof(float) ||
        payload.size() - offset - channels * sizeof(float) != serf_size)
      throw std::runtime_error("Serf smoothing payload shape or size mismatch");
    auto coefficients = std::vector<float>(channels);
    std::memcpy(
        coefficients.data(), payload.data() + offset, channels * sizeof(float));
    offset += channels * sizeof(float);
    const auto bytes = Array<std::uint8_t>(std::vector<std::uint8_t>(
        payload.begin() + static_cast<std::ptrdiff_t>(offset), payload.end()));
    const auto values = SerfQtDecompressor32().Decompress(bytes);
    if (values.size() != reconstructed.size())
      throw std::runtime_error("embedded official Serf payload decoded an invalid value count");
    const auto base = std::vector<float>(values.begin(), values.end());
    reconstructed = apply_serf_neighbor_smoothing(
        base, settings.channels, settings.time, coefficients);
  }
  else if (base_mode == 8 || base_mode == 9) {
    if (settings.channels != 16 ||
        payload.size() < kSemanticBlockMagic.size() + 1 ||
        !std::equal(
            kSemanticBlockMagic.begin(), kSemanticBlockMagic.end(), payload.begin()))
      throw std::runtime_error("invalid semantic-block payload header");
    std::size_t offset = kSemanticBlockMagic.size();
    const auto schema_id = payload[offset++];
    if (schema_id != kSemanticSchemaLegacy &&
        schema_id != kSemanticSchemaStrictConstants)
      throw std::runtime_error("unsupported semantic-block schema");
    for (const auto [channel_begin, channel_count] : kVehicleSignalGroups) {
      if (offset + 2 > payload.size())
        throw std::runtime_error("truncated semantic-block descriptor");
      const auto block_mode = static_cast<std::uint32_t>(payload[offset++]);
      const auto bound_index = payload[offset++];
      double block_bound = 0.0;
      if (block_mode == kSemanticConstantOnlyMode &&
          schema_id == kSemanticSchemaStrictConstants && bound_index == 0xffU)
        block_bound = 0.0;
      else if (block_mode == 3 && bound_index < kSperrBoundRatios.size())
        block_bound = settings.pointwise_tolerance * kSperrBoundRatios[bound_index];
      else if (block_mode == 6 && bound_index < kSerfBoundRatios.size())
        block_bound = settings.pointwise_tolerance * kSerfBoundRatios[bound_index];
      else if (block_mode == 4 && bound_index == 0xffU)
        block_bound = settings.pointwise_tolerance;
      else
        throw std::runtime_error("invalid semantic-block codec or bound index");
      const auto correction_count = base_mode == 8 ? read_varint(payload, offset) : 0;
      const auto block_size = read_varint(payload, offset);
      const auto correction_size = base_mode == 8 ? read_varint(payload, offset) : 0;
      if (correction_count > std::numeric_limits<std::uint32_t>::max() ||
          block_size == 0 || block_size > std::numeric_limits<std::size_t>::max() ||
          correction_size > std::numeric_limits<std::size_t>::max() ||
          offset > payload.size() || block_size > payload.size() - offset ||
          correction_size > payload.size() - offset - block_size)
        throw std::runtime_error("invalid semantic-block payload size");
      if (base_mode == 8 && semantic_correction_count != nullptr) {
        if (correction_count >
            std::numeric_limits<std::uint32_t>::max() -
                *semantic_correction_count)
          throw std::overflow_error(
              "semantic inner correction count exceeds uint32 metadata");
        *semantic_correction_count += static_cast<std::uint32_t>(correction_count);
      }
      const auto block_end = offset + static_cast<std::size_t>(block_size);
      auto block_payload = std::vector<unsigned char>(
          payload.begin() + static_cast<std::ptrdiff_t>(offset),
          payload.begin() + static_cast<std::ptrdiff_t>(block_end));
      offset = block_end;
      auto correction_payload = std::vector<unsigned char>(
          payload.begin() + static_cast<std::ptrdiff_t>(offset),
          payload.begin() + static_cast<std::ptrdiff_t>(
              offset + static_cast<std::size_t>(correction_size)));
      offset += static_cast<std::size_t>(correction_size);
      auto block_settings = settings;
      block_settings.channels = channel_count;
      const auto block = schema_id == kSemanticSchemaLegacy
          ? decode_payload(block_payload, block_settings, block_bound, block_mode)
          : decode_strict_constant_semantic_block(
                block_payload, block_settings, block_bound, block_mode);
      auto repaired = block;
      if (base_mode == 8)
        apply_correction_payload(
            repaired,
            correction_payload,
            static_cast<std::uint32_t>(correction_count),
            channel_count,
            settings.time,
            stream_version);
      std::copy(
          repaired.begin(),
          repaired.end(),
          reconstructed.begin() + channel_begin * settings.time);
    }
    if (offset != payload.size())
      throw std::runtime_error("semantic-block payload has trailing bytes");
  }
  else {
    throw std::runtime_error("unsupported QPET base compressor mode");
  }
  return reconstructed;
}

std::vector<float> decode_strict_constant_semantic_block(
    const std::vector<unsigned char>& payload,
    const Settings& settings,
    double global_bound,
    std::uint32_t block_mode)
{
  const auto parsed = parse_strict_constant_block_payload(
      payload, settings.channels, block_mode);
  if (block_mode == kSemanticConstantOnlyMode)
    return scatter_strict_constant_block(
        parsed, {}, settings.channels, settings.time);
  auto active_settings = settings;
  active_settings.channels = parsed.active_count;
  const auto active = decode_payload(
      parsed.official_payload, active_settings, global_bound, block_mode);
  return scatter_strict_constant_block(
      parsed, active, settings.channels, settings.time);
}

std::vector<unsigned char> compress_sperr_payload(
    const std::vector<float>& original,
    const Settings& settings,
    double pointwise_bound)
{
  void* compressed = nullptr;
  std::size_t compressed_size = 0;
  const auto status = C_API::sperr_comp_2d(
      original.data(),
      1,
      settings.time,
      settings.channels,
      3,
      pointwise_bound,
      1,
      &compressed,
      &compressed_size);
  if (status != 0 || compressed == nullptr || compressed_size <= 10) {
    std::free(compressed);
    throw BackendCandidateFailure(
        "official sperr_comp_2d failed with code " + std::to_string(status));
  }
  auto payload = std::vector<unsigned char>(
      static_cast<unsigned char*>(compressed),
      static_cast<unsigned char*>(compressed) + compressed_size);
  std::free(compressed);
  return payload;
}

std::vector<unsigned char> compress_serf_payload(
    const std::vector<float>& original,
    double pointwise_bound,
    std::uint32_t base_mode)
{
  if (!(pointwise_bound > 0.0) ||
      pointwise_bound > std::numeric_limits<float>::max() ||
      original.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("Serf point-wise bound or window size is invalid");
  Array<std::uint8_t> encoded;
  if (base_mode == 5) {
    auto compressor = SerfXORCompressor32(
        static_cast<int>(original.size()), static_cast<float>(pointwise_bound));
    for (const auto value : original)
      compressor.AddValue(value);
    compressor.Close();
    encoded = compressor.compressed_bytes_last_block();
  }
  else if (base_mode == 6) {
    auto compressor = SerfQtCompressor32(
        static_cast<int>(original.size()), static_cast<float>(pointwise_bound));
    for (const auto value : original)
      compressor.AddValue(value);
    compressor.Close();
    encoded = compressor.compressed_bytes();
  }
  else {
    throw std::invalid_argument("invalid Serf base compressor mode");
  }
  if (encoded.length() <= 0)
    throw BackendCandidateFailure("official Serf compressor returned an empty stream");
  return std::vector<unsigned char>(encoded.begin(), encoded.end());
}

std::vector<unsigned char> compress_serf_smooth_payload(
    const std::vector<float>& original,
    const Settings& settings,
    double pointwise_bound,
    double strength)
{
  if (!(strength > 0.0) || !std::isfinite(strength))
    throw std::invalid_argument("Serf smoothing strength must be finite and positive");
  const auto serf = compress_serf_payload(original, pointwise_bound, 6);
  const auto base = decode_payload(serf, settings, pointwise_bound, 6);
  auto coefficients = std::vector<float>(settings.channels, 0.0F);
  for (std::size_t channel = 0; channel < settings.channels; ++channel) {
    const auto begin = channel * settings.time;
    long double numerator = 0.0L;
    long double denominator = 0.0L;
    for (std::size_t index = 1; index + 1 < settings.time; ++index) {
      const auto center = static_cast<long double>(base[begin + index]);
      const auto feature = 0.5L * (
          static_cast<long double>(base[begin + index - 1]) +
          static_cast<long double>(base[begin + index + 1])) - center;
      const auto residual =
          static_cast<long double>(original[begin + index]) - center;
      numerator += feature * residual;
      denominator += feature * feature;
    }
    const auto fitted = denominator > 1e-30L
        ? strength * static_cast<double>(numerator / denominator)
        : 0.0;
    coefficients[channel] = static_cast<float>(std::clamp(fitted, -2.0, 2.0));
  }

  auto payload = std::vector<unsigned char>();
  payload.reserve(
      kSerfSmoothMagic.size() + 3 * sizeof(std::uint32_t) +
      coefficients.size() * sizeof(float) + serf.size());
  payload.insert(payload.end(), kSerfSmoothMagic.begin(), kSerfSmoothMagic.end());
  append_pod(payload, static_cast<std::uint32_t>(settings.channels));
  append_pod(payload, static_cast<std::uint32_t>(settings.time));
  if (serf.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("embedded Serf payload is too large");
  append_pod(payload, static_cast<std::uint32_t>(serf.size()));
  const auto* coefficient_bytes = reinterpret_cast<const unsigned char*>(
      coefficients.data());
  payload.insert(
      payload.end(),
      coefficient_bytes,
      coefficient_bytes + coefficients.size() * sizeof(float));
  payload.insert(payload.end(), serf.begin(), serf.end());
  return payload;
}

std::vector<unsigned char> compress_lossless_zstd_payload(
    const std::vector<float>& original)
{
  auto zstd = QoZ::Lossless_zstd();
  auto* raw = reinterpret_cast<unsigned char*>(
      const_cast<float*>(original.data()));
  const auto raw_size = original.size() * sizeof(float);
  std::size_t compressed_size = 0;
  auto* compressed = zstd.compress(raw, raw_size, compressed_size);
  if (compressed == nullptr || compressed_size == 0) {
    zstd.postcompress_data(compressed);
    throw BackendCandidateFailure(
        "official Zstd lossless base compressor returned an empty stream");
  }
  auto payload = std::vector<unsigned char>(
      compressed, compressed + compressed_size);
  zstd.postcompress_data(compressed);
  return payload;
}

std::vector<unsigned char> compress_zstd_correction(
    const std::vector<unsigned char>& raw)
{
  if (raw.empty())
    throw std::invalid_argument("cannot compress an empty QPET correction field");
  auto zstd = QoZ::Lossless_zstd();
  std::size_t compressed_size = 0;
  auto* compressed = zstd.compress(
      const_cast<unsigned char*>(raw.data()), raw.size(), compressed_size);
  if (compressed == nullptr || compressed_size == 0) {
    zstd.postcompress_data(compressed);
    throw std::runtime_error("official Zstd correction compressor returned an empty stream");
  }
  auto payload = std::vector<unsigned char>(compressed, compressed + compressed_size);
  zstd.postcompress_data(compressed);
  return payload;
}

std::vector<unsigned char> compress_direct_zstd_frame(
    const std::vector<unsigned char>& raw)
{
  if (raw.empty())
    throw std::invalid_argument("cannot directly compress an empty correction frame");
  const auto bound = ZSTD_compressBound(raw.size());
  if (ZSTD_isError(bound) || bound == 0)
    throw std::runtime_error("official ZSTD_compressBound failed");
  auto compressed = std::vector<unsigned char>(bound);
  const auto compressed_size = ZSTD_compress(
      compressed.data(),
      compressed.size(),
      raw.data(),
      raw.size(),
      ZSTD_CLEVEL_DEFAULT);
  if (ZSTD_isError(compressed_size) || compressed_size == 0)
    throw std::runtime_error(
        "official ZSTD_compress failed: " +
        std::string(ZSTD_getErrorName(compressed_size)));
  compressed.resize(compressed_size);
  return compressed;
}

std::vector<unsigned char> decompress_direct_zstd_frame(
    const unsigned char* compressed,
    std::size_t compressed_size,
    std::size_t maximum_raw_size)
{
  if (compressed == nullptr || compressed_size == 0 || maximum_raw_size == 0)
    throw std::runtime_error("packed correction ZSTD frame is empty");
  const auto frame_size = ZSTD_findFrameCompressedSize(compressed, compressed_size);
  if (ZSTD_isError(frame_size) || frame_size != compressed_size)
    throw std::runtime_error(
        "packed correction must contain exactly one complete ZSTD frame");
  const auto declared_size = ZSTD_getFrameContentSize(compressed, compressed_size);
  if (declared_size == ZSTD_CONTENTSIZE_ERROR ||
      declared_size == ZSTD_CONTENTSIZE_UNKNOWN || declared_size == 0 ||
      declared_size > maximum_raw_size ||
      declared_size > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("packed correction ZSTD raw size is invalid");
  auto raw = std::vector<unsigned char>(static_cast<std::size_t>(declared_size));
  const auto decoded_size = ZSTD_decompress(
      raw.data(), raw.size(), compressed, compressed_size);
  if (ZSTD_isError(decoded_size) || decoded_size != raw.size())
    throw std::runtime_error("packed correction ZSTD decompression size mismatch");
  return raw;
}

std::uint8_t zigzag_int8(std::int8_t value)
{
  const auto promoted = static_cast<int>(value);
  const auto encoded = promoted >= 0
      ? 2 * promoted
      : 2 * (-promoted) - 1;
  if (encoded < 0 || encoded > std::numeric_limits<std::uint8_t>::max())
    throw std::logic_error("int8 zigzag code is out of range");
  return static_cast<std::uint8_t>(encoded);
}

std::int8_t unzigzag_int8(std::uint8_t value)
{
  const auto decoded = (value & 1U) == 0
      ? static_cast<int>(value / 2U)
      : -static_cast<int>((static_cast<unsigned int>(value) + 1U) / 2U);
  if (decoded < std::numeric_limits<std::int8_t>::min() ||
      decoded > std::numeric_limits<std::int8_t>::max())
    throw std::runtime_error("packed correction zigzag code is invalid");
  return static_cast<std::int8_t>(decoded);
}

struct PackedInt8Codes {
  std::uint8_t bit_width = 1;
  std::vector<unsigned char> bytes;
};

PackedInt8Codes pack_int8_codes(const std::vector<std::int8_t>& codes)
{
  if (codes.empty())
    throw std::invalid_argument("cannot pack an empty correction code vector");
  std::uint8_t maximum = 0;
  auto zigzag = std::vector<std::uint8_t>();
  zigzag.reserve(codes.size());
  for (const auto code : codes) {
    const auto value = zigzag_int8(code);
    maximum = std::max(maximum, value);
    zigzag.push_back(value);
  }
  std::uint8_t bit_width = 1;
  while (bit_width < 8 && (maximum >> bit_width) != 0)
    ++bit_width;
  if (codes.size() >
      (std::numeric_limits<std::size_t>::max() - 7U) / bit_width)
    throw std::overflow_error("packed correction code size overflow");
  const auto bit_count = codes.size() * bit_width;
  auto bytes = std::vector<unsigned char>((bit_count + 7U) / 8U, 0);
  for (std::size_t index = 0; index < zigzag.size(); ++index) {
    for (std::uint8_t bit = 0; bit < bit_width; ++bit) {
      if ((zigzag[index] & static_cast<std::uint8_t>(1U << bit)) == 0)
        continue;
      const auto bit_index = index * bit_width + bit;
      bytes[bit_index / 8U] |=
          static_cast<unsigned char>(1U << (bit_index % 8U));
    }
  }
  return {bit_width, std::move(bytes)};
}

std::vector<std::int8_t> unpack_int8_codes(
    const unsigned char* packed,
    std::size_t packed_size,
    std::size_t code_count,
    std::uint8_t bit_width)
{
  if (packed == nullptr || code_count == 0 || bit_width < 1 || bit_width > 8 ||
      code_count >
          (std::numeric_limits<std::size_t>::max() - 7U) / bit_width)
    throw std::runtime_error("packed correction code framing is invalid");
  const auto bit_count = code_count * bit_width;
  const auto expected_size = (bit_count + 7U) / 8U;
  if (packed_size != expected_size)
    throw std::runtime_error("packed correction code byte count is invalid");
  const auto used_bits = bit_count % 8U;
  if (used_bits != 0) {
    const auto padding_mask = static_cast<unsigned char>(
        0xffU << static_cast<unsigned int>(used_bits));
    if ((packed[packed_size - 1] & padding_mask) != 0)
      throw std::runtime_error("packed correction code padding is nonzero");
  }
  auto codes = std::vector<std::int8_t>();
  codes.reserve(code_count);
  for (std::size_t index = 0; index < code_count; ++index) {
    std::uint8_t zigzag = 0;
    for (std::uint8_t bit = 0; bit < bit_width; ++bit) {
      const auto bit_index = index * bit_width + bit;
      if ((packed[bit_index / 8U] &
           static_cast<unsigned char>(1U << (bit_index % 8U))) != 0)
        zigzag |= static_cast<std::uint8_t>(1U << bit);
    }
    codes.push_back(unzigzag_int8(zigzag));
  }
  return codes;
}

struct PackedUint9Codes {
  std::uint8_t bit_width = 1;
  std::vector<unsigned char> bytes;
};

PackedUint9Codes pack_uint9_codes(const std::vector<std::uint16_t>& values)
{
  if (values.empty())
    throw std::invalid_argument("cannot pack an empty uint9 code vector");
  auto maximum = std::uint16_t{0};
  for (const auto value : values) {
    if (value > 510U)
      throw std::invalid_argument("predictive correction delta exceeds uint9");
    maximum = std::max(maximum, value);
  }
  auto bit_width = std::uint8_t{1};
  while (bit_width < 9U && (maximum >> bit_width) != 0U)
    ++bit_width;
  if (values.size() >
      (std::numeric_limits<std::size_t>::max() - 7U) / bit_width)
    throw std::overflow_error("predictive packed delta size overflow");
  const auto bit_count = values.size() * bit_width;
  auto bytes = std::vector<unsigned char>((bit_count + 7U) / 8U, 0U);
  for (std::size_t index = 0; index < values.size(); ++index) {
    for (std::uint8_t bit = 0; bit < bit_width; ++bit) {
      if ((values[index] & static_cast<std::uint16_t>(1U << bit)) == 0U)
        continue;
      const auto bit_index = index * bit_width + bit;
      bytes[bit_index / 8U] |=
          static_cast<unsigned char>(1U << (bit_index % 8U));
    }
  }
  return {bit_width, std::move(bytes)};
}

std::vector<std::uint16_t> unpack_uint9_codes(
    const unsigned char* packed,
    std::size_t packed_size,
    std::size_t code_count,
    std::uint8_t bit_width)
{
  if (packed == nullptr || code_count == 0 || bit_width < 1U ||
      bit_width > 9U ||
      code_count >
          (std::numeric_limits<std::size_t>::max() - 7U) / bit_width)
    throw std::runtime_error("predictive packed delta framing is invalid");
  const auto bit_count = code_count * bit_width;
  const auto expected_size = (bit_count + 7U) / 8U;
  if (packed_size != expected_size)
    throw std::runtime_error("predictive packed delta byte count is invalid");
  const auto used_bits = bit_count % 8U;
  if (used_bits != 0U) {
    const auto padding_mask = static_cast<unsigned char>(
        0xffU << static_cast<unsigned int>(used_bits));
    if ((packed[packed_size - 1U] & padding_mask) != 0U)
      throw std::runtime_error("predictive packed delta padding is nonzero");
  }
  auto values = std::vector<std::uint16_t>();
  values.reserve(code_count);
  for (std::size_t index = 0; index < code_count; ++index) {
    auto value = std::uint16_t{0};
    for (std::uint8_t bit = 0; bit < bit_width; ++bit) {
      const auto bit_index = index * bit_width + bit;
      if ((packed[bit_index / 8U] &
           static_cast<unsigned char>(1U << (bit_index % 8U))) != 0U)
        value |= static_cast<std::uint16_t>(1U << bit);
    }
    if (value > 510U)
      throw std::runtime_error("predictive packed delta is out of range");
    values.push_back(value);
  }
  return values;
}

std::uint16_t zigzag_int8_delta(int delta)
{
  if (delta < -255 || delta > 255)
    throw std::invalid_argument("int8 predictive delta is out of range");
  return static_cast<std::uint16_t>(
      delta >= 0 ? 2 * delta : -2 * delta - 1);
}

int unzigzag_int8_delta(std::uint16_t value)
{
  if (value > 510U)
    throw std::runtime_error("predictive ZigZag delta is out of range");
  return (value & 1U) == 0U
      ? static_cast<int>(value / 2U)
      : -static_cast<int>((value + 1U) / 2U);
}

struct QuantizedRepairTargets {
  std::vector<float> values;
  std::vector<float> channel_scales;
};

QuantizedRepairTargets make_quantized_repair_targets(
    const std::vector<float>& original,
    const std::vector<float>& base,
    const Settings& settings,
    int denominator)
{
  if (original.size() != base.size() ||
      original.size() != settings.channels * settings.time)
    throw std::invalid_argument("quantized repair target shape mismatch");
  if (denominator < 1 || denominator > 127)
    throw std::invalid_argument("quantized repair denominator must be in [1,127]");
  QuantizedRepairTargets result{base, std::vector<float>(settings.channels, 0.0F)};
  for (std::size_t channel = 0; channel < settings.channels; ++channel) {
    const auto begin = channel * settings.time;
    double largest_residual = 0.0;
    for (std::size_t index = 0; index < settings.time; ++index) {
      largest_residual = std::max(
          largest_residual,
          std::abs(static_cast<double>(original[begin + index]) - base[begin + index]));
    }
    if (!(largest_residual > 0.0))
      continue;
    const auto scale = static_cast<float>(
        largest_residual / static_cast<double>(denominator));
    if (!(scale > 0.0F) || !std::isfinite(scale))
      continue;
    result.channel_scales[channel] = scale;
    for (std::size_t index = 0; index < settings.time; ++index) {
      const auto offset = begin + index;
      const auto residual = static_cast<double>(original[offset]) - base[offset];
      auto code = static_cast<int>(std::llround(residual / scale));
      code = std::clamp(code, -denominator, denominator);
      if (code == 0)
        continue;
      const auto target = static_cast<float>(
          base[offset] + static_cast<float>(code) * scale);
      const auto old_error = std::abs(
          static_cast<double>(original[offset]) - base[offset]);
      const auto new_error = std::abs(
          static_cast<double>(original[offset]) - target);
      if (std::isfinite(target) && new_error <= old_error)
        result.values[offset] = target;
    }

    const auto original_series = std::vector<float>(
        original.begin() + begin, original.begin() + begin + settings.time);
    const auto target_series = std::vector<float>(
        result.values.begin() + begin, result.values.begin() + begin + settings.time);
    const auto report = qpet_correlation::verify_constraints(
        original_series,
        target_series,
        1,
        settings.time,
        settings.pointwise_tolerance,
        settings.acf_tolerance,
        settings.minimum_correlation,
        settings.max_lag);
    if (!report.all_ok()) {
      // Ill-conditioned channels use explicit exact escapes. This preserves
      // deterministic feasibility without forcing every other channel to be exact.
      std::copy(
          original.begin() + begin,
          original.begin() + begin + settings.time,
          result.values.begin() + begin);
    }
  }
  return result;
}

struct QuantizedCorrectionFields {
  std::vector<unsigned char> mask;
  std::vector<std::int8_t> codes;
  std::vector<float> exact_values;
};

QuantizedCorrectionFields make_quantized_correction_fields(
    const std::vector<float>& base,
    const std::vector<float>& corrected,
    const std::vector<float>& channel_scales,
    std::size_t channels,
    std::size_t time)
{
  if (channels == 0 || time == 0 ||
      channels > std::numeric_limits<std::size_t>::max() / time ||
      base.size() != corrected.size() || base.size() != channels * time ||
      channel_scales.size() != channels)
    throw std::invalid_argument("quantized correction vectors have different sizes");
  QuantizedCorrectionFields fields;
  fields.mask.assign(base.size() / 8U + (base.size() % 8U != 0), 0);
  for (std::size_t index = 0; index < base.size(); ++index) {
    if (base[index] == corrected[index])
      continue;
    fields.mask[index / 8] |= static_cast<unsigned char>(1U << (index % 8));
    const auto scale = channel_scales[index / time];
    auto code = 0;
    if (scale > 0.0F && std::isfinite(scale)) {
      const auto ratio =
          (static_cast<double>(corrected[index]) - base[index]) / scale;
      if (std::isfinite(ratio) &&
          ratio >= static_cast<double>(std::numeric_limits<std::int8_t>::min()) - 0.5 &&
          ratio <= static_cast<double>(std::numeric_limits<std::int8_t>::max()) + 0.5) {
        const auto rounded = std::llround(ratio);
        if (rounded >= std::numeric_limits<std::int8_t>::min() &&
            rounded <= std::numeric_limits<std::int8_t>::max() && rounded != 0) {
          const auto candidate = static_cast<float>(
              base[index] + static_cast<float>(rounded) * scale);
          if (candidate == corrected[index])
            code = static_cast<int>(rounded);
        }
      }
    }
    fields.codes.push_back(static_cast<std::int8_t>(code));
    if (code == 0)
      fields.exact_values.push_back(corrected[index]);
  }
  return fields;
}

std::vector<unsigned char> encode_quantized_correction_format4(
    const std::vector<float>& channel_scales,
    const QuantizedCorrectionFields& fields)
{
  if (fields.codes.empty())
    return {};
  if (channel_scales.size() >
      std::numeric_limits<std::size_t>::max() / sizeof(float))
    throw std::overflow_error("quantized correction scale size overflow");
  const auto scale_bytes = channel_scales.size() * sizeof(float);
  const auto code_bytes = fields.codes.size() * sizeof(std::int8_t);
  if (fields.exact_values.size() >
      std::numeric_limits<std::size_t>::max() / sizeof(float))
    throw std::overflow_error("quantized correction exact size overflow");
  const auto exact_bytes = fields.exact_values.size() * sizeof(float);
  if (scale_bytes > std::numeric_limits<std::size_t>::max() - fields.mask.size() ||
      scale_bytes + fields.mask.size() >
          std::numeric_limits<std::size_t>::max() - code_bytes ||
      scale_bytes + fields.mask.size() + code_bytes >
          std::numeric_limits<std::size_t>::max() - exact_bytes)
    throw std::overflow_error("quantized correction payload size overflow");
  auto raw = std::vector<unsigned char>(
      scale_bytes + fields.mask.size() + code_bytes + exact_bytes);
  auto* cursor = raw.data();
  std::memcpy(cursor, channel_scales.data(), scale_bytes);
  cursor += scale_bytes;
  std::memcpy(cursor, fields.mask.data(), fields.mask.size());
  cursor += fields.mask.size();
  std::memcpy(cursor, fields.codes.data(), code_bytes);
  cursor += code_bytes;
  if (exact_bytes != 0)
    std::memcpy(cursor, fields.exact_values.data(), exact_bytes);
  auto compressed = compress_zstd_correction(raw);
  auto payload = std::vector<unsigned char>();
  payload.reserve(compressed.size() + 1);
  payload.push_back(4U);
  payload.insert(payload.end(), compressed.begin(), compressed.end());
  return payload;
}

std::vector<unsigned char> encode_quantized_correction_format6(
    const std::vector<float>& channel_scales,
    const QuantizedCorrectionFields& fields)
{
  if (fields.codes.empty())
    return {};
  if (channel_scales.size() >
          std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      fields.exact_values.size() >
          std::numeric_limits<std::size_t>::max() / sizeof(float))
    throw std::overflow_error("packed correction field size overflow");
  const auto packed = pack_int8_codes(fields.codes);
  const auto scale_bytes = channel_scales.size() * sizeof(float);
  const auto exact_bytes = fields.exact_values.size() * sizeof(float);
  std::size_t raw_size = 1;
  auto add_size = [&](std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - raw_size)
      throw std::overflow_error("packed correction payload size overflow");
    raw_size += bytes;
  };
  add_size(scale_bytes);
  add_size(fields.mask.size());
  add_size(packed.bytes.size());
  add_size(exact_bytes);
  auto raw = std::vector<unsigned char>(raw_size);
  auto* cursor = raw.data();
  *cursor++ = packed.bit_width;
  std::memcpy(cursor, channel_scales.data(), scale_bytes);
  cursor += scale_bytes;
  std::memcpy(cursor, fields.mask.data(), fields.mask.size());
  cursor += fields.mask.size();
  std::memcpy(cursor, packed.bytes.data(), packed.bytes.size());
  cursor += packed.bytes.size();
  if (exact_bytes != 0)
    std::memcpy(cursor, fields.exact_values.data(), exact_bytes);
  const auto compressed = compress_direct_zstd_frame(raw);
  auto payload = std::vector<unsigned char>();
  payload.reserve(1 + compressed.size());
  payload.push_back(kPackedQuantizedCorrectionFormat);
  payload.insert(payload.end(), compressed.begin(), compressed.end());
  return payload;
}

std::vector<unsigned char> encode_adaptive_packed_correction_format7(
    const std::vector<float>& base,
    const std::vector<float>& corrected,
    const std::vector<float>& channel_scales,
    const QuantizedCorrectionFields& fields,
    std::size_t channels,
    std::size_t time)
{
  if (fields.codes.empty())
    return {};
  if (channels == 0 || time == 0 ||
      channels > std::numeric_limits<std::size_t>::max() / time ||
      base.size() != channels * time || corrected.size() != base.size() ||
      channel_scales.size() != channels)
    throw std::invalid_argument("adaptive packed correction dimensions are invalid");
  auto bit_widths = std::vector<std::uint8_t>(channels, 0);
  auto packed_codes = std::vector<unsigned char>();
  auto exact_xors = std::vector<std::uint32_t>();
  std::size_t code_offset = 0;
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto begin = channel * time;
    auto channel_codes = std::vector<std::int8_t>();
    for (std::size_t index = 0; index < time; ++index) {
      const auto offset = begin + index;
      const auto selected =
          (fields.mask[offset / 8U] &
           static_cast<unsigned char>(1U << (offset % 8U))) != 0;
      if (!selected)
        continue;
      if (code_offset >= fields.codes.size())
        throw std::logic_error("adaptive packed correction code accounting diverged");
      const auto code = fields.codes[code_offset++];
      channel_codes.push_back(code);
      if (code == 0)
        exact_xors.push_back(float32_bits(base[offset]) ^
            float32_bits(corrected[offset]));
    }
    if (channel_codes.empty())
      continue;
    const auto packed = pack_int8_codes(channel_codes);
    bit_widths[channel] = packed.bit_width;
    packed_codes.insert(
        packed_codes.end(), packed.bytes.begin(), packed.bytes.end());
  }
  if (code_offset != fields.codes.size() ||
      exact_xors.size() != fields.exact_values.size())
    throw std::logic_error("adaptive packed correction fields are inconsistent");
  if (channel_scales.size() >
          std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      exact_xors.size() >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t))
    throw std::overflow_error("adaptive packed correction field size overflow");
  const auto scale_bytes = channel_scales.size() * sizeof(float);
  const auto exact_bytes = exact_xors.size() * sizeof(std::uint32_t);
  std::size_t raw_size = 0;
  auto add_size = [&](std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - raw_size)
      throw std::overflow_error("adaptive packed correction payload size overflow");
    raw_size += bytes;
  };
  add_size(scale_bytes);
  add_size(fields.mask.size());
  add_size(bit_widths.size());
  add_size(packed_codes.size());
  add_size(exact_bytes);
  auto raw = std::vector<unsigned char>(raw_size);
  auto* cursor = raw.data();
  std::memcpy(cursor, channel_scales.data(), scale_bytes);
  cursor += scale_bytes;
  std::memcpy(cursor, fields.mask.data(), fields.mask.size());
  cursor += fields.mask.size();
  std::memcpy(cursor, bit_widths.data(), bit_widths.size());
  cursor += bit_widths.size();
  if (!packed_codes.empty()) {
    std::memcpy(cursor, packed_codes.data(), packed_codes.size());
    cursor += packed_codes.size();
  }
  if (exact_bytes != 0)
    std::memcpy(cursor, exact_xors.data(), exact_bytes);
  const auto compressed = compress_direct_zstd_frame(raw);
  auto payload = std::vector<unsigned char>();
  payload.reserve(1 + compressed.size());
  payload.push_back(kAdaptivePackedCorrectionFormat);
  payload.insert(payload.end(), compressed.begin(), compressed.end());
  return payload;
}

enum class FactorizedPositionMode : std::uint8_t {
  none = 0,
  all = 1,
  bitmap = 2,
  gaps = 3,
  complement_gaps = 4,
};

enum class FactorizedValueMode : std::uint8_t {
  none = 0,
  constant = 1,
  bitpack = 2,
  run_length = 3,
  modal_patched = 4,
  delta_bitpack = 5,
  delta_modal_patched = 6,
};

struct FactorizedFieldEncoding {
  std::uint8_t mode = 0;
  std::vector<unsigned char> bytes;
};

bool factorized_field_less(
    const FactorizedFieldEncoding& left,
    const FactorizedFieldEncoding& right)
{
  if (left.bytes.size() != right.bytes.size())
    return left.bytes.size() < right.bytes.size();
  return left.mode < right.mode;
}

FactorizedFieldEncoding encode_factorized_positions(
    const std::vector<std::size_t>& positions,
    std::size_t time)
{
  if (time == 0 || positions.size() > time)
    throw std::invalid_argument("factorized correction positions are invalid");
  for (std::size_t index = 0; index < positions.size(); ++index) {
    if (positions[index] >= time ||
        (index != 0 && positions[index - 1] >= positions[index]))
      throw std::invalid_argument(
          "factorized correction positions are not strictly increasing");
  }
  if (positions.empty())
    return {static_cast<std::uint8_t>(FactorizedPositionMode::none), {}};
  if (positions.size() == time)
    return {static_cast<std::uint8_t>(FactorizedPositionMode::all), {}};

  auto candidates = std::vector<FactorizedFieldEncoding>();
  auto bitmap = std::vector<unsigned char>(
      time / 8U + (time % 8U != 0), 0);
  for (const auto position : positions)
    bitmap[position / 8U] |=
        static_cast<unsigned char>(1U << (position % 8U));
  candidates.push_back(
      {static_cast<std::uint8_t>(FactorizedPositionMode::bitmap),
       std::move(bitmap)});

  auto gaps = std::vector<unsigned char>();
  append_varint(gaps, positions.size());
  auto previous = std::size_t{0};
  for (std::size_t index = 0; index < positions.size(); ++index) {
    const auto delta = index == 0
        ? positions[index] + 1U
        : positions[index] - previous;
    append_varint(gaps, delta);
    previous = positions[index];
  }
  candidates.push_back(
      {static_cast<std::uint8_t>(FactorizedPositionMode::gaps),
       std::move(gaps)});

  auto complement = std::vector<std::size_t>();
  complement.reserve(time - positions.size());
  auto selected_index = std::size_t{0};
  for (std::size_t position = 0; position < time; ++position) {
    if (selected_index < positions.size() &&
        positions[selected_index] == position) {
      ++selected_index;
    }
    else {
      complement.push_back(position);
    }
  }
  auto complement_gaps = std::vector<unsigned char>();
  append_varint(complement_gaps, complement.size());
  previous = 0;
  for (std::size_t index = 0; index < complement.size(); ++index) {
    const auto delta = index == 0
        ? complement[index] + 1U
        : complement[index] - previous;
    append_varint(complement_gaps, delta);
    previous = complement[index];
  }
  candidates.push_back(
      {static_cast<std::uint8_t>(FactorizedPositionMode::complement_gaps),
       std::move(complement_gaps)});

  return *std::min_element(
      candidates.begin(), candidates.end(), factorized_field_less);
}

FactorizedFieldEncoding encode_factorized_values(
    const std::vector<std::int8_t>& codes)
{
  if (codes.empty())
    return {static_cast<std::uint8_t>(FactorizedValueMode::none), {}};
  auto candidates = std::vector<FactorizedFieldEncoding>();
  if (std::all_of(codes.begin(), codes.end(), [&](const auto code) {
        return code == codes.front();
      })) {
    candidates.push_back(
        {static_cast<std::uint8_t>(FactorizedValueMode::constant),
         {zigzag_int8(codes.front())}});
  }

  const auto packed = pack_int8_codes(codes);
  auto bitpacked = std::vector<unsigned char>{packed.bit_width};
  bitpacked.insert(
      bitpacked.end(), packed.bytes.begin(), packed.bytes.end());
  candidates.push_back(
      {static_cast<std::uint8_t>(FactorizedValueMode::bitpack),
       std::move(bitpacked)});

  struct Run {
    std::int8_t code = 0;
    std::size_t length = 0;
  };
  auto runs = std::vector<Run>();
  for (const auto code : codes) {
    if (!runs.empty() && runs.back().code == code) {
      ++runs.back().length;
    }
    else {
      runs.push_back({code, 1});
    }
  }
  auto run_length = std::vector<unsigned char>();
  append_varint(run_length, runs.size());
  for (const auto& run : runs) {
    run_length.push_back(zigzag_int8(run.code));
    append_varint(run_length, run.length);
  }
  candidates.push_back(
      {static_cast<std::uint8_t>(FactorizedValueMode::run_length),
       std::move(run_length)});

  // QPC v14/tag 8 freezes the original modal rule: select the most frequent
  // ZigZag code, breaking frequency ties by the smallest ZigZag byte.  Any
  // future exhaustive-default optimization must use a new format/version;
  // changing this rule in place would make persisted v14 streams fail their
  // canonical replay check.
  auto frequencies = std::array<std::size_t, 256>{};
  for (const auto code : codes)
    ++frequencies[zigzag_int8(code)];
  const auto modal = static_cast<std::uint8_t>(std::distance(
      frequencies.begin(),
      std::max_element(frequencies.begin(), frequencies.end())));
  const auto modal_code = unzigzag_int8(modal);
  auto exception_mask = std::vector<unsigned char>(
      codes.size() / 8U + (codes.size() % 8U != 0), 0);
  auto exceptions = std::vector<std::int8_t>();
  exceptions.reserve(codes.size());
  for (std::size_t index = 0; index < codes.size(); ++index) {
    if (codes[index] == modal_code)
      continue;
    exception_mask[index / 8U] |=
        static_cast<unsigned char>(1U << (index % 8U));
    exceptions.push_back(codes[index]);
  }
  if (!exceptions.empty()) {
    const auto packed_exceptions = pack_int8_codes(exceptions);
    auto patched = std::vector<unsigned char>{modal};
    patched.insert(
        patched.end(), exception_mask.begin(), exception_mask.end());
    patched.push_back(packed_exceptions.bit_width);
    patched.insert(
        patched.end(),
        packed_exceptions.bytes.begin(),
        packed_exceptions.bytes.end());
    candidates.push_back(
        {static_cast<std::uint8_t>(FactorizedValueMode::modal_patched),
         std::move(patched)});
  }

  return *std::min_element(
      candidates.begin(), candidates.end(), factorized_field_less);
}

FactorizedFieldEncoding encode_predictive_values(
    const std::vector<std::int8_t>& codes)
{
  // Tag 10 is a strict extension of the frozen tag-9 value portfolio.  By
  // seeding the competition with the canonical legacy winner and retaining
  // the lower mode on equal byte counts, tag 10 can never replace an equally
  // small legacy representation with a predictive one.
  auto best = encode_factorized_values(codes);
  if (codes.size() < 2U)
    return best;

  auto deltas = std::vector<std::uint16_t>();
  deltas.reserve(codes.size() - 1U);
  for (std::size_t index = 1; index < codes.size(); ++index) {
    deltas.push_back(zigzag_int8_delta(
        static_cast<int>(codes[index]) -
        static_cast<int>(codes[index - 1U])));
  }

  const auto packed_deltas = pack_uint9_codes(deltas);
  auto delta_bitpack = std::vector<unsigned char>{
      zigzag_int8(codes.front()), packed_deltas.bit_width};
  delta_bitpack.insert(
      delta_bitpack.end(),
      packed_deltas.bytes.begin(),
      packed_deltas.bytes.end());
  auto candidate = FactorizedFieldEncoding{
      static_cast<std::uint8_t>(FactorizedValueMode::delta_bitpack),
      std::move(delta_bitpack)};
  if (factorized_field_less(candidate, best))
    best = std::move(candidate);

  auto frequencies = std::array<std::size_t, 511>{};
  for (const auto delta : deltas)
    ++frequencies[delta];
  const auto modal = static_cast<std::uint16_t>(std::distance(
      frequencies.begin(),
      std::max_element(frequencies.begin(), frequencies.end())));
  auto exception_mask = std::vector<unsigned char>(
      deltas.size() / 8U + (deltas.size() % 8U != 0U), 0U);
  auto exceptions = std::vector<std::uint16_t>();
  exceptions.reserve(deltas.size());
  for (std::size_t index = 0; index < deltas.size(); ++index) {
    if (deltas[index] == modal)
      continue;
    exception_mask[index / 8U] |=
        static_cast<unsigned char>(1U << (index % 8U));
    exceptions.push_back(deltas[index]);
  }
  auto delta_modal = std::vector<unsigned char>{zigzag_int8(codes.front())};
  append_pod(delta_modal, modal);
  delta_modal.insert(
      delta_modal.end(), exception_mask.begin(), exception_mask.end());
  if (exceptions.empty()) {
    // A width byte is retained even for the all-modal case, making the field
    // self-delimiting and giving canonical replay a single representation.
    delta_modal.push_back(1U);
  }
  else {
    const auto packed_exceptions = pack_uint9_codes(exceptions);
    delta_modal.push_back(packed_exceptions.bit_width);
    delta_modal.insert(
        delta_modal.end(),
        packed_exceptions.bytes.begin(),
        packed_exceptions.bytes.end());
  }
  candidate = {
      static_cast<std::uint8_t>(FactorizedValueMode::delta_modal_patched),
      std::move(delta_modal)};
  if (factorized_field_less(candidate, best))
    best = std::move(candidate);
  return best;
}

std::vector<unsigned char> encode_factorized_packed_correction_format8(
    const std::vector<float>& base,
    const std::vector<float>& corrected,
    const std::vector<float>& channel_scales,
    const QuantizedCorrectionFields& fields,
    std::size_t channels,
    std::size_t time)
{
  if (fields.codes.empty())
    return {};
  if (channels == 0 || time == 0 ||
      channels > std::numeric_limits<std::size_t>::max() / time ||
      base.size() != channels * time || corrected.size() != base.size() ||
      channel_scales.size() != channels)
    throw std::invalid_argument(
        "factorized packed correction dimensions are invalid");

  auto positions = std::vector<std::vector<std::size_t>>(channels);
  auto codes = std::vector<std::vector<std::int8_t>>(channels);
  auto exact_xors = std::vector<std::uint32_t>();
  auto code_offset = std::size_t{0};
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto begin = channel * time;
    for (std::size_t index = 0; index < time; ++index) {
      const auto offset = begin + index;
      const auto selected =
          (fields.mask[offset / 8U] &
           static_cast<unsigned char>(1U << (offset % 8U))) != 0;
      if (!selected)
        continue;
      if (code_offset >= fields.codes.size())
        throw std::logic_error(
            "factorized packed correction code accounting diverged");
      const auto code = fields.codes[code_offset++];
      positions[channel].push_back(index);
      codes[channel].push_back(code);
      if (code == 0)
        exact_xors.push_back(
            float32_bits(base[offset]) ^ float32_bits(corrected[offset]));
    }
  }
  if (code_offset != fields.codes.size() ||
      exact_xors.size() != fields.exact_values.size())
    throw std::logic_error(
        "factorized packed correction fields are inconsistent");

  auto position_fields = std::vector<FactorizedFieldEncoding>();
  auto value_fields = std::vector<FactorizedFieldEncoding>();
  position_fields.reserve(channels);
  value_fields.reserve(channels);
  auto descriptors = std::vector<unsigned char>();
  descriptors.reserve(channels);
  auto active_channels = std::size_t{0};
  for (std::size_t channel = 0; channel < channels; ++channel) {
    auto position_field = encode_factorized_positions(positions[channel], time);
    auto value_field = encode_factorized_values(codes[channel]);
    const auto active = !positions[channel].empty();
    if (active != !codes[channel].empty() ||
        (!active &&
         (position_field.mode !=
              static_cast<std::uint8_t>(FactorizedPositionMode::none) ||
          value_field.mode !=
              static_cast<std::uint8_t>(FactorizedValueMode::none))))
      throw std::logic_error(
          "factorized packed correction channel activity diverged");
    active_channels += static_cast<std::size_t>(active);
    descriptors.push_back(static_cast<unsigned char>(
        position_field.mode | static_cast<unsigned char>(value_field.mode << 3U)));
    position_fields.push_back(std::move(position_field));
    value_fields.push_back(std::move(value_field));
  }

  if (active_channels >
          std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      exact_xors.size() >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t))
    throw std::overflow_error("factorized packed fixed field size overflow");
  auto raw_size = channels;
  raw_size = checked_add_size(
      raw_size,
      active_channels * sizeof(float),
      "factorized packed scale size overflow");
  for (std::size_t channel = 0; channel < channels; ++channel) {
    raw_size = checked_add_size(
        raw_size,
        position_fields[channel].bytes.size(),
        "factorized packed position size overflow");
    raw_size = checked_add_size(
        raw_size,
        value_fields[channel].bytes.size(),
        "factorized packed value size overflow");
  }
  raw_size = checked_add_size(
      raw_size,
      exact_xors.size() * sizeof(std::uint32_t),
      "factorized packed exact size overflow");
  auto raw = std::vector<unsigned char>();
  raw.reserve(raw_size);
  raw.insert(raw.end(), descriptors.begin(), descriptors.end());
  // Frozen QPC v14/tag-8 layout, as persisted by the v3 artifacts: active
  // scales, position fields, and value fields are three channel-ordered
  // columns after the descriptors.  Exact XOR words are last.  Do not reorder
  // these fields without a new correction tag and stream version.
  for (std::size_t channel = 0; channel < channels; ++channel) {
    if (positions[channel].empty())
      continue;
    append_pod(raw, channel_scales[channel]);
  }
  for (std::size_t channel = 0; channel < channels; ++channel) {
    raw.insert(
        raw.end(),
        position_fields[channel].bytes.begin(),
        position_fields[channel].bytes.end());
  }
  for (std::size_t channel = 0; channel < channels; ++channel) {
    raw.insert(
        raw.end(),
        value_fields[channel].bytes.begin(),
        value_fields[channel].bytes.end());
  }
  for (const auto value : exact_xors)
    append_pod(raw, value);
  if (raw.size() != raw_size)
    throw std::logic_error("factorized packed raw-size accounting diverged");

  const auto compressed = compress_direct_zstd_frame(raw);
  auto payload = std::vector<unsigned char>();
  payload.reserve(1 + compressed.size());
  payload.push_back(kFactorizedPackedCorrectionFormat);
  payload.insert(payload.end(), compressed.begin(), compressed.end());
  return payload;
}

std::vector<unsigned char> encode_subset_rank_packed_correction_impl(
    const std::vector<float>& base,
    const std::vector<float>& corrected,
    const std::vector<float>& channel_scales,
    const QuantizedCorrectionFields& fields,
    std::size_t channels,
    std::size_t time,
    std::uint8_t output_format,
    bool predictive_values)
{
  if ((output_format == kSubsetRankPackedCorrectionFormat) == predictive_values ||
      (output_format != kSubsetRankPackedCorrectionFormat &&
       output_format != kPredictivePackedCorrectionFormat))
    throw std::invalid_argument(
        "subset-rank packed correction format/value policy is invalid");
  if (fields.codes.empty())
    return {};
  if (channels == 0 || time == 0 ||
      channels > std::numeric_limits<std::size_t>::max() / time ||
      base.size() != channels * time || corrected.size() != base.size() ||
      channel_scales.size() != channels)
    throw std::invalid_argument(
        "subset-rank packed correction dimensions are invalid");
  // Tag 9 is optional.  Inputs outside the deliberately bounded combinadic
  // domain retain the complete tag-8 fallback rather than exposing a
  // multiprecision CPU/memory denial-of-service surface to the decoder.
  if (time < 3 || time > qpet_correlation::subset_rank::kMaximumTime)
    return {};

  auto positions = std::vector<std::vector<std::size_t>>(channels);
  auto codes = std::vector<std::vector<std::int8_t>>(channels);
  auto exact_xors = std::vector<std::uint32_t>();
  auto code_offset = std::size_t{0};
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto begin = channel * time;
    for (std::size_t index = 0; index < time; ++index) {
      const auto offset = begin + index;
      const auto selected =
          (fields.mask[offset / 8U] &
           static_cast<unsigned char>(1U << (offset % 8U))) != 0;
      if (!selected)
        continue;
      if (code_offset >= fields.codes.size())
        throw std::logic_error(
            "subset-rank packed correction code accounting diverged");
      const auto code = fields.codes[code_offset++];
      positions[channel].push_back(index);
      codes[channel].push_back(code);
      if (code == 0)
        exact_xors.push_back(
            float32_bits(base[offset]) ^ float32_bits(corrected[offset]));
    }
  }
  if (code_offset != fields.codes.size() ||
      exact_xors.size() != fields.exact_values.size())
    throw std::logic_error(
        "subset-rank packed correction fields are inconsistent");
  if (fields.codes.size() >
      qpet_correlation::subset_rank::kMaximumDecodedPositions)
    return {};
  auto combinadic_work = std::size_t{0};
  for (const auto& selected : positions) {
    if (selected.empty() || selected.size() == time)
      continue;
    const auto channel_work = std::min(selected.size(), time - selected.size());
    if (channel_work >
        qpet_correlation::subset_rank::kMaximumCombinadicWork -
            combinadic_work)
      return {};
    combinadic_work += channel_work;
  }

  auto value_fields = std::vector<FactorizedFieldEncoding>();
  value_fields.reserve(channels);
  auto descriptors = std::vector<unsigned char>();
  descriptors.reserve(channels);
  auto active_channels = std::size_t{0};
  for (std::size_t channel = 0; channel < channels; ++channel) {
    auto value_field = predictive_values
        ? encode_predictive_values(codes[channel])
        : encode_factorized_values(codes[channel]);
    const auto active = !positions[channel].empty();
    if (active != !codes[channel].empty() ||
        (!active &&
         value_field.mode !=
             static_cast<std::uint8_t>(FactorizedValueMode::none)))
      throw std::logic_error(
          "subset-rank packed correction channel activity diverged");
    const auto position_mode = !active
        ? qpet_correlation::subset_rank::kPositionModeNone
        : (positions[channel].size() == time
               ? qpet_correlation::subset_rank::kPositionModeAll
               : qpet_correlation::subset_rank::kPositionModeSubsetRank);
    active_channels += static_cast<std::size_t>(active);
    descriptors.push_back(static_cast<unsigned char>(
        position_mode | static_cast<unsigned char>(value_field.mode << 3U)));
    value_fields.push_back(std::move(value_field));
  }
  const auto position_stream =
      qpet_correlation::subset_rank::encode_interleaved_position_stream(
          time, descriptors, positions);

  if (active_channels >
          std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      exact_xors.size() >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t))
    throw std::overflow_error(
        "subset-rank packed fixed field size overflow");
  auto raw_size = channels;
  raw_size = checked_add_size(
      raw_size,
      active_channels * sizeof(float),
      "subset-rank packed scale size overflow");
  raw_size = checked_add_size(
      raw_size,
      position_stream.size(),
      "subset-rank packed position size overflow");
  for (const auto& value_field : value_fields) {
    raw_size = checked_add_size(
        raw_size,
        value_field.bytes.size(),
        "subset-rank packed value size overflow");
  }
  raw_size = checked_add_size(
      raw_size,
      exact_xors.size() * sizeof(std::uint32_t),
      "subset-rank packed exact size overflow");

  auto raw = std::vector<unsigned char>();
  raw.reserve(raw_size);
  raw.insert(raw.end(), descriptors.begin(), descriptors.end());
  for (std::size_t channel = 0; channel < channels; ++channel) {
    if (positions[channel].empty())
      continue;
    const auto exact_only = std::all_of(
        codes[channel].begin(),
        codes[channel].end(),
        [](std::int8_t code) { return code == 0; });
    const auto stored_scale = exact_only ? 0.0F : channel_scales[channel];
    if (!exact_only && (!(stored_scale > 0.0F) || !std::isfinite(stored_scale)))
      throw std::logic_error(
          "subset-rank packed non-exact channel has an invalid scale");
    append_pod(raw, stored_scale);
  }
  raw.insert(raw.end(), position_stream.begin(), position_stream.end());
  for (const auto& value_field : value_fields) {
    raw.insert(
        raw.end(), value_field.bytes.begin(), value_field.bytes.end());
  }
  for (const auto value : exact_xors)
    append_pod(raw, value);
  if (raw.size() != raw_size)
    throw std::logic_error(
        "subset-rank packed raw-size accounting diverged");

  const auto compressed = compress_direct_zstd_frame(raw);
  auto payload = std::vector<unsigned char>();
  payload.reserve(1 + compressed.size());
  payload.push_back(output_format);
  payload.insert(payload.end(), compressed.begin(), compressed.end());
  return payload;
}

std::vector<unsigned char> encode_subset_rank_packed_correction_format9(
    const std::vector<float>& base,
    const std::vector<float>& corrected,
    const std::vector<float>& channel_scales,
    const QuantizedCorrectionFields& fields,
    std::size_t channels,
    std::size_t time)
{
  return encode_subset_rank_packed_correction_impl(
      base,
      corrected,
      channel_scales,
      fields,
      channels,
      time,
      kSubsetRankPackedCorrectionFormat,
      false);
}

std::vector<unsigned char> encode_predictive_packed_correction_format10(
    const std::vector<float>& base,
    const std::vector<float>& corrected,
    const std::vector<float>& channel_scales,
    const QuantizedCorrectionFields& fields,
    std::size_t channels,
    std::size_t time)
{
  return encode_subset_rank_packed_correction_impl(
      base,
      corrected,
      channel_scales,
      fields,
      channels,
      time,
      kPredictivePackedCorrectionFormat,
      true);
}

std::vector<unsigned char> compress_quantized_correction_payload(
    const std::vector<float>& base,
    const std::vector<float>& corrected,
    const std::vector<float>& channel_scales,
    std::size_t channels,
    std::size_t time)
{
  const auto fields = make_quantized_correction_fields(
      base, corrected, channel_scales, channels, time);
  if (fields.codes.empty())
    return {};
  auto format4 = encode_quantized_correction_format4(channel_scales, fields);
  auto format6 = encode_quantized_correction_format6(channel_scales, fields);
  if (format6.size() < format4.size())
    return format6;
  return format4;
}

std::vector<unsigned char> compress_correction_payload(
    const std::vector<float>& base,
    const std::vector<float>& corrected,
    const std::vector<float>* channel_scales,
    std::size_t channels,
    std::size_t time);

std::vector<unsigned char> compress_adaptive_correction_payload(
    const std::vector<float>& base,
    const std::vector<float>& corrected,
    const std::vector<float>& channel_scales,
    std::size_t channels,
    std::size_t time,
    std::uint8_t correction_format_ceiling)
{
  if (correction_format_ceiling != kPackedQuantizedCorrectionFormat &&
      correction_format_ceiling != kAdaptivePackedCorrectionFormat &&
      correction_format_ceiling != kFactorizedPackedCorrectionFormat &&
      correction_format_ceiling != kSubsetRankPackedCorrectionFormat &&
      correction_format_ceiling != kPredictivePackedCorrectionFormat)
    throw std::invalid_argument(
        "adaptive correction format ceiling must be exactly 6, 7, 8, 9, or 10");
  auto best = compress_correction_payload(
      base, corrected, &channel_scales, channels, time);
  if (!best.empty() && best.front() > kPackedQuantizedCorrectionFormat)
    throw std::logic_error(
        "legacy correction portfolio selected a format above ceiling 6");
  if (correction_format_ceiling == kPackedQuantizedCorrectionFormat)
    return best;
  const auto fields = make_quantized_correction_fields(
      base, corrected, channel_scales, channels, time);
  auto adaptive = encode_adaptive_packed_correction_format7(
      base,
      corrected,
      channel_scales,
      fields,
      channels,
      time);
  if (!adaptive.empty() && adaptive.size() < best.size())
    best = std::move(adaptive);
  if (correction_format_ceiling == kAdaptivePackedCorrectionFormat)
    return best;
  auto factorized = encode_factorized_packed_correction_format8(
      base,
      corrected,
      channel_scales,
      fields,
      channels,
      time);
  if (!factorized.empty() && factorized.size() < best.size())
    best = std::move(factorized);
  if (correction_format_ceiling == kFactorizedPackedCorrectionFormat)
    return best;
  auto subset_rank = encode_subset_rank_packed_correction_format9(
      base,
      corrected,
      channel_scales,
      fields,
      channels,
      time);
  if (!subset_rank.empty() && subset_rank.size() < best.size())
    best = std::move(subset_rank);
  if (correction_format_ceiling == kSubsetRankPackedCorrectionFormat)
    return best;
  auto predictive = encode_predictive_packed_correction_format10(
      base,
      corrected,
      channel_scales,
      fields,
      channels,
      time);
  if (!predictive.empty() && predictive.size() < best.size())
    best = std::move(predictive);
  if (!best.empty() && best.front() > correction_format_ceiling)
    throw std::logic_error(
        "adaptive correction portfolio exceeded its format ceiling");
  return best;
}

std::vector<unsigned char> compress_basic_correction_payload(
    const std::vector<float>& base,
    const std::vector<float>& corrected,
    const std::vector<float>* channel_scales = nullptr,
    std::size_t channels = 0,
    std::size_t time = 0)
{
  if (base.size() != corrected.size())
    throw std::invalid_argument("QPET correction vectors have different sizes");
  auto float_residual = std::vector<float>(base.size(), 0.0F);
  auto double_residual = std::vector<double>(base.size(), 0.0);
  auto exact_mask = std::vector<unsigned char>((base.size() + 7) / 8, 0);
  auto exact_values = std::vector<float>();
  auto sparse_mask = std::vector<unsigned char>((base.size() + 7) / 8, 0);
  auto sparse_values = std::vector<float>();
  auto sparse_xor_values = std::vector<std::uint32_t>();
  std::size_t correction_count = 0;
  bool double_is_exact = true;
  for (std::size_t index = 0; index < base.size(); ++index) {
    if (base[index] == corrected[index])
      continue;
    ++correction_count;
    sparse_mask[index / 8] |= static_cast<unsigned char>(1U << (index % 8));
    sparse_values.push_back(corrected[index]);
    std::uint32_t base_bits = 0;
    std::uint32_t corrected_bits = 0;
    std::memcpy(&base_bits, base.data() + index, sizeof(base_bits));
    std::memcpy(&corrected_bits, corrected.data() + index, sizeof(corrected_bits));
    sparse_xor_values.push_back(base_bits ^ corrected_bits);
    float_residual[index] = corrected[index] - base[index];
    double_residual[index] = static_cast<double>(corrected[index]) -
        static_cast<double>(base[index]);
    double_is_exact = double_is_exact &&
        static_cast<float>(static_cast<double>(base[index]) + double_residual[index]) ==
            corrected[index];
    const auto restored = static_cast<float>(base[index] + float_residual[index]);
    if (restored != corrected[index]) {
      exact_mask[index / 8] |= static_cast<unsigned char>(1U << (index % 8));
      exact_values.push_back(corrected[index]);
    }
  }
  if (correction_count == 0)
    return {};

  const auto residual_bytes = float_residual.size() * sizeof(float);
  const auto exact_bytes = exact_values.size() * sizeof(float);
  if (residual_bytes > std::numeric_limits<std::size_t>::max() - exact_mask.size() ||
      residual_bytes + exact_mask.size() >
          std::numeric_limits<std::size_t>::max() - exact_bytes)
    throw std::overflow_error("QPET correction payload size overflow");
  auto float_raw = std::vector<unsigned char>(
      residual_bytes + exact_mask.size() + exact_bytes);
  std::memcpy(float_raw.data(), float_residual.data(), residual_bytes);
  std::memcpy(
      float_raw.data() + residual_bytes, exact_mask.data(), exact_mask.size());
  if (exact_bytes != 0) {
    std::memcpy(
        float_raw.data() + residual_bytes + exact_mask.size(),
        exact_values.data(),
        exact_bytes);
  }
  auto double_raw = std::vector<unsigned char>(
      double_residual.size() * sizeof(double));
  std::memcpy(
      double_raw.data(), double_residual.data(), double_raw.size());
  auto sparse_raw = std::vector<unsigned char>(
      sparse_mask.size() + sparse_values.size() * sizeof(float));
  std::memcpy(sparse_raw.data(), sparse_mask.data(), sparse_mask.size());
  std::memcpy(
      sparse_raw.data() + sparse_mask.size(),
      sparse_values.data(),
      sparse_values.size() * sizeof(float));
  auto sparse_xor_raw = std::vector<unsigned char>(
      sparse_mask.size() + sparse_xor_values.size() * sizeof(std::uint32_t));
  std::memcpy(
      sparse_xor_raw.data(), sparse_mask.data(), sparse_mask.size());
  std::memcpy(
      sparse_xor_raw.data() + sparse_mask.size(),
      sparse_xor_values.data(),
      sparse_xor_values.size() * sizeof(std::uint32_t));

  auto float_payload = compress_zstd_correction(float_raw);
  auto double_payload = double_is_exact
      ? compress_zstd_correction(double_raw)
      : std::vector<unsigned char>();
  auto sparse_payload = compress_zstd_correction(sparse_raw);
  auto sparse_xor_payload = compress_zstd_correction(sparse_xor_raw);
  unsigned char format = 1U;
  const auto* selected = &float_payload;
  if (double_is_exact && double_payload.size() <= selected->size()) {
    format = 0U;
    selected = &double_payload;
  }
  if (sparse_payload.size() < selected->size()) {
    format = 2U;
    selected = &sparse_payload;
  }
  if (sparse_xor_payload.size() < selected->size()) {
    format = 3U;
    selected = &sparse_xor_payload;
  }
  auto payload = std::vector<unsigned char>();
  payload.reserve(selected->size() + 1);
  payload.push_back(format);
  payload.insert(payload.end(), selected->begin(), selected->end());
  if (channel_scales != nullptr) {
    auto quantized = compress_quantized_correction_payload(
        base, corrected, *channel_scales, channels, time);
    if (!quantized.empty() && quantized.size() < payload.size())
      return quantized;
  }
  return payload;
}

std::vector<unsigned char> compress_correction_payload(
    const std::vector<float>& base,
    const std::vector<float>& corrected,
    const std::vector<float>* channel_scales = nullptr,
    std::size_t channels = 0,
    std::size_t time = 0)
{
  auto best = compress_basic_correction_payload(
      base, corrected, channel_scales, channels, time);
  if (best.empty() || channels == 0 || time == 0 ||
      base.size() != channels * time || corrected.size() != base.size())
    return best;

  // Pearson correlation is undefined for a zero-variance source.  The hard
  // constraint therefore requires an exactly constant reconstruction for an
  // exactly constant source series.  Encoding 128 identical point repairs is
  // wasteful: store one float32 channel value and reproduce the same repaired
  // tensor at decode time.  This is only an alternative lossless representation
  // of the already selected repair; it does not change any constraint or value.
  const auto mask_bytes = (channels + 7) / 8;
  auto channel_mask = std::vector<unsigned char>(mask_bytes, 0);
  auto constant_values = std::vector<float>();
  auto overridden_base = base;
  std::uint64_t override_corrections = 0;
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto begin = channel * time;
    const auto value = corrected[begin];
    const auto value_bits = float32_bits(value);
    bool is_constant = true;
    bool changes_base = false;
    for (std::size_t index = 0; index < time; ++index) {
      is_constant =
          is_constant && float32_bits(corrected[begin + index]) == value_bits;
      changes_base = changes_base || base[begin + index] != corrected[begin + index];
    }
    if (!is_constant || !changes_base)
      continue;
    channel_mask[channel / 8] |=
        static_cast<unsigned char>(1U << (channel % 8));
    constant_values.push_back(value);
    for (std::size_t index = 0; index < time; ++index) {
      if (overridden_base[begin + index] != value)
        ++override_corrections;
      overridden_base[begin + index] = value;
    }
  }
  if (constant_values.empty())
    return best;

  std::uint64_t inner_corrections = 0;
  for (std::size_t index = 0; index < base.size(); ++index)
    inner_corrections += overridden_base[index] != corrected[index];
  if (override_corrections + inner_corrections >
      std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("QPET constant-channel correction count overflow");
  auto inner = compress_basic_correction_payload(
      overridden_base, corrected, channel_scales, channels, time);
  if ((inner_corrections == 0) != inner.empty())
    throw std::logic_error("QPET constant-channel nested correction is inconsistent");

  auto compact = std::vector<unsigned char>();
  compact.reserve(
      1 + varint_size(override_corrections) + varint_size(inner_corrections) +
      channel_mask.size() + constant_values.size() * sizeof(float) + inner.size());
  compact.push_back(5U);
  append_varint(compact, override_corrections);
  append_varint(compact, inner_corrections);
  compact.insert(compact.end(), channel_mask.begin(), channel_mask.end());
  const auto* values = reinterpret_cast<const unsigned char*>(constant_values.data());
  compact.insert(
      compact.end(), values, values + constant_values.size() * sizeof(float));
  compact.insert(compact.end(), inner.begin(), inner.end());
  return compact.size() < best.size() ? compact : best;
}

std::pair<unsigned char, std::vector<unsigned char>> decode_correction_payload(
    const std::vector<unsigned char>& payload,
    std::uint32_t expected_corrections)
{
  if (payload.empty()) {
    if (expected_corrections != 0)
      throw std::runtime_error("QPET correction metadata requires a missing payload");
    return {0U, {}};
  }
  if (expected_corrections == 0)
    throw std::runtime_error("QPET correction payload has zero correction metadata");
  const auto format = payload.front();
  if (format > 4 || payload.size() == 1)
    throw std::runtime_error("QPET correction payload has an invalid format");
  auto zstd = QoZ::Lossless_zstd();
  auto compressed_size = payload.size() - 1;
  auto* decoded = zstd.decompress(payload.data() + 1, compressed_size);
  if (decoded == nullptr || compressed_size == 0) {
    zstd.postdecompress_data(decoded);
    throw std::runtime_error("QPET correction residual is empty");
  }
  auto raw = std::vector<unsigned char>(decoded, decoded + compressed_size);
  zstd.postdecompress_data(decoded);
  return {format, std::move(raw)};
}

std::size_t packed_correction_maximum_raw_size(
    std::size_t values,
    std::size_t channels,
    std::size_t corrections)
{
  if (corrections == 0 || corrections > values ||
      channels > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      corrections > std::numeric_limits<std::size_t>::max() / sizeof(float))
    throw std::runtime_error("packed correction dimensions are invalid");
  const auto mask_bytes = values / 8U + (values % 8U != 0);
  std::size_t maximum = 1;
  auto add_size = [&](std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - maximum)
      throw std::runtime_error("packed correction raw-size bound overflows");
    maximum += bytes;
  };
  add_size(channels * sizeof(float));
  add_size(mask_bytes);
  add_size(corrections);  // worst case: eight packed bits per code
  add_size(corrections * sizeof(float));  // worst case: every code is an escape
  return maximum;
}

void apply_packed_quantized_correction(
    std::vector<float>& reconstruction,
    const std::vector<unsigned char>& payload,
    std::uint32_t expected_corrections,
    std::size_t channels,
    std::size_t time)
{
  if (payload.size() <= 1 ||
      payload.front() != kPackedQuantizedCorrectionFormat ||
      expected_corrections == 0 ||
      expected_corrections > reconstruction.size())
    throw std::runtime_error("packed correction framing is invalid");
  const auto maximum_raw_size = packed_correction_maximum_raw_size(
      reconstruction.size(), channels, expected_corrections);
  const auto scale_bytes = channels * sizeof(float);
  const auto mask_bytes =
      reconstruction.size() / 8U + (reconstruction.size() % 8U != 0);
  const auto raw = decompress_direct_zstd_frame(
      payload.data() + 1, payload.size() - 1, maximum_raw_size);
  if (raw.empty())
    throw std::runtime_error("packed correction raw field is empty");
  const auto bit_width = raw.front();
  if (bit_width < 1 || bit_width > 8 ||
      expected_corrections >
          (std::numeric_limits<std::size_t>::max() - 7U) / bit_width)
    throw std::runtime_error("packed correction bit width is invalid");
  const auto packed_code_bytes =
      (static_cast<std::size_t>(expected_corrections) * bit_width + 7U) / 8U;
  std::size_t minimum_raw_size = 1;
  auto add_minimum = [&](std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - minimum_raw_size)
      throw std::runtime_error("packed correction raw size overflows");
    minimum_raw_size += bytes;
  };
  add_minimum(scale_bytes);
  add_minimum(mask_bytes);
  add_minimum(packed_code_bytes);
  if (raw.size() < minimum_raw_size ||
      (raw.size() - minimum_raw_size) % sizeof(float) != 0)
    throw std::runtime_error("packed correction raw field has an invalid size");

  auto scales = std::vector<float>(channels);
  if (scale_bytes != 0)
    std::memcpy(scales.data(), raw.data() + 1, scale_bytes);
  for (const auto scale : scales) {
    if (!(scale >= 0.0F) || !std::isfinite(scale))
      throw std::runtime_error("packed correction scale is invalid");
  }
  const auto* mask = raw.data() + 1 + scale_bytes;
  if (reconstruction.size() % 8U != 0) {
    const auto valid_bits = static_cast<unsigned char>(
        (1U << (reconstruction.size() % 8U)) - 1U);
    if ((mask[mask_bytes - 1] & static_cast<unsigned char>(~valid_bits)) != 0)
      throw std::runtime_error("packed correction mask has out-of-range bits");
  }
  std::size_t mask_corrections = 0;
  for (std::size_t index = 0; index < reconstruction.size(); ++index) {
    mask_corrections +=
        (mask[index / 8U] & static_cast<unsigned char>(1U << (index % 8U))) != 0;
  }
  if (mask_corrections != expected_corrections)
    throw std::runtime_error("packed correction mask popcount is inconsistent");

  const auto* packed_codes = mask + mask_bytes;
  const auto codes = unpack_int8_codes(
      packed_codes,
      packed_code_bytes,
      expected_corrections,
      bit_width);
  if (pack_int8_codes(codes).bit_width != bit_width)
    throw std::runtime_error("packed correction bit width is not minimal");
  const auto* exact_data = packed_codes + packed_code_bytes;
  const auto exact_count = (raw.size() - minimum_raw_size) / sizeof(float);
  const auto escape_count = static_cast<std::size_t>(std::count(
      codes.begin(), codes.end(), static_cast<std::int8_t>(0)));
  if (escape_count != exact_count)
    throw std::runtime_error("packed correction escape count is inconsistent");
  for (std::size_t index = 0; index < exact_count; ++index) {
    float value = 0.0F;
    std::memcpy(&value, exact_data + index * sizeof(float), sizeof(float));
    if (!std::isfinite(value))
      throw std::runtime_error("packed correction exact value is not finite");
  }
  for (const auto value : reconstruction) {
    if (!std::isfinite(value))
      throw std::runtime_error("packed correction base value is not finite");
  }

  auto candidate = reconstruction;
  std::size_t correction_index = 0;
  std::size_t exact_index = 0;
  for (std::size_t index = 0; index < candidate.size(); ++index) {
    const auto selected =
        (mask[index / 8U] & static_cast<unsigned char>(1U << (index % 8U))) != 0;
    if (!selected)
      continue;
    const auto before = candidate[index];
    const auto code = codes[correction_index++];
    if (code == 0) {
      std::memcpy(
          candidate.data() + index,
          exact_data + exact_index * sizeof(float),
          sizeof(float));
      ++exact_index;
    }
    else {
      const auto scale = scales[index / time];
      if (!(scale > 0.0F))
        throw std::runtime_error("packed correction uses a zero scale");
      candidate[index] = static_cast<float>(
          before + static_cast<float>(code) * scale);
    }
    if (!std::isfinite(candidate[index]) || candidate[index] == before)
      throw std::runtime_error("packed correction result is invalid or unchanged");
  }
  if (correction_index != expected_corrections || exact_index != exact_count)
    throw std::runtime_error("packed correction metadata is inconsistent");
  for (std::size_t index = 0; index < candidate.size(); ++index) {
    const auto selected =
        (mask[index / 8U] & static_cast<unsigned char>(1U << (index % 8U))) != 0;
    if (!selected && float32_bits(candidate[index]) != float32_bits(reconstruction[index]))
      throw std::runtime_error("packed correction changed an unmasked value");
  }
  reconstruction.swap(candidate);
}

void apply_adaptive_packed_correction(
    std::vector<float>& reconstruction,
    const std::vector<unsigned char>& payload,
    std::uint32_t expected_corrections,
    std::size_t channels,
    std::size_t time)
{
  if (payload.size() <= 1 ||
      payload.front() != kAdaptivePackedCorrectionFormat ||
      expected_corrections == 0 ||
      expected_corrections > reconstruction.size() ||
      channels == 0 || time == 0 ||
      channels > std::numeric_limits<std::size_t>::max() / time ||
      reconstruction.size() != channels * time)
    throw std::runtime_error("adaptive packed correction framing is invalid");
  const auto scale_bytes = channels * sizeof(float);
  const auto mask_bytes =
      reconstruction.size() / 8U + (reconstruction.size() % 8U != 0);
  std::size_t maximum_raw_size = 0;
  auto add_maximum = [&](std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - maximum_raw_size)
      throw std::runtime_error("adaptive packed correction size bound overflows");
    maximum_raw_size += bytes;
  };
  add_maximum(scale_bytes);
  add_maximum(mask_bytes);
  add_maximum(channels);
  add_maximum(expected_corrections);
  add_maximum(static_cast<std::size_t>(expected_corrections) *
      sizeof(std::uint32_t));
  const auto raw = decompress_direct_zstd_frame(
      payload.data() + 1, payload.size() - 1, maximum_raw_size);
  std::size_t prefix_size = 0;
  auto add_prefix = [&](std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - prefix_size)
      throw std::runtime_error("adaptive packed correction prefix overflows");
    prefix_size += bytes;
  };
  add_prefix(scale_bytes);
  add_prefix(mask_bytes);
  add_prefix(channels);
  if (raw.size() < prefix_size)
    throw std::runtime_error("adaptive packed correction raw field is truncated");

  auto scales = std::vector<float>(channels);
  std::memcpy(scales.data(), raw.data(), scale_bytes);
  for (const auto scale : scales) {
    if (!(scale >= 0.0F) || !std::isfinite(scale))
      throw std::runtime_error("adaptive packed correction scale is invalid");
  }
  const auto* mask = raw.data() + scale_bytes;
  if (reconstruction.size() % 8U != 0) {
    const auto valid_bits = static_cast<unsigned char>(
        (1U << (reconstruction.size() % 8U)) - 1U);
    if ((mask[mask_bytes - 1] & static_cast<unsigned char>(~valid_bits)) != 0)
      throw std::runtime_error(
          "adaptive packed correction mask has out-of-range bits");
  }
  const auto* bit_widths = mask + mask_bytes;
  auto channel_counts = std::vector<std::size_t>(channels, 0);
  std::size_t mask_corrections = 0;
  for (std::size_t index = 0; index < reconstruction.size(); ++index) {
    const auto selected =
        (mask[index / 8U] &
         static_cast<unsigned char>(1U << (index % 8U))) != 0;
    if (!selected)
      continue;
    ++mask_corrections;
    ++channel_counts[index / time];
  }
  if (mask_corrections != expected_corrections)
    throw std::runtime_error(
        "adaptive packed correction mask popcount is inconsistent");
  std::size_t packed_bytes = 0;
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto width = bit_widths[channel];
    if (channel_counts[channel] == 0) {
      if (width != 0)
        throw std::runtime_error(
            "adaptive packed correction has a width for an empty channel");
      continue;
    }
    if (width < 1 || width > 8 ||
        channel_counts[channel] >
            (std::numeric_limits<std::size_t>::max() - 7U) / width)
      throw std::runtime_error(
          "adaptive packed correction channel width is invalid");
    const auto bytes = (channel_counts[channel] * width + 7U) / 8U;
    if (bytes > std::numeric_limits<std::size_t>::max() - packed_bytes)
      throw std::runtime_error(
          "adaptive packed correction code size overflows");
    packed_bytes += bytes;
  }
  if (packed_bytes > raw.size() - prefix_size)
    throw std::runtime_error("adaptive packed correction codes are truncated");
  const auto exact_bytes = raw.size() - prefix_size - packed_bytes;
  if (exact_bytes % sizeof(std::uint32_t) != 0)
    throw std::runtime_error("adaptive packed correction exact field is invalid");
  const auto exact_count = exact_bytes / sizeof(std::uint32_t);
  const auto* packed_cursor = raw.data() + prefix_size;
  auto codes = std::vector<std::int8_t>();
  codes.reserve(expected_corrections);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    if (channel_counts[channel] == 0)
      continue;
    const auto width = bit_widths[channel];
    const auto bytes = (channel_counts[channel] * width + 7U) / 8U;
    const auto channel_codes = unpack_int8_codes(
        packed_cursor, bytes, channel_counts[channel], width);
    if (pack_int8_codes(channel_codes).bit_width != width)
      throw std::runtime_error(
          "adaptive packed correction channel width is not minimal");
    codes.insert(codes.end(), channel_codes.begin(), channel_codes.end());
    packed_cursor += bytes;
  }
  if (codes.size() != expected_corrections ||
      static_cast<std::size_t>(std::count(
          codes.begin(), codes.end(), static_cast<std::int8_t>(0))) !=
          exact_count)
    throw std::runtime_error(
        "adaptive packed correction exact count is inconsistent");
  const auto* exact_data = raw.data() + prefix_size + packed_bytes;
  auto candidate = reconstruction;
  std::size_t correction_index = 0;
  std::size_t exact_index = 0;
  for (std::size_t index = 0; index < candidate.size(); ++index) {
    const auto selected =
        (mask[index / 8U] &
         static_cast<unsigned char>(1U << (index % 8U))) != 0;
    if (!selected)
      continue;
    const auto before = candidate[index];
    const auto code = codes[correction_index++];
    if (code == 0) {
      std::uint32_t xor_bits = 0;
      std::memcpy(
          &xor_bits,
          exact_data + exact_index * sizeof(std::uint32_t),
          sizeof(xor_bits));
      ++exact_index;
      const auto corrected_bits = float32_bits(before) ^ xor_bits;
      std::memcpy(candidate.data() + index, &corrected_bits, sizeof(corrected_bits));
    }
    else {
      const auto scale = scales[index / time];
      if (!(scale > 0.0F))
        throw std::runtime_error(
            "adaptive packed correction uses a zero scale");
      candidate[index] = static_cast<float>(
          before + static_cast<float>(code) * scale);
    }
    if (!std::isfinite(candidate[index]) || candidate[index] == before)
      throw std::runtime_error(
          "adaptive packed correction result is invalid or unchanged");
  }
  if (correction_index != expected_corrections || exact_index != exact_count)
    throw std::runtime_error(
        "adaptive packed correction metadata is inconsistent");
  for (std::size_t index = 0; index < candidate.size(); ++index) {
    const auto selected =
        (mask[index / 8U] &
         static_cast<unsigned char>(1U << (index % 8U))) != 0;
    if (!selected &&
        float32_bits(candidate[index]) != float32_bits(reconstruction[index]))
      throw std::runtime_error(
          "adaptive packed correction changed an unmasked value");
  }
  reconstruction.swap(candidate);
}

std::uint64_t read_canonical_factorized_varint(
    const std::vector<unsigned char>& input,
    std::size_t& offset)
{
  const auto begin = offset;
  const auto value = read_varint(input, offset);
  if (offset - begin != varint_size(value))
    throw std::runtime_error(
        "factorized packed correction varint is not canonical");
  return value;
}

void apply_factorized_packed_correction(
    std::vector<float>& reconstruction,
    const std::vector<unsigned char>& payload,
    std::uint32_t expected_corrections,
    std::size_t channels,
    std::size_t time)
{
  if (payload.size() <= 1 ||
      payload.front() != kFactorizedPackedCorrectionFormat ||
      expected_corrections == 0 ||
      expected_corrections > reconstruction.size() || channels == 0 ||
      time == 0 || channels > std::numeric_limits<std::size_t>::max() / time ||
      reconstruction.size() != channels * time)
    throw std::runtime_error(
        "factorized packed correction framing is invalid");

  auto maximum_raw_size = channels;
  auto add_product = [&](std::size_t count,
                         std::size_t bytes,
                         const char* message) {
    if (bytes != 0 && count > std::numeric_limits<std::size_t>::max() / bytes)
      throw std::runtime_error(message);
    maximum_raw_size = checked_add_size(
        maximum_raw_size, count * bytes, message);
  };
  // For every partial active channel the canonical position field is no
  // larger than its bitmap candidate, and the canonical value field is no
  // larger than width-byte plus 8-bit codes.  This gives a tight allocation
  // bound independent of hostile varint claims in the compressed frame.
  const auto maximum_active = std::min(
      channels, static_cast<std::size_t>(expected_corrections));
  const auto bitmap_bytes = time / 8U + (time % 8U != 0);
  add_product(
      maximum_active,
      checked_add_size(
          bitmap_bytes,
          5U,
          "factorized packed per-channel size bound overflows"),
      "factorized packed active-channel size bound overflows");
  add_product(
      expected_corrections,
      5U,
      "factorized packed value size bound overflows");
  const auto raw = decompress_direct_zstd_frame(
      payload.data() + 1, payload.size() - 1, maximum_raw_size);
  if (raw.size() < channels)
    throw std::runtime_error(
        "factorized packed correction descriptors are truncated");

  const auto descriptors = std::vector<unsigned char>(
      raw.begin(), raw.begin() + static_cast<std::ptrdiff_t>(channels));
  auto position_modes = std::vector<std::uint8_t>(channels);
  auto value_modes = std::vector<std::uint8_t>(channels);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto descriptor = descriptors[channel];
    if ((descriptor & 0xc0U) != 0)
      throw std::runtime_error(
          "factorized packed correction descriptor has reserved bits");
    position_modes[channel] = descriptor & 0x07U;
    value_modes[channel] = (descriptor >> 3U) & 0x07U;
    if (position_modes[channel] >
            static_cast<std::uint8_t>(FactorizedPositionMode::complement_gaps) ||
        value_modes[channel] >
            static_cast<std::uint8_t>(FactorizedValueMode::modal_patched) ||
        ((position_modes[channel] ==
          static_cast<std::uint8_t>(FactorizedPositionMode::none)) !=
         (value_modes[channel] ==
          static_cast<std::uint8_t>(FactorizedValueMode::none))))
      throw std::runtime_error(
          "factorized packed correction descriptor is invalid");
  }
  auto offset = channels;
  auto scales = std::vector<float>(channels, 0.0F);
  // Frozen QPC v14/tag-8 artifacts store one contiguous scale block for all
  // active channels before any position/value field.
  for (std::size_t channel = 0; channel < channels; ++channel) {
    if (position_modes[channel] ==
        static_cast<std::uint8_t>(FactorizedPositionMode::none))
      continue;
    if (raw.size() - offset < sizeof(float))
      throw std::runtime_error(
          "factorized packed correction scale is truncated");
    scales[channel] = read_pod<float>(raw, offset);
    if (!(scales[channel] >= 0.0F) || !std::isfinite(scales[channel]))
      throw std::runtime_error(
          "factorized packed correction scale is invalid");
  }

  auto positions = std::vector<std::vector<std::size_t>>(channels);
  auto codes = std::vector<std::vector<std::int8_t>>(channels);
  auto total_corrections = std::size_t{0};
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto field_begin = offset;
    const auto mode = position_modes[channel];
    if (mode == static_cast<std::uint8_t>(FactorizedPositionMode::none)) {
      // No bytes.
    }
    else if (mode == static_cast<std::uint8_t>(FactorizedPositionMode::all)) {
      positions[channel].resize(time);
      std::iota(positions[channel].begin(), positions[channel].end(), 0U);
    }
    else if (mode ==
             static_cast<std::uint8_t>(FactorizedPositionMode::bitmap)) {
      const auto bitmap_bytes = time / 8U + (time % 8U != 0);
      if (bitmap_bytes > raw.size() - offset)
        throw std::runtime_error(
            "factorized packed position bitmap is truncated");
      if (time % 8U != 0) {
        const auto valid_bits = static_cast<unsigned char>(
            (1U << (time % 8U)) - 1U);
        if ((raw[offset + bitmap_bytes - 1] &
             static_cast<unsigned char>(~valid_bits)) != 0)
          throw std::runtime_error(
              "factorized packed position bitmap padding is nonzero");
      }
      for (std::size_t index = 0; index < time; ++index) {
        if ((raw[offset + index / 8U] &
             static_cast<unsigned char>(1U << (index % 8U))) != 0)
          positions[channel].push_back(index);
      }
      offset += bitmap_bytes;
    }
    else {
      const auto count64 = read_canonical_factorized_varint(raw, offset);
      if (count64 == 0 || count64 >= time ||
          count64 > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error(
            "factorized packed gap count is invalid");
      const auto count = static_cast<std::size_t>(count64);
      auto encoded_positions = std::vector<std::size_t>();
      encoded_positions.reserve(count);
      auto previous = std::size_t{0};
      for (std::size_t index = 0; index < count; ++index) {
        const auto delta64 = read_canonical_factorized_varint(raw, offset);
        if (delta64 == 0 || delta64 > std::numeric_limits<std::size_t>::max())
          throw std::runtime_error(
              "factorized packed position delta is invalid");
        const auto delta = static_cast<std::size_t>(delta64);
        std::size_t position = 0;
        if (index == 0) {
          if (delta > time)
            throw std::runtime_error(
                "factorized packed first position is out of range");
          position = delta - 1U;
        }
        else {
          if (previous >= time || delta > time - 1U - previous)
            throw std::runtime_error(
                "factorized packed position delta is out of range");
          position = previous + delta;
        }
        encoded_positions.push_back(position);
        previous = position;
      }
      if (mode == static_cast<std::uint8_t>(FactorizedPositionMode::gaps)) {
        positions[channel] = std::move(encoded_positions);
      }
      else if (
          mode == static_cast<std::uint8_t>(
              FactorizedPositionMode::complement_gaps)) {
        positions[channel].reserve(time - encoded_positions.size());
        auto omitted_index = std::size_t{0};
        for (std::size_t position = 0; position < time; ++position) {
          if (omitted_index < encoded_positions.size() &&
              encoded_positions[omitted_index] == position) {
            ++omitted_index;
          }
          else {
            positions[channel].push_back(position);
          }
        }
      }
      else {
        throw std::runtime_error(
            "factorized packed correction has an unknown position mode");
      }
    }
    const auto canonical = encode_factorized_positions(positions[channel], time);
    if (canonical.mode != mode ||
        canonical.bytes.size() != offset - field_begin ||
        !std::equal(
            canonical.bytes.begin(),
            canonical.bytes.end(),
            raw.begin() + static_cast<std::ptrdiff_t>(field_begin)))
      throw std::runtime_error(
          "factorized packed position field is not canonical");
  }
  for (std::size_t channel = 0; channel < channels; ++channel) {
    {
      const auto value_field_begin = offset;
      const auto count = positions[channel].size();
      const auto mode = value_modes[channel];
    if (mode == static_cast<std::uint8_t>(FactorizedValueMode::none)) {
      if (count != 0)
        throw std::runtime_error(
            "factorized packed active positions have no value mode");
    }
    else if (count == 0) {
      throw std::runtime_error(
          "factorized packed value mode has no positions");
    }
    else if (mode ==
             static_cast<std::uint8_t>(FactorizedValueMode::constant)) {
      if (offset >= raw.size())
        throw std::runtime_error(
            "factorized packed constant code is truncated");
      codes[channel].assign(count, unzigzag_int8(raw[offset++]));
    }
    else if (mode ==
             static_cast<std::uint8_t>(FactorizedValueMode::bitpack)) {
      if (offset >= raw.size())
        throw std::runtime_error(
            "factorized packed code width is truncated");
      const auto width = raw[offset++];
      if (width < 1 || width > 8 ||
          count > (std::numeric_limits<std::size_t>::max() - 7U) / width)
        throw std::runtime_error(
            "factorized packed code width is invalid");
      const auto bytes = (count * width + 7U) / 8U;
      if (bytes > raw.size() - offset)
        throw std::runtime_error(
            "factorized packed code field is truncated");
      codes[channel] = unpack_int8_codes(
          raw.data() + offset, bytes, count, width);
      offset += bytes;
    }
    else if (mode ==
             static_cast<std::uint8_t>(FactorizedValueMode::run_length)) {
      const auto run_count64 = read_canonical_factorized_varint(raw, offset);
      if (run_count64 == 0 || run_count64 > count)
        throw std::runtime_error(
            "factorized packed run count is invalid");
      const auto run_count = static_cast<std::size_t>(run_count64);
      codes[channel].reserve(count);
      auto previous_code = std::optional<std::int8_t>();
      for (std::size_t run = 0; run < run_count; ++run) {
        if (offset >= raw.size())
          throw std::runtime_error(
              "factorized packed run code is truncated");
        const auto code = unzigzag_int8(raw[offset++]);
        if (previous_code.has_value() && *previous_code == code)
          throw std::runtime_error(
              "factorized packed adjacent runs are not canonical");
        const auto length64 = read_canonical_factorized_varint(raw, offset);
        if (length64 == 0 || length64 > count - codes[channel].size())
          throw std::runtime_error(
              "factorized packed run length is invalid");
        codes[channel].insert(
            codes[channel].end(), static_cast<std::size_t>(length64), code);
        previous_code = code;
      }
      if (codes[channel].size() != count)
        throw std::runtime_error(
            "factorized packed runs do not cover the channel");
    }
    else if (mode ==
             static_cast<std::uint8_t>(FactorizedValueMode::modal_patched)) {
      if (offset >= raw.size())
        throw std::runtime_error(
            "factorized packed modal code is truncated");
      const auto modal_code = unzigzag_int8(raw[offset++]);
      const auto mask_bytes = count / 8U + (count % 8U != 0);
      if (mask_bytes > raw.size() - offset)
        throw std::runtime_error(
            "factorized packed exception mask is truncated");
      const auto mask_offset = offset;
      if (count % 8U != 0) {
        const auto valid_bits = static_cast<unsigned char>(
            (1U << (count % 8U)) - 1U);
        if ((raw[offset + mask_bytes - 1] &
             static_cast<unsigned char>(~valid_bits)) != 0)
          throw std::runtime_error(
              "factorized packed exception-mask padding is nonzero");
      }
      auto exception_count = std::size_t{0};
      for (std::size_t index = 0; index < count; ++index)
        exception_count +=
            (raw[offset + index / 8U] &
             static_cast<unsigned char>(1U << (index % 8U))) != 0;
      offset += mask_bytes;
      if (exception_count == 0 || exception_count >= count ||
          offset >= raw.size())
        throw std::runtime_error(
            "factorized packed exception count is invalid");
      const auto width = raw[offset++];
      if (width < 1 || width > 8 ||
          exception_count >
              (std::numeric_limits<std::size_t>::max() - 7U) / width)
        throw std::runtime_error(
            "factorized packed exception width is invalid");
      const auto bytes = (exception_count * width + 7U) / 8U;
      if (bytes > raw.size() - offset)
        throw std::runtime_error(
            "factorized packed exception codes are truncated");
      const auto exceptions = unpack_int8_codes(
          raw.data() + offset, bytes, exception_count, width);
      offset += bytes;
      codes[channel].assign(count, modal_code);
      auto exception_index = std::size_t{0};
      for (std::size_t index = 0; index < count; ++index) {
        if ((raw[mask_offset + index / 8U] &
             static_cast<unsigned char>(1U << (index % 8U))) == 0)
          continue;
        if (exceptions[exception_index] == modal_code)
          throw std::runtime_error(
              "factorized packed exception repeats the modal code");
        codes[channel][index] = exceptions[exception_index++];
      }
      if (exception_index != exceptions.size())
        throw std::runtime_error(
            "factorized packed exception accounting diverged");
    }
    else {
      throw std::runtime_error(
          "factorized packed correction has an unknown value mode");
    }
      const auto canonical = encode_factorized_values(codes[channel]);
      if (canonical.mode != mode ||
          canonical.bytes.size() != offset - value_field_begin ||
          !std::equal(
              canonical.bytes.begin(),
              canonical.bytes.end(),
              raw.begin() + static_cast<std::ptrdiff_t>(value_field_begin)))
        throw std::runtime_error(
            "factorized packed value field is not canonical");
    }
    total_corrections = checked_add_size(
        total_corrections,
        codes[channel].size(),
        "factorized packed correction count overflows");
  }
  if (total_corrections != expected_corrections)
    throw std::runtime_error(
        "factorized packed correction count disagrees with metadata");

  auto exact_count = std::size_t{0};
  for (const auto& channel_codes : codes)
    exact_count += static_cast<std::size_t>(std::count(
        channel_codes.begin(), channel_codes.end(), std::int8_t{0}));
  if (exact_count >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) ||
      raw.size() - offset != exact_count * sizeof(std::uint32_t))
    throw std::runtime_error(
        "factorized packed exact field has an invalid size");
  const auto exact_offset = offset;

  auto candidate = reconstruction;
  auto exact_index = std::size_t{0};
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto begin = channel * time;
    if (positions[channel].size() != codes[channel].size())
      throw std::logic_error(
          "factorized packed position/code accounting diverged");
    for (std::size_t local = 0; local < positions[channel].size(); ++local) {
      const auto index = begin + positions[channel][local];
      const auto before = candidate[index];
      const auto code = codes[channel][local];
      if (code == 0) {
        std::uint32_t xor_bits = 0;
        std::memcpy(
            &xor_bits,
            raw.data() + exact_offset + exact_index * sizeof(std::uint32_t),
            sizeof(xor_bits));
        ++exact_index;
        const auto corrected_bits = float32_bits(before) ^ xor_bits;
        std::memcpy(candidate.data() + index, &corrected_bits, sizeof(corrected_bits));
      }
      else {
        if (!(scales[channel] > 0.0F))
          throw std::runtime_error(
              "factorized packed correction uses a zero scale");
        candidate[index] = static_cast<float>(
            before + static_cast<float>(code) * scales[channel]);
      }
      if (!std::isfinite(candidate[index]) || candidate[index] == before)
        throw std::runtime_error(
            "factorized packed correction result is invalid or unchanged");
    }
  }
  if (exact_index != exact_count)
    throw std::runtime_error(
        "factorized packed exact accounting diverged");
  reconstruction.swap(candidate);
}

void apply_subset_rank_packed_correction(
    std::vector<float>& reconstruction,
    const std::vector<unsigned char>& payload,
    std::uint32_t expected_corrections,
    std::size_t channels,
    std::size_t time,
    std::uint8_t expected_format)
{
  const auto predictive_values =
      expected_format == kPredictivePackedCorrectionFormat;
  if (payload.size() <= 1 ||
      payload.front() != expected_format ||
      (expected_format != kSubsetRankPackedCorrectionFormat &&
       expected_format != kPredictivePackedCorrectionFormat) ||
      expected_corrections == 0 ||
      expected_corrections > reconstruction.size() || channels == 0 ||
      time < 3 || time > qpet_correlation::subset_rank::kMaximumTime ||
      channels > std::numeric_limits<std::size_t>::max() / time ||
      reconstruction.size() != channels * time)
    throw std::runtime_error(
        "subset-rank packed correction framing is invalid");

  const auto maximum_active = std::min(
      channels, static_cast<std::size_t>(expected_corrections));
  const auto count_width =
      qpet_correlation::subset_rank::count_bit_width(time);
  if (time > std::numeric_limits<std::size_t>::max() - count_width ||
      maximum_active >
          std::numeric_limits<std::size_t>::max() / (time + count_width))
    throw std::runtime_error(
        "subset-rank packed position bound overflows");
  const auto maximum_position_bits = maximum_active * (time + count_width);
  const auto maximum_position_bytes =
      maximum_position_bits / 8U + (maximum_position_bits % 8U != 0);
  auto maximum_raw_size = channels;
  auto add_product = [&](std::size_t count,
                         std::size_t bytes,
                         const char* message) {
    if (bytes != 0 && count > std::numeric_limits<std::size_t>::max() / bytes)
      throw std::runtime_error(message);
    maximum_raw_size = checked_add_size(
        maximum_raw_size, count * bytes, message);
  };
  add_product(
      maximum_active,
      sizeof(float) + 1U,
      "subset-rank packed active-channel size bound overflows");
  maximum_raw_size = checked_add_size(
      maximum_raw_size,
      maximum_position_bytes,
      "subset-rank packed position size bound overflows");
  add_product(
      expected_corrections,
      5U,
      "subset-rank packed value size bound overflows");
  const auto raw = decompress_direct_zstd_frame(
      payload.data() + 1, payload.size() - 1, maximum_raw_size);
  if (raw.size() < channels)
    throw std::runtime_error(
        "subset-rank packed correction descriptors are truncated");

  const auto descriptors = std::vector<unsigned char>(
      raw.begin(), raw.begin() + static_cast<std::ptrdiff_t>(channels));
  auto position_modes = std::vector<std::uint8_t>(channels);
  auto value_modes = std::vector<std::uint8_t>(channels);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto descriptor = descriptors[channel];
    if ((descriptor & 0xc0U) != 0)
      throw std::runtime_error(
          "subset-rank packed correction descriptor has reserved bits");
    position_modes[channel] = descriptor & 0x07U;
    value_modes[channel] = (descriptor >> 3U) & 0x07U;
    const auto known_position =
        position_modes[channel] ==
            qpet_correlation::subset_rank::kPositionModeNone ||
        position_modes[channel] ==
            qpet_correlation::subset_rank::kPositionModeAll ||
        position_modes[channel] ==
            qpet_correlation::subset_rank::kPositionModeSubsetRank;
    const auto maximum_value_mode = predictive_values
        ? static_cast<std::uint8_t>(
              FactorizedValueMode::delta_modal_patched)
        : static_cast<std::uint8_t>(FactorizedValueMode::modal_patched);
    if (!known_position || value_modes[channel] > maximum_value_mode ||
        ((position_modes[channel] ==
          qpet_correlation::subset_rank::kPositionModeNone) !=
         (value_modes[channel] ==
          static_cast<std::uint8_t>(FactorizedValueMode::none))))
      throw std::runtime_error(
          "subset-rank packed correction descriptor is invalid");
  }

  auto offset = channels;
  auto scales = std::vector<float>(channels, 0.0F);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    if (position_modes[channel] ==
        qpet_correlation::subset_rank::kPositionModeNone)
      continue;
    if (raw.size() - offset < sizeof(float))
      throw std::runtime_error(
          "subset-rank packed correction scale is truncated");
    scales[channel] = read_pod<float>(raw, offset);
    if (!(scales[channel] >= 0.0F) || !std::isfinite(scales[channel]) ||
        std::signbit(scales[channel]))
      throw std::runtime_error(
          "subset-rank packed correction scale is invalid");
  }

  const auto position_offset = offset;
  qpet_correlation::subset_rank::DecodedPositionPrefix decoded_positions;
  try {
    decoded_positions =
        qpet_correlation::subset_rank::decode_interleaved_position_prefix(
            time,
            descriptors,
            raw,
            offset,
            static_cast<std::size_t>(expected_corrections));
  }
  catch (const std::exception& error) {
    throw std::runtime_error(
        "subset-rank packed position stream is invalid: " +
        std::string(error.what()));
  }
  if (decoded_positions.bytes_consumed > raw.size() - offset)
    throw std::runtime_error(
        "subset-rank packed position stream exceeds the raw frame");
  offset += decoded_positions.bytes_consumed;
  auto positions = std::move(decoded_positions.positions);
  const auto canonical_positions =
      qpet_correlation::subset_rank::encode_interleaved_position_stream(
          time, descriptors, positions);
  if (canonical_positions.size() != offset - position_offset ||
      !std::equal(
          canonical_positions.begin(),
          canonical_positions.end(),
          raw.begin() + static_cast<std::ptrdiff_t>(position_offset)))
    throw std::runtime_error(
        "subset-rank packed position stream is not canonical");

  auto codes = std::vector<std::vector<std::int8_t>>(channels);
  auto total_corrections = std::size_t{0};
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto value_field_begin = offset;
    const auto count = positions[channel].size();
    const auto mode = value_modes[channel];
    if (mode == static_cast<std::uint8_t>(FactorizedValueMode::none)) {
      if (count != 0)
        throw std::runtime_error(
            "subset-rank packed active positions have no value mode");
    }
    else if (count == 0) {
      throw std::runtime_error(
          "subset-rank packed value mode has no positions");
    }
    else if (mode ==
             static_cast<std::uint8_t>(FactorizedValueMode::constant)) {
      if (offset >= raw.size())
        throw std::runtime_error(
            "subset-rank packed constant code is truncated");
      codes[channel].assign(count, unzigzag_int8(raw[offset++]));
    }
    else if (mode ==
             static_cast<std::uint8_t>(FactorizedValueMode::bitpack)) {
      if (offset >= raw.size())
        throw std::runtime_error(
            "subset-rank packed code width is truncated");
      const auto width = raw[offset++];
      if (width < 1 || width > 8 ||
          count > (std::numeric_limits<std::size_t>::max() - 7U) / width)
        throw std::runtime_error(
            "subset-rank packed code width is invalid");
      const auto bytes = (count * width + 7U) / 8U;
      if (bytes > raw.size() - offset)
        throw std::runtime_error(
            "subset-rank packed code field is truncated");
      codes[channel] = unpack_int8_codes(
          raw.data() + offset, bytes, count, width);
      offset += bytes;
    }
    else if (mode ==
             static_cast<std::uint8_t>(FactorizedValueMode::run_length)) {
      const auto run_count64 = read_canonical_factorized_varint(raw, offset);
      if (run_count64 == 0 || run_count64 > count)
        throw std::runtime_error(
            "subset-rank packed run count is invalid");
      const auto run_count = static_cast<std::size_t>(run_count64);
      codes[channel].reserve(count);
      auto previous_code = std::optional<std::int8_t>();
      for (std::size_t run = 0; run < run_count; ++run) {
        if (offset >= raw.size())
          throw std::runtime_error(
              "subset-rank packed run code is truncated");
        const auto code = unzigzag_int8(raw[offset++]);
        if (previous_code.has_value() && *previous_code == code)
          throw std::runtime_error(
              "subset-rank packed adjacent runs are not canonical");
        const auto length64 = read_canonical_factorized_varint(raw, offset);
        if (length64 == 0 || length64 > count - codes[channel].size())
          throw std::runtime_error(
              "subset-rank packed run length is invalid");
        codes[channel].insert(
            codes[channel].end(), static_cast<std::size_t>(length64), code);
        previous_code = code;
      }
      if (codes[channel].size() != count)
        throw std::runtime_error(
            "subset-rank packed runs do not cover the channel");
    }
    else if (mode ==
             static_cast<std::uint8_t>(FactorizedValueMode::modal_patched)) {
      if (offset >= raw.size())
        throw std::runtime_error(
            "subset-rank packed modal code is truncated");
      const auto modal_code = unzigzag_int8(raw[offset++]);
      const auto mask_bytes = count / 8U + (count % 8U != 0);
      if (mask_bytes > raw.size() - offset)
        throw std::runtime_error(
            "subset-rank packed exception mask is truncated");
      const auto mask_offset = offset;
      if (count % 8U != 0) {
        const auto valid_bits = static_cast<unsigned char>(
            (1U << (count % 8U)) - 1U);
        if ((raw[offset + mask_bytes - 1] &
             static_cast<unsigned char>(~valid_bits)) != 0)
          throw std::runtime_error(
              "subset-rank packed exception-mask padding is nonzero");
      }
      auto exception_count = std::size_t{0};
      for (std::size_t index = 0; index < count; ++index)
        exception_count +=
            (raw[offset + index / 8U] &
             static_cast<unsigned char>(1U << (index % 8U))) != 0;
      offset += mask_bytes;
      if (exception_count == 0 || exception_count >= count ||
          offset >= raw.size())
        throw std::runtime_error(
            "subset-rank packed exception count is invalid");
      const auto width = raw[offset++];
      if (width < 1 || width > 8 ||
          exception_count >
              (std::numeric_limits<std::size_t>::max() - 7U) / width)
        throw std::runtime_error(
            "subset-rank packed exception width is invalid");
      const auto bytes = (exception_count * width + 7U) / 8U;
      if (bytes > raw.size() - offset)
        throw std::runtime_error(
            "subset-rank packed exception codes are truncated");
      const auto exceptions = unpack_int8_codes(
          raw.data() + offset, bytes, exception_count, width);
      offset += bytes;
      codes[channel].assign(count, modal_code);
      auto exception_index = std::size_t{0};
      for (std::size_t index = 0; index < count; ++index) {
        if ((raw[mask_offset + index / 8U] &
             static_cast<unsigned char>(1U << (index % 8U))) == 0)
          continue;
        if (exceptions[exception_index] == modal_code)
          throw std::runtime_error(
              "subset-rank packed exception repeats the modal code");
        codes[channel][index] = exceptions[exception_index++];
      }
      if (exception_index != exceptions.size())
        throw std::runtime_error(
            "subset-rank packed exception accounting diverged");
    }
    else if (mode ==
             static_cast<std::uint8_t>(FactorizedValueMode::delta_bitpack)) {
      if (count < 2U || raw.size() - offset < 2U)
        throw std::runtime_error(
            "predictive packed delta-bitpack header is truncated");
      const auto first = unzigzag_int8(raw[offset++]);
      const auto width = raw[offset++];
      const auto delta_count = count - 1U;
      if (width < 1U || width > 9U ||
          delta_count >
              (std::numeric_limits<std::size_t>::max() - 7U) / width)
        throw std::runtime_error(
            "predictive packed delta-bitpack width is invalid");
      const auto bytes = (delta_count * width + 7U) / 8U;
      if (bytes > raw.size() - offset)
        throw std::runtime_error(
            "predictive packed delta-bitpack field is truncated");
      const auto deltas = unpack_uint9_codes(
          raw.data() + offset, bytes, delta_count, width);
      offset += bytes;
      codes[channel].reserve(count);
      codes[channel].push_back(first);
      for (const auto encoded_delta : deltas) {
        const auto next = static_cast<int>(codes[channel].back()) +
            unzigzag_int8_delta(encoded_delta);
        if (next < std::numeric_limits<std::int8_t>::min() ||
            next > std::numeric_limits<std::int8_t>::max())
          throw std::runtime_error(
              "predictive packed delta leaves the int8 domain");
        codes[channel].push_back(static_cast<std::int8_t>(next));
      }
    }
    else if (mode == static_cast<std::uint8_t>(
                         FactorizedValueMode::delta_modal_patched)) {
      if (count < 2U || raw.size() - offset < 4U)
        throw std::runtime_error(
            "predictive packed delta-modal header is truncated");
      const auto first = unzigzag_int8(raw[offset++]);
      const auto modal = read_pod<std::uint16_t>(raw, offset);
      if (modal > 510U)
        throw std::runtime_error(
            "predictive packed modal delta is out of range");
      const auto delta_count = count - 1U;
      const auto mask_bytes =
          delta_count / 8U + (delta_count % 8U != 0U);
      if (mask_bytes > raw.size() - offset)
        throw std::runtime_error(
            "predictive packed delta exception mask is truncated");
      const auto mask_offset = offset;
      if (delta_count % 8U != 0U) {
        const auto valid_bits = static_cast<unsigned char>(
            (1U << (delta_count % 8U)) - 1U);
        if ((raw[offset + mask_bytes - 1U] &
             static_cast<unsigned char>(~valid_bits)) != 0U)
          throw std::runtime_error(
              "predictive packed delta exception-mask padding is nonzero");
      }
      auto exception_count = std::size_t{0};
      for (std::size_t index = 0; index < delta_count; ++index)
        exception_count +=
            (raw[offset + index / 8U] &
             static_cast<unsigned char>(1U << (index % 8U))) != 0U;
      offset += mask_bytes;
      if (exception_count >= delta_count || offset >= raw.size())
        throw std::runtime_error(
            "predictive packed delta exception count is invalid");
      const auto width = raw[offset++];
      if (width < 1U || width > 9U ||
          exception_count >
              (std::numeric_limits<std::size_t>::max() - 7U) / width)
        throw std::runtime_error(
            "predictive packed delta exception width is invalid");
      const auto bytes = (exception_count * width + 7U) / 8U;
      if (bytes > raw.size() - offset ||
          (exception_count == 0U && width != 1U))
        throw std::runtime_error(
            "predictive packed delta exception field is noncanonical");
      auto exceptions = std::vector<std::uint16_t>();
      if (exception_count != 0U) {
        exceptions = unpack_uint9_codes(
            raw.data() + offset, bytes, exception_count, width);
      }
      offset += bytes;
      auto deltas = std::vector<std::uint16_t>(delta_count, modal);
      auto exception_index = std::size_t{0};
      for (std::size_t index = 0; index < delta_count; ++index) {
        if ((raw[mask_offset + index / 8U] &
             static_cast<unsigned char>(1U << (index % 8U))) == 0U)
          continue;
        if (exceptions[exception_index] == modal)
          throw std::runtime_error(
              "predictive packed exception repeats the modal delta");
        deltas[index] = exceptions[exception_index++];
      }
      if (exception_index != exceptions.size())
        throw std::runtime_error(
            "predictive packed delta exception accounting diverged");
      codes[channel].reserve(count);
      codes[channel].push_back(first);
      for (const auto encoded_delta : deltas) {
        const auto next = static_cast<int>(codes[channel].back()) +
            unzigzag_int8_delta(encoded_delta);
        if (next < std::numeric_limits<std::int8_t>::min() ||
            next > std::numeric_limits<std::int8_t>::max())
          throw std::runtime_error(
              "predictive packed delta leaves the int8 domain");
        codes[channel].push_back(static_cast<std::int8_t>(next));
      }
    }
    else {
      throw std::runtime_error(
          "subset-rank packed correction has an unknown value mode");
    }
    const auto canonical = predictive_values
        ? encode_predictive_values(codes[channel])
        : encode_factorized_values(codes[channel]);
    if (canonical.mode != mode ||
        canonical.bytes.size() != offset - value_field_begin ||
        !std::equal(
            canonical.bytes.begin(),
            canonical.bytes.end(),
            raw.begin() + static_cast<std::ptrdiff_t>(value_field_begin)))
      throw std::runtime_error(
          "subset-rank packed value field is not canonical");
    const auto has_nonzero_code = std::any_of(
        codes[channel].begin(),
        codes[channel].end(),
        [](std::int8_t code) { return code != 0; });
    if ((has_nonzero_code && !(scales[channel] > 0.0F)) ||
        (!has_nonzero_code && float32_bits(scales[channel]) != 0U))
      throw std::runtime_error(
          "subset-rank packed scale is not canonical for its value field");
    total_corrections = checked_add_size(
        total_corrections,
        codes[channel].size(),
        "subset-rank packed correction count overflows");
  }
  if (total_corrections != expected_corrections)
    throw std::runtime_error(
        "subset-rank packed correction count disagrees with metadata");

  auto exact_count = std::size_t{0};
  for (const auto& channel_codes : codes)
    exact_count += static_cast<std::size_t>(std::count(
        channel_codes.begin(), channel_codes.end(), std::int8_t{0}));
  if (exact_count >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) ||
      raw.size() - offset != exact_count * sizeof(std::uint32_t))
    throw std::runtime_error(
        "subset-rank packed exact field has an invalid size");
  const auto exact_offset = offset;

  auto candidate = reconstruction;
  auto exact_index = std::size_t{0};
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto begin = channel * time;
    if (positions[channel].size() != codes[channel].size())
      throw std::logic_error(
          "subset-rank packed position/code accounting diverged");
    for (std::size_t local = 0; local < positions[channel].size(); ++local) {
      const auto index = begin + positions[channel][local];
      const auto before = candidate[index];
      const auto code = codes[channel][local];
      if (code == 0) {
        std::uint32_t xor_bits = 0;
        std::memcpy(
            &xor_bits,
            raw.data() + exact_offset + exact_index * sizeof(std::uint32_t),
            sizeof(xor_bits));
        ++exact_index;
        const auto corrected_bits = float32_bits(before) ^ xor_bits;
        std::memcpy(
            candidate.data() + index, &corrected_bits, sizeof(corrected_bits));
      }
      else {
        if (!(scales[channel] > 0.0F))
          throw std::runtime_error(
              "subset-rank packed correction uses a zero scale");
        candidate[index] = static_cast<float>(
            before + static_cast<float>(code) * scales[channel]);
      }
      if (!std::isfinite(candidate[index]) || candidate[index] == before)
        throw std::runtime_error(
            "subset-rank packed correction result is invalid or unchanged");
    }
  }
  if (exact_index != exact_count)
    throw std::runtime_error(
        "subset-rank packed exact accounting diverged");
  reconstruction.swap(candidate);
}

void apply_correction_payload(
    std::vector<float>& reconstruction,
    const std::vector<unsigned char>& payload,
    std::uint32_t expected_corrections,
    std::size_t channels,
    std::size_t time,
    std::uint32_t stream_version)
{
  if (channels == 0 || time == 0 ||
      channels > std::numeric_limits<std::size_t>::max() / time ||
      reconstruction.size() != channels * time)
    throw std::runtime_error("QPET correction dimensions do not match reconstruction");
  if (!payload.empty() &&
      payload.front() == kPredictivePackedCorrectionFormat) {
    if (stream_version < kPredictivePackedVersion)
      throw std::runtime_error(
          "predictive packed correction format 10 requires QPC version 16");
    apply_subset_rank_packed_correction(
        reconstruction,
        payload,
        expected_corrections,
        channels,
        time,
        kPredictivePackedCorrectionFormat);
    return;
  }
  if (!payload.empty() &&
      payload.front() == kSubsetRankPackedCorrectionFormat) {
    if (stream_version < kSubsetRankPackedVersion)
      throw std::runtime_error(
          "subset-rank packed correction format 9 requires QPC version 15");
    apply_subset_rank_packed_correction(
        reconstruction,
        payload,
        expected_corrections,
        channels,
        time,
        kSubsetRankPackedCorrectionFormat);
    return;
  }
  if (!payload.empty() && payload.front() == kFactorizedPackedCorrectionFormat) {
    if (stream_version < kFactorizedPackedVersion)
      throw std::runtime_error(
          "factorized packed correction format 8 requires QPC version 14");
    apply_factorized_packed_correction(
        reconstruction, payload, expected_corrections, channels, time);
    return;
  }
  if (!payload.empty() && payload.front() == kAdaptivePackedCorrectionFormat) {
    if (stream_version < kAdaptivePackedVersion)
      throw std::runtime_error(
          "adaptive packed correction format 7 requires QPC version 13");
    apply_adaptive_packed_correction(
        reconstruction, payload, expected_corrections, channels, time);
    return;
  }
  if (!payload.empty() && payload.front() == kPackedQuantizedCorrectionFormat) {
    if (stream_version < 12)
      throw std::runtime_error("packed correction format 6 requires QPC version 12");
    apply_packed_quantized_correction(
        reconstruction, payload, expected_corrections, channels, time);
    return;
  }
  if (!payload.empty() && payload.front() == 5U) {
    if (stream_version < 10)
      throw std::runtime_error(
          "constant-channel correction format 5 requires QPC version 10");
    if (expected_corrections == 0)
      throw std::runtime_error("QPET constant-channel payload has zero corrections");
    auto candidate = reconstruction;
    std::size_t offset = 1;
    const auto override_corrections = read_varint(payload, offset);
    const auto inner_corrections = read_varint(payload, offset);
    if (override_corrections > expected_corrections ||
        inner_corrections > expected_corrections ||
        override_corrections + inner_corrections != expected_corrections)
      throw std::runtime_error("QPET constant-channel correction count is invalid");
    const auto mask_bytes = channels / 8U + (channels % 8U != 0);
    if (offset > payload.size() || mask_bytes > payload.size() - offset)
      throw std::runtime_error("truncated QPET constant-channel mask");
    const auto* channel_mask = payload.data() + offset;
    offset += mask_bytes;
    if (channels % 8 != 0) {
      const auto valid_bits = static_cast<unsigned char>((1U << (channels % 8)) - 1U);
      if ((channel_mask[mask_bytes - 1] & static_cast<unsigned char>(~valid_bits)) != 0)
        throw std::runtime_error("QPET constant-channel mask has out-of-range bits");
    }
    std::uint64_t observed_overrides = 0;
    std::size_t selected_constant_channels = 0;
    for (std::size_t channel = 0; channel < channels; ++channel) {
      if ((channel_mask[channel / 8] &
           static_cast<unsigned char>(1U << (channel % 8))) == 0)
        continue;
      ++selected_constant_channels;
      if (offset > payload.size() || sizeof(float) > payload.size() - offset)
        throw std::runtime_error("truncated QPET constant-channel value");
      float value = 0.0F;
      std::memcpy(&value, payload.data() + offset, sizeof(float));
      offset += sizeof(float);
      if (!std::isfinite(value))
        throw std::runtime_error("QPET constant-channel value is not finite");
      const auto begin = channel * time;
      for (std::size_t index = 0; index < time; ++index) {
        observed_overrides += candidate[begin + index] != value;
        candidate[begin + index] = value;
      }
    }
    if (selected_constant_channels == 0 || override_corrections == 0 ||
        observed_overrides != override_corrections)
      throw std::runtime_error("QPET constant-channel correction metadata is inconsistent");
    auto inner = std::vector<unsigned char>(
        payload.begin() + static_cast<std::ptrdiff_t>(offset), payload.end());
    if ((inner_corrections == 0) != inner.empty())
      throw std::runtime_error("QPET constant-channel nested payload is inconsistent");
    if (!inner.empty() && inner.front() == 5U)
      throw std::runtime_error("QPET constant-channel payload cannot nest format 5");
    apply_correction_payload(
        candidate,
        inner,
        static_cast<std::uint32_t>(inner_corrections),
        channels,
        time,
        stream_version);
    reconstruction.swap(candidate);
    return;
  }
  auto [format, raw] = decode_correction_payload(payload, expected_corrections);
  if (raw.empty())
    return;
  if (format == 0) {
    if (raw.size() != reconstruction.size() * sizeof(double))
      throw std::runtime_error("QPET double correction residual has an invalid size");
    auto residual = std::vector<double>(reconstruction.size());
    std::memcpy(residual.data(), raw.data(), raw.size());
    std::size_t observed_corrections = 0;
    for (std::size_t index = 0; index < reconstruction.size(); ++index) {
      if (residual[index] != 0.0)
        ++observed_corrections;
      reconstruction[index] = static_cast<float>(
          static_cast<double>(reconstruction[index]) + residual[index]);
    }
    if (observed_corrections != expected_corrections)
      throw std::runtime_error("QPET double correction metadata is inconsistent");
    return;
  }
  const auto mask_bytes = (reconstruction.size() + 7) / 8;
  if (format == 4) {
    const auto scale_bytes = channels * sizeof(float);
    const auto code_bytes = static_cast<std::size_t>(expected_corrections);
    if (raw.size() < scale_bytes + mask_bytes + code_bytes ||
        (raw.size() - scale_bytes - mask_bytes - code_bytes) % sizeof(float) != 0)
      throw std::runtime_error("QPET quantized correction field has an invalid size");
    auto scales = std::vector<float>(channels);
    std::memcpy(scales.data(), raw.data(), scale_bytes);
    for (const auto scale : scales) {
      if (!(scale >= 0.0F) || !std::isfinite(scale))
        throw std::runtime_error("QPET quantized correction scale is invalid");
    }
    const auto* mask = raw.data() + scale_bytes;
    const auto* codes = reinterpret_cast<const std::int8_t*>(mask + mask_bytes);
    const auto* exact_data = reinterpret_cast<const unsigned char*>(codes + code_bytes);
    const auto exact_count =
        (raw.size() - scale_bytes - mask_bytes - code_bytes) / sizeof(float);
    std::size_t correction_index = 0;
    std::size_t exact_index = 0;
    for (std::size_t index = 0; index < reconstruction.size(); ++index) {
      if ((mask[index / 8] & static_cast<unsigned char>(1U << (index % 8))) == 0)
        continue;
      if (correction_index >= expected_corrections)
        throw std::runtime_error("QPET quantized correction mask is inconsistent");
      const auto code = codes[correction_index++];
      if (code == 0) {
        if (exact_index >= exact_count)
          throw std::runtime_error("QPET quantized correction escape is truncated");
        std::memcpy(
            reconstruction.data() + index,
            exact_data + exact_index * sizeof(float),
            sizeof(float));
        ++exact_index;
      }
      else {
        const auto scale = scales[index / time];
        if (!(scale > 0.0F))
          throw std::runtime_error("QPET quantized correction uses a zero scale");
        reconstruction[index] = static_cast<float>(
            reconstruction[index] + static_cast<float>(code) * scale);
      }
    }
    if (correction_index != expected_corrections || exact_index != exact_count)
      throw std::runtime_error("QPET quantized correction metadata is inconsistent");
    return;
  }
  if (format == 2 || format == 3) {
    if (raw.size() < mask_bytes ||
        (raw.size() - mask_bytes) % sizeof(float) != 0)
      throw std::runtime_error("QPET sparse correction field has an invalid size");
    const auto* sparse_mask = raw.data();
    const auto* sparse_data = sparse_mask + mask_bytes;
    const auto sparse_count = (raw.size() - mask_bytes) / sizeof(float);
    std::size_t sparse_index = 0;
    for (std::size_t index = 0; index < reconstruction.size(); ++index) {
      if ((sparse_mask[index / 8] &
           static_cast<unsigned char>(1U << (index % 8))) == 0)
        continue;
      if (sparse_index >= sparse_count)
        throw std::runtime_error("QPET sparse correction mask is inconsistent");
      if (format == 2) {
        std::memcpy(
            reconstruction.data() + index,
            sparse_data + sparse_index * sizeof(float),
            sizeof(float));
      }
      else {
        std::uint32_t base_bits = 0;
        std::uint32_t xor_bits = 0;
        std::memcpy(&base_bits, reconstruction.data() + index, sizeof(base_bits));
        std::memcpy(
            &xor_bits,
            sparse_data + sparse_index * sizeof(std::uint32_t),
            sizeof(xor_bits));
        const auto corrected_bits = base_bits ^ xor_bits;
        std::memcpy(
            reconstruction.data() + index,
            &corrected_bits,
            sizeof(corrected_bits));
      }
      ++sparse_index;
    }
    if (sparse_index != sparse_count || sparse_index != expected_corrections)
      throw std::runtime_error("QPET sparse correction metadata is inconsistent");
    return;
  }
  const auto residual_bytes = reconstruction.size() * sizeof(float);
  const auto minimum_size = residual_bytes + mask_bytes;
  if (raw.size() < minimum_size ||
      (raw.size() - minimum_size) % sizeof(float) != 0)
    throw std::runtime_error("QPET float correction residual has an invalid size");
  auto residual = std::vector<float>(reconstruction.size());
  std::memcpy(residual.data(), raw.data(), residual_bytes);
  const auto* exact_mask = raw.data() + residual_bytes;
  const auto* exact_data = exact_mask + mask_bytes;
  const auto exact_bytes = raw.size() - residual_bytes - mask_bytes;
  const auto exact_count = exact_bytes / sizeof(float);
  std::size_t observed_corrections = 0;
  std::size_t exact_index = 0;
  for (std::size_t index = 0; index < reconstruction.size(); ++index) {
    if (residual[index] != 0.0F)
      ++observed_corrections;
    reconstruction[index] = static_cast<float>(reconstruction[index] + residual[index]);
    if ((exact_mask[index / 8] & static_cast<unsigned char>(1U << (index % 8))) != 0) {
      if (exact_index >= exact_count)
        throw std::runtime_error("QPET correction exact-value mask is inconsistent");
      std::memcpy(
          reconstruction.data() + index,
          exact_data + exact_index * sizeof(float),
          sizeof(float));
      ++exact_index;
    }
  }
  if (observed_corrections != expected_corrections || exact_index != exact_count)
    throw std::runtime_error("QPET correction metadata is inconsistent");
}

struct LegacyRepairSeed {
  std::size_t ordinal = 0;
  std::vector<float> targets;
  std::vector<float> channel_scales;
  std::uint32_t corrections = 0;
  std::vector<unsigned char> correction_payload;
  std::vector<float> reconstruction;
  qpet_correlation::ConstraintReport report;
  bool fvro_derived = false;
  bool ssbb_derived = false;
};

struct ZrcarState {
  std::vector<std::size_t> blocked;
  std::uint32_t corrections = 0;
  std::vector<unsigned char> correction_payload;
  std::vector<float> reconstruction;
  qpet_correlation::ConstraintReport report;
  bool feasible = true;
};

struct RollbackRank {
  std::size_t index = 0;
  double violation = 0.0;
};

void finalize_posthoc_repair(
    const std::vector<float>& original,
    const Settings& settings,
    EncodedWindow& result,
    std::vector<LegacyRepairSeed>* captured_seeds = nullptr)
{
  const auto base_reconstruction = result.reconstruction;
  bool has_repair = false;
  std::size_t repair_candidates = 0;
  auto retain_repair = [&](const std::vector<float>* targets,
                           const std::vector<float>* channel_scales) {
    auto corrected = base_reconstruction;
    const auto repair = qpet_correlation::repair_constraints(
        original,
        corrected,
        settings.channels,
        settings.time,
        settings.acf_tolerance,
        settings.minimum_correlation,
        settings.max_lag,
        settings.correction_mode == 2,
        targets,
        settings.pointwise_tolerance);
    if (repair.corrected_values > std::numeric_limits<std::uint32_t>::max())
      throw std::overflow_error("QPET correction count exceeds uint32 metadata");
    const auto correction_count = static_cast<std::uint32_t>(repair.corrected_values);
    auto correction_payload = compress_correction_payload(
        base_reconstruction,
        corrected,
        channel_scales,
        settings.channels,
        settings.time);
    auto decoded = base_reconstruction;
    apply_correction_payload(
        decoded,
        correction_payload,
        correction_count,
        settings.channels,
        settings.time);
    const auto report = qpet_correlation::verify_constraints(
        original,
        decoded,
        settings.channels,
        settings.time,
        settings.pointwise_tolerance,
        settings.acf_tolerance,
        settings.minimum_correlation,
        settings.max_lag);
    if (!report.all_ok())
      throw std::runtime_error("decoded repair candidate failed a hard constraint");
    if (captured_seeds != nullptr) {
      LegacyRepairSeed seed;
      seed.ordinal = repair_candidates;
      seed.targets = targets == nullptr ? original : *targets;
      if (channel_scales != nullptr)
        seed.channel_scales = *channel_scales;
      seed.corrections = correction_count;
      seed.correction_payload = correction_payload;
      seed.reconstruction = decoded;
      seed.report = report;
      captured_seeds->push_back(std::move(seed));
    }
    ++repair_candidates;
    if (!has_repair || correction_payload.size() < result.correction_payload.size()) {
      result.corrections = correction_count;
      result.correction_payload = std::move(correction_payload);
      result.reconstruction = std::move(decoded);
      result.report = report;
      result.correction_channel_scales = channel_scales == nullptr
          ? std::vector<float>()
          : *channel_scales;
      has_repair = true;
    }
  };

  // This block is the frozen legacy algorithm. Do not reorder its target
  // families or change its strict byte comparator: ZRCAR is layered on top.
  retain_repair(nullptr, nullptr);
  if (result.corrections != 0) {
    for (const auto denominator : settings.repair_denominators) {
      const auto quantized = make_quantized_repair_targets(
          original, base_reconstruction, settings, denominator);
      retain_repair(&quantized.values, &quantized.channel_scales);
    }
  }
  if (repair_candidates > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("QPET repair candidate count exceeds uint32 diagnostics");
  result.repair_candidates_tested = static_cast<std::uint32_t>(repair_candidates);
}

std::vector<RollbackRank> rank_zrcar_rollbacks(
    const std::vector<float>& original,
    const std::vector<float>& base,
    const std::vector<float>& current,
    const std::vector<std::size_t>& blocked,
    const Settings& settings)
{
  if (original.size() != base.size() || original.size() != current.size() ||
      original.size() != settings.channels * settings.time ||
      !std::is_sorted(blocked.begin(), blocked.end()))
    throw std::invalid_argument("ZRCAR rollback ranking shape/order mismatch");
  auto ranked = std::vector<RollbackRank>();
  ranked.reserve(current.size());
  for (std::size_t channel = 0; channel < settings.channels; ++channel) {
    const auto begin = channel * settings.time;
    auto channel_current = std::vector<float>(
        current.begin() + static_cast<std::ptrdiff_t>(begin),
        current.begin() + static_cast<std::ptrdiff_t>(begin + settings.time));
    qpet_correlation::detail::SeriesRepairState rollback(
        original.data() + begin,
        channel_current.data(),
        base.data() + begin,
        settings.time,
        settings.acf_tolerance,
        settings.minimum_correlation,
        settings.max_lag);
    for (std::size_t local = 0; local < settings.time; ++local) {
      const auto index = begin + local;
      if (std::binary_search(blocked.begin(), blocked.end(), index) ||
          current[index] == base[index] || !rollback.can_repair(local))
        continue;
      const auto pointwise_error = std::abs(
          static_cast<double>(original[index]) - base[index]);
      // This predicate must be byte-for-byte semantically identical to the
      // strict target check in repair_constraints. A ranking-only epsilon
      // would admit a blocked target that later aborts the whole encode.
      if (pointwise_error > settings.pointwise_tolerance)
        continue;
      const auto structural = rollback.objective_if_repaired(local);
      if (std::isnan(structural))
        throw std::runtime_error("ZRCAR rollback objective is NaN");
      ranked.push_back({
          index,
          std::max(pointwise_error / settings.pointwise_tolerance, structural)});
    }
  }
  std::stable_sort(
      ranked.begin(), ranked.end(), [](const RollbackRank& left, const RollbackRank& right) {
        if (left.violation != right.violation)
          return left.violation < right.violation;
        return left.index < right.index;
      });
  return ranked;
}

std::vector<std::size_t> zrcar_blocked_with(
    const std::vector<std::size_t>& blocked,
    std::size_t index)
{
  auto result = blocked;
  const auto position = std::lower_bound(result.begin(), result.end(), index);
  if (position == result.end() || *position != index)
    result.insert(position, index);
  return result;
}

void require_float32_bitwise_equal(
    const std::vector<float>& observed,
    const std::vector<float>& expected,
    const char* message)
{
  if (observed.size() != expected.size() ||
      (!observed.empty() &&
       std::memcmp(
           observed.data(), expected.data(), observed.size() * sizeof(float)) != 0))
    throw std::runtime_error(message);
}

bool same_float32_tensor_bits(
    const std::vector<float>& left,
    const std::vector<float>& right)
{
  return left.size() == right.size() &&
      (left.empty() ||
       std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0);
}

struct FvroRepairEvaluation {
  bool feasible = false;
  LegacyRepairSeed seed;
};

struct FvroLocalOption {
  std::size_t ordinal = 0;
  float scale = 0.0F;
  std::vector<float> targets;
  std::vector<float> reconstruction;
  std::size_t exact_escapes = 0;
  std::size_t packed_raw_proxy = 0;
  std::size_t corrections = 0;
};

struct QuantizedEscapeSummary {
  bool applicable = false;
  std::uint8_t format = 0;
  std::size_t exact_escapes = 0;
};

std::vector<float> make_fvro_channel_scales(
    const std::vector<float>& original,
    const std::vector<float>& base,
    std::size_t channels,
    std::size_t time,
    int denominator)
{
  if (channels == 0 || time == 0 || original.size() != base.size() ||
      original.size() != channels * time || denominator < 1 || denominator > 127)
    throw std::invalid_argument("FVRO scale dimensions or denominator are invalid");
  auto scales = std::vector<float>(channels, 0.0F);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto begin = channel * time;
    double largest_residual = 0.0;
    for (std::size_t index = 0; index < time; ++index) {
      largest_residual = std::max(
          largest_residual,
          std::abs(
              static_cast<double>(original[begin + index]) -
              static_cast<double>(base[begin + index])));
    }
    if (!(largest_residual > 0.0))
      continue;
    const auto scale = static_cast<float>(
        largest_residual / static_cast<double>(denominator));
    if (scale > 0.0F && std::isfinite(scale))
      scales[channel] = scale;
  }
  return scales;
}

std::vector<std::size_t> quantized_escape_counts_by_channel(
    const std::vector<float>& base,
    const std::vector<float>& corrected,
    const std::vector<float>& channel_scales,
    std::size_t channels,
    std::size_t time)
{
  const auto fields = make_quantized_correction_fields(
      base, corrected, channel_scales, channels, time);
  auto counts = std::vector<std::size_t>(channels, 0);
  std::size_t correction_index = 0;
  for (std::size_t index = 0; index < base.size(); ++index) {
    const auto selected =
        (fields.mask[index / 8U] &
         static_cast<unsigned char>(1U << (index % 8U))) != 0;
    if (!selected)
      continue;
    if (correction_index >= fields.codes.size())
      throw std::logic_error("FVRO quantized mask/code accounting diverged");
    if (fields.codes[correction_index++] == 0)
      ++counts[index / time];
  }
  if (correction_index != fields.codes.size())
    throw std::logic_error("FVRO quantized code stream has trailing entries");
  return counts;
}

QuantizedEscapeSummary quantized_escape_summary(
    const std::vector<float>& base,
    const LegacyRepairSeed& seed,
    std::size_t channels,
    std::size_t time)
{
  if (seed.channel_scales.size() != channels || seed.correction_payload.empty() ||
      (seed.correction_payload.front() != 4U &&
       seed.correction_payload.front() != kPackedQuantizedCorrectionFormat &&
       seed.correction_payload.front() != kAdaptivePackedCorrectionFormat &&
       seed.correction_payload.front() != kFactorizedPackedCorrectionFormat &&
       seed.correction_payload.front() != kSubsetRankPackedCorrectionFormat &&
       seed.correction_payload.front() != kPredictivePackedCorrectionFormat))
    return {};
  const auto counts = quantized_escape_counts_by_channel(
      base, seed.reconstruction, seed.channel_scales, channels, time);
  const auto total = std::accumulate(
      counts.begin(), counts.end(), std::size_t{0});
  return {true, seed.correction_payload.front(), total};
}

std::size_t count_correction_values(
    const std::vector<float>& base,
    const std::vector<float>& corrected)
{
  if (base.size() != corrected.size())
    throw std::invalid_argument("FVRO correction tensors have different sizes");
  auto count = std::size_t{0};
  for (std::size_t index = 0; index < base.size(); ++index)
    count += base[index] != corrected[index];
  return count;
}

LegacyRepairSeed serialize_fvro_reconstruction(
    const std::vector<float>& original,
    const std::vector<float>& base,
    const std::vector<float>& targets,
    const std::vector<float>& channel_scales,
    const std::vector<float>& corrected,
    const Settings& settings,
    std::size_t ordinal,
    bool enable_adaptive_format = false)
{
  if (targets.size() != base.size() || corrected.size() != base.size() ||
      channel_scales.size() != settings.channels)
    throw std::invalid_argument("FVRO serialized candidate dimensions are invalid");
  const auto corrections = count_correction_values(base, corrected);
  if (corrections > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("FVRO correction count exceeds uint32 metadata");
  auto payload = enable_adaptive_format
      ? compress_adaptive_correction_payload(
            base,
            corrected,
            channel_scales,
            settings.channels,
            settings.time,
            settings.correction_format_ceiling)
      : compress_correction_payload(
            base,
            corrected,
            &channel_scales,
            settings.channels,
            settings.time);
  if ((corrections == 0) != payload.empty())
    throw std::logic_error("FVRO correction count/payload presence is inconsistent");
  auto decoded = base;
  apply_correction_payload(
      decoded,
      payload,
      static_cast<std::uint32_t>(corrections),
      settings.channels,
      settings.time,
      correction_payload_required_stream_version(payload));
  require_float32_bitwise_equal(
      decoded,
      corrected,
      "FVRO serializer replay is not float32-bitwise identical to its candidate");
  const auto report = qpet_correlation::verify_constraints(
      original,
      decoded,
      settings.channels,
      settings.time,
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag);
  if (!report.all_ok())
    throw std::runtime_error("FVRO serializer replay failed a hard constraint");
  LegacyRepairSeed seed;
  seed.ordinal = ordinal;
  seed.targets = targets;
  seed.channel_scales = channel_scales;
  seed.corrections = static_cast<std::uint32_t>(corrections);
  seed.correction_payload = std::move(payload);
  seed.reconstruction = std::move(decoded);
  seed.report = report;
  seed.fvro_derived = true;
  return seed;
}

FvroRepairEvaluation evaluate_fvro_targets(
    const std::vector<float>& original,
    const std::vector<float>& base,
    const std::vector<float>& targets,
    const std::vector<float>& channel_scales,
    const Settings& settings,
    std::size_t ordinal)
{
  auto corrected = base;
  const auto repair = qpet_correlation::repair_constraints(
      original,
      corrected,
      settings.channels,
      settings.time,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag,
      settings.correction_mode == 2,
      &targets,
      settings.pointwise_tolerance,
      false);
  if (!repair.feasible) {
    const auto report = qpet_correlation::verify_constraints(
        original,
        corrected,
        settings.channels,
        settings.time,
        settings.pointwise_tolerance,
        settings.acf_tolerance,
        settings.minimum_correlation,
        settings.max_lag);
    if (report.all_ok())
      throw std::logic_error(
          "FVRO repair reported exhaustion for a directly feasible target");
    return {};
  }
  if (repair.corrected_values != count_correction_values(base, corrected))
    throw std::logic_error("FVRO repair and serializer correction counts disagree");
  return {
      true,
      serialize_fvro_reconstruction(
          original,
          base,
          targets,
          channel_scales,
          corrected,
          settings,
          ordinal)};
}

void retain_unique_fvro_seed(
    std::vector<LegacyRepairSeed>& seeds,
    LegacyRepairSeed seed)
{
  const auto duplicate = std::find_if(
      seeds.begin(), seeds.end(), [&](const LegacyRepairSeed& retained) {
        return same_float32_tensor_bits(retained.targets, seed.targets) &&
            same_float32_tensor_bits(
                retained.channel_scales, seed.channel_scales);
      });
  if (duplicate != seeds.end()) {
    if (duplicate->corrections != seed.corrections ||
        duplicate->correction_payload != seed.correction_payload ||
        !same_float32_tensor_bits(
            duplicate->reconstruction, seed.reconstruction))
      throw std::logic_error(
          "identical FVRO seed identity produced different serialized repair");
    // Provenance is counterfactual: if the same target/scale seed already
    // exists in the frozen legacy bank, a later FVRO rediscovery is not
    // necessary for that seed. Keeping the conjunction prevents a ZRCAR win
    // from being falsely attributed to FVRO merely because both paths reached
    // identical bytes and float32 reconstruction.
    duplicate->fvro_derived = duplicate->fvro_derived && seed.fvro_derived;
    return;
  }
  seeds.push_back(std::move(seed));
}

std::size_t next_fvro_seed_ordinal(const std::vector<LegacyRepairSeed>& seeds)
{
  std::size_t next = 0;
  for (const auto& seed : seeds) {
    if (seed.ordinal == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("FVRO seed ordinal overflow");
    next = std::max(next, seed.ordinal + 1);
  }
  return next;
}

std::size_t matching_seed_index(
    const std::vector<LegacyRepairSeed>& seeds,
    const EncodedWindow& result)
{
  const auto match = std::find_if(
      seeds.begin(), seeds.end(), [&](const LegacyRepairSeed& seed) {
        return seed.corrections == result.corrections &&
            seed.correction_payload == result.correction_payload &&
            same_float32_tensor_bits(
                seed.reconstruction, result.reconstruction);
      });
  if (match == seeds.end())
    throw std::logic_error("FVRO could not identify the frozen legacy incumbent seed");
  return static_cast<std::size_t>(std::distance(seeds.begin(), match));
}

std::vector<qpet_correlation::feasible_value::TargetCandidate>
make_fvro_global_target_family(
    const qpet_correlation::feasible_value::TargetBuilder& builder)
{
  using qpet_correlation::feasible_value::HomotopyLevel;
  using qpet_correlation::feasible_value::TargetCandidate;
  auto candidates = std::vector<TargetCandidate>();
  auto append_unique = [&](TargetCandidate candidate) {
    const auto duplicate = std::any_of(
        candidates.begin(), candidates.end(), [&](const TargetCandidate& retained) {
          return same_float32_tensor_bits(retained.values, candidate.values);
        });
    if (!duplicate)
      candidates.push_back(std::move(candidate));
  };
  append_unique(builder.minimum_candidate());
  for (const auto level :
       std::array<HomotopyLevel, 3>{{{1, 4}, {1, 2}, {3, 4}}})
    append_unique(builder.homotopy_candidate(level));
  const auto nearest = builder.packed_bit_width_candidate(8);
  std::uint8_t nearest_width = 1;
  while (nearest_width < 8 &&
         (nearest.maximum_zigzag >> nearest_width) != 0)
    ++nearest_width;
  const auto first_width = static_cast<std::uint8_t>(
      nearest_width > 3 ? nearest_width - 3 : 1);
  for (auto bit_width = first_width; bit_width <= nearest_width; ++bit_width)
    append_unique(builder.packed_bit_width_candidate(bit_width));
  if (candidates.size() > kFvroGlobalSerializerBudget)
    throw std::logic_error("FVRO global target family exceeded its fixed budget");
  return candidates;
}

std::vector<FvroLocalOption> make_fvro_local_options(
    const std::vector<float>& original,
    const std::vector<float>& base,
    const Settings& settings,
    std::size_t& proposals_tested,
    std::size_t& infeasible_proposals)
{
  if (original.size() != settings.time || base.size() != settings.time)
    throw std::invalid_argument("FVRO local channel dimensions are invalid");
  auto options = std::vector<FvroLocalOption>();
  std::size_t ordinal = 0;
  for (const auto denominator : kFvroLatticeDenominators) {
    const auto scales = make_fvro_channel_scales(
        original, base, 1, settings.time, denominator);
    qpet_correlation::feasible_value::TargetBuilder builder(
        original,
        base,
        scales,
        1,
        settings.time,
        settings.pointwise_tolerance);
    auto targets = std::vector<qpet_correlation::feasible_value::TargetCandidate>();
    targets.push_back(builder.minimum_candidate());
    targets.push_back(builder.homotopy_candidate({1, 2}));
    targets.push_back(builder.packed_bit_width_candidate(8));
    for (auto& target : targets) {
      ++proposals_tested;
      auto corrected = base;
      const auto repair = qpet_correlation::repair_constraints(
          original,
          corrected,
          1,
          settings.time,
          settings.acf_tolerance,
          settings.minimum_correlation,
          settings.max_lag,
          settings.correction_mode == 2,
          &target.values,
          settings.pointwise_tolerance,
          false);
      if (!repair.feasible) {
        const auto report = qpet_correlation::verify_constraints(
            original,
            corrected,
            1,
            settings.time,
            settings.pointwise_tolerance,
            settings.acf_tolerance,
            settings.minimum_correlation,
            settings.max_lag);
        if (report.all_ok())
          throw std::logic_error(
              "FVRO local repair reported exhaustion for a feasible target");
        ++infeasible_proposals;
        ++ordinal;
        continue;
      }
      const auto report = qpet_correlation::verify_constraints(
          original,
          corrected,
          1,
          settings.time,
          settings.pointwise_tolerance,
          settings.acf_tolerance,
          settings.minimum_correlation,
          settings.max_lag);
      if (!report.all_ok())
        throw std::logic_error("FVRO local repair reported a false feasible state");
      if (repair.corrected_values != count_correction_values(base, corrected))
        throw std::logic_error(
            "FVRO local repair and correction counts disagree");
      const auto fields = make_quantized_correction_fields(
          base, corrected, scales, 1, settings.time);
      const auto exact_escapes = static_cast<std::size_t>(std::count(
          fields.codes.begin(), fields.codes.end(), static_cast<std::int8_t>(0)));
      std::size_t packed_raw = 0;
      if (!fields.codes.empty()) {
        const auto packed = pack_int8_codes(fields.codes);
        packed_raw = checked_add_size(
            1 + sizeof(float) + fields.mask.size(),
            packed.bytes.size(),
            "FVRO local packed proxy overflow");
        packed_raw = checked_add_size(
            packed_raw,
            fields.exact_values.size() * sizeof(float),
            "FVRO local exact proxy overflow");
      }
      options.push_back(FvroLocalOption{
          ordinal,
          scales.front(),
          std::move(target.values),
          std::move(corrected),
          exact_escapes,
          packed_raw,
          repair.corrected_values});
      ++ordinal;
    }
  }
  std::stable_sort(
      options.begin(), options.end(), [](const FvroLocalOption& left,
                                         const FvroLocalOption& right) {
        if (left.exact_escapes != right.exact_escapes)
          return left.exact_escapes < right.exact_escapes;
        if (left.packed_raw_proxy != right.packed_raw_proxy)
          return left.packed_raw_proxy < right.packed_raw_proxy;
        if (left.corrections != right.corrections)
          return left.corrections < right.corrections;
        return left.ordinal < right.ordinal;
      });
  auto unique = std::vector<FvroLocalOption>();
  for (auto& option : options) {
    const auto duplicate = std::any_of(
        unique.begin(), unique.end(), [&](const FvroLocalOption& retained) {
          return float32_bits(retained.scale) == float32_bits(option.scale) &&
              same_float32_tensor_bits(
                  retained.reconstruction, option.reconstruction);
        });
    if (duplicate)
      continue;
    unique.push_back(std::move(option));
    if (unique.size() == kFvroChannelAlternativeCount)
      break;
  }
  return unique;
}

void copy_channel(
    std::vector<float>& destination,
    std::size_t channel,
    std::size_t time,
    const std::vector<float>& source)
{
  if (source.size() != time ||
      channel >= destination.size() / time || destination.size() % time != 0)
    throw std::invalid_argument("FVRO channel copy dimensions are invalid");
  std::copy(
      source.begin(),
      source.end(),
      destination.begin() + static_cast<std::ptrdiff_t>(channel * time));
}

void retain_fvro_if_strictly_smaller(
    const LegacyRepairSeed& candidate,
    EncodedWindow& result)
{
  if (candidate.correction_payload.size() >= result.correction_payload.size())
    return;
  result.corrections = candidate.corrections;
  result.correction_payload = candidate.correction_payload;
  result.reconstruction = candidate.reconstruction;
  result.report = candidate.report;
  result.correction_channel_scales = candidate.channel_scales;
  result.fvro_selected = true;
}

void run_fvro_value_search(
    const std::vector<float>& original,
    const std::vector<float>& base,
    const Settings& settings,
    std::vector<LegacyRepairSeed>& seeds,
    EncodedWindow& result)
{
  if (!uses_fvro(settings.repair_optimizer))
    return;
  if (seeds.empty())
    throw std::logic_error("FVRO received no deterministic legacy seed");
  const auto legacy_seed_index = matching_seed_index(seeds, result);
  const auto legacy_seed = seeds[legacy_seed_index];
  const auto reference_summary = quantized_escape_summary(
      base, legacy_seed, settings.channels, settings.time);
  result.fvro_reference_quantized = reference_summary.applicable;
  result.fvro_reference_quantized_format = reference_summary.format;
  if (reference_summary.exact_escapes > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("FVRO reference escape count exceeds uint32");
  result.fvro_reference_exact_escapes =
      static_cast<std::uint32_t>(reference_summary.exact_escapes);

  auto selected_quantized = seeds.end();
  for (auto iterator = seeds.begin(); iterator != seeds.end(); ++iterator) {
    if (iterator->channel_scales.size() != settings.channels)
      continue;
    if (selected_quantized == seeds.end() ||
        iterator->correction_payload.size() <
            selected_quantized->correction_payload.size() ||
        (iterator->correction_payload.size() ==
             selected_quantized->correction_payload.size() &&
         iterator->ordinal < selected_quantized->ordinal))
      selected_quantized = iterator;
  }
  auto global_scales = selected_quantized == seeds.end()
      ? make_fvro_channel_scales(
            original, base, settings.channels, settings.time, 127)
      : selected_quantized->channel_scales;
  auto next_ordinal = next_fvro_seed_ordinal(seeds);
  auto current_seed = legacy_seed;
  if (current_seed.channel_scales.size() != settings.channels)
    current_seed.channel_scales = global_scales;
  const auto original_seed_count = seeds.size();
  std::size_t global_generated = 0;
  std::size_t global_serialized = 0;
  std::size_t global_infeasible = 0;
  std::size_t channel_generated = 0;
  std::size_t channel_serialized = 0;
  std::size_t channel_infeasible = 0;
  std::size_t local_proposals = 0;
  std::size_t local_infeasible = 0;

  qpet_correlation::feasible_value::TargetBuilder builder(
      original,
      base,
      global_scales,
      settings.channels,
      settings.time,
      settings.pointwise_tolerance);
  auto global_targets = make_fvro_global_target_family(builder);
  global_generated = global_targets.size();
  for (auto& target : global_targets) {
    auto evaluated = evaluate_fvro_targets(
        original,
        base,
        target.values,
        global_scales,
        settings,
        next_ordinal++);
    if (!evaluated.feasible) {
      ++global_infeasible;
      continue;
    }
    ++global_serialized;
    const auto candidate = evaluated.seed;
    const auto strictly_smaller =
        candidate.correction_payload.size() < result.correction_payload.size();
    retain_fvro_if_strictly_smaller(candidate, result);
    if (strictly_smaller)
      current_seed = candidate;
    retain_unique_fvro_seed(seeds, std::move(evaluated.seed));
  }

  auto difficulty_scales = current_seed.channel_scales.size() == settings.channels
      ? current_seed.channel_scales
      : global_scales;
  const auto escape_counts = quantized_escape_counts_by_channel(
      base,
      current_seed.reconstruction,
      difficulty_scales,
      settings.channels,
      settings.time);
  auto difficult_channels = std::vector<std::size_t>();
  for (std::size_t channel = 0; channel < settings.channels; ++channel) {
    if (escape_counts[channel] != 0)
      difficult_channels.push_back(channel);
  }
  std::stable_sort(
      difficult_channels.begin(),
      difficult_channels.end(),
      [&](std::size_t left, std::size_t right) {
        if (escape_counts[left] != escape_counts[right])
          return escape_counts[left] > escape_counts[right];
        return left < right;
      });
  if (difficult_channels.size() > kFvroMaximumDifficultChannels)
    difficult_channels.resize(kFvroMaximumDifficultChannels);

  auto selected_channels = std::size_t{0};
  auto strict_channel_steps = std::size_t{0};
  for (const auto channel : difficult_channels) {
    const auto begin = channel * settings.time;
    const auto local_original = std::vector<float>(
        original.begin() + static_cast<std::ptrdiff_t>(begin),
        original.begin() + static_cast<std::ptrdiff_t>(begin + settings.time));
    const auto local_base = std::vector<float>(
        base.begin() + static_cast<std::ptrdiff_t>(begin),
        base.begin() + static_cast<std::ptrdiff_t>(begin + settings.time));
    auto options = make_fvro_local_options(
        local_original,
        local_base,
        settings,
        local_proposals,
        local_infeasible);
    bool accepted_channel = false;
    for (const auto& option : options) {
      ++channel_generated;
      auto targets = current_seed.targets;
      auto scales = current_seed.channel_scales;
      auto reconstruction = current_seed.reconstruction;
      if (targets.size() != original.size())
        throw std::logic_error("FVRO current seed lost its target tensor");
      if (scales.size() != settings.channels)
        scales = global_scales;
      copy_channel(targets, channel, settings.time, option.targets);
      copy_channel(
          reconstruction, channel, settings.time, option.reconstruction);
      scales[channel] = option.scale;
      auto candidate = serialize_fvro_reconstruction(
          original,
          base,
          targets,
          scales,
          reconstruction,
          settings,
          next_ordinal++);
      ++channel_serialized;
      const auto strictly_smaller =
          candidate.correction_payload.size() < result.correction_payload.size();
      retain_fvro_if_strictly_smaller(candidate, result);
      if (strictly_smaller) {
        current_seed = candidate;
        accepted_channel = true;
        ++strict_channel_steps;
      }
      retain_unique_fvro_seed(seeds, std::move(candidate));
    }
    if (accepted_channel)
      ++selected_channels;
  }

  if (global_serialized > kFvroGlobalSerializerBudget ||
      channel_serialized > kFvroChannelSerializerBudget ||
      local_proposals > kFvroLocalProposalBudgetPerBase ||
      global_serialized + channel_serialized >
          kFvroFullSerializerBudgetPerBase)
    throw std::logic_error("FVRO exceeded a fixed v1 search budget");
  if (global_generated != global_serialized + global_infeasible ||
      channel_generated != channel_serialized + channel_infeasible ||
      local_infeasible > local_proposals)
    throw std::logic_error(
        "FVRO generated/infeasible/serialized diagnostics diverged");
  if (global_generated > std::numeric_limits<std::uint32_t>::max() ||
      global_serialized > std::numeric_limits<std::uint32_t>::max() ||
      global_infeasible > std::numeric_limits<std::uint32_t>::max() ||
      channel_generated > std::numeric_limits<std::uint32_t>::max() ||
      channel_serialized > std::numeric_limits<std::uint32_t>::max() ||
      channel_infeasible > std::numeric_limits<std::uint32_t>::max() ||
      local_proposals > std::numeric_limits<std::uint32_t>::max() ||
      local_infeasible > std::numeric_limits<std::uint32_t>::max() ||
      selected_channels > std::numeric_limits<std::uint32_t>::max() ||
      strict_channel_steps > std::numeric_limits<std::uint32_t>::max() ||
      seeds.size() - original_seed_count >
          std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("FVRO diagnostics exceed uint32");
  result.fvro_global_candidates_generated =
      static_cast<std::uint32_t>(global_generated);
  result.fvro_global_candidates_serialized =
      static_cast<std::uint32_t>(global_serialized);
  result.fvro_global_candidates_infeasible =
      static_cast<std::uint32_t>(global_infeasible);
  result.fvro_channel_candidates_generated =
      static_cast<std::uint32_t>(channel_generated);
  result.fvro_channel_candidates_serialized =
      static_cast<std::uint32_t>(channel_serialized);
  result.fvro_channel_candidates_infeasible =
      static_cast<std::uint32_t>(channel_infeasible);
  result.fvro_local_proposals_generated =
      static_cast<std::uint32_t>(local_proposals);
  result.fvro_local_proposals_infeasible =
      static_cast<std::uint32_t>(local_infeasible);
  result.fvro_selected_channels =
      static_cast<std::uint32_t>(selected_channels);
  result.fvro_strict_channel_steps =
      static_cast<std::uint32_t>(strict_channel_steps);
  if (result.fvro_selected_channels > result.fvro_strict_channel_steps ||
      result.fvro_strict_channel_steps >
          result.fvro_channel_candidates_serialized ||
      ((result.fvro_selected_channels == 0) !=
       (result.fvro_strict_channel_steps == 0)))
    throw std::logic_error("FVRO strict channel-step accounting diverged");
  result.fvro_seeds_retained = static_cast<std::uint32_t>(
      seeds.size() - original_seed_count);
  result.repair_candidates_tested = checked_add_u32(
      result.repair_candidates_tested,
      static_cast<std::uint32_t>(global_serialized + channel_serialized),
      "QPET legacy plus FVRO repair candidate count overflow");

  const auto final_seed_index = matching_seed_index(seeds, result);
  const auto final_summary = quantized_escape_summary(
      base, seeds[final_seed_index], settings.channels, settings.time);
  result.fvro_final_quantized = final_summary.applicable;
  result.fvro_final_quantized_format = final_summary.format;
  if (final_summary.exact_escapes > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("FVRO final escape count exceeds uint32");
  result.fvro_final_exact_escapes =
      static_cast<std::uint32_t>(final_summary.exact_escapes);
}

std::vector<std::size_t> select_zrcar_seed_order(
    const std::vector<LegacyRepairSeed>& legacy_seeds,
    RepairOptimizer optimizer = RepairOptimizer::zrcar)
{
  if (optimizer != RepairOptimizer::zrcar &&
      optimizer != RepairOptimizer::fvro &&
      !uses_full_ssbb(optimizer))
    throw std::invalid_argument(
        "ZRCAR seed selection requires zrcar, full fvro, full ssbb, or CJRF mode");
  auto ranked = std::vector<std::size_t>(legacy_seeds.size());
  std::iota(ranked.begin(), ranked.end(), std::size_t{0});
  std::stable_sort(
      ranked.begin(), ranked.end(), [&](std::size_t left, std::size_t right) {
        const auto left_bytes = legacy_seeds[left].correction_payload.size();
        const auto right_bytes = legacy_seeds[right].correction_payload.size();
        if (left_bytes != right_bytes)
          return left_bytes < right_bytes;
        return legacy_seeds[left].ordinal < legacy_seeds[right].ordinal;
      });
  auto selected = std::vector<std::size_t>();
  const auto maximum_seeds = optimizer != RepairOptimizer::zrcar
      ? kFullFvroZrcarSeedCount
      : kZrcarSeedCount;
  selected.reserve(std::min(ranked.size(), maximum_seeds));
  auto append_category = [&](bool require_fvro_derived, std::size_t limit) {
    auto appended = std::size_t{0};
    for (const auto index : ranked) {
      if (legacy_seeds[index].ssbb_derived)
        continue;
      if (legacy_seeds[index].fvro_derived != require_fvro_derived)
        continue;
      const auto duplicate = std::find_if(
          selected.begin(), selected.end(), [&](std::size_t retained_index) {
            return same_float32_tensor_bits(
                       legacy_seeds[retained_index].targets,
                       legacy_seeds[index].targets) &&
                same_float32_tensor_bits(
                       legacy_seeds[retained_index].channel_scales,
                       legacy_seeds[index].channel_scales);
          });
      if (duplicate != selected.end()) {
        const auto& retained = legacy_seeds[*duplicate];
        const auto& candidate = legacy_seeds[index];
        if (retained.corrections != candidate.corrections ||
            retained.correction_payload != candidate.correction_payload ||
            !same_float32_tensor_bits(
                retained.reconstruction, candidate.reconstruction))
          throw std::logic_error(
              "identical ZRCAR seed identity produced different repair bytes");
        continue;
      }
      selected.push_back(index);
      ++appended;
      if (appended == limit)
        break;
    }
  };
  // Full FVRO freezes the exact two non-FVRO seeds selected by standalone
  // ZRCAR, then adds at most one genuinely FVRO-derived seed. This makes the
  // full candidate set a superset of both ablations without allowing new
  // seeds to evict a legacy rollback path.
  append_category(false, kZrcarSeedCount);
  if (optimizer != RepairOptimizer::zrcar)
    append_category(true, kFvroAdditionalZrcarSeedCount);
  if (selected.size() > maximum_seeds)
    throw std::logic_error("ZRCAR seed selection exceeded its fixed budget");
  if (optimizer == RepairOptimizer::zrcar &&
      std::any_of(selected.begin(), selected.end(), [&](std::size_t index) {
        return legacy_seeds[index].fvro_derived;
      }))
    throw std::logic_error("standalone ZRCAR selected an FVRO-derived seed");
  if (optimizer != RepairOptimizer::zrcar &&
      std::count_if(selected.begin(), selected.end(), [&](std::size_t index) {
        return legacy_seeds[index].fvro_derived;
      }) > static_cast<std::ptrdiff_t>(kFvroAdditionalZrcarSeedCount))
    throw std::logic_error("full FVRO selected too many FVRO-derived seeds");
  if (optimizer != RepairOptimizer::zrcar &&
      std::count_if(selected.begin(), selected.end(), [&](std::size_t index) {
        return !legacy_seeds[index].fvro_derived;
      }) > static_cast<std::ptrdiff_t>(kZrcarSeedCount))
    throw std::logic_error("full FVRO selected too many frozen legacy seeds");
  if (optimizer == RepairOptimizer::zrcar && selected.size() > kZrcarSeedCount)
    throw std::logic_error("standalone ZRCAR selected too many seeds");
  if (optimizer != RepairOptimizer::zrcar &&
      selected.size() > kFullFvroZrcarSeedCount)
    throw std::logic_error("full FVRO selected too many ZRCAR seeds");
  if (selected.empty() && !legacy_seeds.empty()) {
    // Every standalone input is legacy-derived; full FVRO also always starts
    // from the frozen repair bank. An empty selection therefore indicates a
    // provenance/accounting bug rather than a valid no-op.
    throw std::logic_error("ZRCAR seed selector dropped every available seed");
  }
  return selected;
}

ZrcarState evaluate_zrcar_state(
    const std::vector<float>& original,
    const std::vector<float>& base,
    const LegacyRepairSeed& seed,
    const std::vector<std::size_t>& blocked,
    const Settings& settings)
{
  if (!std::is_sorted(blocked.begin(), blocked.end()) ||
      std::adjacent_find(blocked.begin(), blocked.end()) != blocked.end() ||
      seed.targets.size() != base.size())
    throw std::invalid_argument("ZRCAR state has invalid blocked indices or targets");
  auto effective_targets = seed.targets;
  for (const auto index : blocked) {
    if (index >= effective_targets.size())
      throw std::out_of_range("ZRCAR blocked index exceeds the repair tensor");
    effective_targets[index] = base[index];
  }
  auto corrected = base;
  const auto repair = qpet_correlation::repair_constraints(
      original,
      corrected,
      settings.channels,
      settings.time,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag,
      settings.correction_mode == 2,
      &effective_targets,
      settings.pointwise_tolerance,
      false);
  if (repair.corrected_values > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("ZRCAR correction count exceeds uint32 metadata");
  const auto correction_count = static_cast<std::uint32_t>(repair.corrected_values);
  if (!repair.feasible) {
    const auto report = qpet_correlation::verify_constraints(
        original,
        corrected,
        settings.channels,
        settings.time,
        settings.pointwise_tolerance,
        settings.acf_tolerance,
        settings.minimum_correlation,
        settings.max_lag);
    if (report.all_ok())
      throw std::logic_error(
          "ZRCAR repair reported exhaustion for a directly feasible state");
    return ZrcarState{
        blocked,
        correction_count,
        {},
        std::move(corrected),
        report,
        false};
  }
  const auto* scales = seed.channel_scales.empty() ? nullptr : &seed.channel_scales;
  auto payload = compress_correction_payload(
      base, corrected, scales, settings.channels, settings.time);
  auto decoded = base;
  apply_correction_payload(
      decoded,
      payload,
      correction_count,
      settings.channels,
      settings.time,
      kVersion);
  require_float32_bitwise_equal(
      decoded,
      corrected,
      "ZRCAR serializer replay is not float32-bitwise identical to its candidate");
  const auto report = qpet_correlation::verify_constraints(
      original,
      decoded,
      settings.channels,
      settings.time,
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag);
  if (!report.all_ok())
    throw std::runtime_error("ZRCAR serializer replay failed a hard constraint");
  return ZrcarState{
      blocked,
      correction_count,
      std::move(payload),
      std::move(decoded),
      report};
}

void retain_zrcar_beam(std::vector<ZrcarState>& states)
{
  if (std::any_of(states.begin(), states.end(), [](const ZrcarState& state) {
        return !state.feasible;
      }))
    throw std::logic_error("feasible ZRCAR beam received an infeasible state");
  std::stable_sort(
      states.begin(), states.end(), [](const ZrcarState& left, const ZrcarState& right) {
        if (left.correction_payload.size() != right.correction_payload.size())
          return left.correction_payload.size() < right.correction_payload.size();
        if (left.corrections != right.corrections)
          return left.corrections < right.corrections;
        return left.blocked < right.blocked;
      });
  auto unique = std::vector<ZrcarState>();
  unique.reserve(std::min(states.size(), kZrcarBeamWidth));
  for (auto& state : states) {
    const auto duplicate = std::any_of(
        unique.begin(), unique.end(), [&](const ZrcarState& retained) {
          return state.blocked == retained.blocked;
        });
    if (duplicate)
      continue;
    unique.push_back(std::move(state));
    if (unique.size() == kZrcarBeamWidth)
      break;
  }
  states = std::move(unique);
}

void retain_zrcar_infeasible_beam(std::vector<ZrcarState>& states)
{
  if (std::any_of(states.begin(), states.end(), [](const ZrcarState& state) {
        return state.feasible;
      }))
    throw std::logic_error("infeasible ZRCAR beam received a feasible state");
  std::stable_sort(
      states.begin(), states.end(), [](const ZrcarState& left, const ZrcarState& right) {
        if (left.blocked.size() != right.blocked.size())
          return left.blocked.size() > right.blocked.size();
        return left.blocked < right.blocked;
      });
  auto unique = std::vector<ZrcarState>();
  unique.reserve(std::min(states.size(), kZrcarInfeasibleBeamWidth));
  for (auto& state : states) {
    const auto duplicate = std::any_of(
        unique.begin(), unique.end(), [&](const ZrcarState& retained) {
          return state.blocked == retained.blocked;
        });
    if (duplicate)
      continue;
    unique.push_back(std::move(state));
    if (unique.size() == kZrcarInfeasibleBeamWidth)
      break;
  }
  states = std::move(unique);
}

std::vector<std::size_t> zrcar_geometric_prefixes(std::size_t count)
{
  auto prefixes = std::vector<std::size_t>();
  if (count == 0)
    return prefixes;
  for (std::size_t value = 1;
       value <= count && prefixes.size() < kZrcarPrefixBudgetPerSeed;) {
    prefixes.push_back(value);
    if (value > count / 2)
      break;
    value *= 2;
  }
  if (prefixes.back() != count) {
    if (prefixes.size() == kZrcarPrefixBudgetPerSeed)
      prefixes.back() = count;
    else
      prefixes.push_back(count);
  }
  prefixes.erase(std::unique(prefixes.begin(), prefixes.end()), prefixes.end());
  return prefixes;
}

void run_zrcar_search(
    const std::vector<float>& original,
    const std::vector<float>& base,
    const Settings& settings,
    const std::vector<LegacyRepairSeed>& legacy_seeds,
    EncodedWindow& result)
{
  if (!uses_zrcar(settings.repair_optimizer))
    return;
  if (legacy_seeds.empty())
    throw std::logic_error("ZRCAR received no deterministic legacy seed");
  result.legacy_repair_bytes = result.correction_payload.size();
  const auto seed_order = select_zrcar_seed_order(
      legacy_seeds, settings.repair_optimizer);
  const auto legacy_seed_count = static_cast<std::size_t>(std::count_if(
      seed_order.begin(), seed_order.end(), [&](std::size_t index) {
        return !legacy_seeds[index].fvro_derived &&
            !legacy_seeds[index].ssbb_derived;
      }));
  const auto fvro_seed_count = seed_order.size() - legacy_seed_count;
  if (legacy_seed_count > kZrcarSeedCount ||
      fvro_seed_count > kFvroAdditionalZrcarSeedCount ||
      (settings.repair_optimizer == RepairOptimizer::zrcar &&
       fvro_seed_count != 0) ||
      seed_order.size() >
          (settings.repair_optimizer != RepairOptimizer::zrcar
               ? kFullFvroZrcarSeedCount
               : kZrcarSeedCount))
    throw std::logic_error("ZRCAR selected-seed accounting exceeded its budget");
  result.zrcar_legacy_seeds_selected =
      static_cast<std::uint32_t>(legacy_seed_count);
  result.zrcar_fvro_seeds_selected =
      static_cast<std::uint32_t>(fvro_seed_count);

  auto prefix_evaluated = std::size_t{0};
  auto local_evaluated = std::size_t{0};
  auto states_generated = std::size_t{0};
  auto infeasible_states = std::size_t{0};
  auto infeasible_frontier_expansions = std::size_t{0};
  auto infeasible_frontier_recoveries = std::size_t{0};
  auto retain_if_strictly_smaller = [&](const ZrcarState& candidate,
                                         const LegacyRepairSeed& seed) {
    if (candidate.correction_payload.size() >= result.correction_payload.size())
      return;
    result.corrections = candidate.corrections;
    result.correction_payload = candidate.correction_payload;
    result.reconstruction = candidate.reconstruction;
    result.report = candidate.report;
    result.correction_channel_scales = seed.channel_scales;
    result.zrcar_selected = true;
    result.zrcar_from_fvro_seed = seed.fvro_derived;
    if (uses_fvro(settings.repair_optimizer)) {
      // Provenance must describe the final winning repair, not an earlier
      // intermediate FVRO incumbent. A later ZRCAR win rooted in a frozen
      // legacy seed therefore clears FVRO-selected, while a tied FVRO seed
      // remains available to the search without replacing output bytes.
      // Final-stage attribution is mutually exclusive: this output was
      // selected by ZRCAR. The separate composed flag above records whether
      // its seed originated uniquely from FVRO, so saved bytes are never
      // counted in both fvro_saved_bytes and zrcar_saved_bytes.
      result.fvro_selected = false;
      LegacyRepairSeed selected_seed;
      selected_seed.channel_scales = seed.channel_scales;
      selected_seed.correction_payload = candidate.correction_payload;
      selected_seed.reconstruction = candidate.reconstruction;
      const auto summary = quantized_escape_summary(
          base, selected_seed, settings.channels, settings.time);
      result.fvro_final_quantized = summary.applicable;
      result.fvro_final_quantized_format = summary.format;
      if (summary.exact_escapes > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("FVRO/ZRCAR escape count exceeds uint32");
      result.fvro_final_exact_escapes =
          static_cast<std::uint32_t>(summary.exact_escapes);
    }
  };

  for (const auto seed_index : seed_order) {
    const auto& seed = legacy_seeds[seed_index];
    const auto generated_before_seed = states_generated;
    const auto frontier_expansions_before_seed = infeasible_frontier_expansions;
    ZrcarState root{
        {}, seed.corrections, seed.correction_payload, seed.reconstruction, seed.report};
    const auto root_ranked = rank_zrcar_rollbacks(
        original, base, root.reconstruction, root.blocked, settings);
    if (root_ranked.empty())
      continue;
    auto seen = std::vector<std::vector<std::size_t>>{{}};
    auto feasible_beam = std::vector<ZrcarState>{root};
    auto infeasible_beam = std::vector<ZrcarState>();
    auto seed_evaluated = std::size_t{0};

    // Evaluate every registered geometric prefix independently. Feasibility
    // is not assumed monotone, so one rejected prefix never suppresses a
    // larger, still-unmeasured prefix.
    for (const auto prefix : zrcar_geometric_prefixes(root_ranked.size())) {
      auto blocked = std::vector<std::size_t>();
      blocked.reserve(prefix);
      for (std::size_t index = 0; index < prefix; ++index)
        blocked.push_back(root_ranked[index].index);
      std::sort(blocked.begin(), blocked.end());
      if (std::find(seen.begin(), seen.end(), blocked) != seen.end())
        continue;
      seen.push_back(blocked);
      ++states_generated;
      auto candidate = evaluate_zrcar_state(
          original, base, seed, blocked, settings);
      if (!candidate.feasible) {
        ++infeasible_states;
        infeasible_beam.push_back(std::move(candidate));
        continue;
      }
      ++seed_evaluated;
      ++prefix_evaluated;
      retain_if_strictly_smaller(candidate, seed);
      feasible_beam.push_back(std::move(candidate));
    }
    if (seed_evaluated > kZrcarPrefixBudgetPerSeed)
      throw std::logic_error("ZRCAR geometric prefix budget was exceeded");
    retain_zrcar_beam(feasible_beam);
    retain_zrcar_infeasible_beam(infeasible_beam);

    for (std::size_t depth = 0;
         depth < kZrcarMaximumDepth &&
         seed_evaluated < kZrcarBudgetPerSeed &&
         (!feasible_beam.empty() || !infeasible_beam.empty());
         ++depth) {
      auto feasible_children = std::vector<ZrcarState>();
      auto infeasible_children = std::vector<ZrcarState>();
      auto expand_parent = [&](const ZrcarState& parent) {
        const auto parent_infeasible = !parent.feasible;
        if (parent_infeasible)
          ++infeasible_frontier_expansions;
        const auto ranked = rank_zrcar_rollbacks(
            original, base, parent.reconstruction, parent.blocked, settings);
        const auto branches = std::min(kZrcarBranching, ranked.size());
        for (std::size_t branch = 0;
             branch < branches && seed_evaluated < kZrcarBudgetPerSeed;
             ++branch) {
          auto blocked = zrcar_blocked_with(parent.blocked, ranked[branch].index);
          if (std::find(seen.begin(), seen.end(), blocked) != seen.end())
            continue;
          seen.push_back(blocked);
          ++states_generated;
          auto candidate = evaluate_zrcar_state(
              original, base, seed, blocked, settings);
          if (!candidate.feasible) {
            ++infeasible_states;
            infeasible_children.push_back(std::move(candidate));
            continue;
          }
          ++seed_evaluated;
          ++local_evaluated;
          if (parent_infeasible)
            ++infeasible_frontier_recoveries;
          retain_if_strictly_smaller(candidate, seed);
          feasible_children.push_back(std::move(candidate));
        }
      };
      // Expand retained infeasible states first so a feasible grandchild is
      // not starved by feasible parents consuming the serializer budget.
      for (const auto& parent : infeasible_beam) {
        expand_parent(parent);
        if (seed_evaluated == kZrcarBudgetPerSeed)
          break;
      }
      if (seed_evaluated < kZrcarBudgetPerSeed) {
        for (const auto& parent : feasible_beam) {
          expand_parent(parent);
          if (seed_evaluated == kZrcarBudgetPerSeed)
            break;
        }
      }
      retain_zrcar_beam(feasible_children);
      retain_zrcar_infeasible_beam(infeasible_children);
      feasible_beam = std::move(feasible_children);
      infeasible_beam = std::move(infeasible_children);
    }
    if (seed_evaluated > kZrcarBudgetPerSeed)
      throw std::logic_error("ZRCAR per-seed serializer budget was exceeded");
    if (states_generated - generated_before_seed >
        kZrcarMaximumGeneratedPerSeed)
      throw std::logic_error("ZRCAR per-seed generated-state budget was exceeded");
    if (infeasible_frontier_expansions - frontier_expansions_before_seed >
        kZrcarInfeasibleBeamWidth * kZrcarMaximumDepth)
      throw std::logic_error(
          "ZRCAR per-seed infeasible-frontier budget was exceeded");
  }
  const auto candidates = checked_add_size(
      prefix_evaluated, local_evaluated, "ZRCAR diagnostic count overflow");
  if (candidates > std::numeric_limits<std::uint32_t>::max() ||
      prefix_evaluated > std::numeric_limits<std::uint32_t>::max() ||
      local_evaluated > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("ZRCAR diagnostic count exceeds uint32");
  result.zrcar_candidates_tested = static_cast<std::uint32_t>(candidates);
  result.zrcar_prefix_candidates_tested =
      static_cast<std::uint32_t>(prefix_evaluated);
  result.zrcar_local_candidates_tested =
      static_cast<std::uint32_t>(local_evaluated);
  if (states_generated > std::numeric_limits<std::uint32_t>::max() ||
      infeasible_states > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("ZRCAR state diagnostic count exceeds uint32");
  result.zrcar_states_generated = static_cast<std::uint32_t>(states_generated);
  result.zrcar_infeasible_states =
      static_cast<std::uint32_t>(infeasible_states);
  if (infeasible_frontier_expansions >
          std::numeric_limits<std::uint32_t>::max() ||
      infeasible_frontier_recoveries >
          std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("ZRCAR frontier diagnostic count exceeds uint32");
  result.zrcar_infeasible_frontier_expansions =
      static_cast<std::uint32_t>(infeasible_frontier_expansions);
  result.zrcar_infeasible_frontier_recoveries =
      static_cast<std::uint32_t>(infeasible_frontier_recoveries);
  result.repair_candidates_tested = checked_add_u32(
      result.repair_candidates_tested,
      result.zrcar_candidates_tested,
      "QPET legacy plus ZRCAR repair candidate count overflow");
}

struct SsbbChannelRank {
  std::size_t channel = 0;
  std::size_t exact_escapes = 0;
  std::size_t corrections = 0;
};

struct SsbbChannelOption {
  std::vector<float> reconstruction;
};

struct SsbbChannelBank {
  std::size_t channel = 0;
  std::vector<SsbbChannelOption> options;
};

struct SsbbCompositionState {
  std::vector<std::uint8_t> choices;
  std::uint8_t changed_channels = 0;
  LegacyRepairSeed seed;
};

bool ssbb_composition_state_less(
    const SsbbCompositionState& left,
    const SsbbCompositionState& right)
{
  if (left.seed.correction_payload.size() !=
      right.seed.correction_payload.size())
    return left.seed.correction_payload.size() <
        right.seed.correction_payload.size();
  return std::lexicographical_compare(
      left.choices.begin(),
      left.choices.end(),
      right.choices.begin(),
      right.choices.end());
}

std::optional<std::size_t> ssbb_correction_format(
    const LegacyRepairSeed& seed)
{
  if (seed.correction_payload.empty())
    return std::nullopt;
  const auto format = static_cast<std::size_t>(seed.correction_payload.front());
  if (format >= kSsbbCompositionFormatSlots)
    throw std::logic_error("SSBB composition observed an unknown correction format");
  return format;
}

void run_ssbb_value_search_single_ceiling(
    const std::vector<float>& original,
    const std::vector<float>& base,
    const Settings& settings,
    std::vector<LegacyRepairSeed>& seeds,
    EncodedWindow& result)
{
  if (!uses_ssbb(settings.repair_optimizer))
    return;
  if (!result.report.all_ok() || result.reconstruction.size() != original.size() ||
      base.size() != original.size())
    throw std::invalid_argument("SSBB requires a complete feasible v1 incumbent");

  result.ssbb_base_records_tested = 1;
  result.ssbb_reference_bytes = result.encoded_size();
  auto scales = result.correction_channel_scales;
  if (scales.size() != settings.channels)
    scales = make_fvro_channel_scales(
        original, base, settings.channels, settings.time, 127);
  if (scales.size() != settings.channels)
    throw std::logic_error("SSBB failed to construct a complete scale vector");

  LegacyRepairSeed current;
  current.ordinal = next_fvro_seed_ordinal(seeds);
  current.targets = result.reconstruction;
  current.channel_scales = scales;
  current.corrections = result.corrections;
  current.correction_payload = result.correction_payload;
  current.reconstruction = result.reconstruction;
  current.report = result.report;
  const auto output_reference = current;
  auto normalized = serialize_fvro_reconstruction(
      original,
      base,
      current.reconstruction,
      current.channel_scales,
      current.reconstruction,
      settings,
      current.ordinal,
      true);
  normalized.fvro_derived = false;
  normalized.ssbb_derived = true;
  result.ssbb_representation_candidates_serialized = 1;
  const auto normalized_size = checked_add_size(
      result.payload.size(),
      normalized.correction_payload.size(),
      "SSBB normalized-reference size overflow");
  if (normalized_size < result.ssbb_reference_bytes) {
    result.ssbb_representation_improved_starts = 1;
    result.ssbb_representation_gross_saved_bytes =
        result.ssbb_reference_bytes - normalized_size;
    current = std::move(normalized);
  }
  result.ssbb_search_reference_bytes = checked_add_size(
      result.payload.size(),
      current.correction_payload.size(),
      "SSBB search-reference size overflow");
  const auto reference = current;

  const auto escape_counts = quantized_escape_counts_by_channel(
      base,
      current.reconstruction,
      current.channel_scales,
      settings.channels,
      settings.time);
  auto ranked = std::vector<SsbbChannelRank>();
  ranked.reserve(settings.channels);
  for (std::size_t channel = 0; channel < settings.channels; ++channel) {
    const auto begin = channel * settings.time;
    auto corrections = std::size_t{0};
    for (std::size_t index = 0; index < settings.time; ++index)
      corrections += base[begin + index] != current.reconstruction[begin + index];
    if (corrections != 0)
      ranked.push_back({channel, escape_counts[channel], corrections});
  }
  std::stable_sort(
      ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
        if (left.exact_escapes != right.exact_escapes)
          return left.exact_escapes > right.exact_escapes;
        if (left.corrections != right.corrections)
          return left.corrections > right.corrections;
        return left.channel < right.channel;
      });
  if (ranked.size() > kSsbbMaximumDifficultChannels)
    ranked.resize(kSsbbMaximumDifficultChannels);

  auto positions_processed = std::size_t{0};
  auto actions_generated = std::size_t{0};
  auto states_generated = std::size_t{0};
  auto states_deduplicated = std::size_t{0};
  auto statistics_feasible = std::size_t{0};
  auto statistics_infeasible = std::size_t{0};
  auto feasible_retained = std::size_t{0};
  auto infeasible_retained = std::size_t{0};
  auto states_pruned = std::size_t{0};
  auto terminal_states = std::size_t{0};
  auto terminal_feasible = std::size_t{0};
  auto terminal_infeasible = std::size_t{0};
  auto outputs_returned = std::size_t{0};
  auto serialized = std::size_t{0};
  auto selected_channels = std::size_t{0};
  auto channel_banks = std::vector<SsbbChannelBank>();
  channel_banks.reserve(ranked.size());
  auto next_ordinal = current.ordinal;
  if (next_ordinal == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("SSBB seed ordinal overflow");
  ++next_ordinal;

  const qpet_correlation::ssbb::Constraints constraints{
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag};
  auto accumulate = [](std::size_t& destination,
                       std::size_t value,
                       const char* message) {
    destination = checked_add_size(destination, value, message);
  };

  for (const auto& entry : ranked) {
    const auto begin = entry.channel * settings.time;
    const auto local_original = std::vector<float>(
        original.begin() + static_cast<std::ptrdiff_t>(begin),
        original.begin() + static_cast<std::ptrdiff_t>(begin + settings.time));
    const auto local_base = std::vector<float>(
        base.begin() + static_cast<std::ptrdiff_t>(begin),
        base.begin() + static_cast<std::ptrdiff_t>(begin + settings.time));
    const auto local_start = std::vector<float>(
        reference.reconstruction.begin() + static_cast<std::ptrdiff_t>(begin),
        reference.reconstruction.begin() +
            static_cast<std::ptrdiff_t>(begin + settings.time));
    const auto searched = qpet_correlation::ssbb::search(
        local_original,
        local_base,
        local_start,
        current.channel_scales[entry.channel],
        constraints,
        qpet_correlation::ssbb::SearchBudget{
            qpet_correlation::ssbb::kProductionFeasibleBeamWidth,
            qpet_correlation::ssbb::kProductionInfeasibleBeamWidth,
            kSsbbMaximumSerializedPerChannel});
    const auto& diagnostics = searched.diagnostics;
    accumulate(positions_processed, diagnostics.positions_processed,
               "SSBB position total overflow");
    accumulate(actions_generated, diagnostics.actions_generated,
               "SSBB action total overflow");
    accumulate(states_generated, diagnostics.states_generated,
               "SSBB generated-state total overflow");
    accumulate(states_deduplicated, diagnostics.states_deduplicated,
               "SSBB deduplicated-state total overflow");
    accumulate(statistics_feasible, diagnostics.statistics_feasible_states,
               "SSBB statistics-feasible total overflow");
    accumulate(statistics_infeasible, diagnostics.statistics_infeasible_states,
               "SSBB statistics-infeasible total overflow");
    accumulate(feasible_retained, diagnostics.feasible_states_retained,
               "SSBB feasible-retained total overflow");
    accumulate(infeasible_retained, diagnostics.infeasible_states_retained,
               "SSBB infeasible-retained total overflow");
    accumulate(states_pruned, diagnostics.states_pruned_by_beam,
               "SSBB beam-pruned total overflow");
    accumulate(terminal_states, diagnostics.terminal_states,
               "SSBB terminal-state total overflow");
    accumulate(terminal_feasible, diagnostics.terminal_direct_feasible,
               "SSBB terminal-feasible total overflow");
    accumulate(terminal_infeasible, diagnostics.terminal_direct_infeasible,
               "SSBB terminal-infeasible total overflow");
    accumulate(outputs_returned, diagnostics.outputs_returned,
               "SSBB output total overflow");

    SsbbChannelBank bank;
    bank.channel = entry.channel;
    auto best_channel = std::optional<LegacyRepairSeed>();
    for (const auto& terminal : searched.candidates) {
      const auto duplicate_option = std::any_of(
          bank.options.begin(), bank.options.end(), [&](const auto& option) {
            return same_float32_tensor_bits(
                option.reconstruction, terminal.reconstruction);
          });
      if (!duplicate_option &&
          !same_float32_tensor_bits(local_start, terminal.reconstruction))
        bank.options.push_back({terminal.reconstruction});
      if (next_ordinal == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("SSBB seed ordinal overflow");
      auto reconstruction = current.reconstruction;
      copy_channel(
          reconstruction,
          entry.channel,
          settings.time,
          terminal.reconstruction);
      auto targets = reconstruction;
      auto candidate = serialize_fvro_reconstruction(
          original,
          base,
          targets,
          current.channel_scales,
          reconstruction,
          settings,
          next_ordinal,
          true);
      ++next_ordinal;
      candidate.fvro_derived = false;
      candidate.ssbb_derived = true;
      ++serialized;
      if (candidate.correction_payload.size() >=
          current.correction_payload.size())
        continue;
      if (!best_channel.has_value() ||
          candidate.correction_payload.size() <
              best_channel->correction_payload.size())
        best_channel = std::move(candidate);
    }
    channel_banks.push_back(std::move(bank));
    if (!best_channel.has_value())
      continue;
    current = std::move(*best_channel);
    ++selected_channels;
  }

  // The local search above deliberately preserves the historical greedy
  // incumbent.  The composition stage starts again from the immutable
  // pre-SSBB reference and keeps every singleton as an anchor.  Consequently
  // every pair of local channel options is sent through the real nested
  // correction portfolio (0--6, 0--7, 0--8, or 0--9 according to the frozen
  // ablation ceiling), while higher-order combinations use a bounded,
  // deterministic format/rate-stratified beam.
  auto composition_local_options = std::size_t{0};
  for (const auto& bank : channel_banks)
    composition_local_options = checked_add_size(
        composition_local_options,
        bank.options.size(),
        "SSBB composition local-option total overflow");
  auto composition_transitions_generated = std::size_t{0};
  auto composition_transitions_deduplicated = std::size_t{0};
  auto composition_direct_feasible = std::size_t{0};
  auto composition_direct_infeasible = std::size_t{0};
  auto composition_candidates_serialized = std::size_t{0};
  auto composition_singletons_serialized = std::size_t{0};
  auto composition_pairs_serialized = std::size_t{0};
  auto composition_frontier_carries = std::size_t{0};
  auto composition_anchor_retention_events = std::size_t{0};
  auto composition_format_retention_events = std::size_t{0};
  auto composition_rate_retention_events = std::size_t{0};
  auto composition_states_pruned = std::size_t{0};
  auto composition_max_frontier = std::size_t{1};
  auto composition_selected_channels = std::size_t{0};
  const auto composition_reference_bytes = checked_add_size(
      result.payload.size(),
      current.correction_payload.size(),
      "SSBB composition reference size overflow");
  auto composition_winner = current;
  auto composition_selected = false;

  SsbbCompositionState root;
  root.choices.assign(channel_banks.size(), std::uint8_t{0});
  root.seed = reference;
  auto frontier = std::vector<SsbbCompositionState>{root};
  for (std::size_t depth = 0; depth < channel_banks.size(); ++depth) {
    composition_frontier_carries = checked_add_size(
        composition_frontier_carries,
        frontier.size(),
        "SSBB composition carry total overflow");
    auto pool = frontier;
    for (const auto& parent : frontier) {
      for (std::size_t option_index = 0;
           option_index < channel_banks[depth].options.size();
           ++option_index) {
        composition_transitions_generated = checked_add_size(
            composition_transitions_generated,
            1,
            "SSBB composition transition total overflow");
        auto choices = parent.choices;
        if (option_index + 1 > std::numeric_limits<std::uint8_t>::max())
          throw std::logic_error("SSBB composition option index exceeds uint8");
        choices[depth] = static_cast<std::uint8_t>(option_index + 1);
        const auto duplicate = std::any_of(
            pool.begin(), pool.end(), [&](const auto& retained) {
              return retained.choices == choices;
            });
        if (duplicate) {
          composition_transitions_deduplicated = checked_add_size(
              composition_transitions_deduplicated,
              1,
              "SSBB composition duplicate total overflow");
          continue;
        }

        auto reconstruction = parent.seed.reconstruction;
        copy_channel(
            reconstruction,
            channel_banks[depth].channel,
            settings.time,
            channel_banks[depth].options[option_index].reconstruction);
        if (next_ordinal == std::numeric_limits<std::size_t>::max())
          throw std::overflow_error("SSBB composition seed ordinal overflow");
        auto candidate = serialize_fvro_reconstruction(
            original,
            base,
            reconstruction,
            reference.channel_scales,
            reconstruction,
            settings,
            next_ordinal,
            true);
        ++next_ordinal;
        candidate.fvro_derived = false;
        candidate.ssbb_derived = true;
        composition_direct_feasible = checked_add_size(
            composition_direct_feasible,
            1,
            "SSBB composition feasible total overflow");
        composition_candidates_serialized = checked_add_size(
            composition_candidates_serialized,
            1,
            "SSBB composition serializer total overflow");
        const auto changed_channels = static_cast<std::size_t>(
            parent.changed_channels) + 1;
        if (changed_channels > std::numeric_limits<std::uint8_t>::max())
          throw std::logic_error("SSBB composition channel count exceeds uint8");
        if (changed_channels == 1)
          composition_singletons_serialized = checked_add_size(
              composition_singletons_serialized,
              1,
              "SSBB composition singleton total overflow");
        else if (changed_channels == 2)
          composition_pairs_serialized = checked_add_size(
              composition_pairs_serialized,
              1,
              "SSBB composition pair total overflow");
        if (candidate.correction_payload.size() <
            composition_winner.correction_payload.size()) {
          composition_winner = candidate;
          composition_selected = true;
          composition_selected_channels = changed_channels;
        }
        SsbbCompositionState child;
        child.choices = std::move(choices);
        child.changed_channels = static_cast<std::uint8_t>(changed_channels);
        child.seed = std::move(candidate);
        pool.push_back(std::move(child));
      }
    }

    auto selected = std::vector<bool>(pool.size(), false);
    auto next_frontier = std::vector<SsbbCompositionState>();
    next_frontier.reserve(kSsbbCompositionBeamWidth);
    auto retain = [&](std::size_t index, std::size_t& diagnostic) {
      if (selected[index])
        return;
      selected[index] = true;
      next_frontier.push_back(pool[index]);
      diagnostic = checked_add_size(
          diagnostic, 1, "SSBB composition retention total overflow");
    };

    auto anchor_indices = std::vector<std::size_t>();
    for (std::size_t index = 0; index < pool.size(); ++index) {
      if (pool[index].changed_channels <= 1)
        anchor_indices.push_back(index);
    }
    std::stable_sort(
        anchor_indices.begin(), anchor_indices.end(), [&](auto left, auto right) {
          return std::lexicographical_compare(
              pool[left].choices.begin(),
              pool[left].choices.end(),
              pool[right].choices.begin(),
              pool[right].choices.end());
        });
    for (const auto index : anchor_indices)
      retain(index, composition_anchor_retention_events);
    if (next_frontier.size() >
        1 + kSsbbMaximumDifficultChannels *
            kSsbbMaximumSerializedPerChannel)
      throw std::logic_error("SSBB composition anchor budget was exceeded");

    for (std::size_t format = 0;
         format < kSsbbCompositionFormatSlots;
         ++format) {
      auto best_index = std::optional<std::size_t>();
      for (std::size_t index = 0; index < pool.size(); ++index) {
        if (selected[index])
          continue;
        const auto candidate_format = ssbb_correction_format(pool[index].seed);
        if (!candidate_format.has_value() || *candidate_format != format)
          continue;
        if (!best_index.has_value() ||
            ssbb_composition_state_less(pool[index], pool[*best_index]))
          best_index = index;
      }
      if (best_index.has_value())
        retain(*best_index, composition_format_retention_events);
    }

    auto rate_indices = std::vector<std::size_t>();
    for (std::size_t index = 0; index < pool.size(); ++index) {
      if (!selected[index])
        rate_indices.push_back(index);
    }
    std::stable_sort(
        rate_indices.begin(), rate_indices.end(), [&](auto left, auto right) {
          return ssbb_composition_state_less(pool[left], pool[right]);
        });
    const auto rate_count = std::min(
        rate_indices.size(), kSsbbCompositionRateSlots);
    for (std::size_t index = 0; index < rate_count; ++index)
      retain(rate_indices[index], composition_rate_retention_events);
    if (next_frontier.size() > kSsbbCompositionBeamWidth)
      throw std::logic_error("SSBB composition frontier exceeded its beam");
    composition_states_pruned = checked_add_size(
        composition_states_pruned,
        pool.size() - next_frontier.size(),
        "SSBB composition pruned-state total overflow");
    composition_max_frontier = std::max(
        composition_max_frontier, next_frontier.size());
    frontier = std::move(next_frontier);
  }

  auto expected_singletons = std::size_t{0};
  auto expected_pairs = std::size_t{0};
  for (std::size_t left = 0; left < channel_banks.size(); ++left) {
    expected_singletons = checked_add_size(
        expected_singletons,
        channel_banks[left].options.size(),
        "SSBB expected-singleton total overflow");
    for (std::size_t right = left + 1; right < channel_banks.size(); ++right) {
      const auto product = channel_banks[left].options.size() *
          channel_banks[right].options.size();
      expected_pairs = checked_add_size(
          expected_pairs, product, "SSBB expected-pair total overflow");
    }
  }
  if (composition_singletons_serialized != expected_singletons ||
      composition_pairs_serialized != expected_pairs)
    throw std::logic_error("SSBB composition did not exhaust every singleton/pair");
  if (composition_transitions_generated != checked_add_size(
          composition_transitions_deduplicated,
          checked_add_size(
              composition_direct_feasible,
              composition_direct_infeasible,
              "SSBB composition direct partition overflow"),
          "SSBB composition transition partition overflow") ||
      composition_direct_feasible != composition_candidates_serialized ||
      checked_add_size(
          composition_frontier_carries,
          composition_direct_feasible,
          "SSBB composition frontier partition overflow") !=
          checked_add_size(
              checked_add_size(
                  composition_anchor_retention_events,
                  composition_format_retention_events,
                  "SSBB composition retention partition overflow"),
              checked_add_size(
                  composition_rate_retention_events,
                  composition_states_pruned,
                  "SSBB composition prune partition overflow"),
              "SSBB composition complete partition overflow"))
    throw std::logic_error("SSBB composition diagnostics do not partition exactly");
  if (composition_transitions_generated >
          kSsbbCompositionMaximumTransitionsPerStart ||
      composition_max_frontier > kSsbbCompositionBeamWidth)
    throw std::logic_error("SSBB composition exceeded its fixed search budget");
  if (composition_winner.correction_payload.size() >
      current.correction_payload.size())
    throw std::logic_error("SSBB composition regressed the greedy incumbent");
  if (composition_selected) {
    if (checked_add_size(
            result.payload.size(),
            composition_winner.correction_payload.size(),
            "SSBB composition winner size overflow") >=
        composition_reference_bytes)
      throw std::logic_error("SSBB composition provenance has no strict byte win");
    current = std::move(composition_winner);
  }

  if (serialized > kSsbbMaximumSerializedPerBase ||
      ranked.size() > kSsbbMaximumDifficultChannels ||
      outputs_returned >
          ranked.size() * kSsbbMaximumSerializedPerChannel)
    throw std::logic_error("SSBB exceeded a fixed v2 search budget");
  if (states_generated != checked_add_size(
          states_deduplicated,
          checked_add_size(
              statistics_feasible,
              statistics_infeasible,
              "SSBB statistics partition overflow"),
          "SSBB generated-state partition overflow") ||
      checked_add_size(
          statistics_feasible,
          statistics_infeasible,
          "SSBB statistics partition overflow") !=
          checked_add_size(
              checked_add_size(
                  feasible_retained,
                  infeasible_retained,
                  "SSBB retained partition overflow"),
              states_pruned,
              "SSBB beam partition overflow") ||
      terminal_states != checked_add_size(
          terminal_feasible,
          terminal_infeasible,
          "SSBB terminal partition overflow"))
    throw std::logic_error("SSBB aggregate diagnostics do not partition exactly");

  const auto assign_u32 = [](std::size_t value, const char* message) {
    if (value > std::numeric_limits<std::uint32_t>::max())
      throw std::overflow_error(message);
    return static_cast<std::uint32_t>(value);
  };
  result.ssbb_channels_searched = assign_u32(
      ranked.size(), "SSBB channel total exceeds uint32");
  result.ssbb_positions_processed = assign_u32(
      positions_processed, "SSBB position total exceeds uint32");
  result.ssbb_actions_generated = assign_u32(
      actions_generated, "SSBB action total exceeds uint32");
  result.ssbb_states_generated = assign_u32(
      states_generated, "SSBB generated-state total exceeds uint32");
  result.ssbb_states_deduplicated = assign_u32(
      states_deduplicated, "SSBB deduplicated-state total exceeds uint32");
  result.ssbb_statistics_feasible_states = assign_u32(
      statistics_feasible, "SSBB statistics-feasible total exceeds uint32");
  result.ssbb_statistics_infeasible_states = assign_u32(
      statistics_infeasible, "SSBB statistics-infeasible total exceeds uint32");
  result.ssbb_feasible_states_retained = assign_u32(
      feasible_retained, "SSBB feasible-retained total exceeds uint32");
  result.ssbb_infeasible_states_retained = assign_u32(
      infeasible_retained, "SSBB infeasible-retained total exceeds uint32");
  result.ssbb_states_pruned_by_beam = assign_u32(
      states_pruned, "SSBB beam-pruned total exceeds uint32");
  result.ssbb_terminal_states = assign_u32(
      terminal_states, "SSBB terminal-state total exceeds uint32");
  result.ssbb_terminal_direct_feasible = assign_u32(
      terminal_feasible, "SSBB terminal-feasible total exceeds uint32");
  result.ssbb_terminal_direct_infeasible = assign_u32(
      terminal_infeasible, "SSBB terminal-infeasible total exceeds uint32");
  result.ssbb_outputs_returned = assign_u32(
      outputs_returned, "SSBB output total exceeds uint32");
  result.ssbb_candidates_serialized = assign_u32(
      serialized, "SSBB serializer total exceeds uint32");
  result.ssbb_selected_channels = assign_u32(
      selected_channels, "SSBB selected-channel total exceeds uint32");
  result.ssbb_composition_starts = 1;
  result.ssbb_composition_local_options = assign_u32(
      composition_local_options,
      "SSBB composition local-option total exceeds uint32");
  result.ssbb_composition_transitions_generated = assign_u32(
      composition_transitions_generated,
      "SSBB composition transition total exceeds uint32");
  result.ssbb_composition_transitions_deduplicated = assign_u32(
      composition_transitions_deduplicated,
      "SSBB composition duplicate total exceeds uint32");
  result.ssbb_composition_direct_feasible = assign_u32(
      composition_direct_feasible,
      "SSBB composition feasible total exceeds uint32");
  result.ssbb_composition_direct_infeasible = assign_u32(
      composition_direct_infeasible,
      "SSBB composition infeasible total exceeds uint32");
  result.ssbb_composition_candidates_serialized = assign_u32(
      composition_candidates_serialized,
      "SSBB composition serializer total exceeds uint32");
  result.ssbb_composition_singletons_serialized = assign_u32(
      composition_singletons_serialized,
      "SSBB composition singleton total exceeds uint32");
  result.ssbb_composition_pairs_serialized = assign_u32(
      composition_pairs_serialized,
      "SSBB composition pair total exceeds uint32");
  result.ssbb_composition_frontier_carries = assign_u32(
      composition_frontier_carries,
      "SSBB composition carry total exceeds uint32");
  result.ssbb_composition_anchor_retention_events = assign_u32(
      composition_anchor_retention_events,
      "SSBB composition anchor total exceeds uint32");
  result.ssbb_composition_format_retention_events = assign_u32(
      composition_format_retention_events,
      "SSBB composition format total exceeds uint32");
  result.ssbb_composition_rate_retention_events = assign_u32(
      composition_rate_retention_events,
      "SSBB composition rate total exceeds uint32");
  result.ssbb_composition_states_pruned = assign_u32(
      composition_states_pruned,
      "SSBB composition pruned total exceeds uint32");
  result.ssbb_composition_max_frontier = assign_u32(
      composition_max_frontier,
      "SSBB composition frontier exceeds uint32");
  result.ssbb_composition_selected_channels = assign_u32(
      composition_selected_channels,
      "SSBB composition selected-channel count exceeds uint32");
  result.ssbb_composition_selected = composition_selected;
  result.ssbb_composition_reference_bytes = composition_reference_bytes;
  if (current.correction_payload.size() <
      output_reference.correction_payload.size()) {
    result.corrections = current.corrections;
    result.correction_payload = current.correction_payload;
    result.reconstruction = current.reconstruction;
    result.report = current.report;
    result.correction_channel_scales = current.channel_scales;
    result.fvro_selected = false;
    result.zrcar_selected = false;
    result.zrcar_from_fvro_seed = false;
    result.ssbb_selected = true;
    const auto duplicate = std::find_if(
        seeds.begin(), seeds.end(), [&](const LegacyRepairSeed& seed) {
          return same_float32_tensor_bits(seed.targets, current.targets) &&
              same_float32_tensor_bits(
                  seed.channel_scales, current.channel_scales);
        });
    if (duplicate == seeds.end()) {
      seeds.push_back(current);
      result.ssbb_seeds_retained = 1;
    }
    const auto summary = quantized_escape_summary(
        base, current, settings.channels, settings.time);
    result.fvro_final_quantized = summary.applicable;
    result.fvro_final_quantized_format = summary.format;
    if (summary.exact_escapes > std::numeric_limits<std::uint32_t>::max())
      throw std::overflow_error("SSBB final escape count exceeds uint32");
    result.fvro_final_exact_escapes =
        static_cast<std::uint32_t>(summary.exact_escapes);
  }
  result.repair_candidates_tested = checked_add_u32(
      result.repair_candidates_tested,
      result.ssbb_representation_candidates_serialized,
      "QPET plus SSBB representation candidate count overflow");
  result.repair_candidates_tested = checked_add_u32(
      result.repair_candidates_tested,
      result.ssbb_candidates_serialized,
      "QPET v1 plus SSBB repair candidate count overflow");
  result.repair_candidates_tested = checked_add_u32(
      result.repair_candidates_tested,
      result.ssbb_composition_candidates_serialized,
      "QPET plus SSBB composition candidate count overflow");
  if (result.encoded_size() > result.ssbb_reference_bytes)
    throw std::logic_error("SSBB regressed its complete v1 incumbent");
  if (result.ssbb_search_reference_bytes > result.ssbb_reference_bytes ||
      result.encoded_size() > result.ssbb_search_reference_bytes ||
      (result.ssbb_representation_improved_starts == 0) !=
          (result.ssbb_representation_gross_saved_bytes == 0) ||
      result.ssbb_representation_gross_saved_bytes !=
          result.ssbb_reference_bytes - result.ssbb_search_reference_bytes)
    throw std::logic_error(
        "SSBB representation/search byte decomposition diverged");
}

void accumulate_ssbb_path_work(
    EncodedWindow& total,
    const EncodedWindow& path)
{
  auto add = [](std::uint32_t& destination,
                std::uint32_t value,
                const char* message) {
    destination = checked_add_u32(destination, value, message);
  };
  add(total.ssbb_channels_searched, path.ssbb_channels_searched,
      "SSBB ceiling-path channel total overflow");
  add(total.ssbb_positions_processed, path.ssbb_positions_processed,
      "SSBB ceiling-path position total overflow");
  add(total.ssbb_actions_generated, path.ssbb_actions_generated,
      "SSBB ceiling-path action total overflow");
  add(total.ssbb_states_generated, path.ssbb_states_generated,
      "SSBB ceiling-path state total overflow");
  add(total.ssbb_states_deduplicated, path.ssbb_states_deduplicated,
      "SSBB ceiling-path duplicate total overflow");
  add(total.ssbb_statistics_feasible_states,
      path.ssbb_statistics_feasible_states,
      "SSBB ceiling-path statistics-feasible total overflow");
  add(total.ssbb_statistics_infeasible_states,
      path.ssbb_statistics_infeasible_states,
      "SSBB ceiling-path statistics-infeasible total overflow");
  add(total.ssbb_feasible_states_retained,
      path.ssbb_feasible_states_retained,
      "SSBB ceiling-path feasible-retained total overflow");
  add(total.ssbb_infeasible_states_retained,
      path.ssbb_infeasible_states_retained,
      "SSBB ceiling-path infeasible-retained total overflow");
  add(total.ssbb_states_pruned_by_beam, path.ssbb_states_pruned_by_beam,
      "SSBB ceiling-path pruned total overflow");
  add(total.ssbb_terminal_states, path.ssbb_terminal_states,
      "SSBB ceiling-path terminal total overflow");
  add(total.ssbb_terminal_direct_feasible,
      path.ssbb_terminal_direct_feasible,
      "SSBB ceiling-path terminal-feasible total overflow");
  add(total.ssbb_terminal_direct_infeasible,
      path.ssbb_terminal_direct_infeasible,
      "SSBB ceiling-path terminal-infeasible total overflow");
  add(total.ssbb_outputs_returned, path.ssbb_outputs_returned,
      "SSBB ceiling-path output total overflow");
  add(total.ssbb_candidates_serialized, path.ssbb_candidates_serialized,
      "SSBB ceiling-path serializer total overflow");
  add(total.ssbb_seeds_retained, path.ssbb_seeds_retained,
      "SSBB ceiling-path retained-seed total overflow");
  add(total.ssbb_selected_channels, path.ssbb_selected_channels,
      "SSBB ceiling-path selected-channel total overflow");
  add(total.ssbb_representation_candidates_serialized,
      path.ssbb_representation_candidates_serialized,
      "SSBB ceiling-path representation-candidate total overflow");
  add(total.ssbb_representation_improved_starts,
      path.ssbb_representation_improved_starts,
      "SSBB ceiling-path representation-win total overflow");
  total.ssbb_representation_gross_saved_bytes = checked_add_size(
      total.ssbb_representation_gross_saved_bytes,
      path.ssbb_representation_gross_saved_bytes,
      "SSBB ceiling-path representation-byte total overflow");
  add(total.ssbb_composition_starts, path.ssbb_composition_starts,
      "SSBB ceiling-path composition-start total overflow");
  add(total.ssbb_composition_local_options,
      path.ssbb_composition_local_options,
      "SSBB ceiling-path composition-option total overflow");
  add(total.ssbb_composition_transitions_generated,
      path.ssbb_composition_transitions_generated,
      "SSBB ceiling-path composition-transition total overflow");
  add(total.ssbb_composition_transitions_deduplicated,
      path.ssbb_composition_transitions_deduplicated,
      "SSBB ceiling-path composition-duplicate total overflow");
  add(total.ssbb_composition_direct_feasible,
      path.ssbb_composition_direct_feasible,
      "SSBB ceiling-path composition-feasible total overflow");
  add(total.ssbb_composition_direct_infeasible,
      path.ssbb_composition_direct_infeasible,
      "SSBB ceiling-path composition-infeasible total overflow");
  add(total.ssbb_composition_candidates_serialized,
      path.ssbb_composition_candidates_serialized,
      "SSBB ceiling-path composition-serializer total overflow");
  add(total.ssbb_composition_singletons_serialized,
      path.ssbb_composition_singletons_serialized,
      "SSBB ceiling-path composition-singleton total overflow");
  add(total.ssbb_composition_pairs_serialized,
      path.ssbb_composition_pairs_serialized,
      "SSBB ceiling-path composition-pair total overflow");
  add(total.ssbb_composition_frontier_carries,
      path.ssbb_composition_frontier_carries,
      "SSBB ceiling-path composition-carry total overflow");
  add(total.ssbb_composition_anchor_retention_events,
      path.ssbb_composition_anchor_retention_events,
      "SSBB ceiling-path composition-anchor total overflow");
  add(total.ssbb_composition_format_retention_events,
      path.ssbb_composition_format_retention_events,
      "SSBB ceiling-path composition-format total overflow");
  add(total.ssbb_composition_rate_retention_events,
      path.ssbb_composition_rate_retention_events,
      "SSBB ceiling-path composition-rate total overflow");
  add(total.ssbb_composition_states_pruned,
      path.ssbb_composition_states_pruned,
      "SSBB ceiling-path composition-pruned total overflow");
  total.ssbb_composition_max_frontier = std::max(
      total.ssbb_composition_max_frontier,
      path.ssbb_composition_max_frontier);
}

void overwrite_ssbb_path_work(
    EncodedWindow& destination,
    const EncodedWindow& source)
{
  destination.ssbb_channels_searched = source.ssbb_channels_searched;
  destination.ssbb_positions_processed = source.ssbb_positions_processed;
  destination.ssbb_actions_generated = source.ssbb_actions_generated;
  destination.ssbb_states_generated = source.ssbb_states_generated;
  destination.ssbb_states_deduplicated = source.ssbb_states_deduplicated;
  destination.ssbb_statistics_feasible_states =
      source.ssbb_statistics_feasible_states;
  destination.ssbb_statistics_infeasible_states =
      source.ssbb_statistics_infeasible_states;
  destination.ssbb_feasible_states_retained =
      source.ssbb_feasible_states_retained;
  destination.ssbb_infeasible_states_retained =
      source.ssbb_infeasible_states_retained;
  destination.ssbb_states_pruned_by_beam =
      source.ssbb_states_pruned_by_beam;
  destination.ssbb_terminal_states = source.ssbb_terminal_states;
  destination.ssbb_terminal_direct_feasible =
      source.ssbb_terminal_direct_feasible;
  destination.ssbb_terminal_direct_infeasible =
      source.ssbb_terminal_direct_infeasible;
  destination.ssbb_outputs_returned = source.ssbb_outputs_returned;
  destination.ssbb_candidates_serialized =
      source.ssbb_candidates_serialized;
  destination.ssbb_seeds_retained = source.ssbb_seeds_retained;
  destination.ssbb_selected_channels = source.ssbb_selected_channels;
  destination.ssbb_representation_candidates_serialized =
      source.ssbb_representation_candidates_serialized;
  destination.ssbb_representation_improved_starts =
      source.ssbb_representation_improved_starts;
  destination.ssbb_representation_gross_saved_bytes =
      source.ssbb_representation_gross_saved_bytes;
  destination.ssbb_composition_starts = source.ssbb_composition_starts;
  destination.ssbb_composition_local_options =
      source.ssbb_composition_local_options;
  destination.ssbb_composition_transitions_generated =
      source.ssbb_composition_transitions_generated;
  destination.ssbb_composition_transitions_deduplicated =
      source.ssbb_composition_transitions_deduplicated;
  destination.ssbb_composition_direct_feasible =
      source.ssbb_composition_direct_feasible;
  destination.ssbb_composition_direct_infeasible =
      source.ssbb_composition_direct_infeasible;
  destination.ssbb_composition_candidates_serialized =
      source.ssbb_composition_candidates_serialized;
  destination.ssbb_composition_singletons_serialized =
      source.ssbb_composition_singletons_serialized;
  destination.ssbb_composition_pairs_serialized =
      source.ssbb_composition_pairs_serialized;
  destination.ssbb_composition_frontier_carries =
      source.ssbb_composition_frontier_carries;
  destination.ssbb_composition_anchor_retention_events =
      source.ssbb_composition_anchor_retention_events;
  destination.ssbb_composition_format_retention_events =
      source.ssbb_composition_format_retention_events;
  destination.ssbb_composition_rate_retention_events =
      source.ssbb_composition_rate_retention_events;
  destination.ssbb_composition_states_pruned =
      source.ssbb_composition_states_pruned;
  destination.ssbb_composition_max_frontier =
      source.ssbb_composition_max_frontier;
}

void realize_ssbb_winner_against_incumbent(
    EncodedWindow& candidate,
    std::size_t incumbent_bytes,
    bool force_cross_start_rebase)
{
  if (!candidate.ssbb_selected)
    return;
  const auto final_bytes = candidate.encoded_size();
  if (final_bytes >= incumbent_bytes)
    throw std::logic_error(
        "SSBB realized-provenance rebase requires a strict incumbent win");
  if (candidate.ssbb_search_reference_bytes >
          candidate.ssbb_reference_bytes ||
      final_bytes > candidate.ssbb_search_reference_bytes)
    throw std::logic_error(
        "SSBB candidate has invalid local representation/search provenance");
  if (candidate.ssbb_composition_selected) {
    if (candidate.ssbb_composition_selected_channels == 0 ||
        candidate.ssbb_composition_reference_bytes <= final_bytes ||
        candidate.ssbb_composition_reference_bytes >
            candidate.ssbb_search_reference_bytes)
      throw std::logic_error(
          "SSBB candidate has invalid local composition provenance");
  }
  else if (candidate.ssbb_composition_selected_channels != 0) {
    throw std::logic_error(
        "SSBB candidate has composition channels without a selected composition");
  }
  if (!force_cross_start_rebase &&
      candidate.ssbb_reference_bytes <= incumbent_bytes)
    return;

  // A candidate reached from a different start/base can have a large local
  // improvement while only narrowly beating the already available global
  // incumbent.  That counterfactual local saving remains in the aggregate
  // work diagnostics, but it is not a realized representation/search saving.
  // The actually selected cross-branch gain is therefore measured from the
  // incumbent and attributed to search/selection, with representation zero.
  candidate.ssbb_reference_bytes = incumbent_bytes;
  candidate.ssbb_search_reference_bytes = incumbent_bytes;
  if (candidate.ssbb_composition_selected)
    candidate.ssbb_composition_reference_bytes = std::min(
        candidate.ssbb_composition_reference_bytes, incumbent_bytes);
}

void run_ssbb_value_search(
    const std::vector<float>& original,
    const std::vector<float>& base,
    const Settings& settings,
    std::vector<LegacyRepairSeed>& seeds,
    EncodedWindow& result)
{
  if (!uses_ssbb(settings.repair_optimizer))
    return;
  const auto path_count = correction_format_ceiling_path_count(
      settings.correction_format_ceiling);
  const auto input = result;
  const auto input_seeds = seeds;
  auto aggregate_work = EncodedWindow();
  auto winner = input;
  auto winner_seeds = input_seeds;
  auto has_winner_path = false;
  for (std::uint8_t ceiling = kPackedQuantizedCorrectionFormat;
       ceiling <= settings.correction_format_ceiling;
       ++ceiling) {
    auto path_settings = settings;
    path_settings.correction_format_ceiling = ceiling;
    auto path = input;
    auto path_seeds = input_seeds;
    run_ssbb_value_search_single_ceiling(
        original, base, path_settings, path_seeds, path);
    if (!path.correction_payload.empty() &&
        path.correction_payload.front() > ceiling)
      throw std::logic_error(
          "SSBB complete ceiling path selected a forbidden correction tag");
    accumulate_ssbb_path_work(aggregate_work, path);
    // Iterating from 6 upward and retaining byte ties makes each lower-ceiling
    // complete result an immutable candidate in every larger portfolio.
    if (!has_winner_path || path.encoded_size() < winner.encoded_size()) {
      winner = std::move(path);
      winner_seeds = std::move(path_seeds);
      has_winner_path = true;
    }
  }
  if (!has_winner_path ||
      aggregate_work.ssbb_composition_starts != path_count ||
      aggregate_work.ssbb_representation_candidates_serialized != path_count)
    throw std::logic_error(
        "SSBB nested ceiling paths did not execute exactly once");
  result = std::move(winner);
  seeds = std::move(winner_seeds);
  result.ssbb_base_records_tested = 1;
  overwrite_ssbb_path_work(result, aggregate_work);
  result.repair_candidates_tested = input.repair_candidates_tested;
  result.repair_candidates_tested = checked_add_u32(
      result.repair_candidates_tested,
      result.ssbb_representation_candidates_serialized,
      "QPET plus nested SSBB representation candidate count overflow");
  result.repair_candidates_tested = checked_add_u32(
      result.repair_candidates_tested,
      result.ssbb_candidates_serialized,
      "QPET plus nested SSBB repair candidate count overflow");
  result.repair_candidates_tested = checked_add_u32(
      result.repair_candidates_tested,
      result.ssbb_composition_candidates_serialized,
      "QPET plus nested SSBB composition candidate count overflow");
  if (result.ssbb_reference_bytes != input.encoded_size() ||
      result.ssbb_search_reference_bytes > result.ssbb_reference_bytes ||
      result.encoded_size() > result.ssbb_search_reference_bytes ||
      (!result.correction_payload.empty() &&
       result.correction_payload.front() > settings.correction_format_ceiling))
    throw std::logic_error(
        "SSBB nested ceiling winner lost its reference provenance, regressed "
        "bytes, or exceeded its tag bound");
}

struct DecodedBaseCandidate {
  EncodedWindow encoded;
  std::shared_ptr<qpet_correlation::CorrelationQoI> qoi;
};

DecodedBaseCandidate encode_decode_base_candidate_unchecked(
    const std::vector<float>& original,
    const Settings& settings,
    double global_bound,
    double estimator_scale,
    std::uint32_t base_mode,
    bool canonicalize_hpez_metadata = false)
{
  if (base_mode <= 2U && settings.channels < 2U)
    throw QoZ::CodecInvariantError(
        "official two-dimensional HPEZ modes 0/1/2 are undefined for one channel");
  auto in_place_reconstruction = original;
  std::size_t compressed_size = 0;
  unsigned char* compressed = nullptr;
  auto qoi = std::shared_ptr<qpet_correlation::CorrelationQoI>();
  auto result = EncodedWindow();
  result.base_mode = base_mode;
  result.selected_global_bound = global_bound;
  result.selected_estimator_scale = estimator_scale;
  if (base_mode == 0 || base_mode == 2) {
    auto config = tune_official_hpez(
        original, settings, global_bound, canonicalize_hpez_metadata);
    const auto allocation = base_mode == 2
        ? qpet_correlation::BoundAllocation::gradient_balanced
        : qpet_correlation::BoundAllocation::uniform;
    qoi = make_qoi(
        original, settings, global_bound, estimator_scale, allocation);
    auto compressor = make_compressor(qoi, global_bound);
    compressed = compressor.compress(
        config, in_place_reconstruction.data(), compressed_size);
  }
  else if (base_mode == 1) {
    if (settings.channels < 2U)
      throw QoZ::CodecInvariantError(
          "official two-dimensional HPEZ mode is undefined for one channel");
    auto config = tune_official_hpez(
        original, settings, global_bound, canonicalize_hpez_metadata);
    config.qoi = 0;
    config.use_global_eb = true;
    auto compressor = make_global_compressor(global_bound);
    compressed = compressor.compress(
        config, in_place_reconstruction.data(), compressed_size);
  }
  else if (base_mode == 3) {
    result.payload = compress_sperr_payload(original, settings, global_bound);
    result.reconstruction = decode_payload(
        result.payload, settings, global_bound, base_mode);
  }
  else if (base_mode == 4) {
    result.payload = compress_lossless_zstd_payload(original);
    result.reconstruction = decode_payload(
        result.payload, settings, global_bound, base_mode);
  }
  else if (base_mode == 5 || base_mode == 6) {
    result.payload = compress_serf_payload(original, global_bound, base_mode);
    result.reconstruction = decode_payload(
        result.payload, settings, global_bound, base_mode);
  }
  else if (base_mode == 7) {
    result.payload = compress_serf_smooth_payload(
        original, settings, global_bound, estimator_scale);
    result.reconstruction = decode_payload(
        result.payload, settings, global_bound, base_mode);
  }
  else {
    throw std::invalid_argument("invalid QPET base compressor mode");
  }
  if (base_mode < 3 && (compressed == nullptr || compressed_size == 0)) {
    delete[] compressed;
    throw BackendCandidateFailure("official QPET compressor returned an empty stream");
  }

  if (base_mode < 3) {
    result.payload.assign(compressed, compressed + compressed_size);
    delete[] compressed;

    // Decode from the official compressed bytes. Metrics never use the
    // compressor's in-place array as a shortcut.
    result.reconstruction = decode_payload(
        result.payload, settings, global_bound, base_mode);
    if (result.reconstruction.size() != in_place_reconstruction.size() ||
        std::memcmp(
            result.reconstruction.data(),
            in_place_reconstruction.data(),
            result.reconstruction.size() * sizeof(float)) != 0)
      throw BackendCandidateFailure(
          "official QPET byte-stream round trip disagrees with encoder state: mode=" +
          std::to_string(base_mode) + ",shape=" +
          std::to_string(settings.channels) + "x" +
          std::to_string(settings.time));
  }

  return {std::move(result), std::move(qoi)};
}

DecodedBaseCandidate encode_decode_base_candidate(
    const std::vector<float>& original,
    const Settings& settings,
    double global_bound,
    double estimator_scale,
    std::uint32_t base_mode,
    bool canonicalize_hpez_metadata = false)
{
  try {
    return encode_decode_base_candidate_unchecked(
        original,
        settings,
        global_bound,
        estimator_scale,
        base_mode,
        canonicalize_hpez_metadata);
  }
  catch (const QoZ::CodecError& error) {
    throw BackendCandidateFailure(
        "official QPET codec rejected one candidate: mode=" +
        std::to_string(base_mode) + ",shape=" +
        std::to_string(settings.channels) + "x" +
        std::to_string(settings.time) + ": " + error.what());
  }
}

EncodedWindow finalize_decoded_base_candidate(
    const std::vector<float>& original,
    const Settings& settings,
    DecodedBaseCandidate decoded)
{
  auto& result = decoded.encoded;
  qpet_correlation::ConstraintReport paper_base_report;
  if (settings.paper_budget_audit) {
    paper_base_report = qpet_correlation::verify_constraints(
        original, result.reconstruction, settings.channels, settings.time,
        settings.pointwise_tolerance, settings.acf_tolerance,
        settings.minimum_correlation, settings.max_lag);
    if (settings.minimum_correlation < 0.0) {
      const auto complete = qpet_correlation::verify_constraints(
          original, result.reconstruction, settings.channels, settings.time,
          settings.pointwise_tolerance, settings.acf_tolerance, 0.0, settings.max_lag);
      paper_base_report.min_series_correlation = complete.min_series_correlation;
    }
    ++settings.paper_budget_audit->base_candidates;
    if (paper_base_report.all_ok())
      ++settings.paper_budget_audit->base_passes;
    if (result.base_mode != 4U && paper_base_report.max_abs_error >
        result.selected_global_bound + 64.0 * std::numeric_limits<double>::epsilon())
      ++settings.paper_budget_audit->base_bound_violations;
  }
  if (settings.correction_mode == 0) {
    const auto base_mode = result.base_mode;
    if (base_mode != 0)
      throw std::logic_error("online correction requires the variable-EB QPET path");
    if (!decoded.qoi)
      throw std::logic_error("online QPET base omitted its QoI diagnostics");
    result.online_corrections = static_cast<std::uint32_t>(std::min<std::size_t>(
        decoded.qoi->correction_count(),
        std::numeric_limits<std::uint32_t>::max()));
    // Online rejections are already represented inside the official QPET
    // reconstruction and payload.  They are diagnostics, not an outer repair
    // field, so the serialized correction count must remain zero.
    result.corrections = 0;
  }
  else
    finalize_posthoc_repair(original, settings, result);

  result.report = qpet_correlation::verify_constraints(
      original,
      result.reconstruction,
      settings.channels,
      settings.time,
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag);
  if (!result.report.all_ok()) {
    auto message = std::ostringstream();
    message << "QPET candidate failed an independently recomputed hard constraint: "
            << "global_bound=" << result.selected_global_bound
            << ", estimator_scale=" << result.selected_estimator_scale
            << ", max_abs=" << result.report.max_abs_error
            << ", max_acf=" << result.report.max_acf_error
            << ", min_corr=" << result.report.min_series_correlation;
    throw std::runtime_error(message.str());
  }
  result.paper_base_report = paper_base_report;
  return std::move(decoded.encoded);
}

EncodedWindow compress_candidate(
    const std::vector<float>& original,
    const Settings& settings,
    double global_bound,
    double estimator_scale,
    std::uint32_t base_mode)
{
  try {
    return finalize_decoded_base_candidate(
        original, settings,
        encode_decode_base_candidate(original, settings, global_bound, estimator_scale, base_mode));
  }
  catch (const BackendCandidateFailure& error) {
    if (settings.paper_budget_audit) {
      auto& audit = *settings.paper_budget_audit;
      ++audit.base_failures;
      auto message = std::string(error.what());
      std::replace(message.begin(), message.end(), '\t', ' ');
      std::replace(message.begin(), message.end(), '\n', ' ');
      audit.failures_output << settings.paper_budget_window << '\t' << base_mode
          << '\t' << estimator_scale << '\t' << global_bound << '\t' << message << '\n';
    }
    throw;
  }
}

EncodedWindow optimize_stored_legacy_candidate(
    const std::vector<float>& original,
    const Settings& settings,
    const EncodedWindow& legacy)
{
  if (!uses_second_stage(settings.repair_optimizer) ||
      settings.correction_mode == 0 || settings.enable_vehicle_semantic ||
      legacy.base_mode >= 8)
    throw std::invalid_argument("invalid stored legacy candidate for FVRO/ZRCAR");
  auto base = decode_payload(
      legacy.payload,
      settings,
      legacy.selected_global_bound,
      legacy.base_mode,
      nullptr,
      kVersion);
  auto optimized = legacy;
  optimized.corrections = 0;
  optimized.correction_payload.clear();
  optimized.reconstruction = base;
  optimized.report = {};
  optimized.zrcar_candidates_tested = 0;
  optimized.zrcar_prefix_candidates_tested = 0;
  optimized.zrcar_local_candidates_tested = 0;
  optimized.zrcar_base_records_tested =
      uses_zrcar(settings.repair_optimizer) ? 1U : 0U;
  optimized.zrcar_legacy_seeds_selected = 0;
  optimized.zrcar_fvro_seeds_selected = 0;
  optimized.zrcar_states_generated = 0;
  optimized.zrcar_infeasible_states = 0;
  optimized.zrcar_infeasible_frontier_expansions = 0;
  optimized.zrcar_infeasible_frontier_recoveries = 0;
  optimized.zrcar_selected = false;
  optimized.zrcar_from_fvro_seed = false;
  optimized.legacy_repair_bytes = 0;
  optimized.zrcar_legacy_reference_bytes = 0;
  optimized.fvro_global_candidates_generated = 0;
  optimized.fvro_global_candidates_serialized = 0;
  optimized.fvro_global_candidates_infeasible = 0;
  optimized.fvro_channel_candidates_generated = 0;
  optimized.fvro_channel_candidates_serialized = 0;
  optimized.fvro_channel_candidates_infeasible = 0;
  optimized.fvro_local_proposals_generated = 0;
  optimized.fvro_local_proposals_infeasible = 0;
  optimized.fvro_selected_channels = 0;
  optimized.fvro_strict_channel_steps = 0;
  optimized.fvro_seeds_retained = 0;
  optimized.fvro_base_records_tested =
      uses_fvro(settings.repair_optimizer) ? 1U : 0U;
  optimized.fvro_reference_exact_escapes = 0;
  optimized.fvro_final_exact_escapes = 0;
  optimized.fvro_reference_quantized_format = 0;
  optimized.fvro_final_quantized_format = 0;
  optimized.fvro_reference_quantized = false;
  optimized.fvro_final_quantized = false;
  optimized.fvro_selected = false;
  optimized.fvro_legacy_reference_bytes = legacy.encoded_size();
  optimized.ssbb_base_records_tested = 0;
  optimized.ssbb_channels_searched = 0;
  optimized.ssbb_positions_processed = 0;
  optimized.ssbb_actions_generated = 0;
  optimized.ssbb_states_generated = 0;
  optimized.ssbb_states_deduplicated = 0;
  optimized.ssbb_statistics_feasible_states = 0;
  optimized.ssbb_statistics_infeasible_states = 0;
  optimized.ssbb_feasible_states_retained = 0;
  optimized.ssbb_infeasible_states_retained = 0;
  optimized.ssbb_states_pruned_by_beam = 0;
  optimized.ssbb_terminal_states = 0;
  optimized.ssbb_terminal_direct_feasible = 0;
  optimized.ssbb_terminal_direct_infeasible = 0;
  optimized.ssbb_outputs_returned = 0;
  optimized.ssbb_candidates_serialized = 0;
  optimized.ssbb_seeds_retained = 0;
  optimized.ssbb_selected_channels = 0;
  optimized.ssbb_representation_candidates_serialized = 0;
  optimized.ssbb_representation_improved_starts = 0;
  optimized.ssbb_representation_gross_saved_bytes = 0;
  optimized.ssbb_composition_starts = 0;
  optimized.ssbb_composition_local_options = 0;
  optimized.ssbb_composition_transitions_generated = 0;
  optimized.ssbb_composition_transitions_deduplicated = 0;
  optimized.ssbb_composition_direct_feasible = 0;
  optimized.ssbb_composition_direct_infeasible = 0;
  optimized.ssbb_composition_candidates_serialized = 0;
  optimized.ssbb_composition_singletons_serialized = 0;
  optimized.ssbb_composition_pairs_serialized = 0;
  optimized.ssbb_composition_frontier_carries = 0;
  optimized.ssbb_composition_anchor_retention_events = 0;
  optimized.ssbb_composition_format_retention_events = 0;
  optimized.ssbb_composition_rate_retention_events = 0;
  optimized.ssbb_composition_states_pruned = 0;
  optimized.ssbb_composition_max_frontier = 0;
  optimized.ssbb_composition_selected_channels = 0;
  optimized.ssbb_composition_selected = false;
  optimized.ssbb_composition_reference_bytes = 0;
  optimized.ssbb_selected = false;
  optimized.ssbb_reference_bytes = legacy.encoded_size();
  optimized.ssbb_search_reference_bytes = legacy.encoded_size();
  optimized.cjrf_selected = false;
  optimized.cjrf_winner_kind = CjrfWinnerKind::none;
  optimized.cjrf_relaxed_base_attempts = 0;
  optimized.cjrf_relaxed_base_failures = 0;
  optimized.cjrf_relaxed_base_pruned = 0;
  optimized.cjrf_relaxed_base_duplicates = 0;
  optimized.cjrf_relaxed_base_records = 0;
  optimized.cjrf_relaxed_base_outside_pointwise = 0;
  optimized.cjrf_relaxed_legacy_repaired = 0;
  optimized.cjrf_relaxed_legacy_repair_candidates = 0;
  optimized.cjrf_reference_bytes = 0;
  optimized.cfsr_selected = false;
  optimized.cfsr_winner_kind = CfsrWinnerKind::none;
  optimized.cfsr_surrogate_sources = 0;
  optimized.cfsr_surrogate_base_attempts = 0;
  optimized.cfsr_surrogate_base_failures = 0;
  optimized.cfsr_surrogate_base_pruned = 0;
  optimized.cfsr_surrogate_base_duplicates = 0;
  optimized.cfsr_surrogate_base_records = 0;
  optimized.cfsr_surrogate_base_outside_original_pointwise = 0;
  optimized.cfsr_surrogate_legacy_repaired = 0;
  optimized.cfsr_surrogate_legacy_repair_candidates = 0;
  optimized.cfsr_reference_bytes = 0;
  auto seeds = std::vector<LegacyRepairSeed>();
  finalize_posthoc_repair(original, settings, optimized, &seeds);
  if (optimized.corrections != legacy.corrections ||
      optimized.correction_payload != legacy.correction_payload)
    throw std::runtime_error(
        "second-stage legacy replay disagrees with the first-stage repair bytes");
  require_float32_bitwise_equal(
      optimized.reconstruction,
      legacy.reconstruction,
      "second-stage legacy replay reconstruction changed at the float32 bit level");
  run_fvro_value_search(original, base, settings, seeds, optimized);
  auto value_ssbb_branch = std::optional<EncodedWindow>();
  if (uses_full_ssbb(settings.repair_optimizer)) {
    auto branch = optimized;
    auto branch_seeds = seeds;
    auto value_settings = settings;
    value_settings.repair_optimizer = RepairOptimizer::ssbb_value;
    run_ssbb_value_search(
        original, base, value_settings, branch_seeds, branch);
    value_ssbb_branch = std::move(branch);
  }
  run_zrcar_search(original, base, settings, seeds, optimized);
  // The full mode searches both certified starts: the FVRO-value incumbent
  // above and the complete FVRO+ZRCAR incumbent here. Keeping the two paths
  // separate makes the final candidate set a superset of ssbb_value and full
  // FVRO without allowing either beam to evict the other path.
  run_ssbb_value_search(original, base, settings, seeds, optimized);
  if (value_ssbb_branch.has_value()) {
    auto& branch = *value_ssbb_branch;
    auto add_u32 = [](std::uint32_t left,
                      std::uint32_t right,
                      const char* message) {
      return checked_add_u32(left, right, message);
    };
    optimized.ssbb_channels_searched = add_u32(
        optimized.ssbb_channels_searched,
        branch.ssbb_channels_searched,
        "SSBB dual-start channel total overflow");
    optimized.ssbb_positions_processed = add_u32(
        optimized.ssbb_positions_processed,
        branch.ssbb_positions_processed,
        "SSBB dual-start position total overflow");
    optimized.ssbb_actions_generated = add_u32(
        optimized.ssbb_actions_generated,
        branch.ssbb_actions_generated,
        "SSBB dual-start action total overflow");
    optimized.ssbb_states_generated = add_u32(
        optimized.ssbb_states_generated,
        branch.ssbb_states_generated,
        "SSBB dual-start generated-state total overflow");
    optimized.ssbb_states_deduplicated = add_u32(
        optimized.ssbb_states_deduplicated,
        branch.ssbb_states_deduplicated,
        "SSBB dual-start deduplicated-state total overflow");
    optimized.ssbb_statistics_feasible_states = add_u32(
        optimized.ssbb_statistics_feasible_states,
        branch.ssbb_statistics_feasible_states,
        "SSBB dual-start statistics-feasible total overflow");
    optimized.ssbb_statistics_infeasible_states = add_u32(
        optimized.ssbb_statistics_infeasible_states,
        branch.ssbb_statistics_infeasible_states,
        "SSBB dual-start statistics-infeasible total overflow");
    optimized.ssbb_feasible_states_retained = add_u32(
        optimized.ssbb_feasible_states_retained,
        branch.ssbb_feasible_states_retained,
        "SSBB dual-start feasible-retained total overflow");
    optimized.ssbb_infeasible_states_retained = add_u32(
        optimized.ssbb_infeasible_states_retained,
        branch.ssbb_infeasible_states_retained,
        "SSBB dual-start infeasible-retained total overflow");
    optimized.ssbb_states_pruned_by_beam = add_u32(
        optimized.ssbb_states_pruned_by_beam,
        branch.ssbb_states_pruned_by_beam,
        "SSBB dual-start beam-pruned total overflow");
    optimized.ssbb_terminal_states = add_u32(
        optimized.ssbb_terminal_states,
        branch.ssbb_terminal_states,
        "SSBB dual-start terminal-state total overflow");
    optimized.ssbb_terminal_direct_feasible = add_u32(
        optimized.ssbb_terminal_direct_feasible,
        branch.ssbb_terminal_direct_feasible,
        "SSBB dual-start terminal-feasible total overflow");
    optimized.ssbb_terminal_direct_infeasible = add_u32(
        optimized.ssbb_terminal_direct_infeasible,
        branch.ssbb_terminal_direct_infeasible,
        "SSBB dual-start terminal-infeasible total overflow");
    optimized.ssbb_outputs_returned = add_u32(
        optimized.ssbb_outputs_returned,
        branch.ssbb_outputs_returned,
        "SSBB dual-start output total overflow");
    optimized.ssbb_candidates_serialized = add_u32(
        optimized.ssbb_candidates_serialized,
        branch.ssbb_candidates_serialized,
        "SSBB dual-start serializer total overflow");
    optimized.ssbb_seeds_retained = add_u32(
        optimized.ssbb_seeds_retained,
        branch.ssbb_seeds_retained,
        "SSBB dual-start retained-seed total overflow");
    optimized.ssbb_selected_channels = add_u32(
        optimized.ssbb_selected_channels,
        branch.ssbb_selected_channels,
        "SSBB dual-start selected-channel total overflow");
    optimized.ssbb_representation_candidates_serialized = add_u32(
        optimized.ssbb_representation_candidates_serialized,
        branch.ssbb_representation_candidates_serialized,
        "SSBB dual-start representation-candidate total overflow");
    optimized.ssbb_representation_improved_starts = add_u32(
        optimized.ssbb_representation_improved_starts,
        branch.ssbb_representation_improved_starts,
        "SSBB dual-start representation-win total overflow");
    optimized.ssbb_representation_gross_saved_bytes = checked_add_size(
        optimized.ssbb_representation_gross_saved_bytes,
        branch.ssbb_representation_gross_saved_bytes,
        "SSBB dual-start representation saved-byte total overflow");
    optimized.ssbb_composition_starts = add_u32(
        optimized.ssbb_composition_starts,
        branch.ssbb_composition_starts,
        "SSBB dual-start composition-start total overflow");
    optimized.ssbb_composition_local_options = add_u32(
        optimized.ssbb_composition_local_options,
        branch.ssbb_composition_local_options,
        "SSBB dual-start composition-option total overflow");
    optimized.ssbb_composition_transitions_generated = add_u32(
        optimized.ssbb_composition_transitions_generated,
        branch.ssbb_composition_transitions_generated,
        "SSBB dual-start composition-transition total overflow");
    optimized.ssbb_composition_transitions_deduplicated = add_u32(
        optimized.ssbb_composition_transitions_deduplicated,
        branch.ssbb_composition_transitions_deduplicated,
        "SSBB dual-start composition-duplicate total overflow");
    optimized.ssbb_composition_direct_feasible = add_u32(
        optimized.ssbb_composition_direct_feasible,
        branch.ssbb_composition_direct_feasible,
        "SSBB dual-start composition-feasible total overflow");
    optimized.ssbb_composition_direct_infeasible = add_u32(
        optimized.ssbb_composition_direct_infeasible,
        branch.ssbb_composition_direct_infeasible,
        "SSBB dual-start composition-infeasible total overflow");
    optimized.ssbb_composition_candidates_serialized = add_u32(
        optimized.ssbb_composition_candidates_serialized,
        branch.ssbb_composition_candidates_serialized,
        "SSBB dual-start composition-serializer total overflow");
    optimized.ssbb_composition_singletons_serialized = add_u32(
        optimized.ssbb_composition_singletons_serialized,
        branch.ssbb_composition_singletons_serialized,
        "SSBB dual-start composition-singleton total overflow");
    optimized.ssbb_composition_pairs_serialized = add_u32(
        optimized.ssbb_composition_pairs_serialized,
        branch.ssbb_composition_pairs_serialized,
        "SSBB dual-start composition-pair total overflow");
    optimized.ssbb_composition_frontier_carries = add_u32(
        optimized.ssbb_composition_frontier_carries,
        branch.ssbb_composition_frontier_carries,
        "SSBB dual-start composition-carry total overflow");
    optimized.ssbb_composition_anchor_retention_events = add_u32(
        optimized.ssbb_composition_anchor_retention_events,
        branch.ssbb_composition_anchor_retention_events,
        "SSBB dual-start composition-anchor total overflow");
    optimized.ssbb_composition_format_retention_events = add_u32(
        optimized.ssbb_composition_format_retention_events,
        branch.ssbb_composition_format_retention_events,
        "SSBB dual-start composition-format total overflow");
    optimized.ssbb_composition_rate_retention_events = add_u32(
        optimized.ssbb_composition_rate_retention_events,
        branch.ssbb_composition_rate_retention_events,
        "SSBB dual-start composition-rate total overflow");
    optimized.ssbb_composition_states_pruned = add_u32(
        optimized.ssbb_composition_states_pruned,
        branch.ssbb_composition_states_pruned,
        "SSBB dual-start composition-pruned total overflow");
    optimized.ssbb_composition_max_frontier = std::max(
        optimized.ssbb_composition_max_frontier,
        branch.ssbb_composition_max_frontier);
    optimized.repair_candidates_tested = add_u32(
        optimized.repair_candidates_tested,
        branch.ssbb_representation_candidates_serialized,
        "SSBB dual-start representation-candidate total overflow");
    optimized.repair_candidates_tested = add_u32(
        optimized.repair_candidates_tested,
        branch.ssbb_candidates_serialized,
        "SSBB dual-start repair-candidate total overflow");
    optimized.repair_candidates_tested = add_u32(
        optimized.repair_candidates_tested,
        branch.ssbb_composition_candidates_serialized,
        "SSBB dual-start composition-candidate total overflow");
    if (branch.encoded_size() < optimized.encoded_size()) {
      if (!branch.ssbb_selected)
        throw std::logic_error(
            "SSBB value-start strict winner has no SSBB provenance");
      realize_ssbb_winner_against_incumbent(
          branch, optimized.encoded_size(), true);
      optimized.corrections = branch.corrections;
      optimized.correction_payload = branch.correction_payload;
      optimized.reconstruction = branch.reconstruction;
      optimized.report = branch.report;
      optimized.correction_channel_scales =
          branch.correction_channel_scales;
      optimized.fvro_final_exact_escapes =
          branch.fvro_final_exact_escapes;
      optimized.fvro_final_quantized_format =
          branch.fvro_final_quantized_format;
      optimized.fvro_final_quantized = branch.fvro_final_quantized;
      optimized.fvro_selected = false;
      optimized.zrcar_selected = false;
      optimized.zrcar_from_fvro_seed = false;
      optimized.ssbb_selected = true;
      optimized.ssbb_reference_bytes = branch.ssbb_reference_bytes;
      optimized.ssbb_search_reference_bytes =
          branch.ssbb_search_reference_bytes;
      optimized.ssbb_composition_selected =
          branch.ssbb_composition_selected;
      optimized.ssbb_composition_selected_channels =
          branch.ssbb_composition_selected_channels;
      optimized.ssbb_composition_reference_bytes =
          branch.ssbb_composition_reference_bytes;
    }
    const auto full_ceiling_paths = checked_add_size(
        correction_format_ceiling_path_count(
            settings.correction_format_ceiling),
        correction_format_ceiling_path_count(
            settings.correction_format_ceiling),
        "SSBB dual-start ceiling-path total overflow");
    if (optimized.ssbb_channels_searched >
            full_ceiling_paths * kSsbbMaximumDifficultChannels ||
        optimized.ssbb_candidates_serialized >
            full_ceiling_paths * kSsbbMaximumSerializedPerBase ||
        optimized.ssbb_composition_starts != full_ceiling_paths ||
        optimized.ssbb_representation_candidates_serialized !=
            optimized.ssbb_composition_starts ||
        optimized.ssbb_representation_improved_starts >
            optimized.ssbb_representation_candidates_serialized ||
        optimized.ssbb_composition_candidates_serialized >
            full_ceiling_paths *
                kSsbbCompositionMaximumTransitionsPerStart ||
        optimized.ssbb_composition_max_frontier >
            kSsbbCompositionBeamWidth ||
        optimized.ssbb_seeds_retained > full_ceiling_paths)
      throw std::logic_error("SSBB dual-start search exceeded its fixed budget");
  }
  optimized.report = qpet_correlation::verify_constraints(
      original,
      optimized.reconstruction,
      settings.channels,
      settings.time,
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag);
  if (!optimized.report.all_ok())
    throw std::runtime_error(
        "stored-base FVRO/ZRCAR/SSBB winner failed final verification");
  if (optimized.encoded_size() > legacy.encoded_size())
    throw std::logic_error(
        "stored-base FVRO/ZRCAR/SSBB regressed frozen legacy bytes");
  return optimized;
}

struct TuningCandidate {
  double global_bound = 0.0;
  double estimator_scale = 1.0;
  std::uint32_t base_mode = 0;
  double proposed_bound = 0.0;
};

#include "PaperBudgetCandidates.inc"

std::vector<TuningCandidate> tuning_candidates(
    const std::vector<float>& original,
    const Settings& settings)
{
  if (settings.paper_budget_profile != 0)
    return paper_budget_candidates(original, settings);
  auto candidates = std::vector<TuningCandidate>();
  const auto scales = settings.correction_mode == 0
      ? std::vector<double>{1.0}
      : settings.estimator_scales;
  for (const auto scale : scales) {
    // The pinned official QPET/HPEZ implementation instantiates a fixed 2-D
    // compressor.  Its C=1 degenerate shape is not an official 1-D path and
    // is rejected before backend entry (see encode/decode guards above).
    if (settings.channels < 2U)
      continue;
    // The gradient-balanced allocation remains available for controlled
    // ablation, but validation showed no rate win and a large runtime cost.
    // Formal candidate generation therefore retains the official uniform
    // QPET allocation and adds the independent official SPERR portfolio arm.
    const auto variable_modes = std::vector<std::uint32_t>{0};
    for (const auto base_mode : variable_modes) {
      const auto allocation = base_mode == 2
          ? qpet_correlation::BoundAllocation::gradient_balanced
          : qpet_correlation::BoundAllocation::uniform;
      auto analyzer = make_qoi(
          original,
          settings,
          settings.pointwise_tolerance,
          scale,
          allocation);
      auto sorted = analyzer->local_bounds();
      std::sort(sorted.begin(), sorted.end());
      auto global_bounds = std::vector<double>{settings.pointwise_tolerance};
      const auto k = std::max<std::size_t>(
          1,
          std::min(
              sorted.size() - 1,
              static_cast<std::size_t>(std::ceil(0.2 * sorted.size()))));
      // Same r_list used by the official QPET QoI tuner. Unlike its sampled
      // bitrate estimator, every candidate here is fully encoded and decoded.
      constexpr std::array<double, 7> rates = {
          1.0, 0.5, 0.25, 0.10, 0.05, 0.025, 0.0125};
      const auto positive_floor = std::max(
          settings.pointwise_tolerance * 1e-9,
          static_cast<double>(std::numeric_limits<float>::min()));
      for (const auto rate : rates) {
        const auto index = std::min(
            sorted.size() - 1,
            static_cast<std::size_t>(rate * static_cast<double>(k)));
        global_bounds.push_back(
            std::clamp(sorted[index], positive_floor, settings.pointwise_tolerance));
      }
      std::sort(global_bounds.begin(), global_bounds.end());
      global_bounds.erase(
          std::unique(
              global_bounds.begin(), global_bounds.end(), [](double left, double right) {
                return std::abs(left - right) <=
                    32.0 * std::numeric_limits<double>::epsilon() *
                        std::max({1.0, left, right});
              }),
          global_bounds.end());
      for (const auto global_bound : global_bounds)
        candidates.push_back({global_bound, scale, base_mode});
    }
  }
  if (settings.correction_mode != 0) {
    auto global_bounds = std::vector<double>();
    for (const auto& candidate : candidates)
      global_bounds.push_back(candidate.global_bound);
    std::sort(global_bounds.begin(), global_bounds.end());
    global_bounds.erase(
        std::unique(global_bounds.begin(), global_bounds.end(), [](double left, double right) {
          return std::abs(left - right) <=
              32.0 * std::numeric_limits<double>::epsilon() *
                  std::max({1.0, left, right});
        }),
        global_bounds.end());
    if (settings.channels >= 2U) {
      for (const auto global_bound : global_bounds)
        candidates.push_back({global_bound, 0.0, 1});
    }
    for (const auto ratio : kSperrBoundRatios)
      candidates.push_back({settings.pointwise_tolerance * ratio, 0.0, 3});
    for (const auto ratio : kSerfBoundRatios) {
      candidates.push_back({settings.pointwise_tolerance * ratio, 0.0, 5});
      candidates.push_back({settings.pointwise_tolerance * ratio, 0.0, 6});
    }
    candidates.push_back({settings.pointwise_tolerance, 0.0, 4});
  }
  return candidates;
}

bool has_strict_constant_channel(
    const std::vector<float>& original,
    std::size_t channels,
    std::size_t time)
{
  return !split_strict_constant_channels(original, channels, time)
              .constant_values.empty();
}

std::vector<unsigned char> compress_semantic_official_payload(
    const StrictConstantLayout& layout,
    const Settings& block_settings,
    double global_bound,
    std::uint32_t block_mode)
{
  if (layout.active_count == 0)
    throw std::invalid_argument("cannot invoke an official codec with zero active channels");
  auto active_settings = block_settings;
  active_settings.channels = layout.active_count;
  if (block_mode == 3) {
    if (layout.active_count < 2)
      throw std::invalid_argument("SPERR requires at least two active semantic channels");
    return compress_sperr_payload(layout.active_values, active_settings, global_bound);
  }
  if (block_mode == 4)
    return compress_lossless_zstd_payload(layout.active_values);
  if (block_mode == 6)
    return compress_serf_payload(layout.active_values, global_bound, block_mode);
  throw std::invalid_argument("unsupported strict-constant semantic codec");
}

EncodedWindow compress_strict_constant_block_candidate(
    const std::vector<float>& block_original,
    const Settings& block_settings,
    const StrictConstantLayout& layout,
    double global_bound,
    std::uint32_t block_mode)
{
  EncodedWindow result;
  result.base_mode = block_mode;
  result.selected_global_bound = global_bound;
  result.selected_estimator_scale = 0.0;
  auto official_payload = std::vector<unsigned char>();
  if (block_mode == kSemanticConstantOnlyMode) {
    if (layout.active_count != 0)
      throw std::invalid_argument("constant-only semantic candidate has active channels");
  }
  else {
    official_payload = compress_semantic_official_payload(
        layout, block_settings, global_bound, block_mode);
  }
  result.payload = make_strict_constant_block_payload(layout, official_payload);
  result.reconstruction = decode_strict_constant_semantic_block(
      result.payload, block_settings, global_bound, block_mode);
  finalize_posthoc_repair(block_original, block_settings, result);
  result.report = qpet_correlation::verify_constraints(
      block_original,
      result.reconstruction,
      block_settings.channels,
      block_settings.time,
      block_settings.pointwise_tolerance,
      block_settings.acf_tolerance,
      block_settings.minimum_correlation,
      block_settings.max_lag);
  if (!result.report.all_ok())
    throw std::runtime_error(
        "strict-constant semantic block failed an independently recomputed hard constraint");
  return result;
}

struct SemanticBlockChoice {
  EncodedWindow encoded;
  std::uint8_t bound_index = 0;
};

struct SemanticBaseChoice {
  std::uint32_t base_mode = 0;
  std::uint8_t bound_index = 0;
  double selected_global_bound = 0.0;
  std::vector<unsigned char> payload;
};

struct SemanticBlockWinners {
  bool has_local = false;
  bool has_base = false;
  std::size_t local_framed_bytes = 0;
  std::size_t base_framed_bytes = 0;
  SemanticBlockChoice local;
  SemanticBaseChoice base;
};

std::size_t semantic_independent_block_bytes(const EncodedWindow& encoded)
{
  auto size = std::size_t{2};
  size = checked_add_size(
      size, varint_size(encoded.corrections),
      "semantic independent descriptor size overflow");
  size = checked_add_size(
      size, varint_size(encoded.payload.size()),
      "semantic independent descriptor size overflow");
  size = checked_add_size(
      size, varint_size(encoded.correction_payload.size()),
      "semantic independent descriptor size overflow");
  return checked_add_size(
      size, encoded.encoded_size(),
      "semantic independent block size overflow");
}

std::size_t semantic_shared_block_bytes(const std::vector<unsigned char>& payload)
{
  auto size = checked_add_size(
      std::size_t{2}, varint_size(payload.size()),
      "semantic shared descriptor size overflow");
  return checked_add_size(
      size, payload.size(), "semantic shared block size overflow");
}

void retain_semantic_block_candidate(
    EncodedWindow current,
    std::uint8_t bound_index,
    SemanticBlockWinners& winners,
    std::size_t& block_candidates_tested,
    const char* count_overflow_message)
{
  block_candidates_tested = checked_add_size(
      block_candidates_tested,
      current.repair_candidates_tested,
      count_overflow_message);
  const auto local_bytes = semantic_independent_block_bytes(current);
  const auto base_bytes = semantic_shared_block_bytes(current.payload);
  const auto local_is_better =
      !winners.has_local || local_bytes < winners.local_framed_bytes ||
      (local_bytes == winners.local_framed_bytes &&
       current.selected_global_bound >
           winners.local.encoded.selected_global_bound);
  const auto base_is_better =
      !winners.has_base || base_bytes < winners.base_framed_bytes ||
      (base_bytes == winners.base_framed_bytes &&
       current.selected_global_bound > winners.base.selected_global_bound);

  // Copy only the base payload when it becomes the base-only winner.  The
  // already completed official encode is reused; no codec is invoked again.
  if (base_is_better) {
    winners.base.base_mode = current.base_mode;
    winners.base.bound_index = bound_index;
    winners.base.selected_global_bound = current.selected_global_bound;
    winners.base.payload = current.payload;
    winners.base_framed_bytes = base_bytes;
    winners.has_base = true;
  }
  if (local_is_better) {
    winners.local.encoded = std::move(current);
    winners.local.bound_index = bound_index;
    winners.local_framed_bytes = local_bytes;
    winners.has_local = true;
  }
}

SemanticBaseChoice semantic_base_choice(const SemanticBlockChoice& local)
{
  return {
      local.encoded.base_mode,
      local.bound_index,
      local.encoded.selected_global_bound,
      local.encoded.payload};
}

bool same_semantic_base_choices(
    const std::vector<SemanticBaseChoice>& left,
    const std::vector<SemanticBaseChoice>& right)
{
  if (left.size() != right.size())
    return false;
  for (std::size_t index = 0; index < left.size(); ++index) {
    // These are exactly the serialized descriptor and official base bytes.
    // selected_global_bound is derived from mode/index by the decoder and is
    // intentionally not a separate equality condition.
    if (left[index].base_mode != right[index].base_mode ||
        left[index].bound_index != right[index].bound_index ||
        left[index].payload != right[index].payload)
      return false;
  }
  return true;
}

EncodedWindow build_semantic_shared_candidate(
    const std::vector<float>& original,
    const Settings& settings,
    std::uint8_t schema_id,
    const std::vector<SemanticBaseChoice>& choices)
{
  if (settings.channels != 16 || choices.size() != kVehicleSignalGroups.size() ||
      (schema_id != kSemanticSchemaLegacy &&
       schema_id != kSemanticSchemaStrictConstants))
    throw std::invalid_argument("invalid semantic shared candidate layout");

  EncodedWindow shared;
  shared.base_mode = 9;
  shared.selected_global_bound = settings.pointwise_tolerance;
  shared.selected_estimator_scale = 0.0;
  auto payload_bytes = checked_add_size(
      kSemanticBlockMagic.size(), std::size_t{1},
      "semantic shared payload size overflow");
  for (const auto& choice : choices)
    payload_bytes = checked_add_size(
        payload_bytes,
        semantic_shared_block_bytes(choice.payload),
        "semantic shared payload size overflow");
  shared.payload.reserve(payload_bytes);
  shared.payload.insert(
      shared.payload.end(), kSemanticBlockMagic.begin(), kSemanticBlockMagic.end());
  shared.payload.push_back(schema_id);
  for (const auto& choice : choices) {
    if (choice.base_mode > std::numeric_limits<std::uint8_t>::max())
      throw std::overflow_error("semantic shared base mode exceeds uint8 framing");
    shared.payload.push_back(static_cast<unsigned char>(choice.base_mode));
    shared.payload.push_back(choice.bound_index);
    append_varint(shared.payload, choice.payload.size());
    shared.payload.insert(
        shared.payload.end(), choice.payload.begin(), choice.payload.end());
  }
  if (shared.payload.size() != payload_bytes)
    throw std::logic_error("semantic shared payload accounting disagrees with framing");

  // Decode the complete SBK payload rather than stitching encoder-side block
  // arrays.  The global repair is therefore optimized against exactly the
  // base reconstruction a QPC decoder observes.
  shared.reconstruction = decode_payload(
      shared.payload,
      settings,
      settings.pointwise_tolerance,
      shared.base_mode,
      nullptr,
      kVersion);
  finalize_posthoc_repair(original, settings, shared);

  // Independently repeat the complete semantic-base and correction decode.
  // Any framing, format-6, or count disagreement fails before byte selection.
  auto decoded = decode_payload(
      shared.payload,
      settings,
      settings.pointwise_tolerance,
      shared.base_mode,
      nullptr,
      kVersion);
  apply_correction_payload(
      decoded,
      shared.correction_payload,
      shared.corrections,
      settings.channels,
      settings.time,
      kVersion);
  if (decoded.size() != shared.reconstruction.size() ||
      std::memcmp(
          decoded.data(),
          shared.reconstruction.data(),
          decoded.size() * sizeof(float)) != 0)
    throw std::runtime_error(
        "semantic shared complete decode disagrees with repair selection");
  shared.reconstruction = std::move(decoded);
  shared.report = qpet_correlation::verify_constraints(
      original,
      shared.reconstruction,
      settings.channels,
      settings.time,
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag);
  if (!shared.report.all_ok())
    throw std::runtime_error(
        "semantic shared candidate failed an independently decoded hard constraint");
  return shared;
}

EncodedWindow select_dual_shared_candidate(
    EncodedWindow shared_local,
    EncodedWindow shared_base)
{
  const auto local_bytes = shared_local.encoded_size();
  const auto base_bytes = shared_base.encoded_size();
  const auto repair_candidates = checked_add_u32(
      shared_local.repair_candidates_tested,
      shared_base.repair_candidates_tested,
      "semantic shared repair candidate count overflow");
  const auto base_wins = base_bytes < local_bytes;
  auto selected = base_wins ? std::move(shared_base) : std::move(shared_local);
  selected.repair_candidates_tested = repair_candidates;
  selected.shared_base_search_groups = base_wins ? 1U : 0U;
  selected.shared_local_reference_bytes = local_bytes;
  if (selected.encoded_size() > local_bytes)
    throw std::logic_error("dual semantic shared search regressed old shared bytes");
  return selected;
}

template <typename Factory, typename Retain>
bool try_retain_candidate(
    Factory&& factory,
    Retain&& retain,
    std::size_t& candidate_failures)
{
  using Candidate = std::decay_t<decltype(factory())>;
  auto candidate = std::optional<Candidate>();
  try {
    candidate.emplace(factory());
  }
  catch (const BackendCandidateFailure&) {
    if (candidate_failures == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("backend candidate failure count overflow");
    ++candidate_failures;
    return false;
  }
  // Candidate storage/selection, byte accounting, integrity verification, and
  // hard-constraint errors are deliberately outside the isolation boundary.
  retain(std::move(*candidate));
  return true;
}

template <typename Factory, typename Retain>
bool try_retain_semantic_candidate(
    Factory&& factory,
    std::uint8_t bound_index,
    Retain&& retain,
    std::size_t& candidate_failures)
{
  return try_retain_candidate(
      std::forward<Factory>(factory),
      [&](EncodedWindow candidate) {
        retain(std::move(candidate), bound_index);
      },
      candidate_failures);
}

std::pair<EncodedWindow, EncodedWindow> compress_semantic_candidates(
    const std::vector<float>& original,
    const Settings& settings)
{
  if (settings.channels != 16 || settings.correction_mode == 0)
    throw std::invalid_argument("semantic-block candidates require 16 channels and post-hoc repair");
  auto choices = std::vector<SemanticBlockChoice>();
  auto base_choices = std::vector<SemanticBaseChoice>();
  choices.reserve(kVehicleSignalGroups.size());
  base_choices.reserve(kVehicleSignalGroups.size());
  std::size_t block_candidates_tested = 0;
  std::size_t isolated_candidate_failures = 0;
  for (const auto [channel_begin, channel_count] : kVehicleSignalGroups) {
    auto block_original = std::vector<float>(channel_count * settings.time);
    std::copy(
        original.begin() + static_cast<std::ptrdiff_t>(channel_begin * settings.time),
        original.begin() + static_cast<std::ptrdiff_t>(
            (channel_begin + channel_count) * settings.time),
        block_original.begin());
    auto block_settings = settings;
    block_settings.channels = channel_count;
    SemanticBlockWinners winners;
    auto retain = [&](EncodedWindow current, std::uint8_t bound_index) {
      retain_semantic_block_candidate(
          std::move(current),
          bound_index,
          winners,
          block_candidates_tested,
          "semantic candidate count overflow");
    };
    for (std::size_t index = 0; index < kSerfBoundRatios.size(); ++index) {
      (void)try_retain_semantic_candidate(
          [&] {
            return compress_candidate(
                block_original,
                block_settings,
                settings.pointwise_tolerance * kSerfBoundRatios[index],
                0.0,
                6);
          },
          static_cast<std::uint8_t>(index),
          retain,
          isolated_candidate_failures);
    }
    for (std::size_t index = 0; index < kSperrBoundRatios.size(); ++index) {
      (void)try_retain_semantic_candidate(
          [&] {
            return compress_candidate(
                block_original,
                block_settings,
                settings.pointwise_tolerance * kSperrBoundRatios[index],
                0.0,
                3);
          },
          static_cast<std::uint8_t>(index),
          retain,
          isolated_candidate_failures);
    }
    retain(
        compress_candidate(
            block_original,
            block_settings,
            settings.pointwise_tolerance,
            0.0,
            4),
        0xffU);
    if (!winners.has_local || !winners.has_base)
      throw std::runtime_error("semantic-block tuner produced no candidate");
    choices.push_back(std::move(winners.local));
    base_choices.push_back(std::move(winners.base));
  }

  block_candidates_tested = checked_add_size(
      block_candidates_tested,
      isolated_candidate_failures,
      "semantic candidate attempt count overflow");
  if (block_candidates_tested > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("semantic candidate count exceeds uint32 diagnostics");

  EncodedWindow independent;
  independent.base_mode = 8;
  independent.selected_global_bound = settings.pointwise_tolerance;
  independent.selected_estimator_scale = 0.0;
  independent.reconstruction.resize(original.size());
  independent.payload.insert(
      independent.payload.end(), kSemanticBlockMagic.begin(), kSemanticBlockMagic.end());
  independent.payload.push_back(kSemanticSchemaLegacy);
  for (std::size_t index = 0; index < choices.size(); ++index) {
    const auto [channel_begin, channel_count] = kVehicleSignalGroups[index];
    const auto& choice = choices[index];
    independent.payload.push_back(static_cast<unsigned char>(choice.encoded.base_mode));
    independent.payload.push_back(choice.bound_index);
    append_varint(independent.payload, choice.encoded.corrections);
    append_varint(independent.payload, choice.encoded.payload.size());
    append_varint(independent.payload, choice.encoded.correction_payload.size());
    independent.payload.insert(
        independent.payload.end(),
        choice.encoded.payload.begin(),
        choice.encoded.payload.end());
    independent.payload.insert(
        independent.payload.end(),
        choice.encoded.correction_payload.begin(),
        choice.encoded.correction_payload.end());
    if (choice.encoded.corrections >
        std::numeric_limits<std::uint32_t>::max() - independent.corrections)
      throw std::overflow_error("semantic independent correction count overflow");
    independent.corrections += choice.encoded.corrections;
    std::copy(
        choice.encoded.reconstruction.begin(),
        choice.encoded.reconstruction.end(),
        independent.reconstruction.begin() + channel_begin * settings.time);
  }
  independent.repair_candidates_tested =
      static_cast<std::uint32_t>(block_candidates_tested);
  independent.candidate_failures =
      static_cast<std::uint32_t>(isolated_candidate_failures);
  independent.report = qpet_correlation::verify_constraints(
      original,
      independent.reconstruction,
      settings.channels,
      settings.time,
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag);
  if (!independent.report.all_ok())
    throw std::runtime_error("semantic independent-block candidate failed a hard constraint");

  auto local_base_choices = std::vector<SemanticBaseChoice>();
  local_base_choices.reserve(choices.size());
  for (const auto& choice : choices)
    local_base_choices.push_back(semantic_base_choice(choice));
  auto shared_local = build_semantic_shared_candidate(
      original, settings, kSemanticSchemaLegacy, local_base_choices);
  shared_local.shared_local_reference_bytes = shared_local.encoded_size();
  if (same_semantic_base_choices(local_base_choices, base_choices))
    return {std::move(independent), std::move(shared_local)};

  auto shared_base = build_semantic_shared_candidate(
      original, settings, kSemanticSchemaLegacy, base_choices);
  auto shared = select_dual_shared_candidate(
      std::move(shared_local), std::move(shared_base));
  return {std::move(independent), std::move(shared)};
}

std::pair<EncodedWindow, EncodedWindow> compress_strict_constant_semantic_candidates(
    const std::vector<float>& original,
    const Settings& settings)
{
  if (settings.channels != 16 || settings.correction_mode == 0 ||
      !has_strict_constant_channel(original, settings.channels, settings.time))
    throw std::invalid_argument(
        "schema-2 semantic candidates require a strict constant channel and post-hoc repair");
  auto choices = std::vector<SemanticBlockChoice>();
  auto base_choices = std::vector<SemanticBaseChoice>();
  choices.reserve(kVehicleSignalGroups.size());
  base_choices.reserve(kVehicleSignalGroups.size());
  std::size_t block_candidates_tested = 0;
  std::size_t isolated_candidate_failures = 0;
  for (const auto [channel_begin, channel_count] : kVehicleSignalGroups) {
    auto block_original = std::vector<float>(channel_count * settings.time);
    std::copy(
        original.begin() + static_cast<std::ptrdiff_t>(channel_begin * settings.time),
        original.begin() + static_cast<std::ptrdiff_t>(
            (channel_begin + channel_count) * settings.time),
        block_original.begin());
    auto block_settings = settings;
    block_settings.channels = channel_count;
    const auto layout = split_strict_constant_channels(
        block_original, channel_count, settings.time);
    SemanticBlockWinners winners;
    auto retain = [&](EncodedWindow current, std::uint8_t bound_index) {
      retain_semantic_block_candidate(
          std::move(current),
          bound_index,
          winners,
          block_candidates_tested,
          "schema-2 semantic candidate count overflow");
    };
    if (layout.active_count == 0) {
      retain(
          compress_strict_constant_block_candidate(
              block_original,
              block_settings,
              layout,
              0.0,
              kSemanticConstantOnlyMode),
          0xffU);
    }
    else {
      for (std::size_t index = 0; index < kSerfBoundRatios.size(); ++index) {
        (void)try_retain_semantic_candidate(
            [&] {
              return compress_strict_constant_block_candidate(
                  block_original,
                  block_settings,
                  layout,
                  settings.pointwise_tolerance * kSerfBoundRatios[index],
                  6);
            },
            static_cast<std::uint8_t>(index),
            retain,
            isolated_candidate_failures);
      }
      if (layout.active_count >= 2) {
        for (std::size_t index = 0; index < kSperrBoundRatios.size(); ++index) {
          (void)try_retain_semantic_candidate(
              [&] {
                return compress_strict_constant_block_candidate(
                    block_original,
                    block_settings,
                    layout,
                    settings.pointwise_tolerance * kSperrBoundRatios[index],
                    3);
              },
              static_cast<std::uint8_t>(index),
              retain,
              isolated_candidate_failures);
        }
      }
      retain(
          compress_strict_constant_block_candidate(
              block_original,
              block_settings,
              layout,
              settings.pointwise_tolerance,
              4),
          0xffU);
    }
    if (!winners.has_local || !winners.has_base)
      throw std::runtime_error("schema-2 semantic tuner produced no candidate");
    choices.push_back(std::move(winners.local));
    base_choices.push_back(std::move(winners.base));
  }

  block_candidates_tested = checked_add_size(
      block_candidates_tested,
      isolated_candidate_failures,
      "schema-2 semantic candidate attempt count overflow");
  if (block_candidates_tested > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error(
        "schema-2 semantic candidate count exceeds uint32 diagnostics");

  EncodedWindow independent;
  independent.base_mode = 8;
  independent.selected_global_bound = settings.pointwise_tolerance;
  independent.selected_estimator_scale = 0.0;
  independent.reconstruction.resize(original.size());
  independent.payload.insert(
      independent.payload.end(), kSemanticBlockMagic.begin(), kSemanticBlockMagic.end());
  independent.payload.push_back(kSemanticSchemaStrictConstants);
  for (std::size_t index = 0; index < choices.size(); ++index) {
    const auto [channel_begin, channel_count] = kVehicleSignalGroups[index];
    const auto& choice = choices[index];
    independent.payload.push_back(static_cast<unsigned char>(choice.encoded.base_mode));
    independent.payload.push_back(choice.bound_index);
    append_varint(independent.payload, choice.encoded.corrections);
    append_varint(independent.payload, choice.encoded.payload.size());
    append_varint(independent.payload, choice.encoded.correction_payload.size());
    independent.payload.insert(
        independent.payload.end(),
        choice.encoded.payload.begin(),
        choice.encoded.payload.end());
    independent.payload.insert(
        independent.payload.end(),
        choice.encoded.correction_payload.begin(),
        choice.encoded.correction_payload.end());
    if (choice.encoded.corrections >
        std::numeric_limits<std::uint32_t>::max() - independent.corrections)
      throw std::overflow_error("schema-2 independent correction count overflow");
    independent.corrections += choice.encoded.corrections;
    std::copy(
        choice.encoded.reconstruction.begin(),
        choice.encoded.reconstruction.end(),
        independent.reconstruction.begin() + channel_begin * settings.time);
  }
  independent.repair_candidates_tested =
      static_cast<std::uint32_t>(block_candidates_tested);
  independent.candidate_failures =
      static_cast<std::uint32_t>(isolated_candidate_failures);
  independent.report = qpet_correlation::verify_constraints(
      original,
      independent.reconstruction,
      settings.channels,
      settings.time,
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag);
  if (!independent.report.all_ok())
    throw std::runtime_error(
        "schema-2 semantic independent candidate failed a hard constraint");

  auto local_base_choices = std::vector<SemanticBaseChoice>();
  local_base_choices.reserve(choices.size());
  for (const auto& choice : choices)
    local_base_choices.push_back(semantic_base_choice(choice));
  auto shared_local = build_semantic_shared_candidate(
      original, settings, kSemanticSchemaStrictConstants, local_base_choices);
  shared_local.shared_local_reference_bytes = shared_local.encoded_size();
  if (same_semantic_base_choices(local_base_choices, base_choices))
    return {std::move(independent), std::move(shared_local)};

  auto shared_base = build_semantic_shared_candidate(
      original, settings, kSemanticSchemaStrictConstants, base_choices);
  auto shared = select_dual_shared_candidate(
      std::move(shared_local), std::move(shared_base));
  return {std::move(independent), std::move(shared)};
}

bool retain_strict_zrcar_whole_window(
    EncodedWindow candidate,
    EncodedWindow& incumbent)
{
  if (candidate.encoded_size() >= incumbent.encoded_size())
    return false;
  incumbent = std::move(candidate);
  return true;
}

struct RetainedLegacyRecord {
  std::size_t ordinal = 0;
  EncodedWindow encoded;
};

bool same_binary64_bits(double left, double right)
{
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  std::uint64_t left_bits = 0;
  std::uint64_t right_bits = 0;
  std::memcpy(&left_bits, &left, sizeof(left_bits));
  std::memcpy(&right_bits, &right, sizeof(right_bits));
  return left_bits == right_bits;
}

bool same_stored_base_identity(
    const EncodedWindow& left,
    const EncodedWindow& right)
{
  // These are exactly the per-record inputs consumed by decode_payload.
  // Estimator scale and legacy correction metadata do not change the stored
  // base reconstruction once mode, bound bits, and payload bytes are fixed.
  return left.base_mode == right.base_mode &&
      same_binary64_bits(
          left.selected_global_bound, right.selected_global_bound) &&
      left.payload == right.payload;
}

bool same_persisted_record_and_reconstruction(
    const EncodedWindow& left,
    const EncodedWindow& right)
{
  return left.logical_windows == right.logical_windows &&
      same_stored_base_identity(left, right) &&
      left.corrections == right.corrections &&
      left.correction_payload == right.correction_payload &&
      same_float32_tensor_bits(left.reconstruction, right.reconstruction);
}

std::vector<float> canonical_cfsr_surrogate_replay(
    const std::vector<float>& original,
    const Settings& settings,
    const EncodedWindow& incumbent)
{
  if (!uses_cfsr(settings.surrogate_frontier) ||
      settings.correction_mode == 0 || settings.enable_vehicle_semantic ||
      settings.group_size != 1 || incumbent.logical_windows != 1 ||
      incumbent.base_mode >= 8U || incumbent.payload.empty() ||
      original.size() != settings.channels * settings.time)
    throw std::invalid_argument(
        "CFSR canonical replay requires one generic post-hoc record");
  auto replayed = decode_payload(
      incumbent.payload,
      settings,
      incumbent.selected_global_bound,
      incumbent.base_mode,
      nullptr,
      correction_payload_required_stream_version(
          incumbent.correction_payload));
  apply_correction_payload(
      replayed,
      incumbent.correction_payload,
      incumbent.corrections,
      settings.channels,
      settings.time,
      correction_payload_required_stream_version(
          incumbent.correction_payload));
  require_float32_bitwise_equal(
      replayed,
      incumbent.reconstruction,
      "CFSR canonical outer replay changed incumbent float32 bits");
  for (const auto value : replayed) {
    if (!std::isfinite(value))
      throw std::runtime_error(
          "CFSR canonical outer replay produced NaN or infinity");
  }
  const auto report = qpet_correlation::verify_constraints(
      original,
      replayed,
      settings.channels,
      settings.time,
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag);
  if (!report.all_ok())
    throw std::runtime_error(
        "CFSR canonical outer replay failed direct original-data constraints");
  return replayed;
}

void retain_unique_stored_base(
    std::vector<RetainedLegacyRecord>& records,
    EncodedWindow current,
    std::size_t ordinal)
{
  const auto duplicate = std::find_if(
      records.begin(), records.end(), [&](const RetainedLegacyRecord& retained) {
        return same_stored_base_identity(retained.encoded, current);
      });
  if (duplicate != records.end()) {
    // The frozen legacy repair is deterministic for a decoded base. Treat a
    // disagreement as corruption rather than silently choosing one identity.
    if (duplicate->encoded.corrections != current.corrections ||
        duplicate->encoded.correction_payload != current.correction_payload ||
        !same_float32_tensor_bits(
            duplicate->encoded.reconstruction, current.reconstruction))
      throw std::logic_error(
          "identical stored base identity produced different legacy repair bytes");
    return;
  }
  records.push_back(RetainedLegacyRecord{ordinal, std::move(current)});
}

void select_zrcar_base_records(
    std::vector<RetainedLegacyRecord>& records,
    std::size_t legacy_incumbent_bytes)
{
  // A correction payload has nonnegative length. A base at or above the
  // independent legacy incumbent therefore cannot produce the strict whole-
  // window byte win required by ZRCAR.
  records.erase(
      std::remove_if(
          records.begin(),
          records.end(),
          [&](const RetainedLegacyRecord& record) {
            return record.encoded.payload.size() >= legacy_incumbent_bytes;
          }),
      records.end());
  std::stable_sort(
      records.begin(),
      records.end(),
      [](const RetainedLegacyRecord& left, const RetainedLegacyRecord& right) {
        const auto left_bytes = left.encoded.encoded_size();
        const auto right_bytes = right.encoded.encoded_size();
        if (left_bytes != right_bytes)
          return left_bytes < right_bytes;
        if (left.encoded.selected_global_bound !=
            right.encoded.selected_global_bound)
          return left.encoded.selected_global_bound >
              right.encoded.selected_global_bound;
        return left.ordinal < right.ordinal;
      });
  if (records.size() > kZrcarBaseRecordCount)
    records.resize(kZrcarBaseRecordCount);
}

struct CjrfRelaxedWork {
  std::size_t attempts = 0;
  std::size_t failures = 0;
  std::size_t pruned = 0;
  std::size_t duplicates = 0;
  std::size_t records = 0;
  std::size_t outside_pointwise = 0;
  std::size_t legacy_repaired = 0;
  std::size_t legacy_repair_candidates = 0;
};

struct CfsrSurrogateWork {
  std::size_t sources = 0;
  std::size_t attempts = 0;
  std::size_t failures = 0;
  std::size_t pruned = 0;
  std::size_t duplicates = 0;
  std::size_t records = 0;
  std::size_t outside_original_pointwise = 0;
  std::size_t legacy_repaired = 0;
  std::size_t legacy_repair_candidates = 0;
};

std::size_t cfsr_enabled_ratio_count(std::uint8_t mask)
{
  auto count = std::size_t{0};
  for (std::size_t bit = 0; bit < kCfsrSurrogateBoundRatios.size(); ++bit)
    count += (mask & (1U << bit)) == 0U ? 0U : 1U;
  return count;
}

std::size_t expected_cfsr_base_attempts(const Settings& settings)
{
  if (!uses_cfsr(settings.surrogate_frontier))
    return 0;
  const auto lossy_modes = kCfsrSurrogateBaseModes.size() -
      2U * static_cast<std::size_t>(settings.channels == 1U);
  return checked_add_size(
      1U,
      cfsr_enabled_ratio_count(settings.cfsr_ratio_mask) * lossy_modes,
      "CFSR expected attempt count overflow");
}

bool base_exceeds_final_pointwise_tolerance(
    const std::vector<float>& original,
    const std::vector<float>& base,
    double tolerance)
{
  if (original.size() != base.size())
    throw std::invalid_argument(
        "CJRF base and source tensors have different sizes");
  for (std::size_t index = 0; index < original.size(); ++index) {
    if (std::abs(
            static_cast<double>(original[index]) -
            static_cast<double>(base[index])) > tolerance)
      return true;
  }
  return false;
}

void append_cjrf_relaxed_base_records(
    const std::vector<float>& original,
    const Settings& settings,
    std::size_t protected_incumbent_record_bytes,
    std::vector<RetainedLegacyRecord>& records,
    std::size_t& next_ordinal,
    CjrfRelaxedWork& work)
{
  if (!uses_cjrf(settings.base_frontier))
    return;
  if (settings.correction_mode == 0 || settings.enable_vehicle_semantic ||
      settings.group_size != 1)
    throw std::invalid_argument("CJRF relaxed bases require generic group-size-one repair");
  for (std::size_t ratio_index = 0;
       ratio_index < kCjrfRelaxedBoundRatios.size();
       ++ratio_index) {
    if ((settings.cjrf_ratio_mask & (1U << ratio_index)) == 0U)
      continue;
    const auto ratio = kCjrfRelaxedBoundRatios[ratio_index];
    const auto global_bound = settings.pointwise_tolerance * ratio;
    if (!(global_bound > settings.pointwise_tolerance) ||
        !std::isfinite(global_bound))
      throw std::overflow_error("CJRF relaxed global bound is not finite");
    for (const auto base_mode : kCjrfRelaxedBaseModes) {
      // SPERR's two-dimensional arm is not a defined candidate for a
      // single-channel tensor. This is a shape rule, not a data/domain rule.
      if (base_mode == 3U && settings.channels < 2)
        continue;
      work.attempts = checked_add_size(
          work.attempts, 1U, "CJRF relaxed attempt total overflow");
      auto decoded = std::optional<DecodedBaseCandidate>();
      try {
        decoded.emplace(encode_decode_base_candidate(
            original, settings, global_bound, 0.0, base_mode));
      }
      catch (const BackendCandidateFailure&) {
        work.failures = checked_add_size(
            work.failures, 1U, "CJRF relaxed failure total overflow");
        continue;
      }
      if (!decoded.has_value())
        throw std::logic_error("CJRF relaxed base isolation lost its result");
      auto& candidate = decoded->encoded;
      candidate.cjrf_relaxed_base = true;
      candidate.cjrf_relaxed_ratio = ratio;

      // Correction bytes are nonnegative. Compare the actual compact-record
      // lower bound, including the fixed 24-byte record header.
      const auto base_record_lower_bound =
          compact_base_record_lower_bound_bytes(candidate);
      if (base_record_lower_bound >= protected_incumbent_record_bytes) {
        work.pruned = checked_add_size(
            work.pruned, 1U, "CJRF relaxed prune total overflow");
        continue;
      }
      const auto duplicate = std::find_if(
          records.begin(), records.end(), [&](const RetainedLegacyRecord& retained) {
            return same_stored_base_identity(
                retained.encoded, candidate);
          });
      if (duplicate != records.end()) {
        work.duplicates = checked_add_size(
            work.duplicates, 1U, "CJRF relaxed duplicate total overflow");
        continue;
      }
      if (base_exceeds_final_pointwise_tolerance(
              original,
              candidate.reconstruction,
              settings.pointwise_tolerance))
        work.outside_pointwise = checked_add_size(
            work.outside_pointwise,
            1U,
            "CJRF outside-pointwise total overflow");

      auto repaired = finalize_decoded_base_candidate(
          original, settings, std::move(*decoded));
      if (!repaired.report.all_ok())
        throw std::logic_error("CJRF legacy repair returned an infeasible candidate");
      work.legacy_repaired = checked_add_size(
          work.legacy_repaired, 1U, "CJRF repaired-base total overflow");
      work.legacy_repair_candidates = checked_add_size(
          work.legacy_repair_candidates,
          repaired.repair_candidates_tested,
          "CJRF legacy repair-candidate total overflow");
      work.records = checked_add_size(
          work.records, 1U, "CJRF retained-base total overflow");
      if (next_ordinal == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("CJRF relaxed record ordinal overflow");
      records.push_back(
          RetainedLegacyRecord{next_ordinal++, std::move(repaired)});
    }
  }
  if (work.records != work.legacy_repaired ||
      work.legacy_repair_candidates < work.legacy_repaired ||
      work.outside_pointwise > work.records ||
      work.failures + work.pruned + work.duplicates + work.records !=
          work.attempts)
    throw std::logic_error("CJRF relaxed-base accounting does not partition attempts");
}

void append_cfsr_surrogate_base_records(
    const std::vector<float>& original,
    const std::vector<float>& surrogate,
    const Settings& settings,
    std::size_t protected_incumbent_record_bytes,
    std::vector<RetainedLegacyRecord>& records,
    std::size_t& next_ordinal,
    CfsrSurrogateWork& work)
{
  if (!uses_cfsr(settings.surrogate_frontier))
    return;
  if (uses_cjrf(settings.base_frontier) ||
      settings.correction_mode != 2U || settings.enable_vehicle_semantic ||
      settings.group_size != 1U ||
      original.size() != settings.channels * settings.time ||
      surrogate.size() != original.size())
    throw std::invalid_argument(
        "CFSR surrogate bases require one certified generic greedy window");
  work.sources = checked_add_size(
      work.sources, 1U, "CFSR surrogate-source total overflow");
  const auto immutable_surrogate = surrogate;
  auto attempt = [&](std::uint32_t base_mode,
                     double global_bound,
                     double surrogate_ratio) {
    work.attempts = checked_add_size(
        work.attempts, 1U, "CFSR surrogate attempt total overflow");
    auto decoded = std::optional<DecodedBaseCandidate>();
    // Serf's public constructor consumes float32.  Round toward zero before
    // calling it so the actual backend bound can never exceed the registered
    // binary64 r*epsilon shell when the nearest float rounds upward.
    const auto official_encode_bound = base_mode == 5U || base_mode == 6U
        ? static_cast<double>(cfsr_float_bound_not_above(global_bound))
        : global_bound;
    try {
      decoded.emplace(encode_decode_base_candidate(
          surrogate,
          settings,
          official_encode_bound,
          0.0,
          base_mode,
          base_mode == 1U));
    }
    catch (const BackendCandidateFailure&) {
      require_float32_bitwise_equal(
          surrogate,
          immutable_surrogate,
          "CFSR failed official codec modified its surrogate source");
      work.failures = checked_add_size(
          work.failures, 1U, "CFSR surrogate failure total overflow");
      return;
    }
    require_float32_bitwise_equal(
        surrogate,
        immutable_surrogate,
        "CFSR official codec modified its surrogate source");
    if (!decoded.has_value())
      throw std::logic_error("CFSR surrogate isolation lost its result");
    auto& candidate = decoded->encoded;
    // Persist the exact effective official-codec input.  The separate fixed
    // surrogate ratio remains the registered-shell provenance, while the QPC
    // record truthfully records a Serf float32 bound narrowed (never widened)
    // from r*epsilon.
    candidate.selected_global_bound = official_encode_bound;
    candidate.cfsr_surrogate_base = true;
    candidate.cfsr_surrogate_ratio = surrogate_ratio;
    for (const auto value : candidate.reconstruction) {
      if (!std::isfinite(value))
        throw std::runtime_error(
            "CFSR official base decode produced NaN or infinity");
    }
    // A backend may be isolated only before it yields a canonical arm.  Once
    // official payload bytes decode successfully, violating the exact bound
    // passed to that arm is an integrity error and aborts the whole window.
    if (base_exceeds_final_pointwise_tolerance(
            surrogate, candidate.reconstruction, official_encode_bound))
      throw std::runtime_error(
          "CFSR official base decode exceeded its registered surrogate bound");
    if (base_mode == 4U)
      require_float32_bitwise_equal(
          candidate.reconstruction,
          surrogate,
          "CFSR lossless anchor changed surrogate float32 bits");

    const auto base_record_lower_bound =
        compact_base_record_lower_bound_bytes(candidate);
    if (base_record_lower_bound >= protected_incumbent_record_bytes) {
      work.pruned = checked_add_size(
          work.pruned, 1U, "CFSR surrogate prune total overflow");
      return;
    }
    const auto duplicate = std::find_if(
        records.begin(), records.end(), [&](const RetainedLegacyRecord& retained) {
          return same_stored_base_identity(retained.encoded, candidate);
        });
    if (duplicate != records.end()) {
      work.duplicates = checked_add_size(
          work.duplicates, 1U, "CFSR surrogate duplicate total overflow");
      return;
    }
    if (base_exceeds_final_pointwise_tolerance(
            original,
            candidate.reconstruction,
            settings.pointwise_tolerance))
      work.outside_original_pointwise = checked_add_size(
          work.outside_original_pointwise,
          1U,
          "CFSR outside-original-pointwise total overflow");

    auto repaired = finalize_decoded_base_candidate(
        original, settings, std::move(*decoded));
    if (!repaired.report.all_ok())
      throw std::logic_error("CFSR legacy repair returned an infeasible candidate");
    if (base_mode == 4U &&
        (repaired.corrections != 0U ||
         !repaired.correction_payload.empty() ||
         !same_float32_tensor_bits(repaired.reconstruction, surrogate)))
      throw std::runtime_error(
          "CFSR lossless feasible-surrogate anchor required a correction");
    work.legacy_repaired = checked_add_size(
        work.legacy_repaired, 1U, "CFSR repaired-base total overflow");
    work.legacy_repair_candidates = checked_add_size(
        work.legacy_repair_candidates,
        repaired.repair_candidates_tested,
        "CFSR legacy repair-candidate total overflow");
    work.records = checked_add_size(
        work.records, 1U, "CFSR retained-base total overflow");
    if (next_ordinal == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("CFSR surrogate record ordinal overflow");
    records.push_back(
        RetainedLegacyRecord{next_ordinal++, std::move(repaired)});
  };

  // A lossless encoding of the canonical feasible surrogate is an explicit
  // correction-free anchor in every nonempty CFSR prefix.
  attempt(4U, settings.pointwise_tolerance, 0.0);
  for (std::size_t ratio_index = 0;
       ratio_index < kCfsrSurrogateBoundRatios.size();
       ++ratio_index) {
    if ((settings.cfsr_ratio_mask & (1U << ratio_index)) == 0U)
      continue;
    const auto ratio = kCfsrSurrogateBoundRatios[ratio_index];
    const auto global_bound = settings.pointwise_tolerance * ratio;
    for (const auto base_mode : kCfsrSurrogateBaseModes) {
      if ((base_mode == 1U || base_mode == 3U) && settings.channels < 2U)
        continue;
      attempt(base_mode, global_bound, ratio);
    }
  }
  if (work.sources != 1U ||
      work.attempts != expected_cfsr_base_attempts(settings) ||
      work.records != work.legacy_repaired ||
      work.legacy_repair_candidates < work.legacy_repaired ||
      work.outside_original_pointwise > work.records ||
      work.failures + work.pruned + work.duplicates + work.records !=
          work.attempts)
    throw std::logic_error(
        "CFSR surrogate-base accounting does not partition attempts");
}

EncodedWindow compress_window(
    const std::vector<float>& original,
    const Settings& settings)
{
  const auto candidates = tuning_candidates(original, settings);
  EncodedWindow best;
  bool has_best = false;
  std::size_t tested = 0;
  std::size_t top_level_candidate_failures = 0;
  std::size_t total_candidate_failures = 0;
  std::size_t successful_legacy_ordinal = 0;
  auto second_stage_base_records = std::vector<RetainedLegacyRecord>();
  auto retain_if_smaller =
      [&best,
       &has_best,
       &tested,
       &total_candidate_failures,
       &successful_legacy_ordinal,
        &second_stage_base_records,
       &settings](EncodedWindow current) {
    tested = checked_add_size(
        tested,
        current.repair_candidates_tested,
        "QPET candidate attempt count overflow");
    total_candidate_failures = checked_add_size(
        total_candidate_failures,
        current.candidate_failures,
        "QPET backend candidate failure count overflow");
    if (uses_second_stage(settings.repair_optimizer)) {
      if (successful_legacy_ordinal == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("FVRO/ZRCAR legacy record ordinal overflow");
      retain_unique_stored_base(
          second_stage_base_records, current, successful_legacy_ordinal++);
    }
    if (!has_best || current.encoded_size() < best.encoded_size() ||
        (current.encoded_size() == best.encoded_size() &&
         current.selected_global_bound > best.selected_global_bound)) {
      best = std::move(current);
      has_best = true;
    }
  };
  for (const auto candidate : candidates) {
    (void)try_retain_candidate(
        [&] {
          if (settings.paper_budget_audit)
            ++settings.paper_budget_audit->base_attempts;
          auto encoded = compress_candidate(
              original, settings, candidate.global_bound,
              candidate.estimator_scale, candidate.base_mode);
          if (settings.paper_budget_audit) {
            auto& output = settings.paper_budget_audit->candidates_output;
            const auto& pre = encoded.paper_base_report;
            output << settings.paper_budget_window << '\t' << candidate.base_mode
                << '\t' << candidate.estimator_scale << '\t' << candidate.proposed_bound
                << '\t' << candidate.global_bound << '\t' << pre.max_abs_error
                << '\t' << pre.min_series_correlation << '\t' << pre.max_acf_error
                << '\t' << pre.all_ok() << '\t' << encoded.corrections
                << '\t' << encoded.payload.size() << '\t' << encoded.correction_payload.size()
                << '\t' << compact_record_bytes(encoded) << '\t' << encoded.report.all_ok()
                << '\n';
          }
          return encoded;
        },
        retain_if_smaller,
        top_level_candidate_failures);
  }
  if (settings.enable_vehicle_semantic &&
      settings.correction_mode != 0 && settings.channels == 16) {
    auto semantic = compress_semantic_candidates(original, settings);
    retain_if_smaller(std::move(semantic.first));
    retain_if_smaller(std::move(semantic.second));
    if (has_strict_constant_channel(original, settings.channels, settings.time)) {
      auto strict_constants =
          compress_strict_constant_semantic_candidates(original, settings);
      retain_if_smaller(std::move(strict_constants.first));
      retain_if_smaller(std::move(strict_constants.second));
    }
  }
  if (!has_best)
    throw std::runtime_error("QPET tuner produced no candidate");

  // QPET Algorithm 4, lines 8-19: when the first-stage winner lies in
  // the lowest 0.5% of point-wise bounds, continue down the sorted tail
  // until the error bound drops below the paper's 0.95 linear slope.
  // The full payload and correction stream are measured for every added
  // candidate, rather than using a sampled bitrate proxy.
  if (settings.correction_mode != 0 &&
      (best.base_mode == 0 || best.base_mode == 2)) {
    const auto allocation = best.base_mode == 2
        ? qpet_correlation::BoundAllocation::gradient_balanced
        : qpet_correlation::BoundAllocation::uniform;
    auto analyzer = make_qoi(
        original,
        settings,
        settings.pointwise_tolerance,
        best.selected_estimator_scale,
        allocation);
    auto sorted = analyzer->local_bounds();
    std::sort(sorted.begin(), sorted.end());
    const auto upper = std::upper_bound(
        sorted.begin(), sorted.end(), best.selected_global_bound);
    const auto selected_count = static_cast<std::size_t>(
        std::distance(sorted.begin(), upper));
    if (selected_count > 1 &&
        static_cast<double>(selected_count) / sorted.size() <= 0.005) {
      constexpr double termination_ratio = 0.95;
      const auto initial_bound = best.selected_global_bound;
      for (auto index = selected_count - 1; index-- > 0;) {
        const auto slope = termination_ratio +
            (static_cast<double>(index) / static_cast<double>(selected_count - 1)) *
                (1.0 - termination_ratio);
        const auto global_bound = sorted[index];
        if (!(global_bound >= slope * initial_bound))
          break;
        if (std::abs(global_bound - initial_bound) <=
            32.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, initial_bound))
          continue;
        (void)try_retain_candidate(
            [&] {
              return compress_candidate(
                  original,
                  settings,
                  global_bound,
                  best.selected_estimator_scale,
                  best.base_mode);
            },
            retain_if_smaller,
            top_level_candidate_failures);
      }
    }
  }
  if (top_level_candidate_failures >
      std::numeric_limits<std::size_t>::max() - tested)
    throw std::overflow_error("QPET candidate attempt count overflow");
  tested += top_level_candidate_failures;
  total_candidate_failures = checked_add_size(
      total_candidate_failures,
      top_level_candidate_failures,
      "QPET backend candidate failure count overflow");

  // FVRO/ZRCAR are second-stage encoder searches. The complete frozen legacy
  // portfolio (including the legacy-winner-driven Algorithm-4 tail above)
  // has already run. Only its top-K stored base records are replayed, without
  // invoking an official base encoder again. The legacy whole-window winner
  // remains the incumbent on every byte tie.
  const auto legacy_best = best;
  if (uses_second_stage(settings.repair_optimizer)) {
    select_zrcar_base_records(
        second_stage_base_records, legacy_best.encoded_size());
    // Freeze the incumbent method's exact top-K before adding any CJRF base.
    // Relaxed bases are appended after this point and can neither evict nor
    // reorder a candidate from the complete pre-CJRF method.
    const auto protected_base_record_count = second_stage_base_records.size();
    auto cjrf_work = CjrfRelaxedWork();
    auto cjrf_protected_incumbent = std::optional<EncodedWindow>();
    auto cjrf_suffix_appended = false;
    auto cfsr_work = CfsrSurrogateWork();
    auto cfsr_protected_incumbent = std::optional<EncodedWindow>();
    auto cfsr_suffix_appended = false;
    auto zrcar_candidates = std::size_t{0};
    auto zrcar_prefix_candidates = std::size_t{0};
    auto zrcar_local_candidates = std::size_t{0};
    auto zrcar_legacy_seeds = std::size_t{0};
    auto zrcar_fvro_seeds = std::size_t{0};
    auto zrcar_states_generated = std::size_t{0};
    auto zrcar_infeasible_states = std::size_t{0};
    auto zrcar_infeasible_frontier_expansions = std::size_t{0};
    auto zrcar_infeasible_frontier_recoveries = std::size_t{0};
    auto fvro_global_generated = std::size_t{0};
    auto fvro_global_serialized = std::size_t{0};
    auto fvro_global_infeasible = std::size_t{0};
    auto fvro_channel_generated = std::size_t{0};
    auto fvro_channel_serialized = std::size_t{0};
    auto fvro_channel_infeasible = std::size_t{0};
    auto fvro_local_generated = std::size_t{0};
    auto fvro_local_infeasible = std::size_t{0};
    auto fvro_selected_channels = std::size_t{0};
    auto fvro_strict_channel_steps = std::size_t{0};
    auto fvro_seeds_retained = std::size_t{0};
    auto ssbb_base_records = std::size_t{0};
    auto ssbb_channels_searched = std::size_t{0};
    auto ssbb_positions_processed = std::size_t{0};
    auto ssbb_actions_generated = std::size_t{0};
    auto ssbb_states_generated = std::size_t{0};
    auto ssbb_states_deduplicated = std::size_t{0};
    auto ssbb_statistics_feasible = std::size_t{0};
    auto ssbb_statistics_infeasible = std::size_t{0};
    auto ssbb_feasible_retained = std::size_t{0};
    auto ssbb_infeasible_retained = std::size_t{0};
    auto ssbb_states_pruned = std::size_t{0};
    auto ssbb_terminal_states = std::size_t{0};
    auto ssbb_terminal_feasible = std::size_t{0};
    auto ssbb_terminal_infeasible = std::size_t{0};
    auto ssbb_outputs_returned = std::size_t{0};
    auto ssbb_candidates_serialized = std::size_t{0};
    auto ssbb_seeds_retained = std::size_t{0};
    auto ssbb_selected_channels = std::size_t{0};
    auto ssbb_representation_candidates_serialized = std::size_t{0};
    auto ssbb_representation_improved_starts = std::size_t{0};
    auto ssbb_representation_gross_saved_bytes = std::size_t{0};
    auto ssbb_composition_starts = std::size_t{0};
    auto ssbb_composition_local_options = std::size_t{0};
    auto ssbb_composition_transitions_generated = std::size_t{0};
    auto ssbb_composition_transitions_deduplicated = std::size_t{0};
    auto ssbb_composition_direct_feasible = std::size_t{0};
    auto ssbb_composition_direct_infeasible = std::size_t{0};
    auto ssbb_composition_candidates_serialized = std::size_t{0};
    auto ssbb_composition_singletons_serialized = std::size_t{0};
    auto ssbb_composition_pairs_serialized = std::size_t{0};
    auto ssbb_composition_frontier_carries = std::size_t{0};
    auto ssbb_composition_anchor_retention_events = std::size_t{0};
    auto ssbb_composition_format_retention_events = std::size_t{0};
    auto ssbb_composition_rate_retention_events = std::size_t{0};
    auto ssbb_composition_states_pruned = std::size_t{0};
    auto ssbb_composition_max_frontier = std::size_t{0};
    auto ssbb_winner_reference_bytes = legacy_best.encoded_size();
    auto ssbb_winner_search_reference_bytes = legacy_best.encoded_size();
    auto legacy_fvro_reference_quantized = false;
    auto legacy_fvro_reference_exact_escapes = std::uint32_t{0};
    auto legacy_fvro_reference_quantized_format = std::uint8_t{0};
    auto captured_legacy_fvro_reference = false;
    for (std::size_t record_index = 0;; ++record_index) {
      // The complete incumbent method must finish before any relaxed base is
      // generated.  Snapshot it at the exact prefix boundary, then append the
      // untruncated CJRF suffix.  Taking the record reference only afterwards
      // avoids retaining a vector reference across possible reallocation.
      if (!cjrf_suffix_appended &&
          record_index == protected_base_record_count) {
        if (uses_cjrf(settings.base_frontier)) {
          cjrf_protected_incumbent = best;
          append_cjrf_relaxed_base_records(
              original,
              settings,
              compact_record_bytes(*cjrf_protected_incumbent),
              second_stage_base_records,
              successful_legacy_ordinal,
              cjrf_work);
          const auto nonretained_attempts = checked_add_size(
              checked_add_size(
                  cjrf_work.failures,
                  cjrf_work.pruned,
                  "CJRF nonretained-attempt total overflow"),
              cjrf_work.duplicates,
              "CJRF nonretained-attempt total overflow");
          tested = checked_add_size(
              tested,
              checked_add_size(
                  nonretained_attempts,
                  cjrf_work.legacy_repair_candidates,
                  "CJRF initial candidate-attempt total overflow"),
              "QPET CJRF candidate attempt count overflow");
          total_candidate_failures = checked_add_size(
              total_candidate_failures,
              cjrf_work.failures,
              "QPET CJRF backend failure count overflow");
        }
        cjrf_suffix_appended = true;
      }
      // CFSR is deliberately a depth-one suffix.  Only after the complete
      // protected method has exhausted every stored base do we replay its
      // exact record into one immutable feasible surrogate and append the
      // fixed official-codec schedule.  The one-shot flag prevents any CFSR
      // winner from recursively becoming a depth-two source.
      if (!cfsr_suffix_appended &&
          record_index == second_stage_base_records.size()) {
        if (uses_cfsr(settings.surrogate_frontier)) {
          cfsr_protected_incumbent = best;
          const auto surrogate = canonical_cfsr_surrogate_replay(
              original, settings, *cfsr_protected_incumbent);
          append_cfsr_surrogate_base_records(
              original,
              surrogate,
              settings,
              compact_record_bytes(*cfsr_protected_incumbent),
              second_stage_base_records,
              successful_legacy_ordinal,
              cfsr_work);
          const auto nonretained_attempts = checked_add_size(
              checked_add_size(
                  cfsr_work.failures,
                  cfsr_work.pruned,
                  "CFSR nonretained-attempt total overflow"),
              cfsr_work.duplicates,
              "CFSR nonretained-attempt total overflow");
          tested = checked_add_size(
              tested,
              checked_add_size(
                  nonretained_attempts,
                  cfsr_work.legacy_repair_candidates,
                  "CFSR initial candidate-attempt total overflow"),
              "QPET CFSR candidate attempt count overflow");
          total_candidate_failures = checked_add_size(
              total_candidate_failures,
              cfsr_work.failures,
              "QPET CFSR backend failure count overflow");
        }
        cfsr_suffix_appended = true;
      }
      if (record_index >= second_stage_base_records.size())
        break;
      const auto& record = second_stage_base_records[record_index];
      auto optimized = optimize_stored_legacy_candidate(
          original, settings, record.encoded);
      if (uses_fvro(settings.repair_optimizer) &&
          same_stored_base_identity(record.encoded, legacy_best)) {
        if (captured_legacy_fvro_reference)
          throw std::logic_error(
              "FVRO captured the frozen legacy base reference twice");
        legacy_fvro_reference_quantized =
            optimized.fvro_reference_quantized;
        legacy_fvro_reference_exact_escapes =
            optimized.fvro_reference_exact_escapes;
        legacy_fvro_reference_quantized_format =
            optimized.fvro_reference_quantized_format;
        captured_legacy_fvro_reference = true;
      }
      tested = checked_add_size(
          tested,
          optimized.repair_candidates_tested,
          "QPET FVRO/ZRCAR candidate attempt count overflow");
      zrcar_candidates = checked_add_size(
          zrcar_candidates,
          optimized.zrcar_candidates_tested,
          "ZRCAR serialized candidate total overflow");
      zrcar_prefix_candidates = checked_add_size(
          zrcar_prefix_candidates,
          optimized.zrcar_prefix_candidates_tested,
          "ZRCAR prefix candidate total overflow");
      zrcar_local_candidates = checked_add_size(
          zrcar_local_candidates,
          optimized.zrcar_local_candidates_tested,
          "ZRCAR local candidate total overflow");
      zrcar_legacy_seeds = checked_add_size(
          zrcar_legacy_seeds,
          optimized.zrcar_legacy_seeds_selected,
          "ZRCAR frozen-seed total overflow");
      zrcar_fvro_seeds = checked_add_size(
          zrcar_fvro_seeds,
          optimized.zrcar_fvro_seeds_selected,
          "ZRCAR FVRO-seed total overflow");
      zrcar_states_generated = checked_add_size(
          zrcar_states_generated,
          optimized.zrcar_states_generated,
          "ZRCAR generated-state total overflow");
      zrcar_infeasible_states = checked_add_size(
          zrcar_infeasible_states,
          optimized.zrcar_infeasible_states,
          "ZRCAR infeasible-state total overflow");
      zrcar_infeasible_frontier_expansions = checked_add_size(
          zrcar_infeasible_frontier_expansions,
          optimized.zrcar_infeasible_frontier_expansions,
          "ZRCAR infeasible-frontier expansion total overflow");
      zrcar_infeasible_frontier_recoveries = checked_add_size(
          zrcar_infeasible_frontier_recoveries,
          optimized.zrcar_infeasible_frontier_recoveries,
          "ZRCAR infeasible-frontier recovery total overflow");
      fvro_global_generated = checked_add_size(
          fvro_global_generated,
          optimized.fvro_global_candidates_generated,
          "FVRO global generated total overflow");
      fvro_global_serialized = checked_add_size(
          fvro_global_serialized,
          optimized.fvro_global_candidates_serialized,
          "FVRO global serialized total overflow");
      fvro_global_infeasible = checked_add_size(
          fvro_global_infeasible,
          optimized.fvro_global_candidates_infeasible,
          "FVRO global infeasible total overflow");
      fvro_channel_generated = checked_add_size(
          fvro_channel_generated,
          optimized.fvro_channel_candidates_generated,
          "FVRO channel generated total overflow");
      fvro_channel_serialized = checked_add_size(
          fvro_channel_serialized,
          optimized.fvro_channel_candidates_serialized,
          "FVRO channel serialized total overflow");
      fvro_channel_infeasible = checked_add_size(
          fvro_channel_infeasible,
          optimized.fvro_channel_candidates_infeasible,
          "FVRO channel infeasible total overflow");
      fvro_local_generated = checked_add_size(
          fvro_local_generated,
          optimized.fvro_local_proposals_generated,
          "FVRO local generated total overflow");
      fvro_local_infeasible = checked_add_size(
          fvro_local_infeasible,
          optimized.fvro_local_proposals_infeasible,
          "FVRO local infeasible total overflow");
      fvro_selected_channels = checked_add_size(
          fvro_selected_channels,
          optimized.fvro_selected_channels,
          "FVRO selected-channel total overflow");
      fvro_strict_channel_steps = checked_add_size(
          fvro_strict_channel_steps,
          optimized.fvro_strict_channel_steps,
          "FVRO strict channel-step total overflow");
      fvro_seeds_retained = checked_add_size(
          fvro_seeds_retained,
          optimized.fvro_seeds_retained,
          "FVRO retained-seed total overflow");
      ssbb_base_records = checked_add_size(
          ssbb_base_records,
          optimized.ssbb_base_records_tested,
          "SSBB base-record total overflow");
      ssbb_channels_searched = checked_add_size(
          ssbb_channels_searched,
          optimized.ssbb_channels_searched,
          "SSBB searched-channel total overflow");
      ssbb_positions_processed = checked_add_size(
          ssbb_positions_processed,
          optimized.ssbb_positions_processed,
          "SSBB processed-position total overflow");
      ssbb_actions_generated = checked_add_size(
          ssbb_actions_generated,
          optimized.ssbb_actions_generated,
          "SSBB generated-action total overflow");
      ssbb_states_generated = checked_add_size(
          ssbb_states_generated,
          optimized.ssbb_states_generated,
          "SSBB generated-state total overflow");
      ssbb_states_deduplicated = checked_add_size(
          ssbb_states_deduplicated,
          optimized.ssbb_states_deduplicated,
          "SSBB deduplicated-state total overflow");
      ssbb_statistics_feasible = checked_add_size(
          ssbb_statistics_feasible,
          optimized.ssbb_statistics_feasible_states,
          "SSBB statistics-feasible total overflow");
      ssbb_statistics_infeasible = checked_add_size(
          ssbb_statistics_infeasible,
          optimized.ssbb_statistics_infeasible_states,
          "SSBB statistics-infeasible total overflow");
      ssbb_feasible_retained = checked_add_size(
          ssbb_feasible_retained,
          optimized.ssbb_feasible_states_retained,
          "SSBB feasible-retained total overflow");
      ssbb_infeasible_retained = checked_add_size(
          ssbb_infeasible_retained,
          optimized.ssbb_infeasible_states_retained,
          "SSBB infeasible-retained total overflow");
      ssbb_states_pruned = checked_add_size(
          ssbb_states_pruned,
          optimized.ssbb_states_pruned_by_beam,
          "SSBB beam-pruned total overflow");
      ssbb_terminal_states = checked_add_size(
          ssbb_terminal_states,
          optimized.ssbb_terminal_states,
          "SSBB terminal-state total overflow");
      ssbb_terminal_feasible = checked_add_size(
          ssbb_terminal_feasible,
          optimized.ssbb_terminal_direct_feasible,
          "SSBB terminal-feasible total overflow");
      ssbb_terminal_infeasible = checked_add_size(
          ssbb_terminal_infeasible,
          optimized.ssbb_terminal_direct_infeasible,
          "SSBB terminal-infeasible total overflow");
      ssbb_outputs_returned = checked_add_size(
          ssbb_outputs_returned,
          optimized.ssbb_outputs_returned,
          "SSBB output total overflow");
      ssbb_candidates_serialized = checked_add_size(
          ssbb_candidates_serialized,
          optimized.ssbb_candidates_serialized,
          "SSBB serializer total overflow");
      ssbb_seeds_retained = checked_add_size(
          ssbb_seeds_retained,
          optimized.ssbb_seeds_retained,
          "SSBB retained-seed total overflow");
      ssbb_selected_channels = checked_add_size(
          ssbb_selected_channels,
          optimized.ssbb_selected_channels,
          "SSBB selected-channel total overflow");
      ssbb_representation_candidates_serialized = checked_add_size(
          ssbb_representation_candidates_serialized,
          optimized.ssbb_representation_candidates_serialized,
          "SSBB representation-candidate total overflow");
      ssbb_representation_improved_starts = checked_add_size(
          ssbb_representation_improved_starts,
          optimized.ssbb_representation_improved_starts,
          "SSBB representation-win total overflow");
      ssbb_representation_gross_saved_bytes = checked_add_size(
          ssbb_representation_gross_saved_bytes,
          optimized.ssbb_representation_gross_saved_bytes,
          "SSBB representation saved-byte total overflow");
      ssbb_composition_starts = checked_add_size(
          ssbb_composition_starts,
          optimized.ssbb_composition_starts,
          "SSBB composition-start total overflow");
      ssbb_composition_local_options = checked_add_size(
          ssbb_composition_local_options,
          optimized.ssbb_composition_local_options,
          "SSBB composition-option total overflow");
      ssbb_composition_transitions_generated = checked_add_size(
          ssbb_composition_transitions_generated,
          optimized.ssbb_composition_transitions_generated,
          "SSBB composition-transition total overflow");
      ssbb_composition_transitions_deduplicated = checked_add_size(
          ssbb_composition_transitions_deduplicated,
          optimized.ssbb_composition_transitions_deduplicated,
          "SSBB composition-duplicate total overflow");
      ssbb_composition_direct_feasible = checked_add_size(
          ssbb_composition_direct_feasible,
          optimized.ssbb_composition_direct_feasible,
          "SSBB composition-feasible total overflow");
      ssbb_composition_direct_infeasible = checked_add_size(
          ssbb_composition_direct_infeasible,
          optimized.ssbb_composition_direct_infeasible,
          "SSBB composition-infeasible total overflow");
      ssbb_composition_candidates_serialized = checked_add_size(
          ssbb_composition_candidates_serialized,
          optimized.ssbb_composition_candidates_serialized,
          "SSBB composition-serializer total overflow");
      ssbb_composition_singletons_serialized = checked_add_size(
          ssbb_composition_singletons_serialized,
          optimized.ssbb_composition_singletons_serialized,
          "SSBB composition-singleton total overflow");
      ssbb_composition_pairs_serialized = checked_add_size(
          ssbb_composition_pairs_serialized,
          optimized.ssbb_composition_pairs_serialized,
          "SSBB composition-pair total overflow");
      ssbb_composition_frontier_carries = checked_add_size(
          ssbb_composition_frontier_carries,
          optimized.ssbb_composition_frontier_carries,
          "SSBB composition-carry total overflow");
      ssbb_composition_anchor_retention_events = checked_add_size(
          ssbb_composition_anchor_retention_events,
          optimized.ssbb_composition_anchor_retention_events,
          "SSBB composition-anchor total overflow");
      ssbb_composition_format_retention_events = checked_add_size(
          ssbb_composition_format_retention_events,
          optimized.ssbb_composition_format_retention_events,
          "SSBB composition-format total overflow");
      ssbb_composition_rate_retention_events = checked_add_size(
          ssbb_composition_rate_retention_events,
          optimized.ssbb_composition_rate_retention_events,
          "SSBB composition-rate total overflow");
      ssbb_composition_states_pruned = checked_add_size(
          ssbb_composition_states_pruned,
          optimized.ssbb_composition_states_pruned,
          "SSBB composition-pruned total overflow");
      ssbb_composition_max_frontier = std::max(
          ssbb_composition_max_frontier,
          static_cast<std::size_t>(optimized.ssbb_composition_max_frontier));
      // Strict total-byte improvement only. This deliberately does not reuse
      // the legacy larger-bound tie breaker: a whole-window tie must retain
      // the byte-identical legacy incumbent.  If a later stored base wins,
      // its SSBB incremental saving is the deterministic selection-path
      // marginal against the incumbent at this step.  It is deliberately not
      // a counterfactual against a separately optimized best-without-SSBB
      // portfolio; total SSBB/second-stage savings below remain measured from
      // the frozen legacy winner.
      if (optimized.encoded_size() < best.encoded_size())
        realize_ssbb_winner_against_incumbent(
            optimized, best.encoded_size(), false);
      if (retain_strict_zrcar_whole_window(std::move(optimized), best)) {
        ssbb_winner_reference_bytes = best.ssbb_reference_bytes;
        ssbb_winner_search_reference_bytes =
            best.ssbb_search_reference_bytes;
      }
    }
    if (uses_cjrf(settings.base_frontier)) {
      if (!cjrf_suffix_appended || !cjrf_protected_incumbent.has_value())
        throw std::logic_error(
            "CJRF did not freeze the complete incumbent method");
      const auto protected_bytes =
          compact_record_bytes(*cjrf_protected_incumbent);
      const auto final_bytes = compact_record_bytes(best);
      if (final_bytes > protected_bytes)
        throw std::logic_error("CJRF regressed its protected incumbent");
      if (final_bytes == protected_bytes &&
          !same_persisted_record_and_reconstruction(
              best, *cjrf_protected_incumbent))
        throw std::logic_error(
            "CJRF replaced a byte tie instead of retaining its incumbent");
      best.cjrf_reference_bytes = protected_bytes;
      best.cjrf_selected = final_bytes < protected_bytes;
      if (best.cjrf_selected) {
        if (!best.cjrf_relaxed_base)
          throw std::logic_error(
              "CJRF strict winner is not identified as a relaxed base");
        best.cjrf_winner_kind = best.ssbb_selected
            ? CjrfWinnerKind::ssbb
            : best.zrcar_selected
                ? CjrfWinnerKind::zrcar
                : best.fvro_selected
                    ? CjrfWinnerKind::fvro
                    : CjrfWinnerKind::legacy;
      }
      else {
        if (best.cjrf_relaxed_base)
          throw std::logic_error(
              "CJRF retained incumbent is marked as a relaxed base");
        best.cjrf_winner_kind = CjrfWinnerKind::none;
      }
    }
    if (uses_cfsr(settings.surrogate_frontier)) {
      if (!cfsr_suffix_appended || !cfsr_protected_incumbent.has_value())
        throw std::logic_error(
            "CFSR did not freeze and replay the complete incumbent method");
      const auto protected_bytes =
          compact_record_bytes(*cfsr_protected_incumbent);
      const auto final_bytes = compact_record_bytes(best);
      if (final_bytes > protected_bytes)
        throw std::logic_error("CFSR regressed its protected incumbent");
      if (final_bytes == protected_bytes &&
          !same_persisted_record_and_reconstruction(
              best, *cfsr_protected_incumbent))
        throw std::logic_error(
            "CFSR replaced a byte tie instead of retaining its incumbent");
      best.cfsr_reference_bytes = protected_bytes;
      best.cfsr_selected = final_bytes < protected_bytes;
      if (best.cfsr_selected) {
        if (!best.cfsr_surrogate_base)
          throw std::logic_error(
              "CFSR strict winner is not identified as a surrogate base");
        best.cfsr_winner_kind = best.ssbb_selected
            ? CfsrWinnerKind::ssbb
            : best.zrcar_selected
                ? CfsrWinnerKind::zrcar
                : best.fvro_selected
                    ? CfsrWinnerKind::fvro
                    : CfsrWinnerKind::legacy;
      }
      else {
        if (best.cfsr_surrogate_base)
          throw std::logic_error(
              "CFSR retained incumbent is marked as a surrogate base");
        best.cfsr_winner_kind = CfsrWinnerKind::none;
      }
    }
    if (best.encoded_size() > legacy_best.encoded_size())
      throw std::logic_error(
          "FVRO/ZRCAR/SSBB regressed the independent legacy winner");
    if (uses_fvro(settings.repair_optimizer) && legacy_best.corrections != 0 &&
        !captured_legacy_fvro_reference)
      throw std::logic_error(
          "FVRO did not retain the frozen legacy winner among its top-K bases");
    const auto assign_u32 = [](std::size_t value, const char* message) {
      if (value > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error(message);
      return static_cast<std::uint32_t>(value);
    };
    best.cjrf_relaxed_base_attempts = assign_u32(
        cjrf_work.attempts,
        "CJRF relaxed attempt total exceeds uint32 diagnostics");
    best.cjrf_relaxed_base_failures = assign_u32(
        cjrf_work.failures,
        "CJRF relaxed failure total exceeds uint32 diagnostics");
    best.cjrf_relaxed_base_pruned = assign_u32(
        cjrf_work.pruned,
        "CJRF relaxed prune total exceeds uint32 diagnostics");
    best.cjrf_relaxed_base_duplicates = assign_u32(
        cjrf_work.duplicates,
        "CJRF relaxed duplicate total exceeds uint32 diagnostics");
    best.cjrf_relaxed_base_records = assign_u32(
        cjrf_work.records,
        "CJRF relaxed record total exceeds uint32 diagnostics");
    best.cjrf_relaxed_base_outside_pointwise = assign_u32(
        cjrf_work.outside_pointwise,
        "CJRF outside-pointwise total exceeds uint32 diagnostics");
    best.cjrf_relaxed_legacy_repaired = assign_u32(
        cjrf_work.legacy_repaired,
        "CJRF legacy-repaired total exceeds uint32 diagnostics");
    best.cjrf_relaxed_legacy_repair_candidates = assign_u32(
        cjrf_work.legacy_repair_candidates,
        "CJRF legacy repair-candidate total exceeds uint32 diagnostics");
    best.cfsr_surrogate_sources = assign_u32(
        cfsr_work.sources,
        "CFSR surrogate-source total exceeds uint32 diagnostics");
    best.cfsr_surrogate_base_attempts = assign_u32(
        cfsr_work.attempts,
        "CFSR surrogate attempt total exceeds uint32 diagnostics");
    best.cfsr_surrogate_base_failures = assign_u32(
        cfsr_work.failures,
        "CFSR surrogate failure total exceeds uint32 diagnostics");
    best.cfsr_surrogate_base_pruned = assign_u32(
        cfsr_work.pruned,
        "CFSR surrogate prune total exceeds uint32 diagnostics");
    best.cfsr_surrogate_base_duplicates = assign_u32(
        cfsr_work.duplicates,
        "CFSR surrogate duplicate total exceeds uint32 diagnostics");
    best.cfsr_surrogate_base_records = assign_u32(
        cfsr_work.records,
        "CFSR surrogate record total exceeds uint32 diagnostics");
    best.cfsr_surrogate_base_outside_original_pointwise = assign_u32(
        cfsr_work.outside_original_pointwise,
        "CFSR outside-original-pointwise total exceeds uint32 diagnostics");
    best.cfsr_surrogate_legacy_repaired = assign_u32(
        cfsr_work.legacy_repaired,
        "CFSR legacy-repaired total exceeds uint32 diagnostics");
    best.cfsr_surrogate_legacy_repair_candidates = assign_u32(
        cfsr_work.legacy_repair_candidates,
        "CFSR legacy repair-candidate total exceeds uint32 diagnostics");
    best.zrcar_candidates_tested = assign_u32(
        zrcar_candidates, "ZRCAR serialized total exceeds uint32 diagnostics");
    best.zrcar_prefix_candidates_tested = assign_u32(
        zrcar_prefix_candidates,
        "ZRCAR prefix total exceeds uint32 diagnostics");
    best.zrcar_local_candidates_tested = assign_u32(
        zrcar_local_candidates,
        "ZRCAR local total exceeds uint32 diagnostics");
    best.zrcar_legacy_seeds_selected = assign_u32(
        zrcar_legacy_seeds,
        "ZRCAR frozen-seed total exceeds uint32 diagnostics");
    best.zrcar_fvro_seeds_selected = assign_u32(
        zrcar_fvro_seeds,
        "ZRCAR FVRO-seed total exceeds uint32 diagnostics");
    best.zrcar_states_generated = assign_u32(
        zrcar_states_generated,
        "ZRCAR generated-state total exceeds uint32 diagnostics");
    best.zrcar_infeasible_states = assign_u32(
        zrcar_infeasible_states,
        "ZRCAR infeasible-state total exceeds uint32 diagnostics");
    best.zrcar_infeasible_frontier_expansions = assign_u32(
        zrcar_infeasible_frontier_expansions,
        "ZRCAR frontier-expansion total exceeds uint32 diagnostics");
    best.zrcar_infeasible_frontier_recoveries = assign_u32(
        zrcar_infeasible_frontier_recoveries,
        "ZRCAR frontier-recovery total exceeds uint32 diagnostics");
    best.zrcar_base_records_tested = assign_u32(
        uses_zrcar(settings.repair_optimizer)
            ? second_stage_base_records.size()
            : std::size_t{0},
        "ZRCAR base-record total exceeds uint32 diagnostics");
    if (best.zrcar_legacy_seeds_selected >
            best.zrcar_base_records_tested * kZrcarSeedCount ||
        best.zrcar_fvro_seeds_selected >
            best.zrcar_base_records_tested *
                (uses_fvro(settings.repair_optimizer) &&
                         uses_zrcar(settings.repair_optimizer)
                     ? kFvroAdditionalZrcarSeedCount
                     : 0U))
      throw std::logic_error(
          "ZRCAR window-level selected-seed budget was exceeded");
    best.zrcar_legacy_reference_bytes = uses_zrcar(settings.repair_optimizer)
        ? legacy_best.encoded_size()
        : std::size_t{0};
    best.fvro_global_candidates_generated = assign_u32(
        fvro_global_generated,
        "FVRO global generated total exceeds uint32 diagnostics");
    best.fvro_global_candidates_serialized = assign_u32(
        fvro_global_serialized,
        "FVRO global serialized total exceeds uint32 diagnostics");
    best.fvro_global_candidates_infeasible = assign_u32(
        fvro_global_infeasible,
        "FVRO global infeasible total exceeds uint32 diagnostics");
    best.fvro_channel_candidates_generated = assign_u32(
        fvro_channel_generated,
        "FVRO channel generated total exceeds uint32 diagnostics");
    best.fvro_channel_candidates_serialized = assign_u32(
        fvro_channel_serialized,
        "FVRO channel serialized total exceeds uint32 diagnostics");
    best.fvro_channel_candidates_infeasible = assign_u32(
        fvro_channel_infeasible,
        "FVRO channel infeasible total exceeds uint32 diagnostics");
    best.fvro_local_proposals_generated = assign_u32(
        fvro_local_generated,
        "FVRO local generated total exceeds uint32 diagnostics");
    best.fvro_local_proposals_infeasible = assign_u32(
        fvro_local_infeasible,
        "FVRO local infeasible total exceeds uint32 diagnostics");
    best.fvro_selected_channels = assign_u32(
        fvro_selected_channels,
        "FVRO selected-channel total exceeds uint32 diagnostics");
    best.fvro_strict_channel_steps = assign_u32(
        fvro_strict_channel_steps,
        "FVRO strict channel-step total exceeds uint32 diagnostics");
    if (best.fvro_selected_channels > best.fvro_strict_channel_steps ||
        best.fvro_strict_channel_steps >
            best.fvro_channel_candidates_serialized ||
        ((best.fvro_selected_channels == 0) !=
         (best.fvro_strict_channel_steps == 0)))
      throw std::logic_error(
          "FVRO window-level strict channel-step accounting diverged");
    best.fvro_seeds_retained = assign_u32(
        fvro_seeds_retained,
        "FVRO retained-seed total exceeds uint32 diagnostics");
    best.fvro_base_records_tested = assign_u32(
        uses_fvro(settings.repair_optimizer)
            ? second_stage_base_records.size()
            : std::size_t{0},
        "FVRO base-record total exceeds uint32 diagnostics");
    best.fvro_legacy_reference_bytes = uses_fvro(settings.repair_optimizer)
        ? legacy_best.encoded_size()
        : std::size_t{0};
    best.ssbb_base_records_tested = assign_u32(
        ssbb_base_records,
        "SSBB base-record total exceeds uint32 diagnostics");
    best.ssbb_channels_searched = assign_u32(
        ssbb_channels_searched,
        "SSBB searched-channel total exceeds uint32 diagnostics");
    best.ssbb_positions_processed = assign_u32(
        ssbb_positions_processed,
        "SSBB processed-position total exceeds uint32 diagnostics");
    best.ssbb_actions_generated = assign_u32(
        ssbb_actions_generated,
        "SSBB generated-action total exceeds uint32 diagnostics");
    best.ssbb_states_generated = assign_u32(
        ssbb_states_generated,
        "SSBB generated-state total exceeds uint32 diagnostics");
    best.ssbb_states_deduplicated = assign_u32(
        ssbb_states_deduplicated,
        "SSBB deduplicated-state total exceeds uint32 diagnostics");
    best.ssbb_statistics_feasible_states = assign_u32(
        ssbb_statistics_feasible,
        "SSBB statistics-feasible total exceeds uint32 diagnostics");
    best.ssbb_statistics_infeasible_states = assign_u32(
        ssbb_statistics_infeasible,
        "SSBB statistics-infeasible total exceeds uint32 diagnostics");
    best.ssbb_feasible_states_retained = assign_u32(
        ssbb_feasible_retained,
        "SSBB feasible-retained total exceeds uint32 diagnostics");
    best.ssbb_infeasible_states_retained = assign_u32(
        ssbb_infeasible_retained,
        "SSBB infeasible-retained total exceeds uint32 diagnostics");
    best.ssbb_states_pruned_by_beam = assign_u32(
        ssbb_states_pruned,
        "SSBB beam-pruned total exceeds uint32 diagnostics");
    best.ssbb_terminal_states = assign_u32(
        ssbb_terminal_states,
        "SSBB terminal-state total exceeds uint32 diagnostics");
    best.ssbb_terminal_direct_feasible = assign_u32(
        ssbb_terminal_feasible,
        "SSBB terminal-feasible total exceeds uint32 diagnostics");
    best.ssbb_terminal_direct_infeasible = assign_u32(
        ssbb_terminal_infeasible,
        "SSBB terminal-infeasible total exceeds uint32 diagnostics");
    best.ssbb_outputs_returned = assign_u32(
        ssbb_outputs_returned,
        "SSBB output total exceeds uint32 diagnostics");
    best.ssbb_candidates_serialized = assign_u32(
        ssbb_candidates_serialized,
        "SSBB serializer total exceeds uint32 diagnostics");
    best.ssbb_seeds_retained = assign_u32(
        ssbb_seeds_retained,
        "SSBB retained-seed total exceeds uint32 diagnostics");
    best.ssbb_selected_channels = assign_u32(
        ssbb_selected_channels,
        "SSBB selected-channel total exceeds uint32 diagnostics");
    best.ssbb_representation_candidates_serialized = assign_u32(
        ssbb_representation_candidates_serialized,
        "SSBB representation-candidate total exceeds uint32 diagnostics");
    best.ssbb_representation_improved_starts = assign_u32(
        ssbb_representation_improved_starts,
        "SSBB representation-win total exceeds uint32 diagnostics");
    best.ssbb_representation_gross_saved_bytes =
        ssbb_representation_gross_saved_bytes;
    best.ssbb_composition_starts = assign_u32(
        ssbb_composition_starts,
        "SSBB composition-start total exceeds uint32 diagnostics");
    best.ssbb_composition_local_options = assign_u32(
        ssbb_composition_local_options,
        "SSBB composition-option total exceeds uint32 diagnostics");
    best.ssbb_composition_transitions_generated = assign_u32(
        ssbb_composition_transitions_generated,
        "SSBB composition-transition total exceeds uint32 diagnostics");
    best.ssbb_composition_transitions_deduplicated = assign_u32(
        ssbb_composition_transitions_deduplicated,
        "SSBB composition-duplicate total exceeds uint32 diagnostics");
    best.ssbb_composition_direct_feasible = assign_u32(
        ssbb_composition_direct_feasible,
        "SSBB composition-feasible total exceeds uint32 diagnostics");
    best.ssbb_composition_direct_infeasible = assign_u32(
        ssbb_composition_direct_infeasible,
        "SSBB composition-infeasible total exceeds uint32 diagnostics");
    best.ssbb_composition_candidates_serialized = assign_u32(
        ssbb_composition_candidates_serialized,
        "SSBB composition-serializer total exceeds uint32 diagnostics");
    best.ssbb_composition_singletons_serialized = assign_u32(
        ssbb_composition_singletons_serialized,
        "SSBB composition-singleton total exceeds uint32 diagnostics");
    best.ssbb_composition_pairs_serialized = assign_u32(
        ssbb_composition_pairs_serialized,
        "SSBB composition-pair total exceeds uint32 diagnostics");
    best.ssbb_composition_frontier_carries = assign_u32(
        ssbb_composition_frontier_carries,
        "SSBB composition-carry total exceeds uint32 diagnostics");
    best.ssbb_composition_anchor_retention_events = assign_u32(
        ssbb_composition_anchor_retention_events,
        "SSBB composition-anchor total exceeds uint32 diagnostics");
    best.ssbb_composition_format_retention_events = assign_u32(
        ssbb_composition_format_retention_events,
        "SSBB composition-format total exceeds uint32 diagnostics");
    best.ssbb_composition_rate_retention_events = assign_u32(
        ssbb_composition_rate_retention_events,
        "SSBB composition-rate total exceeds uint32 diagnostics");
    best.ssbb_composition_states_pruned = assign_u32(
        ssbb_composition_states_pruned,
        "SSBB composition-pruned total exceeds uint32 diagnostics");
    best.ssbb_composition_max_frontier = assign_u32(
        ssbb_composition_max_frontier,
        "SSBB composition-frontier total exceeds uint32 diagnostics");
    best.ssbb_reference_bytes = uses_ssbb(settings.repair_optimizer)
        ? ssbb_winner_reference_bytes
        : std::size_t{0};
    best.ssbb_search_reference_bytes = uses_ssbb(settings.repair_optimizer)
        ? ssbb_winner_search_reference_bytes
        : std::size_t{0};
    if (uses_ssbb(settings.repair_optimizer)) {
      const auto certified_starts_per_base =
          uses_full_ssbb(settings.repair_optimizer)
          ? kSsbbFullSearchStartsPerBase
          : std::size_t{1};
      const auto ceiling_paths_per_base =
          certified_starts_per_base *
          correction_format_ceiling_path_count(
              settings.correction_format_ceiling);
      const auto maximum_channels_per_base =
          ceiling_paths_per_base * kSsbbMaximumDifficultChannels;
      const auto maximum_serialized_per_base =
          ceiling_paths_per_base * kSsbbMaximumSerializedPerBase;
      const auto maximum_seeds_per_base = ceiling_paths_per_base;
      const auto composition_starts_per_base = ceiling_paths_per_base;
      if (best.ssbb_base_records_tested != second_stage_base_records.size() ||
          best.ssbb_channels_searched >
              best.ssbb_base_records_tested * maximum_channels_per_base ||
          best.ssbb_positions_processed !=
              best.ssbb_channels_searched * settings.time ||
          best.ssbb_actions_generated < best.ssbb_positions_processed ||
          best.ssbb_actions_generated >
              best.ssbb_positions_processed *
                  qpet_correlation::ssbb::kMaximumActionsPerPosition ||
          best.ssbb_candidates_serialized >
              best.ssbb_base_records_tested * maximum_serialized_per_base ||
          best.ssbb_composition_starts !=
              best.ssbb_base_records_tested * composition_starts_per_base ||
          best.ssbb_composition_transitions_generated !=
              best.ssbb_composition_transitions_deduplicated +
                  best.ssbb_composition_direct_feasible +
                  best.ssbb_composition_direct_infeasible ||
          best.ssbb_composition_direct_feasible !=
              best.ssbb_composition_candidates_serialized ||
          best.ssbb_composition_singletons_serialized !=
              best.ssbb_composition_local_options ||
          best.ssbb_composition_frontier_carries +
                  best.ssbb_composition_direct_feasible !=
              best.ssbb_composition_anchor_retention_events +
                  best.ssbb_composition_format_retention_events +
                  best.ssbb_composition_rate_retention_events +
                  best.ssbb_composition_states_pruned ||
          best.ssbb_composition_candidates_serialized >
              best.ssbb_composition_starts *
                  kSsbbCompositionMaximumTransitionsPerStart ||
          best.ssbb_representation_candidates_serialized !=
              best.ssbb_composition_starts ||
          best.ssbb_representation_improved_starts >
              best.ssbb_representation_candidates_serialized ||
          (best.ssbb_representation_improved_starts == 0) !=
              (best.ssbb_representation_gross_saved_bytes == 0) ||
          best.ssbb_composition_max_frontier >
              kSsbbCompositionBeamWidth ||
          best.ssbb_seeds_retained >
              best.ssbb_base_records_tested * maximum_seeds_per_base ||
          best.ssbb_search_reference_bytes > best.ssbb_reference_bytes ||
          best.encoded_size() > best.ssbb_search_reference_bytes)
        throw std::logic_error("SSBB window-level budget or no-regression invariant failed");
      if ((best.encoded_size() < best.ssbb_reference_bytes) !=
          best.ssbb_selected)
        throw std::logic_error(
            "SSBB whole-window provenance disagrees with its strict byte win");
      if (best.ssbb_composition_selected) {
        if (!best.ssbb_selected ||
            best.ssbb_composition_selected_channels == 0 ||
            best.ssbb_composition_reference_bytes <= best.encoded_size())
          throw std::logic_error(
              "SSBB composition winner has inconsistent provenance");
      }
      else if (best.ssbb_composition_selected_channels != 0) {
        throw std::logic_error(
            "SSBB composition selected channels have no final winner");
      }
    }
    if (uses_fvro(settings.repair_optimizer)) {
      // Escape diagnostics compare the frozen whole-window legacy incumbent
      // with the actual final winner. They must not accidentally describe a
      // different top-K base that was explored but rejected.
      best.fvro_reference_quantized = legacy_fvro_reference_quantized;
      best.fvro_reference_exact_escapes =
          legacy_fvro_reference_exact_escapes;
      best.fvro_reference_quantized_format =
          legacy_fvro_reference_quantized_format;
      if (best.encoded_size() == legacy_best.encoded_size()) {
        best.fvro_selected = false;
        best.fvro_final_quantized = legacy_fvro_reference_quantized;
        best.fvro_final_exact_escapes =
            legacy_fvro_reference_exact_escapes;
        best.fvro_final_quantized_format =
            legacy_fvro_reference_quantized_format;
      }
      else if (!best.fvro_selected && !best.ssbb_selected &&
               !best.cjrf_selected && !best.cfsr_selected) {
        // A full-FVRO run may be won solely by ZRCAR from a frozen legacy
        // seed. The record already carries the corresponding final escape
        // summary, but it is not an FVRO-value win.
        if (!uses_zrcar(settings.repair_optimizer) || !best.zrcar_selected)
          throw std::logic_error(
              "strict second-stage winner has no FVRO/ZRCAR/SSBB provenance");
      }
    }
  }
  if (tested > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("QPET candidate count exceeds uint32 diagnostics");
  if (total_candidate_failures > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error(
        "QPET backend candidate failures exceed uint32 diagnostics");
  best.candidates_tested = static_cast<std::uint32_t>(tested);
  best.candidate_failures =
      static_cast<std::uint32_t>(total_candidate_failures);
  return best;
}

void write_stream(
    const std::string& path,
    const Settings& settings,
    const std::vector<EncodedWindow>& windows)
{
  validate_settings(settings);
  auto stream_version = kVersion;
  for (const auto& window : windows) {
    if (window.base_mode > 9)
      throw std::runtime_error("QPET writer received an unknown base mode");
    if (!settings.enable_vehicle_semantic && window.base_mode >= 8)
      throw std::runtime_error(
          "generic candidate profile cannot serialize a vehicle-semantic mode");
    if (!window.correction_payload.empty()) {
      const auto correction_format = window.correction_payload.front();
      if (correction_format > kPredictivePackedCorrectionFormat)
        throw std::runtime_error(
            "QPET writer received an unknown correction format");
      if (correction_format > settings.correction_format_ceiling)
        throw std::runtime_error(
            "QPET writer correction format exceeds the encoder ceiling");
    }
    if (!window.correction_payload.empty())
      stream_version = std::max(
          stream_version,
          correction_payload_required_stream_version(
              window.correction_payload));
  }
  auto output = std::ofstream(path, std::ios::binary);
  if (!output)
    throw std::runtime_error("cannot open output stream file: " + path);
  output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  write_scalar(output, stream_version);
  write_scalar(output, static_cast<std::uint32_t>(settings.batch));
  write_scalar(output, static_cast<std::uint32_t>(settings.channels));
  write_scalar(output, static_cast<std::uint32_t>(settings.time));
  write_scalar(output, static_cast<std::uint32_t>(settings.max_lag));
  write_scalar(output, static_cast<std::uint32_t>(settings.qoz_level));
  std::uint32_t flags = 0;
  if (settings.acf_tolerance != kDisabledConstraint)
    flags |= kHasAcf;
  if (settings.minimum_correlation != kDisabledConstraint)
    flags |= kHasPearson;
  write_scalar(output, flags);
  write_scalar(output, settings.correction_mode);
  write_scalar(output, static_cast<std::uint32_t>(settings.group_size));
  write_scalar(output, settings.pointwise_tolerance);
  write_scalar(output, settings.acf_tolerance);
  write_scalar(output, settings.minimum_correlation);
  write_scalar(output, settings.qpet_confidence);
  write_scalar(output, settings.qpet_k);
  std::size_t logical_windows_written = 0;
  for (const auto& window : windows) {
    if (window.logical_windows == 0 ||
        window.logical_windows > settings.group_size ||
        window.logical_windows > settings.batch - logical_windows_written ||
        window.logical_windows > std::numeric_limits<std::uint16_t>::max() ||
        window.base_mode > std::numeric_limits<std::uint8_t>::max() ||
        window.payload.size() > std::numeric_limits<std::uint32_t>::max() ||
        window.correction_payload.size() > std::numeric_limits<std::uint32_t>::max() ||
        !(window.selected_global_bound > 0.0) ||
        !std::isfinite(window.selected_global_bound))
      throw std::runtime_error("QPET compact window record exceeds its integer framing");
    write_scalar(output, static_cast<std::uint16_t>(window.logical_windows));
    write_scalar(output, static_cast<std::uint8_t>(window.base_mode));
    write_scalar(output, static_cast<std::uint8_t>(0));
    write_scalar(output, window.selected_global_bound);
    write_scalar(output, window.corrections);
    write_scalar(output, static_cast<std::uint32_t>(window.payload.size()));
    write_scalar(output, static_cast<std::uint32_t>(window.correction_payload.size()));
    output.write(
        reinterpret_cast<const char*>(window.payload.data()),
        static_cast<std::streamsize>(window.payload.size()));
    if (!output)
      throw std::runtime_error("failed while writing official QPET payload");
    output.write(
        reinterpret_cast<const char*>(window.correction_payload.data()),
        static_cast<std::streamsize>(window.correction_payload.size()));
    if (!output)
      throw std::runtime_error("failed while writing QPET correction payload");
    logical_windows_written += window.logical_windows;
  }
  if (logical_windows_written != settings.batch)
    throw std::runtime_error("QPET window records do not cover the declared batch");
}

std::pair<Settings, std::vector<float>> decode_stream(
    const std::string& path,
    std::vector<DecodedWindowRecord>* decoded_records = nullptr,
    std::uint32_t* decoded_stream_version = nullptr)
{
  if (decoded_records != nullptr)
    decoded_records->clear();
  auto input = std::ifstream(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open QPET correlation stream: " + path);
  const auto end_position = input.tellg();
  const auto end_offset = static_cast<std::streamoff>(end_position);
  if (end_offset < 0 ||
      static_cast<std::uintmax_t>(end_offset) >
          std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("QPET correlation stream size is not representable");
  const auto file_size = static_cast<std::size_t>(end_offset);
  input.seekg(0, std::ios::beg);
  if (!input)
    throw std::runtime_error("cannot seek QPET correlation stream");
  std::array<char, 4> magic{};
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!input || magic != kMagic)
    throw std::runtime_error("invalid QPET correlation stream magic");
  const auto stream_version = read_scalar<std::uint32_t>(input);
  if (stream_version != 8 && stream_version != 9 && stream_version != 10 &&
      stream_version != 11 && stream_version != 12 &&
      stream_version != kAdaptivePackedVersion &&
      stream_version != kFactorizedPackedVersion &&
      stream_version != kSubsetRankPackedVersion &&
      stream_version != kPredictivePackedVersion)
    throw std::runtime_error("unsupported QPET correlation stream version");
  if (decoded_stream_version != nullptr)
    *decoded_stream_version = stream_version;
  Settings settings;
  settings.batch = read_scalar<std::uint32_t>(input);
  settings.channels = read_scalar<std::uint32_t>(input);
  settings.time = read_scalar<std::uint32_t>(input);
  settings.max_lag = read_scalar<std::uint32_t>(input);
  settings.qoz_level = static_cast<int>(read_scalar<std::uint32_t>(input));
  const auto flags = read_scalar<std::uint32_t>(input);
  settings.correction_mode = read_scalar<std::uint32_t>(input);
  settings.group_size = read_scalar<std::uint32_t>(input);
  settings.pointwise_tolerance = read_scalar<double>(input);
  settings.acf_tolerance = read_scalar<double>(input);
  settings.minimum_correlation = read_scalar<double>(input);
  settings.qpet_confidence = read_scalar<double>(input);
  settings.qpet_k = read_scalar<double>(input);
  settings.estimator_scales = {1.0};
  validate_settings(settings, stream_version >= 10);
  if ((flags & ~(kHasAcf | kHasPearson)) != 0 ||
      ((flags & kHasAcf) == 0) !=
          (settings.acf_tolerance == kDisabledConstraint) ||
      ((flags & kHasPearson) == 0) !=
          (settings.minimum_correlation == kDisabledConstraint))
    throw std::runtime_error("inconsistent QPET correlation stream flags");
  if (settings.channels > std::numeric_limits<std::size_t>::max() / settings.time ||
      settings.batch >
          std::numeric_limits<std::size_t>::max() / (settings.channels * settings.time))
    throw std::runtime_error("QPET correlation stream dimensions overflow size_t");

  const auto values_per_window = settings.channels * settings.time;
  const auto total_values = settings.batch * values_per_window;
  if (total_values > kMaximumDecodedValues)
    throw std::runtime_error(
        "QPET decoded tensor exceeds the 64 MiB float32 safety policy");
  auto reconstructed = std::vector<float>(total_values);
  std::size_t decoded_windows = 0;
  auto required_stream_version = kVersion;
  while (decoded_windows < settings.batch) {
    std::uint32_t logical_windows = 0;
    std::uint32_t base_mode = 0;
    double global_bound = 0.0;
    std::uint32_t correction_count = 0;
    std::uint64_t payload_size = 0;
    std::uint64_t correction_size = 0;
    if (stream_version >= 10) {
      logical_windows = read_scalar<std::uint16_t>(input);
      base_mode = read_scalar<std::uint8_t>(input);
      if (read_scalar<std::uint8_t>(input) != 0)
        throw std::runtime_error("invalid QPET compact window flags");
      global_bound = read_scalar<double>(input);
      correction_count = read_scalar<std::uint32_t>(input);
      payload_size = read_scalar<std::uint32_t>(input);
      correction_size = read_scalar<std::uint32_t>(input);
    }
    else {
      logical_windows = read_scalar<std::uint32_t>(input);
      base_mode = read_scalar<std::uint32_t>(input);
      global_bound = read_scalar<double>(input);
      (void)read_scalar<double>(input);  // selected estimator scale metadata
      correction_count = read_scalar<std::uint32_t>(input);
      (void)read_scalar<std::uint32_t>(input);  // candidate count metadata
      payload_size = read_scalar<std::uint64_t>(input);
      correction_size = read_scalar<std::uint64_t>(input);
    }
    if (logical_windows == 0 || logical_windows > settings.group_size ||
        logical_windows > settings.batch - decoded_windows)
      throw std::runtime_error("invalid QPET logical window group size");
    if (base_mode > (stream_version == 8 ? 7U : 9U))
      throw std::runtime_error("invalid QPET base compressor mode");
    if (!(global_bound > 0.0) || !std::isfinite(global_bound))
      throw std::runtime_error("invalid QPET candidate global bound");
    if (payload_size == 0 || payload_size > std::numeric_limits<std::size_t>::max())
      throw std::runtime_error("invalid official QPET payload size");
    if (correction_size > std::numeric_limits<std::size_t>::max())
      throw std::runtime_error("invalid QPET correction payload size");
    const auto payload_bytes = static_cast<std::size_t>(payload_size);
    const auto correction_bytes = static_cast<std::size_t>(correction_size);
    const auto declared_bytes = checked_add_size(
        payload_bytes,
        correction_bytes,
        "QPET declared payload sizes overflow size_t");
    const auto data_position = input.tellg();
    const auto data_offset = static_cast<std::streamoff>(data_position);
    if (data_offset < 0 || static_cast<std::uintmax_t>(data_offset) > file_size)
      throw std::runtime_error("invalid QPET payload data offset");
    const auto remaining = file_size - static_cast<std::size_t>(data_offset);
    if (declared_bytes > remaining)
      throw std::runtime_error(
          "QPET declared payload sizes exceed remaining stream bytes");
    if (payload_bytes >
            static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) ||
        correction_bytes >
            static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
      throw std::runtime_error("QPET payload size exceeds stream read limits");
    auto payload = std::vector<unsigned char>(payload_bytes);
    input.read(
        reinterpret_cast<char*>(payload.data()),
        static_cast<std::streamsize>(payload.size()));
    if (!input)
      throw std::runtime_error("truncated official QPET payload");
    auto correction_payload = std::vector<unsigned char>(correction_bytes);
    input.read(
        reinterpret_cast<char*>(correction_payload.data()),
        static_cast<std::streamsize>(correction_payload.size()));
    if (!input)
      throw std::runtime_error("truncated QPET correction payload");
    if (stream_version >= kVersion && !correction_payload.empty())
      required_stream_version = std::max(
          required_stream_version,
          correction_payload_required_stream_version(correction_payload));
    if (base_mode == 8 || base_mode == 9) {
      if (payload.size() < kSemanticBlockMagic.size() + 1 ||
          !std::equal(
              kSemanticBlockMagic.begin(), kSemanticBlockMagic.end(), payload.begin()))
        throw std::runtime_error("invalid semantic payload in outer QPET stream");
      const auto semantic_schema = payload[kSemanticBlockMagic.size()];
      if (semantic_schema == kSemanticSchemaStrictConstants && stream_version < 11)
        throw std::runtime_error("semantic schema 2 requires QPC stream version 11");
    }
    auto group_settings = settings;
    group_settings.batch = 1;
    group_settings.channels = logical_windows * settings.channels;
    group_settings.group_size = 1;
    std::uint32_t semantic_correction_count = 0;
    auto decoded = decode_payload(
        payload,
        group_settings,
        global_bound,
        base_mode,
        base_mode == 8 ? &semantic_correction_count : nullptr,
        stream_version);
    const auto base_reconstruction = decoded_records == nullptr
        ? std::vector<float>()
        : decoded;
    if (base_mode == 8) {
      if (!correction_payload.empty())
        throw std::runtime_error("semantic independent-block stream has outer correction bytes");
      if (semantic_correction_count != correction_count)
        throw std::runtime_error(
            "semantic inner correction count disagrees with outer metadata");
    }
    else if (settings.correction_mode == 0 && base_mode == 0) {
      if (!correction_payload.empty() || (stream_version >= 10 && correction_count != 0))
        throw std::runtime_error("online QPET stream has an invalid outer correction field");
      // v8/v9 stored the online rejection diagnostic in correction_count even
      // though no outer payload existed.  Ignore that legacy diagnostic.
    }
    else {
      apply_correction_payload(
          decoded,
          correction_payload,
          correction_count,
          group_settings.channels,
          group_settings.time,
          stream_version);
    }
    if (decoded_records != nullptr) {
      DecodedWindowRecord record;
      record.logical_windows = logical_windows;
      record.base_mode = base_mode;
      record.selected_global_bound = global_bound;
      record.corrections = correction_count;
      record.payload = payload;
      record.correction_payload = correction_payload;
      record.base_reconstruction = base_reconstruction;
      record.reconstruction = decoded;
      decoded_records->push_back(std::move(record));
    }
    std::copy(
        decoded.begin(),
        decoded.end(),
        reconstructed.begin() + decoded_windows * values_per_window);
    decoded_windows += logical_windows;
  }
  if (input.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("QPET correlation stream has trailing bytes");
  if (stream_version >= kVersion && stream_version != required_stream_version)
    throw std::runtime_error(
        "QPET correlation stream version is not canonical for its correction tags");
  return {settings, reconstructed};
}

qpet_correlation::ConstraintReport summarize_outer_constraints(
    const std::vector<float>& original,
    const std::vector<float>& reconstructed,
    const Settings& settings)
{
  if (settings.channels > std::numeric_limits<std::size_t>::max() / settings.time)
    throw std::invalid_argument("outer constraint summary dimensions overflow size_t");
  const auto values_per_window = settings.channels * settings.time;
  if (settings.batch >
          std::numeric_limits<std::size_t>::max() / values_per_window ||
      original.size() != settings.batch * values_per_window ||
      reconstructed.size() != original.size())
    throw std::invalid_argument("outer constraint summary shape mismatch");

  auto summary = qpet_correlation::ConstraintReport();
  for (std::size_t window = 0; window < settings.batch; ++window) {
    const auto begin = window * values_per_window;
    const auto original_window = std::vector<float>(
        original.begin() + begin, original.begin() + begin + values_per_window);
    const auto reconstructed_window = std::vector<float>(
        reconstructed.begin() + begin,
        reconstructed.begin() + begin + values_per_window);
    const auto report = qpet_correlation::verify_constraints(
        original_window,
        reconstructed_window,
        settings.channels,
        settings.time,
        settings.pointwise_tolerance,
        settings.acf_tolerance,
        settings.minimum_correlation,
        settings.max_lag);
    summary.max_abs_error = std::max(
        summary.max_abs_error, report.max_abs_error);
    summary.max_acf_error = std::max(
        summary.max_acf_error, report.max_acf_error);
    summary.min_series_correlation = std::min(
        summary.min_series_correlation, report.min_series_correlation);
    summary.pointwise_ok = summary.pointwise_ok && report.pointwise_ok;
    summary.acf_ok = summary.acf_ok && report.acf_ok;
    summary.correlation_ok = summary.correlation_ok && report.correlation_ok;
  }
  return summary;
}

void write_qpet_constraint_metrics(
    std::ostream& output,
    const qpet_correlation::ConstraintReport& report)
{
  constexpr auto metadata_double_precision = 17;
  static_assert(
      std::numeric_limits<double>::max_digits10 == metadata_double_precision,
      "QPET metadata precision assumes IEEE-754 binary64");
  const auto previous_precision = output.precision();
  output << std::setprecision(metadata_double_precision)
         << " max_abs=" << report.max_abs_error
         << " max_acf=" << report.max_acf_error
         << " min_corr=" << report.min_series_correlation;
  output.precision(previous_precision);
}

#include "RefinedBudgetExperiment.inc"
#include "GuidedExperiment.inc"

void compress_mode(int argc, char** argv)
{
  if (argc != 20 && argc != 21 && argc != 22 && argc != 24 && argc != 26)
    throw std::invalid_argument(
        "usage: qpet_correlation_batch compress input.f32 stream.qpc reconstruction.f32 "
        "batch channels time epsilon acf_epsilon min_correlation max_lag qoz_level "
        "correction_mode estimator_scales qpet_confidence qpet_k group_size "
        "repair_denominators candidate_profile [repair_optimizer "
        "[correction_format_ceiling [base_frontier "
        "relaxed_bound_ratio_ceiling [surrogate_frontier "
        "surrogate_bound_ratio_ceiling]]]]");
  Settings settings;
  settings.batch = std::stoull(argv[5]);
  settings.channels = std::stoull(argv[6]);
  settings.time = std::stoull(argv[7]);
  settings.pointwise_tolerance = std::stod(argv[8]);
  settings.acf_tolerance = std::stod(argv[9]);
  settings.minimum_correlation = std::stod(argv[10]);
  settings.max_lag = std::stoull(argv[11]);
  settings.qoz_level = std::stoi(argv[12]);
  settings.correction_mode = parse_correction_mode(argv[13]);
  settings.estimator_scales = parse_estimator_scales(argv[14]);
  settings.qpet_confidence = std::stod(argv[15]);
  settings.qpet_k = std::stod(argv[16]);
  settings.group_size = std::stoull(argv[17]);
  settings.repair_denominators = parse_repair_denominators(argv[18]);
  const auto profile_name = std::string(argv[19]);
  const bool refined_profile = profile_name == "paper_budget_refined";
  const bool guided_profile = profile_name == "guided_grid" ||
      profile_name == "guided_formula" || profile_name == "guided_union" ||
      profile_name == "guided_noformula" || profile_name == "guided_all";
  if (refined_profile || guided_profile) {
    // Separate encoder-only branch; the original scalar portfolio remains intact.
  }
  else if (profile_name == "paper_budget_uniform")
    settings.paper_budget_profile = 1;
  else if (profile_name == "fixed_grid4")
    settings.paper_budget_profile = 2;
  else
    settings.enable_vehicle_semantic = parse_candidate_profile(profile_name);
  if (argc >= 21)
    settings.repair_optimizer = parse_repair_optimizer(argv[20]);
  settings.correction_format_ceiling = argc >= 22
      ? parse_correction_format_ceiling(argv[21])
      : omitted_correction_format_ceiling(settings.repair_optimizer);
  if (argc >= 24) {
    settings.base_frontier = parse_base_frontier(argv[22]);
    settings.cjrf_ratio_mask = parse_cjrf_ratio_mask(argv[23]);
  }
  if (argc == 26) {
    settings.surrogate_frontier = parse_surrogate_frontier(argv[24]);
    settings.cfsr_ratio_mask = parse_cfsr_ratio_mask(argv[25]);
  }
  if (settings.group_size == 0)
    settings.group_size = settings.batch;
  validate_settings(settings);
  if (settings.channels > std::numeric_limits<std::size_t>::max() / settings.time ||
      settings.batch >
          std::numeric_limits<std::size_t>::max() / (settings.channels * settings.time))
    throw std::invalid_argument("input dimensions overflow size_t");

  if (guided_profile) {
    guided_compress_mode(settings, profile_name, argv[2], argv[3], argv[4]);
    return;
  }
  if (refined_profile) {
    refined_compress_mode(settings, argv[2], argv[3], argv[4]);
    return;
  }

  PaperBudgetAudit budget_audit;
  if (settings.paper_budget_profile != 0) {
    if (settings.channels != 1 || settings.group_size != 1 ||
        settings.correction_mode != 2 || settings.repair_optimizer != RepairOptimizer::legacy ||
        settings.correction_format_ceiling != 6 ||
        settings.base_frontier != BaseFrontier::incumbent_only ||
        settings.surrogate_frontier != SurrogateFrontier::none ||
        settings.enable_vehicle_semantic ||
        settings.estimator_scales != std::vector<double>{1.75, 2.0, 2.25, 2.5} ||
        settings.repair_denominators != std::vector<int>{7, 15, 31, 63, 127} ||
        settings.qpet_confidence != 0.999 || settings.qpet_k != 2.0)
      throw std::invalid_argument("paper experiment requires C=1, group=1, posthoc_greedy, "
          "legacy, cf6, alpha=1.75,2,2.25,2.5, beta=.999, k=2, denominators=7,15,31,63,127");
    const auto output_path = std::string(argv[3]);
    budget_audit.plans_output.open(output_path + ".budget.tsv");
    budget_audit.candidates_output.open(output_path + ".budget-candidates.tsv");
    budget_audit.winners_output.open(output_path + ".budget-winners.tsv");
    budget_audit.failures_output.open(output_path + ".budget-failures.tsv");
    if (!budget_audit.plans_output || !budget_audit.candidates_output ||
        !budget_audit.winners_output || !budget_audit.failures_output)
      throw std::runtime_error("cannot open paper budget provenance sidecars");
    budget_audit.plans_output << std::setprecision(17)
        << "window\tprofile\talpha\tunscaled_cap\tproposed_bound\teffective_bound\tpearson_active\tacf_active\tusable\tduplicate\texact_required\n";
    budget_audit.candidates_output << std::setprecision(17)
        << "window\tbase_mode\talpha\tproposed_bound\teffective_bound\tpre_max_abs\tpre_min_corr\tpre_max_acf\tpre_all_ok\tcorrections\tbase_bytes\tcorrection_bytes\trecord_bytes\tfinal_all_ok\n";
    budget_audit.winners_output << std::setprecision(17)
        << "window\tbase_mode\talpha\teffective_bound\tcorrections\tbase_bytes\tcorrection_bytes\trecord_bytes\tcandidates_including_repair\tfinal_all_ok\n";
    budget_audit.failures_output << std::setprecision(17)
        << "window\tbase_mode\talpha\teffective_bound\terror\n";
    settings.paper_budget_audit = &budget_audit;
  }
  const auto values_per_window = settings.channels * settings.time;
  const auto source = read_floats(argv[2], settings.batch * values_per_window);
  auto windows = std::vector<EncodedWindow>();
  windows.reserve(settings.batch);
  std::uint64_t total_corrections = 0;
  std::uint64_t total_online_corrections = 0;
  std::uint64_t total_candidates = 0;
  std::uint64_t total_candidate_failures = 0;
  std::uint64_t total_base_bytes = 0;
  std::uint64_t total_correction_bytes = 0;
  std::uint64_t total_strict_constant_groups = 0;
  std::uint64_t total_packed_quantized_groups = 0;
  std::uint64_t total_adaptive_packed_groups = 0;
  std::uint64_t total_factorized_packed_groups = 0;
  std::uint64_t total_subset_rank_packed_groups = 0;
  std::uint64_t total_predictive_packed_groups = 0;
  std::uint64_t total_shared_base_search_groups = 0;
  std::uint64_t total_zrcar_base_records = 0;
  std::uint64_t total_zrcar_states_generated = 0;
  std::uint64_t total_zrcar_infeasible_states = 0;
  std::uint64_t total_zrcar_infeasible_frontier_expansions = 0;
  std::uint64_t total_zrcar_infeasible_frontier_recoveries = 0;
  std::uint64_t total_zrcar_candidates = 0;
  std::uint64_t total_zrcar_prefix_candidates = 0;
  std::uint64_t total_zrcar_local_candidates = 0;
  std::uint64_t total_zrcar_legacy_seeds = 0;
  std::uint64_t total_zrcar_fvro_seeds = 0;
  std::uint64_t total_zrcar_selected_groups = 0;
  std::uint64_t total_zrcar_saved_bytes = 0;
  std::uint64_t total_zrcar_from_fvro_seed_groups = 0;
  std::uint64_t total_fvro_base_records = 0;
  std::uint64_t total_fvro_global_generated = 0;
  std::uint64_t total_fvro_global_serialized = 0;
  std::uint64_t total_fvro_global_infeasible = 0;
  std::uint64_t total_fvro_channel_generated = 0;
  std::uint64_t total_fvro_channel_serialized = 0;
  std::uint64_t total_fvro_channel_infeasible = 0;
  std::uint64_t total_fvro_local_generated = 0;
  std::uint64_t total_fvro_local_infeasible = 0;
  std::uint64_t total_fvro_selected_channels = 0;
  std::uint64_t total_fvro_strict_channel_steps = 0;
  std::uint64_t total_fvro_seeds_retained = 0;
  std::uint64_t total_fvro_reference_quantized_groups = 0;
  std::uint64_t total_fvro_final_quantized_groups = 0;
  std::uint64_t total_fvro_reference_format4_groups = 0;
  std::uint64_t total_fvro_reference_format6_groups = 0;
  std::uint64_t total_fvro_final_format4_groups = 0;
  std::uint64_t total_fvro_final_format6_groups = 0;
  std::uint64_t total_fvro_final_format7_groups = 0;
  std::uint64_t total_fvro_final_format8_groups = 0;
  std::uint64_t total_fvro_final_format9_groups = 0;
  std::uint64_t total_fvro_final_format10_groups = 0;
  std::uint64_t total_fvro_reference_exact_escapes = 0;
  std::uint64_t total_fvro_final_exact_escapes = 0;
  std::uint64_t total_fvro_reference_format4_exact_escapes = 0;
  std::uint64_t total_fvro_reference_format6_exact_escapes = 0;
  std::uint64_t total_fvro_final_format4_exact_escapes = 0;
  std::uint64_t total_fvro_final_format6_exact_escapes = 0;
  std::uint64_t total_fvro_final_format7_exact_escapes = 0;
  std::uint64_t total_fvro_final_format8_exact_escapes = 0;
  std::uint64_t total_fvro_final_format9_exact_escapes = 0;
  std::uint64_t total_fvro_final_format10_exact_escapes = 0;
  std::uint64_t total_fvro_ref6_to_final6_groups = 0;
  std::uint64_t total_fvro_ref6_to_non6_groups = 0;
  std::uint64_t total_fvro_ref6_to_final6_reference_exact_escapes = 0;
  std::uint64_t total_fvro_ref6_to_final6_final_exact_escapes = 0;
  std::uint64_t total_fvro_ref6_to_non6_reference_exact_escapes = 0;
  std::uint64_t total_fvro_selected_groups = 0;
  std::uint64_t total_fvro_saved_bytes = 0;
  std::uint64_t total_ssbb_base_records = 0;
  std::uint64_t total_ssbb_channels_searched = 0;
  std::uint64_t total_ssbb_positions_processed = 0;
  std::uint64_t total_ssbb_actions_generated = 0;
  std::uint64_t total_ssbb_states_generated = 0;
  std::uint64_t total_ssbb_states_deduplicated = 0;
  std::uint64_t total_ssbb_statistics_feasible_states = 0;
  std::uint64_t total_ssbb_statistics_infeasible_states = 0;
  std::uint64_t total_ssbb_feasible_states_retained = 0;
  std::uint64_t total_ssbb_infeasible_states_retained = 0;
  std::uint64_t total_ssbb_states_pruned_by_beam = 0;
  std::uint64_t total_ssbb_terminal_states = 0;
  std::uint64_t total_ssbb_terminal_direct_feasible = 0;
  std::uint64_t total_ssbb_terminal_direct_infeasible = 0;
  std::uint64_t total_ssbb_outputs_returned = 0;
  std::uint64_t total_ssbb_candidates_serialized = 0;
  std::uint64_t total_ssbb_seeds_retained = 0;
  std::uint64_t total_ssbb_selected_channels = 0;
  std::uint64_t total_ssbb_representation_candidates_serialized = 0;
  std::uint64_t total_ssbb_representation_improved_starts = 0;
  std::uint64_t total_ssbb_representation_gross_saved_bytes = 0;
  std::uint64_t total_ssbb_composition_starts = 0;
  std::uint64_t total_ssbb_composition_local_options = 0;
  std::uint64_t total_ssbb_composition_transitions_generated = 0;
  std::uint64_t total_ssbb_composition_transitions_deduplicated = 0;
  std::uint64_t total_ssbb_composition_direct_feasible = 0;
  std::uint64_t total_ssbb_composition_direct_infeasible = 0;
  std::uint64_t total_ssbb_composition_candidates_serialized = 0;
  std::uint64_t total_ssbb_composition_singletons_serialized = 0;
  std::uint64_t total_ssbb_composition_pairs_serialized = 0;
  std::uint64_t total_ssbb_composition_frontier_carries = 0;
  std::uint64_t total_ssbb_composition_anchor_retention_events = 0;
  std::uint64_t total_ssbb_composition_format_retention_events = 0;
  std::uint64_t total_ssbb_composition_rate_retention_events = 0;
  std::uint64_t total_ssbb_composition_states_pruned = 0;
  std::uint64_t total_ssbb_composition_max_frontier = 0;
  std::uint64_t total_ssbb_composition_selected_channels = 0;
  std::uint64_t total_ssbb_composition_selected_groups = 0;
  std::uint64_t total_ssbb_composition_saved_bytes = 0;
  std::uint64_t total_ssbb_selected_groups = 0;
  std::uint64_t total_ssbb_saved_bytes = 0;
  std::uint64_t total_ssbb_incremental_saved_bytes = 0;
  std::uint64_t total_ssbb_representation_selected_groups = 0;
  std::uint64_t total_ssbb_selected_representation_saved_bytes = 0;
  std::uint64_t total_ssbb_search_selected_groups = 0;
  std::uint64_t total_ssbb_selected_search_saved_bytes = 0;
  std::uint64_t total_second_stage_selected_groups = 0;
  std::uint64_t total_second_stage_saved_bytes = 0;
  std::uint64_t total_cjrf_relaxed_base_attempts = 0;
  std::uint64_t total_cjrf_relaxed_base_failures = 0;
  std::uint64_t total_cjrf_relaxed_base_pruned = 0;
  std::uint64_t total_cjrf_relaxed_base_duplicates = 0;
  std::uint64_t total_cjrf_relaxed_base_records = 0;
  std::uint64_t total_cjrf_relaxed_base_outside_pointwise = 0;
  std::uint64_t total_cjrf_relaxed_legacy_repaired = 0;
  std::uint64_t total_cjrf_relaxed_legacy_repair_candidates = 0;
  std::uint64_t total_cjrf_selected_groups = 0;
  std::uint64_t total_cjrf_saved_bytes = 0;
  std::uint64_t total_cjrf_legacy_winner_groups = 0;
  std::uint64_t total_cjrf_fvro_winner_groups = 0;
  std::uint64_t total_cjrf_zrcar_winner_groups = 0;
  std::uint64_t total_cjrf_ssbb_winner_groups = 0;
  std::uint64_t total_cfsr_surrogate_sources = 0;
  std::uint64_t total_cfsr_surrogate_base_attempts = 0;
  std::uint64_t total_cfsr_surrogate_base_failures = 0;
  std::uint64_t total_cfsr_surrogate_base_pruned = 0;
  std::uint64_t total_cfsr_surrogate_base_duplicates = 0;
  std::uint64_t total_cfsr_surrogate_base_records = 0;
  std::uint64_t total_cfsr_surrogate_base_outside_original_pointwise = 0;
  std::uint64_t total_cfsr_surrogate_legacy_repaired = 0;
  std::uint64_t total_cfsr_surrogate_legacy_repair_candidates = 0;
  std::uint64_t total_cfsr_selected_groups = 0;
  std::uint64_t total_cfsr_saved_bytes = 0;
  std::uint64_t total_cfsr_legacy_winner_groups = 0;
  std::uint64_t total_cfsr_fvro_winner_groups = 0;
  std::uint64_t total_cfsr_zrcar_winner_groups = 0;
  std::uint64_t total_cfsr_ssbb_winner_groups = 0;
  std::uint64_t total_cfsr_mode1_winner_groups = 0;
  std::uint64_t total_cfsr_mode3_winner_groups = 0;
  std::uint64_t total_cfsr_mode4_winner_groups = 0;
  std::uint64_t total_cfsr_mode5_winner_groups = 0;
  std::uint64_t total_cfsr_mode6_winner_groups = 0;
  for (std::size_t window = 0; window < settings.batch; window += settings.group_size) {
    const auto logical_windows = std::min(settings.group_size, settings.batch - window);
    auto original = std::vector<float>(
        source.begin() + window * values_per_window,
        source.begin() + (window + logical_windows) * values_per_window);
    auto group_settings = settings;
    group_settings.batch = 1;
    group_settings.channels = logical_windows * settings.channels;
    group_settings.group_size = 1;
    group_settings.paper_budget_window = window;
    auto encoded = compress_window(original, group_settings);
    if (settings.paper_budget_audit) {
      budget_audit.winners_output << window << '\t' << encoded.base_mode << '\t'
          << encoded.selected_estimator_scale << '\t' << encoded.selected_global_bound
          << '\t' << encoded.corrections << '\t' << encoded.payload.size()
          << '\t' << encoded.correction_payload.size() << '\t' << compact_record_bytes(encoded)
          << '\t' << encoded.candidates_tested << '\t' << encoded.report.all_ok() << '\n';
    }
    encoded.logical_windows = static_cast<std::uint32_t>(logical_windows);
    if ((encoded.base_mode == 8 || encoded.base_mode == 9) &&
        encoded.payload.size() >= kSemanticBlockMagic.size() + 1 &&
        std::equal(
            kSemanticBlockMagic.begin(),
            kSemanticBlockMagic.end(),
            encoded.payload.begin()) &&
        encoded.payload[kSemanticBlockMagic.size()] ==
            kSemanticSchemaStrictConstants)
      ++total_strict_constant_groups;
    if (!encoded.correction_payload.empty() &&
        encoded.correction_payload.front() == kPackedQuantizedCorrectionFormat)
      ++total_packed_quantized_groups;
    if (!encoded.correction_payload.empty() &&
        encoded.correction_payload.front() == kAdaptivePackedCorrectionFormat)
      ++total_adaptive_packed_groups;
    if (!encoded.correction_payload.empty() &&
        encoded.correction_payload.front() == kFactorizedPackedCorrectionFormat)
      ++total_factorized_packed_groups;
    if (!encoded.correction_payload.empty() &&
        encoded.correction_payload.front() == kSubsetRankPackedCorrectionFormat)
      ++total_subset_rank_packed_groups;
    if (!encoded.correction_payload.empty() &&
        encoded.correction_payload.front() == kPredictivePackedCorrectionFormat)
      ++total_predictive_packed_groups;
    if (encoded.shared_base_search_groups >
        std::numeric_limits<std::uint64_t>::max() -
            total_shared_base_search_groups)
      throw std::overflow_error("shared-base search group count overflow");
    total_shared_base_search_groups += encoded.shared_base_search_groups;
    auto add_search_diagnostic = [](std::uint64_t& total,
                                    std::uint64_t value,
                                    const char* message) {
      if (value > std::numeric_limits<std::uint64_t>::max() - total)
        throw std::overflow_error(message);
      total += value;
    };
    add_search_diagnostic(
        total_cjrf_relaxed_base_attempts,
        encoded.cjrf_relaxed_base_attempts,
        "CJRF relaxed-attempt group total overflow");
    add_search_diagnostic(
        total_cjrf_relaxed_base_failures,
        encoded.cjrf_relaxed_base_failures,
        "CJRF relaxed-failure group total overflow");
    add_search_diagnostic(
        total_cjrf_relaxed_base_pruned,
        encoded.cjrf_relaxed_base_pruned,
        "CJRF relaxed-prune group total overflow");
    add_search_diagnostic(
        total_cjrf_relaxed_base_duplicates,
        encoded.cjrf_relaxed_base_duplicates,
        "CJRF relaxed-duplicate group total overflow");
    add_search_diagnostic(
        total_cjrf_relaxed_base_records,
        encoded.cjrf_relaxed_base_records,
        "CJRF relaxed-record group total overflow");
    add_search_diagnostic(
        total_cjrf_relaxed_base_outside_pointwise,
        encoded.cjrf_relaxed_base_outside_pointwise,
        "CJRF outside-pointwise group total overflow");
    add_search_diagnostic(
        total_cjrf_relaxed_legacy_repaired,
        encoded.cjrf_relaxed_legacy_repaired,
        "CJRF legacy-repaired group total overflow");
    add_search_diagnostic(
        total_cjrf_relaxed_legacy_repair_candidates,
        encoded.cjrf_relaxed_legacy_repair_candidates,
        "CJRF legacy repair-candidate group total overflow");
    if (uses_cjrf(settings.base_frontier)) {
      auto enabled_ratios = std::size_t{0};
      for (std::size_t bit = 0; bit < kCjrfRelaxedBoundRatios.size(); ++bit)
        enabled_ratios +=
            (settings.cjrf_ratio_mask & (1U << bit)) == 0U ? 0U : 1U;
      const auto arms_per_ratio = group_settings.channels == 1 ? 3U : 4U;
      const auto expected_attempts = enabled_ratios * arms_per_ratio;
      if (encoded.cjrf_relaxed_base_attempts != expected_attempts ||
          encoded.cjrf_relaxed_base_attempts !=
              encoded.cjrf_relaxed_base_failures +
                  encoded.cjrf_relaxed_base_pruned +
                  encoded.cjrf_relaxed_base_duplicates +
                  encoded.cjrf_relaxed_base_records ||
          encoded.cjrf_relaxed_legacy_repaired !=
              encoded.cjrf_relaxed_base_records ||
          encoded.cjrf_relaxed_legacy_repair_candidates <
              encoded.cjrf_relaxed_legacy_repaired ||
          encoded.cjrf_relaxed_base_outside_pointwise >
              encoded.cjrf_relaxed_base_records ||
          encoded.cjrf_reference_bytes < kCompactRecordHeaderBytes)
        throw std::logic_error(
            "CJRF per-group search accounting is inconsistent");
      const auto final_record_bytes = compact_record_bytes(encoded);
      if (final_record_bytes > encoded.cjrf_reference_bytes ||
          encoded.cjrf_selected !=
              (final_record_bytes < encoded.cjrf_reference_bytes))
        throw std::logic_error(
            "CJRF per-group no-regression provenance is inconsistent");
      if (encoded.cjrf_selected) {
        if (!encoded.cjrf_relaxed_base ||
            encoded.cjrf_winner_kind == CjrfWinnerKind::none)
          throw std::logic_error(
              "CJRF selected group has no relaxed winner provenance");
        auto enabled_winner_ratio = false;
        for (std::size_t bit = 0;
             bit < kCjrfRelaxedBoundRatios.size();
             ++bit) {
          if ((settings.cjrf_ratio_mask & (1U << bit)) != 0U &&
              encoded.cjrf_relaxed_ratio ==
                  kCjrfRelaxedBoundRatios[bit])
            enabled_winner_ratio = true;
        }
        if (!enabled_winner_ratio)
          throw std::logic_error(
              "CJRF winner ratio is outside the enabled nested shell");
        ++total_cjrf_selected_groups;
        add_search_diagnostic(
            total_cjrf_saved_bytes,
            encoded.cjrf_reference_bytes - final_record_bytes,
            "CJRF saved-byte total overflow");
        switch (encoded.cjrf_winner_kind) {
          case CjrfWinnerKind::legacy:
            ++total_cjrf_legacy_winner_groups;
            break;
          case CjrfWinnerKind::fvro:
            ++total_cjrf_fvro_winner_groups;
            break;
          case CjrfWinnerKind::zrcar:
            ++total_cjrf_zrcar_winner_groups;
            break;
          case CjrfWinnerKind::ssbb:
            ++total_cjrf_ssbb_winner_groups;
            break;
          case CjrfWinnerKind::none:
            throw std::logic_error("CJRF selected an empty winner kind");
        }
      }
      else if (encoded.cjrf_relaxed_base ||
               encoded.cjrf_winner_kind != CjrfWinnerKind::none) {
        throw std::logic_error(
            "CJRF retained incumbent carries relaxed winner provenance");
      }
    }
    else if (encoded.cjrf_selected || encoded.cjrf_relaxed_base ||
             encoded.cjrf_winner_kind != CjrfWinnerKind::none ||
             encoded.cjrf_reference_bytes != 0 ||
             encoded.cjrf_relaxed_base_attempts != 0 ||
             encoded.cjrf_relaxed_base_failures != 0 ||
             encoded.cjrf_relaxed_base_pruned != 0 ||
             encoded.cjrf_relaxed_base_duplicates != 0 ||
             encoded.cjrf_relaxed_base_records != 0 ||
             encoded.cjrf_relaxed_base_outside_pointwise != 0 ||
             encoded.cjrf_relaxed_legacy_repaired != 0 ||
             encoded.cjrf_relaxed_legacy_repair_candidates != 0) {
      throw std::logic_error(
          "incumbent-only group carries CJRF diagnostics");
    }
    add_search_diagnostic(
        total_cfsr_surrogate_sources,
        encoded.cfsr_surrogate_sources,
        "CFSR surrogate-source group total overflow");
    add_search_diagnostic(
        total_cfsr_surrogate_base_attempts,
        encoded.cfsr_surrogate_base_attempts,
        "CFSR surrogate-attempt group total overflow");
    add_search_diagnostic(
        total_cfsr_surrogate_base_failures,
        encoded.cfsr_surrogate_base_failures,
        "CFSR surrogate-failure group total overflow");
    add_search_diagnostic(
        total_cfsr_surrogate_base_pruned,
        encoded.cfsr_surrogate_base_pruned,
        "CFSR surrogate-prune group total overflow");
    add_search_diagnostic(
        total_cfsr_surrogate_base_duplicates,
        encoded.cfsr_surrogate_base_duplicates,
        "CFSR surrogate-duplicate group total overflow");
    add_search_diagnostic(
        total_cfsr_surrogate_base_records,
        encoded.cfsr_surrogate_base_records,
        "CFSR surrogate-record group total overflow");
    add_search_diagnostic(
        total_cfsr_surrogate_base_outside_original_pointwise,
        encoded.cfsr_surrogate_base_outside_original_pointwise,
        "CFSR outside-original-pointwise group total overflow");
    add_search_diagnostic(
        total_cfsr_surrogate_legacy_repaired,
        encoded.cfsr_surrogate_legacy_repaired,
        "CFSR legacy-repaired group total overflow");
    add_search_diagnostic(
        total_cfsr_surrogate_legacy_repair_candidates,
        encoded.cfsr_surrogate_legacy_repair_candidates,
        "CFSR legacy repair-candidate group total overflow");
    if (uses_cfsr(settings.surrogate_frontier)) {
      const auto expected_attempts = expected_cfsr_base_attempts(group_settings);
      if (encoded.cfsr_surrogate_sources != 1U ||
          encoded.cfsr_surrogate_base_attempts != expected_attempts ||
          encoded.cfsr_surrogate_base_attempts !=
              encoded.cfsr_surrogate_base_failures +
                  encoded.cfsr_surrogate_base_pruned +
                  encoded.cfsr_surrogate_base_duplicates +
                  encoded.cfsr_surrogate_base_records ||
          encoded.cfsr_surrogate_legacy_repaired !=
              encoded.cfsr_surrogate_base_records ||
          encoded.cfsr_surrogate_legacy_repair_candidates <
              encoded.cfsr_surrogate_legacy_repaired ||
          encoded.cfsr_surrogate_base_outside_original_pointwise >
              encoded.cfsr_surrogate_base_records ||
          encoded.cfsr_reference_bytes < kCompactRecordHeaderBytes)
        throw std::logic_error(
            "CFSR per-group search accounting is inconsistent");
      const auto final_record_bytes = compact_record_bytes(encoded);
      if (final_record_bytes > encoded.cfsr_reference_bytes ||
          encoded.cfsr_selected !=
              (final_record_bytes < encoded.cfsr_reference_bytes))
        throw std::logic_error(
            "CFSR per-group no-regression provenance is inconsistent");
      if (encoded.cfsr_selected) {
        if (!encoded.cfsr_surrogate_base ||
            encoded.cfsr_winner_kind == CfsrWinnerKind::none)
          throw std::logic_error(
              "CFSR selected group has no surrogate winner provenance");
        const auto lossless_winner = encoded.base_mode == 4U &&
            encoded.cfsr_surrogate_ratio == 0.0;
        auto enabled_lossy_winner_ratio = false;
        for (std::size_t bit = 0;
             bit < kCfsrSurrogateBoundRatios.size();
             ++bit) {
          if ((settings.cfsr_ratio_mask & (1U << bit)) != 0U &&
              encoded.cfsr_surrogate_ratio ==
                  kCfsrSurrogateBoundRatios[bit])
            enabled_lossy_winner_ratio = true;
        }
        const auto lossy_winner_mode = encoded.base_mode == 1U ||
            encoded.base_mode == 3U || encoded.base_mode == 5U ||
            encoded.base_mode == 6U;
        if (!lossless_winner &&
            !(lossy_winner_mode && enabled_lossy_winner_ratio))
          throw std::logic_error(
              "CFSR winner mode/ratio is outside the registered frontier");
        if (encoded.base_mode == 3U && group_settings.channels == 1U)
          throw std::logic_error(
              "single-channel CFSR selected the excluded SPERR arm");
        ++total_cfsr_selected_groups;
        add_search_diagnostic(
            total_cfsr_saved_bytes,
            encoded.cfsr_reference_bytes - final_record_bytes,
            "CFSR saved-byte total overflow");
        switch (encoded.cfsr_winner_kind) {
          case CfsrWinnerKind::legacy:
            ++total_cfsr_legacy_winner_groups;
            break;
          case CfsrWinnerKind::fvro:
            ++total_cfsr_fvro_winner_groups;
            break;
          case CfsrWinnerKind::zrcar:
            ++total_cfsr_zrcar_winner_groups;
            break;
          case CfsrWinnerKind::ssbb:
            ++total_cfsr_ssbb_winner_groups;
            break;
          case CfsrWinnerKind::none:
            throw std::logic_error("CFSR selected an empty winner kind");
        }
        switch (encoded.base_mode) {
          case 1U:
            ++total_cfsr_mode1_winner_groups;
            break;
          case 3U:
            ++total_cfsr_mode3_winner_groups;
            break;
          case 4U:
            ++total_cfsr_mode4_winner_groups;
            break;
          case 5U:
            ++total_cfsr_mode5_winner_groups;
            break;
          case 6U:
            ++total_cfsr_mode6_winner_groups;
            break;
          default:
            throw std::logic_error(
                "CFSR selected an unregistered official base mode");
        }
      }
      else if (encoded.cfsr_surrogate_base ||
               encoded.cfsr_winner_kind != CfsrWinnerKind::none) {
        throw std::logic_error(
            "CFSR retained incumbent carries surrogate winner provenance");
      }
    }
    else if (encoded.cfsr_selected || encoded.cfsr_surrogate_base ||
             encoded.cfsr_winner_kind != CfsrWinnerKind::none ||
             encoded.cfsr_surrogate_ratio != 0.0 ||
             encoded.cfsr_reference_bytes != 0 ||
             encoded.cfsr_surrogate_sources != 0 ||
             encoded.cfsr_surrogate_base_attempts != 0 ||
             encoded.cfsr_surrogate_base_failures != 0 ||
             encoded.cfsr_surrogate_base_pruned != 0 ||
             encoded.cfsr_surrogate_base_duplicates != 0 ||
             encoded.cfsr_surrogate_base_records != 0 ||
             encoded.cfsr_surrogate_base_outside_original_pointwise != 0 ||
             encoded.cfsr_surrogate_legacy_repaired != 0 ||
             encoded.cfsr_surrogate_legacy_repair_candidates != 0) {
      throw std::logic_error(
          "disabled surrogate frontier carries CFSR diagnostics");
    }
    add_search_diagnostic(
        total_zrcar_base_records,
        encoded.zrcar_base_records_tested,
        "ZRCAR base-record group total overflow");
    add_search_diagnostic(
        total_zrcar_states_generated,
        encoded.zrcar_states_generated,
        "ZRCAR generated-state group total overflow");
    add_search_diagnostic(
        total_zrcar_infeasible_states,
        encoded.zrcar_infeasible_states,
        "ZRCAR infeasible-state group total overflow");
    add_search_diagnostic(
        total_zrcar_infeasible_frontier_expansions,
        encoded.zrcar_infeasible_frontier_expansions,
        "ZRCAR infeasible-frontier expansion group total overflow");
    add_search_diagnostic(
        total_zrcar_infeasible_frontier_recoveries,
        encoded.zrcar_infeasible_frontier_recoveries,
        "ZRCAR infeasible-frontier recovery group total overflow");
    add_search_diagnostic(
        total_zrcar_candidates,
        encoded.zrcar_candidates_tested,
        "ZRCAR serialized-candidate group total overflow");
    add_search_diagnostic(
        total_zrcar_prefix_candidates,
        encoded.zrcar_prefix_candidates_tested,
        "ZRCAR prefix-candidate group total overflow");
    add_search_diagnostic(
        total_zrcar_local_candidates,
        encoded.zrcar_local_candidates_tested,
        "ZRCAR local-candidate group total overflow");
    add_search_diagnostic(
        total_zrcar_legacy_seeds,
        encoded.zrcar_legacy_seeds_selected,
        "ZRCAR frozen-seed group total overflow");
    add_search_diagnostic(
        total_zrcar_fvro_seeds,
        encoded.zrcar_fvro_seeds_selected,
        "ZRCAR FVRO-seed group total overflow");
    if (encoded.zrcar_selected) {
      if (encoded.fvro_selected || encoded.ssbb_selected)
        throw std::logic_error(
            "FVRO/ZRCAR/SSBB final-stage provenance is not mutually exclusive");
      if (encoded.zrcar_legacy_reference_bytes <= encoded.encoded_size())
        throw std::logic_error(
            "ZRCAR selected group does not have a strict byte saving");
      if (!encoded.cjrf_selected && !encoded.cfsr_selected) {
        ++total_zrcar_selected_groups;
        ++total_second_stage_selected_groups;
        const auto saved =
            encoded.zrcar_legacy_reference_bytes - encoded.encoded_size();
        add_search_diagnostic(
            total_zrcar_saved_bytes,
            saved,
            "ZRCAR saved-byte total overflow");
        add_search_diagnostic(
            total_second_stage_saved_bytes,
            saved,
            "second-stage saved-byte total overflow");
      }
    }
    if (encoded.zrcar_from_fvro_seed) {
      if (!encoded.zrcar_selected || encoded.fvro_selected ||
          encoded.ssbb_selected)
        throw std::logic_error(
            "FVRO-composed ZRCAR provenance is internally inconsistent");
      if (!encoded.cjrf_selected && !encoded.cfsr_selected)
        ++total_zrcar_from_fvro_seed_groups;
    }
    add_search_diagnostic(
        total_fvro_base_records,
        encoded.fvro_base_records_tested,
        "FVRO base-record group total overflow");
    add_search_diagnostic(
        total_fvro_global_generated,
        encoded.fvro_global_candidates_generated,
        "FVRO global generated-candidate group total overflow");
    add_search_diagnostic(
        total_fvro_global_serialized,
        encoded.fvro_global_candidates_serialized,
        "FVRO global serialized-candidate group total overflow");
    add_search_diagnostic(
        total_fvro_global_infeasible,
        encoded.fvro_global_candidates_infeasible,
        "FVRO global infeasible-candidate group total overflow");
    add_search_diagnostic(
        total_fvro_channel_generated,
        encoded.fvro_channel_candidates_generated,
        "FVRO channel generated-candidate group total overflow");
    add_search_diagnostic(
        total_fvro_channel_serialized,
        encoded.fvro_channel_candidates_serialized,
        "FVRO channel serialized-candidate group total overflow");
    add_search_diagnostic(
        total_fvro_channel_infeasible,
        encoded.fvro_channel_candidates_infeasible,
        "FVRO channel infeasible-candidate group total overflow");
    add_search_diagnostic(
        total_fvro_local_generated,
        encoded.fvro_local_proposals_generated,
        "FVRO local generated-proposal group total overflow");
    add_search_diagnostic(
        total_fvro_local_infeasible,
        encoded.fvro_local_proposals_infeasible,
        "FVRO local infeasible-proposal group total overflow");
    add_search_diagnostic(
        total_fvro_selected_channels,
        encoded.fvro_selected_channels,
        "FVRO selected-channel group total overflow");
    add_search_diagnostic(
        total_fvro_strict_channel_steps,
        encoded.fvro_strict_channel_steps,
        "FVRO strict channel-step group total overflow");
    add_search_diagnostic(
        total_fvro_seeds_retained,
        encoded.fvro_seeds_retained,
        "FVRO retained-seed group total overflow");
    if (encoded.fvro_reference_quantized) {
      ++total_fvro_reference_quantized_groups;
      add_search_diagnostic(
          total_fvro_reference_exact_escapes,
          encoded.fvro_reference_exact_escapes,
          "FVRO reference exact-escape group total overflow");
      if (encoded.fvro_reference_quantized_format == 4U) {
        ++total_fvro_reference_format4_groups;
        add_search_diagnostic(
            total_fvro_reference_format4_exact_escapes,
            encoded.fvro_reference_exact_escapes,
            "FVRO reference format-4 escape total overflow");
      }
      else if (encoded.fvro_reference_quantized_format ==
               kPackedQuantizedCorrectionFormat) {
        ++total_fvro_reference_format6_groups;
        add_search_diagnostic(
            total_fvro_reference_format6_exact_escapes,
            encoded.fvro_reference_exact_escapes,
            "FVRO reference format-6 escape total overflow");
      }
      else
        throw std::logic_error(
            "FVRO reference quantized summary has an unknown format");
    }
    else if (encoded.fvro_reference_quantized_format != 0)
      throw std::logic_error(
          "FVRO nonquantized reference carries a quantized format");
    if (encoded.fvro_final_quantized) {
      ++total_fvro_final_quantized_groups;
      add_search_diagnostic(
          total_fvro_final_exact_escapes,
          encoded.fvro_final_exact_escapes,
          "FVRO final exact-escape group total overflow");
      if (encoded.fvro_final_quantized_format == 4U) {
        ++total_fvro_final_format4_groups;
        add_search_diagnostic(
            total_fvro_final_format4_exact_escapes,
            encoded.fvro_final_exact_escapes,
            "FVRO final format-4 escape total overflow");
      }
      else if (encoded.fvro_final_quantized_format ==
               kPackedQuantizedCorrectionFormat) {
        ++total_fvro_final_format6_groups;
        add_search_diagnostic(
            total_fvro_final_format6_exact_escapes,
            encoded.fvro_final_exact_escapes,
            "FVRO final format-6 escape total overflow");
      }
      else if (encoded.fvro_final_quantized_format ==
               kAdaptivePackedCorrectionFormat) {
        ++total_fvro_final_format7_groups;
        add_search_diagnostic(
            total_fvro_final_format7_exact_escapes,
            encoded.fvro_final_exact_escapes,
            "FVRO final format-7 escape total overflow");
      }
      else if (encoded.fvro_final_quantized_format ==
               kFactorizedPackedCorrectionFormat) {
        ++total_fvro_final_format8_groups;
        add_search_diagnostic(
            total_fvro_final_format8_exact_escapes,
            encoded.fvro_final_exact_escapes,
            "FVRO final format-8 escape total overflow");
      }
      else if (encoded.fvro_final_quantized_format ==
               kSubsetRankPackedCorrectionFormat) {
        ++total_fvro_final_format9_groups;
        add_search_diagnostic(
            total_fvro_final_format9_exact_escapes,
            encoded.fvro_final_exact_escapes,
            "FVRO final format-9 escape total overflow");
      }
      else if (encoded.fvro_final_quantized_format ==
               kPredictivePackedCorrectionFormat) {
        ++total_fvro_final_format10_groups;
        add_search_diagnostic(
            total_fvro_final_format10_exact_escapes,
            encoded.fvro_final_exact_escapes,
            "FVRO final format-10 escape total overflow");
      }
      else
        throw std::logic_error(
            "FVRO final quantized summary has an unknown format");
    }
    else if (encoded.fvro_final_quantized_format != 0)
      throw std::logic_error(
          "FVRO nonquantized final result carries a quantized format");
    if (encoded.fvro_reference_quantized_format ==
        kPackedQuantizedCorrectionFormat) {
      if (encoded.fvro_final_quantized_format ==
          kPackedQuantizedCorrectionFormat) {
        ++total_fvro_ref6_to_final6_groups;
        add_search_diagnostic(
            total_fvro_ref6_to_final6_reference_exact_escapes,
            encoded.fvro_reference_exact_escapes,
            "FVRO paired reference format-6 escape total overflow");
        add_search_diagnostic(
            total_fvro_ref6_to_final6_final_exact_escapes,
            encoded.fvro_final_exact_escapes,
            "FVRO paired final format-6 escape total overflow");
      }
      else {
        ++total_fvro_ref6_to_non6_groups;
        add_search_diagnostic(
            total_fvro_ref6_to_non6_reference_exact_escapes,
            encoded.fvro_reference_exact_escapes,
            "FVRO migrated reference format-6 escape total overflow");
      }
    }
    if (encoded.fvro_selected) {
      if (encoded.zrcar_selected || encoded.zrcar_from_fvro_seed ||
          encoded.ssbb_selected)
        throw std::logic_error(
            "FVRO value-stage provenance overlaps final ZRCAR provenance");
      if (encoded.fvro_legacy_reference_bytes <= encoded.encoded_size())
        throw std::logic_error(
            "FVRO selected group does not have a strict byte saving");
      if (!encoded.cjrf_selected && !encoded.cfsr_selected) {
        ++total_fvro_selected_groups;
        ++total_second_stage_selected_groups;
        const auto saved =
            encoded.fvro_legacy_reference_bytes - encoded.encoded_size();
        add_search_diagnostic(
            total_fvro_saved_bytes,
            saved,
            "FVRO saved-byte total overflow");
        add_search_diagnostic(
            total_second_stage_saved_bytes,
            saved,
            "second-stage saved-byte total overflow");
      }
    }
    add_search_diagnostic(
        total_ssbb_base_records,
        encoded.ssbb_base_records_tested,
        "SSBB base-record group total overflow");
    add_search_diagnostic(
        total_ssbb_channels_searched,
        encoded.ssbb_channels_searched,
        "SSBB searched-channel group total overflow");
    add_search_diagnostic(
        total_ssbb_positions_processed,
        encoded.ssbb_positions_processed,
        "SSBB processed-position group total overflow");
    add_search_diagnostic(
        total_ssbb_actions_generated,
        encoded.ssbb_actions_generated,
        "SSBB generated-action group total overflow");
    add_search_diagnostic(
        total_ssbb_states_generated,
        encoded.ssbb_states_generated,
        "SSBB generated-state group total overflow");
    add_search_diagnostic(
        total_ssbb_states_deduplicated,
        encoded.ssbb_states_deduplicated,
        "SSBB deduplicated-state group total overflow");
    add_search_diagnostic(
        total_ssbb_statistics_feasible_states,
        encoded.ssbb_statistics_feasible_states,
        "SSBB statistics-feasible group total overflow");
    add_search_diagnostic(
        total_ssbb_statistics_infeasible_states,
        encoded.ssbb_statistics_infeasible_states,
        "SSBB statistics-infeasible group total overflow");
    add_search_diagnostic(
        total_ssbb_feasible_states_retained,
        encoded.ssbb_feasible_states_retained,
        "SSBB feasible-retained group total overflow");
    add_search_diagnostic(
        total_ssbb_infeasible_states_retained,
        encoded.ssbb_infeasible_states_retained,
        "SSBB infeasible-retained group total overflow");
    add_search_diagnostic(
        total_ssbb_states_pruned_by_beam,
        encoded.ssbb_states_pruned_by_beam,
        "SSBB beam-pruned group total overflow");
    add_search_diagnostic(
        total_ssbb_terminal_states,
        encoded.ssbb_terminal_states,
        "SSBB terminal-state group total overflow");
    add_search_diagnostic(
        total_ssbb_terminal_direct_feasible,
        encoded.ssbb_terminal_direct_feasible,
        "SSBB terminal-feasible group total overflow");
    add_search_diagnostic(
        total_ssbb_terminal_direct_infeasible,
        encoded.ssbb_terminal_direct_infeasible,
        "SSBB terminal-infeasible group total overflow");
    add_search_diagnostic(
        total_ssbb_outputs_returned,
        encoded.ssbb_outputs_returned,
        "SSBB output group total overflow");
    add_search_diagnostic(
        total_ssbb_candidates_serialized,
        encoded.ssbb_candidates_serialized,
        "SSBB serializer group total overflow");
    add_search_diagnostic(
        total_ssbb_seeds_retained,
        encoded.ssbb_seeds_retained,
        "SSBB retained-seed group total overflow");
    add_search_diagnostic(
        total_ssbb_selected_channels,
        encoded.ssbb_selected_channels,
        "SSBB selected-channel group total overflow");
    add_search_diagnostic(
        total_ssbb_representation_candidates_serialized,
        encoded.ssbb_representation_candidates_serialized,
        "SSBB representation-candidate group total overflow");
    add_search_diagnostic(
        total_ssbb_representation_improved_starts,
        encoded.ssbb_representation_improved_starts,
        "SSBB representation-win group total overflow");
    add_search_diagnostic(
        total_ssbb_representation_gross_saved_bytes,
        encoded.ssbb_representation_gross_saved_bytes,
        "SSBB representation saved-byte group total overflow");
    add_search_diagnostic(
        total_ssbb_composition_starts,
        encoded.ssbb_composition_starts,
        "SSBB composition-start group total overflow");
    add_search_diagnostic(
        total_ssbb_composition_local_options,
        encoded.ssbb_composition_local_options,
        "SSBB composition-option group total overflow");
    add_search_diagnostic(
        total_ssbb_composition_transitions_generated,
        encoded.ssbb_composition_transitions_generated,
        "SSBB composition-transition group total overflow");
    add_search_diagnostic(
        total_ssbb_composition_transitions_deduplicated,
        encoded.ssbb_composition_transitions_deduplicated,
        "SSBB composition-duplicate group total overflow");
    add_search_diagnostic(
        total_ssbb_composition_direct_feasible,
        encoded.ssbb_composition_direct_feasible,
        "SSBB composition-feasible group total overflow");
    add_search_diagnostic(
        total_ssbb_composition_direct_infeasible,
        encoded.ssbb_composition_direct_infeasible,
        "SSBB composition-infeasible group total overflow");
    add_search_diagnostic(
        total_ssbb_composition_candidates_serialized,
        encoded.ssbb_composition_candidates_serialized,
        "SSBB composition-serializer group total overflow");
    add_search_diagnostic(
        total_ssbb_composition_singletons_serialized,
        encoded.ssbb_composition_singletons_serialized,
        "SSBB composition-singleton group total overflow");
    add_search_diagnostic(
        total_ssbb_composition_pairs_serialized,
        encoded.ssbb_composition_pairs_serialized,
        "SSBB composition-pair group total overflow");
    add_search_diagnostic(
        total_ssbb_composition_frontier_carries,
        encoded.ssbb_composition_frontier_carries,
        "SSBB composition-carry group total overflow");
    add_search_diagnostic(
        total_ssbb_composition_anchor_retention_events,
        encoded.ssbb_composition_anchor_retention_events,
        "SSBB composition-anchor group total overflow");
    add_search_diagnostic(
        total_ssbb_composition_format_retention_events,
        encoded.ssbb_composition_format_retention_events,
        "SSBB composition-format group total overflow");
    add_search_diagnostic(
        total_ssbb_composition_rate_retention_events,
        encoded.ssbb_composition_rate_retention_events,
        "SSBB composition-rate group total overflow");
    add_search_diagnostic(
        total_ssbb_composition_states_pruned,
        encoded.ssbb_composition_states_pruned,
        "SSBB composition-pruned group total overflow");
    total_ssbb_composition_max_frontier = std::max(
        total_ssbb_composition_max_frontier,
        static_cast<std::uint64_t>(encoded.ssbb_composition_max_frontier));
    add_search_diagnostic(
        total_ssbb_composition_selected_channels,
        encoded.ssbb_composition_selected_channels,
        "SSBB composition selected-channel group total overflow");
    if (uses_ssbb(settings.repair_optimizer) &&
        (encoded.ssbb_representation_candidates_serialized !=
             encoded.ssbb_composition_starts ||
         encoded.ssbb_representation_improved_starts >
             encoded.ssbb_representation_candidates_serialized ||
         (encoded.ssbb_representation_improved_starts == 0) !=
             (encoded.ssbb_representation_gross_saved_bytes == 0) ||
         encoded.ssbb_search_reference_bytes >
             encoded.ssbb_reference_bytes ||
         encoded.encoded_size() > encoded.ssbb_search_reference_bytes))
      throw std::logic_error(
          "SSBB group representation/search accounting diverged");
    if (encoded.ssbb_composition_selected) {
      if (!encoded.ssbb_selected ||
          encoded.ssbb_composition_reference_bytes <= encoded.encoded_size())
        throw std::logic_error(
            "SSBB composition group has inconsistent final provenance");
      if (!encoded.cjrf_selected && !encoded.cfsr_selected) {
        ++total_ssbb_composition_selected_groups;
        add_search_diagnostic(
            total_ssbb_composition_saved_bytes,
            encoded.ssbb_composition_reference_bytes - encoded.encoded_size(),
            "SSBB composition saved-byte total overflow");
      }
    }
    if (encoded.ssbb_selected) {
      if (encoded.zrcar_selected || encoded.zrcar_from_fvro_seed ||
          encoded.fvro_selected)
        throw std::logic_error(
            "SSBB final-stage provenance overlaps FVRO/ZRCAR provenance");
      if (encoded.ssbb_reference_bytes <= encoded.encoded_size())
        throw std::logic_error(
            "SSBB selected group does not have a strict byte saving");
      if (encoded.fvro_legacy_reference_bytes <= encoded.encoded_size())
        throw std::logic_error(
            "SSBB final result does not improve the frozen legacy reference");
      const auto representation_saved =
          encoded.ssbb_reference_bytes - encoded.ssbb_search_reference_bytes;
      const auto search_saved =
          encoded.ssbb_search_reference_bytes - encoded.encoded_size();
      const auto incremental_saved = checked_add_size(
          representation_saved,
          search_saved,
          "SSBB selected incremental saving overflow");
      if (incremental_saved !=
              encoded.ssbb_reference_bytes - encoded.encoded_size() ||
          encoded.fvro_legacy_reference_bytes - encoded.encoded_size() <
              incremental_saved) {
        auto message = std::ostringstream();
        message << "SSBB selected saving decomposition is inconsistent: "
                << "legacy=" << encoded.fvro_legacy_reference_bytes
                << ", reference=" << encoded.ssbb_reference_bytes
                << ", search_reference="
                << encoded.ssbb_search_reference_bytes
                << ", final=" << encoded.encoded_size()
                << ", representation_saved=" << representation_saved
                << ", search_saved=" << search_saved;
        throw std::logic_error(message.str());
      }
      if (!encoded.cjrf_selected && !encoded.cfsr_selected) {
        ++total_ssbb_selected_groups;
        ++total_second_stage_selected_groups;
        if (representation_saved != 0) {
          ++total_ssbb_representation_selected_groups;
          add_search_diagnostic(
              total_ssbb_selected_representation_saved_bytes,
              representation_saved,
              "SSBB selected representation saved-byte total overflow");
        }
        if (search_saved != 0) {
          ++total_ssbb_search_selected_groups;
          add_search_diagnostic(
              total_ssbb_selected_search_saved_bytes,
              search_saved,
              "SSBB selected search saved-byte total overflow");
        }
        add_search_diagnostic(
            total_ssbb_incremental_saved_bytes,
            incremental_saved,
            "SSBB incremental saved-byte total overflow");
        const auto saved =
            encoded.fvro_legacy_reference_bytes - encoded.encoded_size();
        add_search_diagnostic(
            total_ssbb_saved_bytes,
            saved,
            "SSBB saved-byte total overflow");
        add_search_diagnostic(
            total_second_stage_saved_bytes,
            encoded.fvro_legacy_reference_bytes - encoded.encoded_size(),
            "second-stage saved-byte total overflow");
      }
    }
    total_corrections += encoded.corrections;
    total_online_corrections += encoded.online_corrections;
    total_candidates += encoded.candidates_tested;
    if (encoded.candidate_failures >
        std::numeric_limits<std::uint64_t>::max() - total_candidate_failures)
      throw std::overflow_error("QPET backend candidate failure total overflow");
    total_candidate_failures += encoded.candidate_failures;
    total_base_bytes += encoded.payload.size();
    total_correction_bytes += encoded.correction_payload.size();
    windows.push_back(std::move(encoded));
  }
  if (total_fvro_reference_quantized_groups !=
          total_fvro_reference_format4_groups +
              total_fvro_reference_format6_groups ||
      total_fvro_final_quantized_groups !=
          total_fvro_final_format4_groups + total_fvro_final_format6_groups +
              total_fvro_final_format7_groups +
              total_fvro_final_format8_groups +
              total_fvro_final_format9_groups +
              total_fvro_final_format10_groups ||
      total_fvro_reference_exact_escapes !=
          total_fvro_reference_format4_exact_escapes +
              total_fvro_reference_format6_exact_escapes ||
      total_fvro_final_exact_escapes !=
          total_fvro_final_format4_exact_escapes +
              total_fvro_final_format6_exact_escapes +
              total_fvro_final_format7_exact_escapes +
              total_fvro_final_format8_exact_escapes +
              total_fvro_final_format9_exact_escapes +
              total_fvro_final_format10_exact_escapes)
    throw std::logic_error(
        "FVRO quantized-format diagnostics do not partition totals");
  if (total_adaptive_packed_groups != total_fvro_final_format7_groups ||
      total_factorized_packed_groups != total_fvro_final_format8_groups ||
      total_subset_rank_packed_groups != total_fvro_final_format9_groups ||
      total_predictive_packed_groups != total_fvro_final_format10_groups)
    throw std::logic_error(
        "selected correction formats disagree with final format diagnostics");
  if ((settings.correction_format_ceiling <
           kAdaptivePackedCorrectionFormat &&
       total_adaptive_packed_groups != 0) ||
      (settings.correction_format_ceiling <
           kFactorizedPackedCorrectionFormat &&
       total_factorized_packed_groups != 0) ||
      (settings.correction_format_ceiling <
           kSubsetRankPackedCorrectionFormat &&
       total_subset_rank_packed_groups != 0) ||
      (settings.correction_format_ceiling <
           kPredictivePackedCorrectionFormat &&
       total_predictive_packed_groups != 0))
    throw std::logic_error(
        "selected correction formats exceed the batch encoder ceiling");
  if (total_fvro_ref6_to_final6_groups + total_fvro_ref6_to_non6_groups !=
          total_fvro_reference_format6_groups ||
      total_fvro_ref6_to_final6_reference_exact_escapes +
              total_fvro_ref6_to_non6_reference_exact_escapes !=
          total_fvro_reference_format6_exact_escapes)
    throw std::logic_error(
        "FVRO paired reference-format-6 diagnostics do not partition totals");
  if (total_fvro_selected_channels > total_fvro_strict_channel_steps ||
      total_fvro_strict_channel_steps > total_fvro_channel_serialized ||
      ((total_fvro_selected_channels == 0) !=
       (total_fvro_strict_channel_steps == 0)))
    throw std::logic_error(
        "FVRO batch-level strict channel-step accounting diverged");
  if (total_ssbb_composition_transitions_generated !=
          total_ssbb_composition_transitions_deduplicated +
              total_ssbb_composition_direct_feasible +
              total_ssbb_composition_direct_infeasible ||
      total_ssbb_composition_direct_feasible !=
          total_ssbb_composition_candidates_serialized ||
      total_ssbb_composition_singletons_serialized !=
          total_ssbb_composition_local_options ||
      total_ssbb_composition_frontier_carries +
              total_ssbb_composition_direct_feasible !=
          total_ssbb_composition_anchor_retention_events +
              total_ssbb_composition_format_retention_events +
              total_ssbb_composition_rate_retention_events +
              total_ssbb_composition_states_pruned ||
      total_ssbb_composition_max_frontier >
          kSsbbCompositionBeamWidth ||
      (total_ssbb_composition_selected_groups == 0) !=
          (total_ssbb_composition_saved_bytes == 0))
    throw std::logic_error(
        "SSBB composition batch-level diagnostics diverged");
  if (total_ssbb_representation_candidates_serialized !=
          total_ssbb_composition_starts ||
      total_ssbb_representation_improved_starts >
          total_ssbb_representation_candidates_serialized ||
      (total_ssbb_representation_improved_starts == 0) !=
          (total_ssbb_representation_gross_saved_bytes == 0) ||
      total_ssbb_representation_selected_groups >
          total_ssbb_selected_groups ||
      total_ssbb_search_selected_groups > total_ssbb_selected_groups ||
      (total_ssbb_representation_selected_groups == 0) !=
          (total_ssbb_selected_representation_saved_bytes == 0) ||
      (total_ssbb_search_selected_groups == 0) !=
          (total_ssbb_selected_search_saved_bytes == 0) ||
      total_ssbb_incremental_saved_bytes !=
          total_ssbb_selected_representation_saved_bytes +
              total_ssbb_selected_search_saved_bytes ||
      (total_ssbb_selected_groups == 0) !=
          (total_ssbb_incremental_saved_bytes == 0) ||
      total_ssbb_saved_bytes < total_ssbb_incremental_saved_bytes)
    throw std::logic_error(
        "SSBB representation/search batch-level diagnostics diverged");
  if (total_cjrf_relaxed_base_attempts !=
          total_cjrf_relaxed_base_failures +
              total_cjrf_relaxed_base_pruned +
              total_cjrf_relaxed_base_duplicates +
              total_cjrf_relaxed_base_records ||
      total_cjrf_relaxed_legacy_repaired !=
          total_cjrf_relaxed_base_records ||
      total_cjrf_relaxed_legacy_repair_candidates <
          total_cjrf_relaxed_legacy_repaired ||
      total_cjrf_relaxed_base_outside_pointwise >
          total_cjrf_relaxed_base_records ||
      total_cjrf_selected_groups !=
          total_cjrf_legacy_winner_groups +
              total_cjrf_fvro_winner_groups +
              total_cjrf_zrcar_winner_groups +
              total_cjrf_ssbb_winner_groups ||
      (total_cjrf_selected_groups == 0) !=
          (total_cjrf_saved_bytes == 0))
    throw std::logic_error(
        "CJRF batch-level accounting or no-regression provenance diverged");
  auto expected_cfsr_attempts_total = std::uint64_t{0};
  if (uses_cfsr(settings.surrogate_frontier)) {
    const auto attempts_per_window = static_cast<std::uint64_t>(
        expected_cfsr_base_attempts(settings));
    if (attempts_per_window == 0U ||
        static_cast<std::uint64_t>(settings.batch) >
            std::numeric_limits<std::uint64_t>::max() /
                attempts_per_window)
      throw std::overflow_error(
          "CFSR expected batch attempt total overflow");
    expected_cfsr_attempts_total =
        static_cast<std::uint64_t>(settings.batch) * attempts_per_window;
  }
  const auto expected_cfsr_surrogate_sources = uses_cfsr(
      settings.surrogate_frontier)
      ? static_cast<std::uint64_t>(settings.batch)
      : 0U;
  if (total_cfsr_surrogate_sources != expected_cfsr_surrogate_sources ||
      total_cfsr_surrogate_base_attempts != expected_cfsr_attempts_total ||
      total_cfsr_surrogate_base_attempts !=
          total_cfsr_surrogate_base_failures +
              total_cfsr_surrogate_base_pruned +
              total_cfsr_surrogate_base_duplicates +
              total_cfsr_surrogate_base_records ||
      total_cfsr_surrogate_legacy_repaired !=
          total_cfsr_surrogate_base_records ||
      total_cfsr_surrogate_legacy_repair_candidates <
          total_cfsr_surrogate_legacy_repaired ||
      total_cfsr_surrogate_base_outside_original_pointwise >
          total_cfsr_surrogate_base_records ||
      total_cfsr_selected_groups !=
          total_cfsr_legacy_winner_groups +
              total_cfsr_fvro_winner_groups +
              total_cfsr_zrcar_winner_groups +
              total_cfsr_ssbb_winner_groups ||
      total_cfsr_selected_groups !=
          total_cfsr_mode1_winner_groups +
              total_cfsr_mode3_winner_groups +
              total_cfsr_mode4_winner_groups +
              total_cfsr_mode5_winner_groups +
              total_cfsr_mode6_winner_groups ||
      total_cfsr_selected_groups > total_cfsr_surrogate_base_records ||
      (total_cfsr_selected_groups == 0) !=
          (total_cfsr_saved_bytes == 0))
    throw std::logic_error(
        "CFSR batch-level accounting or no-regression provenance diverged");
  write_stream(argv[3], settings, windows);

  // A second, outer-stream decode detects framing/metadata mistakes and is the
  // sole source of the reconstruction returned to Python.
  auto [decoded_settings, reconstructed] = decode_stream(argv[3]);
  if (decoded_settings.batch != settings.batch ||
      decoded_settings.channels != settings.channels || decoded_settings.time != settings.time)
    throw std::runtime_error("decoded outer stream dimensions changed");
  const auto outer_report = summarize_outer_constraints(
      source, reconstructed, decoded_settings);
  if (!outer_report.all_ok())
    throw std::runtime_error("final outer-stream decode violated a hard constraint");
  write_floats(argv[4], reconstructed);
  std::cout << "QPET_META corrections=" << total_corrections
            << " online_corrections=" << total_online_corrections
            << " candidates=" << total_candidates
            << " candidate_failures=" << total_candidate_failures
            << " base_bytes=" << total_base_bytes
            << " correction_bytes=" << total_correction_bytes
            << " strict_constant_groups=" << total_strict_constant_groups
            << " packed_quantized_groups=" << total_packed_quantized_groups
            << " adaptive_packed_groups=" << total_adaptive_packed_groups
            << " factorized_packed_groups=" << total_factorized_packed_groups
            << " subset_rank_packed_groups="
            << total_subset_rank_packed_groups
            << " predictive_packed_groups="
            << total_predictive_packed_groups
            << " shared_base_search_groups=" << total_shared_base_search_groups
            << " candidate_profile=" << profile_name
            << " budget_mode=" << (settings.paper_budget_profile == 1 ? "paper_budget_uniform" :
                (settings.paper_budget_profile == 2 ? "fixed_grid4" : "disabled"))
            << " budget_analyses=" << budget_audit.analyses
            << " budget_plans=" << budget_audit.plans
            << " budget_base_attempts=" << budget_audit.base_attempts
            << " budget_base_failures=" << budget_audit.base_failures
            << " budget_base_candidates=" << budget_audit.base_candidates
            << " budget_base_passes=" << budget_audit.base_passes
            << " budget_base_bound_violations=" << budget_audit.base_bound_violations
            << " budget_analysis_seconds=" << budget_audit.analysis_seconds
            << " repair_optimizer="
            << repair_optimizer_name(settings.repair_optimizer)
            << " correction_format_ceiling="
            << static_cast<unsigned int>(settings.correction_format_ceiling)
            << " base_frontier="
            << base_frontier_name(settings.base_frontier)
            << " cjrf_ratio_mask="
            << static_cast<unsigned int>(settings.cjrf_ratio_mask)
            << " cjrf_relaxed_base_attempts="
            << total_cjrf_relaxed_base_attempts
            << " cjrf_relaxed_base_failures="
            << total_cjrf_relaxed_base_failures
            << " cjrf_relaxed_base_pruned="
            << total_cjrf_relaxed_base_pruned
            << " cjrf_relaxed_base_duplicates="
            << total_cjrf_relaxed_base_duplicates
            << " cjrf_relaxed_base_records="
            << total_cjrf_relaxed_base_records
            << " cjrf_relaxed_base_outside_pointwise="
            << total_cjrf_relaxed_base_outside_pointwise
            << " cjrf_relaxed_legacy_repaired="
            << total_cjrf_relaxed_legacy_repaired
            << " cjrf_relaxed_legacy_repair_candidates="
            << total_cjrf_relaxed_legacy_repair_candidates
            << " cjrf_selected_groups="
            << total_cjrf_selected_groups
            << " cjrf_saved_bytes=" << total_cjrf_saved_bytes
            << " cjrf_legacy_winner_groups="
            << total_cjrf_legacy_winner_groups
            << " cjrf_fvro_winner_groups="
            << total_cjrf_fvro_winner_groups
            << " cjrf_zrcar_winner_groups="
            << total_cjrf_zrcar_winner_groups
            << " cjrf_ssbb_winner_groups="
            << total_cjrf_ssbb_winner_groups
            << " surrogate_frontier="
            << surrogate_frontier_name(settings.surrogate_frontier)
            << " cfsr_ratio_mask="
            << static_cast<unsigned int>(settings.cfsr_ratio_mask)
            << " cfsr_surrogate_sources="
            << total_cfsr_surrogate_sources
            << " cfsr_surrogate_base_attempts="
            << total_cfsr_surrogate_base_attempts
            << " cfsr_surrogate_base_failures="
            << total_cfsr_surrogate_base_failures
            << " cfsr_surrogate_base_pruned="
            << total_cfsr_surrogate_base_pruned
            << " cfsr_surrogate_base_duplicates="
            << total_cfsr_surrogate_base_duplicates
            << " cfsr_surrogate_base_records="
            << total_cfsr_surrogate_base_records
            << " cfsr_surrogate_base_outside_original_pointwise="
            << total_cfsr_surrogate_base_outside_original_pointwise
            << " cfsr_surrogate_legacy_repaired="
            << total_cfsr_surrogate_legacy_repaired
            << " cfsr_surrogate_legacy_repair_candidates="
            << total_cfsr_surrogate_legacy_repair_candidates
            << " cfsr_selected_groups=" << total_cfsr_selected_groups
            << " cfsr_saved_bytes=" << total_cfsr_saved_bytes
            << " cfsr_legacy_winner_groups="
            << total_cfsr_legacy_winner_groups
            << " cfsr_fvro_winner_groups="
            << total_cfsr_fvro_winner_groups
            << " cfsr_zrcar_winner_groups="
            << total_cfsr_zrcar_winner_groups
            << " cfsr_ssbb_winner_groups="
            << total_cfsr_ssbb_winner_groups
            << " cfsr_mode1_winner_groups="
            << total_cfsr_mode1_winner_groups
            << " cfsr_mode3_winner_groups="
            << total_cfsr_mode3_winner_groups
            << " cfsr_mode4_winner_groups="
            << total_cfsr_mode4_winner_groups
            << " cfsr_mode5_winner_groups="
            << total_cfsr_mode5_winner_groups
            << " cfsr_mode6_winner_groups="
            << total_cfsr_mode6_winner_groups
            << " zrcar_base_records=" << total_zrcar_base_records
            << " zrcar_states_generated=" << total_zrcar_states_generated
            << " zrcar_infeasible_states=" << total_zrcar_infeasible_states
            << " zrcar_infeasible_frontier_expansions="
            << total_zrcar_infeasible_frontier_expansions
            << " zrcar_infeasible_frontier_recoveries="
            << total_zrcar_infeasible_frontier_recoveries
            // Compatibility alias for parsers written before the endpoint
            // feasibility gate was removed. The value now means states that
            // exhausted their actual deterministic repair path.
            << " zrcar_target_states_pruned=" << total_zrcar_infeasible_states
            << " zrcar_candidates=" << total_zrcar_candidates
            << " zrcar_prefix_candidates=" << total_zrcar_prefix_candidates
            << " zrcar_local_candidates=" << total_zrcar_local_candidates
            << " zrcar_legacy_seeds=" << total_zrcar_legacy_seeds
            << " zrcar_fvro_seeds=" << total_zrcar_fvro_seeds
            << " zrcar_selected_groups=" << total_zrcar_selected_groups
            << " zrcar_saved_bytes=" << total_zrcar_saved_bytes
            << " zrcar_from_fvro_seed_groups="
            << total_zrcar_from_fvro_seed_groups
            << " fvro_base_records=" << total_fvro_base_records
            << " fvro_global_generated=" << total_fvro_global_generated
            << " fvro_global_infeasible=" << total_fvro_global_infeasible
            << " fvro_global_serialized=" << total_fvro_global_serialized
            << " fvro_channel_generated=" << total_fvro_channel_generated
            << " fvro_channel_infeasible=" << total_fvro_channel_infeasible
            << " fvro_channel_serialized=" << total_fvro_channel_serialized
            << " fvro_local_generated=" << total_fvro_local_generated
            << " fvro_local_infeasible=" << total_fvro_local_infeasible
            << " fvro_selected_channels=" << total_fvro_selected_channels
            << " fvro_strict_channel_steps="
            << total_fvro_strict_channel_steps
            << " fvro_seeds_retained=" << total_fvro_seeds_retained
            << " fvro_reference_quantized_groups="
            << total_fvro_reference_quantized_groups
            << " fvro_final_quantized_groups="
            << total_fvro_final_quantized_groups
            << " fvro_reference_format4_groups="
            << total_fvro_reference_format4_groups
            << " fvro_reference_format6_groups="
            << total_fvro_reference_format6_groups
            << " fvro_final_format4_groups="
            << total_fvro_final_format4_groups
            << " fvro_final_format6_groups="
            << total_fvro_final_format6_groups
            << " fvro_final_format7_groups="
            << total_fvro_final_format7_groups
            << " fvro_final_format8_groups="
            << total_fvro_final_format8_groups
            << " fvro_final_format9_groups="
            << total_fvro_final_format9_groups
            << " fvro_final_format10_groups="
            << total_fvro_final_format10_groups
            << " fvro_reference_exact_escapes="
            << total_fvro_reference_exact_escapes
            << " fvro_final_exact_escapes="
            << total_fvro_final_exact_escapes
            << " fvro_reference_format4_exact_escapes="
            << total_fvro_reference_format4_exact_escapes
            << " fvro_reference_format6_exact_escapes="
            << total_fvro_reference_format6_exact_escapes
            << " fvro_final_format4_exact_escapes="
            << total_fvro_final_format4_exact_escapes
            << " fvro_final_format6_exact_escapes="
            << total_fvro_final_format6_exact_escapes
            << " fvro_final_format7_exact_escapes="
            << total_fvro_final_format7_exact_escapes
            << " fvro_final_format8_exact_escapes="
            << total_fvro_final_format8_exact_escapes
            << " fvro_final_format9_exact_escapes="
            << total_fvro_final_format9_exact_escapes
            << " fvro_final_format10_exact_escapes="
            << total_fvro_final_format10_exact_escapes
            << " fvro_ref6_to_final6_groups="
            << total_fvro_ref6_to_final6_groups
            << " fvro_ref6_to_non6_groups="
            << total_fvro_ref6_to_non6_groups
            << " fvro_ref6_to_final6_reference_exact_escapes="
            << total_fvro_ref6_to_final6_reference_exact_escapes
            << " fvro_ref6_to_final6_final_exact_escapes="
            << total_fvro_ref6_to_final6_final_exact_escapes
            << " fvro_ref6_to_non6_reference_exact_escapes="
            << total_fvro_ref6_to_non6_reference_exact_escapes
            << " fvro_selected_groups=" << total_fvro_selected_groups
            << " fvro_saved_bytes=" << total_fvro_saved_bytes
            << " ssbb_base_records=" << total_ssbb_base_records
            << " ssbb_channels_searched=" << total_ssbb_channels_searched
            << " ssbb_positions_processed=" << total_ssbb_positions_processed
            << " ssbb_actions_generated=" << total_ssbb_actions_generated
            << " ssbb_states_generated=" << total_ssbb_states_generated
            << " ssbb_states_deduplicated="
            << total_ssbb_states_deduplicated
            << " ssbb_statistics_feasible_states="
            << total_ssbb_statistics_feasible_states
            << " ssbb_statistics_infeasible_states="
            << total_ssbb_statistics_infeasible_states
            << " ssbb_feasible_states_retained="
            << total_ssbb_feasible_states_retained
            << " ssbb_infeasible_states_retained="
            << total_ssbb_infeasible_states_retained
            << " ssbb_states_pruned_by_beam="
            << total_ssbb_states_pruned_by_beam
            << " ssbb_terminal_states=" << total_ssbb_terminal_states
            << " ssbb_terminal_direct_feasible="
            << total_ssbb_terminal_direct_feasible
            << " ssbb_terminal_direct_infeasible="
            << total_ssbb_terminal_direct_infeasible
            << " ssbb_outputs_returned=" << total_ssbb_outputs_returned
            << " ssbb_candidates_serialized="
            << total_ssbb_candidates_serialized
            << " ssbb_seeds_retained=" << total_ssbb_seeds_retained
            << " ssbb_selected_channels=" << total_ssbb_selected_channels
            << " ssbb_representation_candidates_serialized="
            << total_ssbb_representation_candidates_serialized
            << " ssbb_representation_improved_starts="
            << total_ssbb_representation_improved_starts
            << " ssbb_representation_gross_saved_bytes="
            << total_ssbb_representation_gross_saved_bytes
            << " ssbb_composition_starts="
            << total_ssbb_composition_starts
            << " ssbb_composition_local_options="
            << total_ssbb_composition_local_options
            << " ssbb_composition_transitions_generated="
            << total_ssbb_composition_transitions_generated
            << " ssbb_composition_transitions_deduplicated="
            << total_ssbb_composition_transitions_deduplicated
            << " ssbb_composition_direct_feasible="
            << total_ssbb_composition_direct_feasible
            << " ssbb_composition_direct_infeasible="
            << total_ssbb_composition_direct_infeasible
            << " ssbb_composition_candidates_serialized="
            << total_ssbb_composition_candidates_serialized
            << " ssbb_composition_singletons_serialized="
            << total_ssbb_composition_singletons_serialized
            << " ssbb_composition_pairs_serialized="
            << total_ssbb_composition_pairs_serialized
            << " ssbb_composition_frontier_carries="
            << total_ssbb_composition_frontier_carries
            << " ssbb_composition_anchor_retention_events="
            << total_ssbb_composition_anchor_retention_events
            << " ssbb_composition_format_retention_events="
            << total_ssbb_composition_format_retention_events
            << " ssbb_composition_rate_retention_events="
            << total_ssbb_composition_rate_retention_events
            << " ssbb_composition_states_pruned="
            << total_ssbb_composition_states_pruned
            << " ssbb_composition_max_frontier="
            << total_ssbb_composition_max_frontier
            << " ssbb_composition_selected_channels="
            << total_ssbb_composition_selected_channels
            << " ssbb_composition_selected_groups="
            << total_ssbb_composition_selected_groups
            << " ssbb_composition_saved_bytes="
            << total_ssbb_composition_saved_bytes
            << " ssbb_selected_groups=" << total_ssbb_selected_groups
            << " ssbb_saved_bytes=" << total_ssbb_saved_bytes
            << " ssbb_incremental_saved_bytes="
            << total_ssbb_incremental_saved_bytes
            << " ssbb_representation_selected_groups="
            << total_ssbb_representation_selected_groups
            << " ssbb_selected_representation_saved_bytes="
            << total_ssbb_selected_representation_saved_bytes
            << " ssbb_search_selected_groups="
            << total_ssbb_search_selected_groups
            << " ssbb_selected_search_saved_bytes="
            << total_ssbb_selected_search_saved_bytes
            << " second_stage_selected_groups="
            << total_second_stage_selected_groups
            << " second_stage_saved_bytes="
            << total_second_stage_saved_bytes;
  write_qpet_constraint_metrics(std::cout, outer_report);
  std::cout << '\n';
}

void decompress_mode(int argc, char** argv)
{
  if (argc != 4)
    throw std::invalid_argument(
        "usage: qpet_correlation_batch decompress stream.qpc reconstruction.f32");
  auto [settings, reconstructed] = decode_stream(argv[2]);
  (void)settings;
  write_floats(argv[3], reconstructed);
}

bool bitwise_equal_floats(
    const std::vector<float>& left,
    const std::vector<float>& right);

struct FactorizedScalePrefix {
  std::vector<float> scales;
  std::size_t raw_bytes = 0;
  std::size_t none_channels = 0;
  std::size_t all_channels = 0;
  std::size_t partial_channels = 0;
};

FactorizedScalePrefix recover_factorized_scale_prefix(
    const std::vector<unsigned char>& payload,
    std::uint32_t expected_corrections,
    std::size_t channels,
    std::size_t time)
{
  if (payload.size() <= 1 ||
      payload.front() != kFactorizedPackedCorrectionFormat ||
      expected_corrections == 0 || channels == 0 || time == 0)
    throw std::runtime_error(
        "fixed serializer source is not a nonempty format 8 payload");
  auto maximum_raw_size = channels;
  const auto maximum_active = std::min(
      channels, static_cast<std::size_t>(expected_corrections));
  const auto bitmap_bytes = time / 8U + (time % 8U != 0);
  auto add_product = [&](std::size_t count,
                         std::size_t bytes,
                         const char* message) {
    if (bytes != 0 && count > std::numeric_limits<std::size_t>::max() / bytes)
      throw std::runtime_error(message);
    maximum_raw_size = checked_add_size(
        maximum_raw_size, count * bytes, message);
  };
  add_product(
      maximum_active,
      checked_add_size(
          bitmap_bytes,
          5U,
          "fixed serializer tag8 channel bound overflows"),
      "fixed serializer tag8 active-channel bound overflows");
  add_product(
      expected_corrections,
      5U,
      "fixed serializer tag8 value bound overflows");
  const auto raw = decompress_direct_zstd_frame(
      payload.data() + 1, payload.size() - 1, maximum_raw_size);
  if (raw.size() < channels)
    throw std::runtime_error(
        "fixed serializer tag8 descriptors are truncated");

  FactorizedScalePrefix result;
  result.scales.assign(channels, 0.0F);
  result.raw_bytes = raw.size();
  auto offset = channels;
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto descriptor = raw[channel];
    const auto position_mode = descriptor & 0x07U;
    const auto value_mode = (descriptor >> 3U) & 0x07U;
    if ((descriptor & 0xc0U) != 0 ||
        position_mode >
            static_cast<std::uint8_t>(
                FactorizedPositionMode::complement_gaps) ||
        value_mode >
            static_cast<std::uint8_t>(FactorizedValueMode::modal_patched) ||
        ((position_mode ==
          static_cast<std::uint8_t>(FactorizedPositionMode::none)) !=
         (value_mode ==
          static_cast<std::uint8_t>(FactorizedValueMode::none))))
      throw std::runtime_error(
          "fixed serializer tag8 descriptor is invalid");
    if (position_mode ==
        static_cast<std::uint8_t>(FactorizedPositionMode::none)) {
      ++result.none_channels;
      continue;
    }
    if (position_mode ==
        static_cast<std::uint8_t>(FactorizedPositionMode::all))
      ++result.all_channels;
    else
      ++result.partial_channels;
    if (raw.size() - offset < sizeof(float))
      throw std::runtime_error(
          "fixed serializer tag8 scale prefix is truncated");
    result.scales[channel] = read_pod<float>(raw, offset);
    if (!(result.scales[channel] >= 0.0F) ||
        !std::isfinite(result.scales[channel]))
      throw std::runtime_error(
          "fixed serializer tag8 scale is invalid");
  }
  if (result.none_channels + result.all_channels + result.partial_channels !=
      channels)
    throw std::logic_error(
        "fixed serializer channel partition diverged");
  return result;
}

std::size_t direct_zstd_declared_size(
    const std::vector<unsigned char>& tagged_payload)
{
  if (tagged_payload.size() <= 1)
    throw std::runtime_error("tagged ZSTD payload is empty");
  const auto declared = ZSTD_getFrameContentSize(
      tagged_payload.data() + 1, tagged_payload.size() - 1);
  if (declared == ZSTD_CONTENTSIZE_ERROR ||
      declared == ZSTD_CONTENTSIZE_UNKNOWN ||
      declared > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error(
        "tagged ZSTD payload has no representable content size");
  return static_cast<std::size_t>(declared);
}

void fixed_serializer_transcode_mode(int argc, char** argv)
{
  if (argc != 4)
    throw std::invalid_argument(
        "usage: qpet_correlation_batch fixed-serializer-transcode "
        "source-v14.qpc output-v15.qpc");
  auto source_records = std::vector<DecodedWindowRecord>();
  auto source_version = std::uint32_t{0};
  auto [settings, source_reconstruction] = decode_stream(
      argv[2], &source_records, &source_version);
  if (source_version != kFactorizedPackedVersion ||
      settings.group_size != 1 || source_records.empty())
    throw std::runtime_error(
        "fixed serializer transcode requires a nonempty group_size=1 QPC v14 stream");

  auto output_windows = std::vector<EncodedWindow>();
  output_windows.reserve(source_records.size());
  auto total_tag8_bytes = std::size_t{0};
  auto total_tag9_bytes = std::size_t{0};
  auto strict_wins = std::size_t{0};
  auto ties = std::size_t{0};
  auto losses = std::size_t{0};
  for (std::size_t record_index = 0;
       record_index < source_records.size();
       ++record_index) {
    const auto& record = source_records[record_index];
    if (record.logical_windows != 1 || record.corrections == 0 ||
        record.correction_payload.empty() ||
        record.correction_payload.front() !=
            kFactorizedPackedCorrectionFormat ||
        record.base_reconstruction.size() != settings.channels * settings.time ||
        record.reconstruction.size() != record.base_reconstruction.size())
      throw std::runtime_error(
          "fixed serializer source record is not an independent tag8 repair");

    const auto prefix = recover_factorized_scale_prefix(
        record.correction_payload,
        record.corrections,
        settings.channels,
        settings.time);
    const auto fields = make_quantized_correction_fields(
        record.base_reconstruction,
        record.reconstruction,
        prefix.scales,
        settings.channels,
        settings.time);
    if (fields.codes.size() != record.corrections)
      throw std::runtime_error(
          "fixed serializer recovered correction count changed");
    const auto canonical_tag8 = encode_factorized_packed_correction_format8(
        record.base_reconstruction,
        record.reconstruction,
        prefix.scales,
        fields,
        settings.channels,
        settings.time);
    if (canonical_tag8 != record.correction_payload)
      throw std::runtime_error(
          "fixed serializer could not reproduce the source tag8 bytes");
    const auto tag9 = encode_subset_rank_packed_correction_format9(
        record.base_reconstruction,
        record.reconstruction,
        prefix.scales,
        fields,
        settings.channels,
        settings.time);
    if (tag9.empty() || tag9.front() != kSubsetRankPackedCorrectionFormat)
      throw std::runtime_error(
          "fixed serializer source lies outside the tag9 production domain");
    auto tag9_reconstruction = record.base_reconstruction;
    apply_correction_payload(
        tag9_reconstruction,
        tag9,
        record.corrections,
        settings.channels,
        settings.time,
        kSubsetRankPackedVersion);
    if (!bitwise_equal_floats(tag9_reconstruction, record.reconstruction))
      throw std::runtime_error(
          "fixed serializer tag9 reconstruction differs from tag8");

    total_tag8_bytes = checked_add_size(
        total_tag8_bytes,
        canonical_tag8.size(),
        "fixed serializer tag8 byte total overflows");
    total_tag9_bytes = checked_add_size(
        total_tag9_bytes,
        tag9.size(),
        "fixed serializer tag9 byte total overflows");
    strict_wins += static_cast<std::size_t>(tag9.size() < canonical_tag8.size());
    ties += static_cast<std::size_t>(tag9.size() == canonical_tag8.size());
    losses += static_cast<std::size_t>(tag9.size() > canonical_tag8.size());

    EncodedWindow output;
    output.logical_windows = record.logical_windows;
    output.base_mode = record.base_mode;
    output.selected_global_bound = record.selected_global_bound;
    output.corrections = record.corrections;
    output.payload = record.payload;
    output.correction_payload = tag9;
    output.reconstruction = tag9_reconstruction;
    output_windows.push_back(std::move(output));

    std::cout << "QPET_FIXED_SERIALIZER_RECORD index=" << record_index
              << " corrections=" << record.corrections
              << " base_bytes=" << record.payload.size()
              << " tag8_bytes=" << canonical_tag8.size()
              << " tag8_raw_bytes=" << prefix.raw_bytes
              << " tag9_bytes=" << tag9.size()
              << " tag9_raw_bytes=" << direct_zstd_declared_size(tag9)
              << " none_channels=" << prefix.none_channels
              << " all_channels=" << prefix.all_channels
              << " partial_channels=" << prefix.partial_channels
              << " exact_escapes=" << fields.exact_values.size()
              << " tag8_byte_equal=1 reconstruction_bitwise_equal=1"
              << " search_performed=0\n";
  }

  auto output_settings = settings;
  output_settings.repair_optimizer = RepairOptimizer::ssbb_value;
  output_settings.correction_format_ceiling =
      kSubsetRankPackedCorrectionFormat;
  validate_settings(output_settings);
  write_stream(argv[3], output_settings, output_windows);
  auto output_records = std::vector<DecodedWindowRecord>();
  auto output_version = std::uint32_t{0};
  const auto [decoded_output_settings, output_reconstruction] = decode_stream(
      argv[3], &output_records, &output_version);
  (void)decoded_output_settings;
  if (output_version != kSubsetRankPackedVersion ||
      output_records.size() != source_records.size() ||
      !bitwise_equal_floats(output_reconstruction, source_reconstruction))
    throw std::runtime_error(
        "fixed serializer persisted QPC v15 replay changed reconstruction");
  for (std::size_t index = 0; index < source_records.size(); ++index) {
    const auto& source = source_records[index];
    const auto& output = output_records[index];
    if (source.logical_windows != output.logical_windows ||
        source.base_mode != output.base_mode ||
        !same_binary64_bits(
            source.selected_global_bound, output.selected_global_bound) ||
        source.corrections != output.corrections ||
        source.payload != output.payload ||
        !bitwise_equal_floats(
            source.base_reconstruction, output.base_reconstruction) ||
        !bitwise_equal_floats(source.reconstruction, output.reconstruction))
      throw std::runtime_error(
          "fixed serializer changed base bytes, record metadata, or reconstruction");
  }
  if (total_tag8_bytes >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      total_tag9_bytes >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error(
        "fixed serializer signed byte delta exceeds int64");
  const auto signed_saved = static_cast<std::int64_t>(total_tag8_bytes) -
      static_cast<std::int64_t>(total_tag9_bytes);
  std::cout << "QPET_FIXED_SERIALIZER_SUMMARY records="
            << source_records.size()
            << " tag8_bytes=" << total_tag8_bytes
            << " tag9_bytes=" << total_tag9_bytes
            << " saved_bytes=" << signed_saved
            << " wins=" << strict_wins
            << " ties=" << ties
            << " losses=" << losses
            << " source_qpc_bytes=" << read_bytes(argv[2]).size()
            << " output_qpc_bytes=" << read_bytes(argv[3]).size()
            << " reconstruction_bitwise_equal=1 search_performed=0\n";
}

void repair_external_mode(int argc, char** argv)
{
  if (argc != 14)
    throw std::invalid_argument(
        "usage: qpet_correlation_batch repair-external original.f32 base.f32 "
        "correction.bin reconstruction.f32 channels time epsilon acf_epsilon "
        "min_correlation max_lag correction_mode repair_denominators");
  Settings settings;
  settings.batch = 1;
  settings.channels = std::stoull(argv[6]);
  settings.time = std::stoull(argv[7]);
  settings.pointwise_tolerance = std::stod(argv[8]);
  settings.acf_tolerance = std::stod(argv[9]);
  settings.minimum_correlation = std::stod(argv[10]);
  settings.max_lag = std::stoull(argv[11]);
  settings.correction_mode = parse_correction_mode(argv[12]);
  settings.repair_denominators = parse_repair_denominators(argv[13]);
  settings.group_size = 1;
  if (settings.correction_mode == 0)
    throw std::invalid_argument("external-base repair requires a post-hoc mode");
  validate_settings(settings);
  if (settings.channels > std::numeric_limits<std::size_t>::max() / settings.time)
    throw std::invalid_argument("external-base dimensions overflow size_t");
  const auto count = settings.channels * settings.time;
  const auto original = read_floats(argv[2], count);
  EncodedWindow repaired;
  repaired.reconstruction = read_floats(argv[3], count);
  finalize_posthoc_repair(original, settings, repaired);
  repaired.report = qpet_correlation::verify_constraints(
      original,
      repaired.reconstruction,
      settings.channels,
      settings.time,
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag);
  if (!repaired.report.all_ok())
    throw std::runtime_error("external-base repair failed a hard constraint");
  write_bytes(argv[4], repaired.correction_payload);
  write_floats(argv[5], repaired.reconstruction);
  std::cout << "QPET_REPAIR_META corrections=" << repaired.corrections
            << " candidates=" << repaired.repair_candidates_tested
            << " correction_bytes=" << repaired.correction_payload.size()
            << " max_abs=" << repaired.report.max_abs_error
            << " max_acf=" << repaired.report.max_acf_error
            << " min_corr=" << repaired.report.min_series_correlation << '\n';
}

void apply_external_mode(int argc, char** argv)
{
  if (argc != 8)
    throw std::invalid_argument(
        "usage: qpet_correlation_batch apply-external base.f32 correction.bin "
        "reconstruction.f32 correction_count channels time");
  const auto correction_count = static_cast<std::uint32_t>(std::stoul(argv[5]));
  const auto channels = std::stoull(argv[6]);
  const auto time = std::stoull(argv[7]);
  if (channels == 0 || time == 0 ||
      channels > std::numeric_limits<std::size_t>::max() / time)
    throw std::invalid_argument("external correction dimensions are invalid");
  auto reconstruction = read_floats(argv[2], channels * time);
  const auto correction = read_bytes(argv[3]);
  apply_correction_payload(
      reconstruction,
      correction,
      correction_count,
      channels,
      time,
      correction_payload_required_stream_version(correction));
  write_floats(argv[4], reconstruction);
}

bool bitwise_equal_floats(
    const std::vector<float>& left,
    const std::vector<float>& right)
{
  return left.size() == right.size() &&
      (left.empty() ||
       std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0);
}

std::vector<unsigned char> frame_packed_correction_raw(
    const std::vector<unsigned char>& raw)
{
  const auto compressed = compress_direct_zstd_frame(raw);
  auto payload = std::vector<unsigned char>();
  payload.reserve(1 + compressed.size());
  payload.push_back(kPackedQuantizedCorrectionFormat);
  payload.insert(payload.end(), compressed.begin(), compressed.end());
  return payload;
}

std::vector<unsigned char> frame_adaptive_packed_correction_raw(
    const std::vector<unsigned char>& raw)
{
  const auto compressed = compress_direct_zstd_frame(raw);
  auto payload = std::vector<unsigned char>();
  payload.reserve(1 + compressed.size());
  payload.push_back(kAdaptivePackedCorrectionFormat);
  payload.insert(payload.end(), compressed.begin(), compressed.end());
  return payload;
}

std::vector<unsigned char> frame_factorized_packed_correction_raw(
    const std::vector<unsigned char>& raw)
{
  const auto compressed = compress_direct_zstd_frame(raw);
  auto payload = std::vector<unsigned char>();
  payload.reserve(1 + compressed.size());
  payload.push_back(kFactorizedPackedCorrectionFormat);
  payload.insert(payload.end(), compressed.begin(), compressed.end());
  return payload;
}

std::vector<unsigned char> frame_subset_rank_packed_correction_raw(
    const std::vector<unsigned char>& raw)
{
  const auto compressed = compress_direct_zstd_frame(raw);
  auto payload = std::vector<unsigned char>();
  payload.reserve(1 + compressed.size());
  payload.push_back(kSubsetRankPackedCorrectionFormat);
  payload.insert(payload.end(), compressed.begin(), compressed.end());
  return payload;
}

void write_legacy_selftest_stream(
    const std::string& path,
    std::uint32_t version,
    const Settings& settings,
    const EncodedWindow& window)
{
  if (version != 8 && version != 9)
    throw std::invalid_argument("legacy self-test writer only supports QPC v8/v9");
  auto output = std::ofstream(path, std::ios::binary);
  if (!output)
    throw std::runtime_error("cannot open legacy QPC self-test stream");
  output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  write_scalar(output, version);
  write_scalar(output, static_cast<std::uint32_t>(settings.batch));
  write_scalar(output, static_cast<std::uint32_t>(settings.channels));
  write_scalar(output, static_cast<std::uint32_t>(settings.time));
  write_scalar(output, static_cast<std::uint32_t>(settings.max_lag));
  write_scalar(output, static_cast<std::uint32_t>(settings.qoz_level));
  std::uint32_t flags = 0;
  if (settings.acf_tolerance != kDisabledConstraint)
    flags |= kHasAcf;
  if (settings.minimum_correlation != kDisabledConstraint)
    flags |= kHasPearson;
  write_scalar(output, flags);
  write_scalar(output, settings.correction_mode);
  write_scalar(output, static_cast<std::uint32_t>(settings.group_size));
  write_scalar(output, settings.pointwise_tolerance);
  write_scalar(output, settings.acf_tolerance);
  write_scalar(output, settings.minimum_correlation);
  write_scalar(output, settings.qpet_confidence);
  write_scalar(output, settings.qpet_k);
  write_scalar(output, static_cast<std::uint32_t>(window.logical_windows));
  write_scalar(output, window.base_mode);
  write_scalar(output, window.selected_global_bound);
  write_scalar(output, window.selected_estimator_scale);
  write_scalar(output, window.corrections);
  write_scalar(output, window.candidates_tested);
  write_scalar(output, static_cast<std::uint64_t>(window.payload.size()));
  write_scalar(output, static_cast<std::uint64_t>(window.correction_payload.size()));
  output.write(
      reinterpret_cast<const char*>(window.payload.data()),
      static_cast<std::streamsize>(window.payload.size()));
  output.write(
      reinterpret_cast<const char*>(window.correction_payload.data()),
      static_cast<std::streamsize>(window.correction_payload.size()));
  if (!output)
    throw std::runtime_error("failed to write legacy QPC self-test stream");
}

void packed_correction_selftest_mode(int argc, char** argv)
{
  if (argc != 3)
    throw std::invalid_argument(
        "usage: qpet_correlation_batch packed-selftest output-prefix");
  if (Settings{}.correction_format_ceiling !=
          kPackedQuantizedCorrectionFormat ||
      omitted_correction_format_ceiling(RepairOptimizer::legacy) !=
          kPackedQuantizedCorrectionFormat ||
      omitted_correction_format_ceiling(RepairOptimizer::fvro) !=
          kPackedQuantizedCorrectionFormat ||
      omitted_correction_format_ceiling(RepairOptimizer::ssbb_value) !=
          kFactorizedPackedCorrectionFormat ||
      omitted_correction_format_ceiling(RepairOptimizer::ssbb) !=
          kFactorizedPackedCorrectionFormat ||
      parse_correction_format_ceiling("6") !=
          kPackedQuantizedCorrectionFormat ||
      parse_correction_format_ceiling("7") !=
          kAdaptivePackedCorrectionFormat ||
      parse_correction_format_ceiling("8") !=
          kFactorizedPackedCorrectionFormat ||
      parse_correction_format_ceiling("9") !=
          kSubsetRankPackedCorrectionFormat ||
      parse_correction_format_ceiling("10") !=
          kPredictivePackedCorrectionFormat)
    throw std::runtime_error(
        "correction format ceiling defaults/parser changed semantics");
  for (const auto& invalid :
       {std::string(), std::string("5"), std::string("11"),
        std::string("07"), std::string("7junk"), std::string("-1")}) {
    auto rejected = false;
    try {
      (void)parse_correction_format_ceiling(invalid);
    }
    catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error(
          "correction format ceiling parser accepted a noncanonical token");
  }
  constexpr std::size_t channels = 4;
  constexpr std::size_t time = 512;
  constexpr std::size_t values = channels * time;
  auto every_int8 = std::vector<std::int8_t>();
  every_int8.reserve(256);
  for (int value = std::numeric_limits<std::int8_t>::min();
       value <= std::numeric_limits<std::int8_t>::max(); ++value)
    every_int8.push_back(static_cast<std::int8_t>(value));
  const auto every_int8_packed = pack_int8_codes(every_int8);
  if (every_int8_packed.bit_width != 8 ||
      unpack_int8_codes(
          every_int8_packed.bytes.data(),
          every_int8_packed.bytes.size(),
          every_int8.size(),
          every_int8_packed.bit_width) != every_int8)
    throw std::runtime_error("packed self-test failed the complete int8 alphabet");
  for (std::uint8_t width = 1; width <= 8; ++width) {
    const auto boundary = static_cast<std::uint8_t>(1U << (width - 1U));
    const auto code = unzigzag_int8(boundary);
    const auto width_codes = std::vector<std::int8_t>{code, 0, code};
    const auto packed = pack_int8_codes(width_codes);
    if (packed.bit_width != width ||
        unpack_int8_codes(
            packed.bytes.data(),
            packed.bytes.size(),
            width_codes.size(),
            packed.bit_width) != width_codes)
      throw std::runtime_error("packed self-test failed a bit-width boundary");
  }
  for (std::uint8_t width = 1; width <= 9; ++width) {
    const auto boundary = static_cast<std::uint16_t>(1U << (width - 1U));
    const auto width_deltas = std::vector<std::uint16_t>{boundary, 0U, boundary};
    const auto packed = pack_uint9_codes(width_deltas);
    if (packed.bit_width != width ||
        unpack_uint9_codes(
            packed.bytes.data(),
            packed.bytes.size(),
            width_deltas.size(),
            packed.bit_width) != width_deltas)
      throw std::runtime_error(
          "predictive packed self-test failed a uint9 bit-width boundary");
  }
  auto base = std::vector<float>(values);
  auto corrected = std::vector<float>(values);
  const auto scales = std::vector<float>{0.125F, 0.25F, 0.5F, 1.0F};
  std::uint32_t state = 0x9e3779b9U;
  for (std::size_t channel = 0; channel < channels; ++channel) {
    for (std::size_t index = 0; index < time; ++index) {
      const auto offset = channel * time + index;
      base[offset] = static_cast<float>(
          static_cast<double>(channel) * 16.0 +
          static_cast<double>(index) / 64.0);
      state = state * 1664525U + 1013904223U;
      auto code = static_cast<int>((state >> 16U) % 15U) - 7;
      if (code == 0)
        code = 7;
      corrected[offset] = offset % 19U == 0
          ? static_cast<float>(base[offset] + scales[channel] / 3.0F)
          : static_cast<float>(base[offset] + static_cast<float>(code) * scales[channel]);
    }
  }
  const auto fields = make_quantized_correction_fields(
      base, corrected, scales, channels, time);
  if (fields.codes.size() != values)
    throw std::runtime_error("packed self-test did not create a dense correction mask");
  const auto format4 = encode_quantized_correction_format4(scales, fields);
  const auto format6 = encode_quantized_correction_format6(scales, fields);
  const auto packed_codes = pack_int8_codes(fields.codes);
  if (format6.size() >= format4.size() || packed_codes.bit_width != 4)
    throw std::runtime_error("packed self-test fixture did not improve format 4");
  const auto selected = compress_quantized_correction_payload(
      base, corrected, scales, channels, time);
  if (selected != format6)
    throw std::runtime_error("format 6 was not selected despite a strict byte win");
  const auto basic_selected = compress_basic_correction_payload(
      base, corrected, &scales, channels, time);
  if (basic_selected.empty() ||
      basic_selected.front() != kPackedQuantizedCorrectionFormat)
    throw std::runtime_error("format 6 did not win the complete correction portfolio");

  auto format4_decoded = base;
  apply_correction_payload(
      format4_decoded,
      format4,
      static_cast<std::uint32_t>(fields.codes.size()),
      channels,
      time,
      11);
  auto format6_decoded = base;
  apply_correction_payload(
      format6_decoded,
      format6,
      static_cast<std::uint32_t>(fields.codes.size()),
      channels,
      time,
      12);
  if (!bitwise_equal_floats(format4_decoded, corrected) ||
      !bitwise_equal_floats(format6_decoded, corrected))
    throw std::runtime_error("format 4/6 A/B changed correction semantics");

  // A separate fixture forces four different per-channel widths and exact
  // XOR escapes.  This makes format 7's defining representation choices
  // observable rather than merely exercising its dispatch branch.
  constexpr std::size_t adaptive_channels = 4;
  constexpr std::size_t adaptive_time = 257;
  constexpr std::size_t adaptive_values = adaptive_channels * adaptive_time;
  auto adaptive_base = std::vector<float>(adaptive_values);
  auto adaptive_corrected = adaptive_base;
  const auto adaptive_scales =
      std::vector<float>{0.125F, 0.125F, 0.125F, 0.125F};
  const auto adaptive_magnitudes = std::array<int, adaptive_channels>{1, 3, 7, 15};
  for (std::size_t channel = 0; channel < adaptive_channels; ++channel) {
    for (std::size_t index = 0; index < adaptive_time; ++index) {
      const auto offset = channel * adaptive_time + index;
      adaptive_base[offset] = static_cast<float>(
          static_cast<double>(channel) * 32.0 +
          static_cast<double>(index) / 16.0);
      if (index % 29U == 0) {
        adaptive_corrected[offset] = static_cast<float>(
            adaptive_base[offset] + adaptive_scales[channel] / 3.0F);
      }
      else {
        const auto sign = ((index * 17U + channel * 13U) & 1U) == 0 ? 1 : -1;
        adaptive_corrected[offset] = static_cast<float>(
            adaptive_base[offset] +
            static_cast<float>(sign * adaptive_magnitudes[channel]) *
                adaptive_scales[channel]);
      }
    }
  }
  const auto adaptive_fields = make_quantized_correction_fields(
      adaptive_base,
      adaptive_corrected,
      adaptive_scales,
      adaptive_channels,
      adaptive_time);
  const auto format7 = encode_adaptive_packed_correction_format7(
      adaptive_base,
      adaptive_corrected,
      adaptive_scales,
      adaptive_fields,
      adaptive_channels,
      adaptive_time);
  if (format7.empty() ||
      format7.front() != kAdaptivePackedCorrectionFormat ||
      adaptive_fields.exact_values.empty())
    throw std::runtime_error(
        "adaptive packed self-test did not create format 7 with exact escapes");
  const auto adaptive_mask_bytes =
      adaptive_values / 8U + (adaptive_values % 8U != 0);
  const auto adaptive_maximum_raw =
      adaptive_channels * sizeof(float) + adaptive_mask_bytes +
      adaptive_channels + adaptive_fields.codes.size() +
      adaptive_fields.codes.size() * sizeof(std::uint32_t);
  const auto adaptive_raw = decompress_direct_zstd_frame(
      format7.data() + 1, format7.size() - 1, adaptive_maximum_raw);
  const auto adaptive_width_offset =
      adaptive_channels * sizeof(float) + adaptive_mask_bytes;
  const auto expected_widths = std::array<std::uint8_t, adaptive_channels>{2, 3, 4, 5};
  if (adaptive_raw.size() < adaptive_width_offset + adaptive_channels ||
      !std::equal(
          expected_widths.begin(),
          expected_widths.end(),
          adaptive_raw.begin() +
              static_cast<std::ptrdiff_t>(adaptive_width_offset)))
    throw std::runtime_error(
        "adaptive packed self-test did not preserve per-channel minimal widths");
  auto format7_decoded = adaptive_base;
  apply_correction_payload(
      format7_decoded,
      format7,
      static_cast<std::uint32_t>(adaptive_fields.codes.size()),
      adaptive_channels,
      adaptive_time,
      kAdaptivePackedVersion);
  if (!bitwise_equal_floats(format7_decoded, adaptive_corrected))
    throw std::runtime_error("format 7 changed correction semantics");

  constexpr std::size_t factorized_channels = 6;
  constexpr std::size_t factorized_time = 128;
  constexpr std::size_t factorized_values =
      factorized_channels * factorized_time;
  auto factorized_base = std::vector<float>(factorized_values);
  auto factorized_corrected = factorized_base;
  const auto factorized_scales =
      std::vector<float>(factorized_channels, 0.125F);
  for (std::size_t channel = 0; channel < factorized_channels; ++channel) {
    for (std::size_t index = 0; index < factorized_time; ++index) {
      const auto offset = channel * factorized_time + index;
      factorized_base[offset] = static_cast<float>(
          static_cast<double>(channel) * 32.0 +
          static_cast<double>(index) / 256.0);
      factorized_corrected[offset] = factorized_base[offset];
      auto code = 0;
      auto selected = false;
      if (channel == 1) {
        selected = true;
        code = 7;
      }
      else if (channel == 2 && (index == 1 || index == 50)) {
        selected = true;
        code = 6;
      }
      else if (channel == 3 && index != 10 && index != 20) {
        selected = true;
        code = 3;
      }
      else if (channel == 4 && index % 2U == 0) {
        selected = true;
        code = index % 20U == 0 ? 0 : (index % 10U == 2 ? 2 : -5);
      }
      else if (channel == 5) {
        selected = true;
        code = static_cast<int>(index / 32U) + 1;
      }
      if (!selected)
        continue;
      factorized_corrected[offset] = code == 0
          ? static_cast<float>(
                factorized_base[offset] + factorized_scales[channel] / 3.0F)
          : static_cast<float>(
                factorized_base[offset] +
                static_cast<float>(code) * factorized_scales[channel]);
    }
  }
  const auto factorized_fields = make_quantized_correction_fields(
      factorized_base,
      factorized_corrected,
      factorized_scales,
      factorized_channels,
      factorized_time);
  const auto format8 = encode_factorized_packed_correction_format8(
      factorized_base,
      factorized_corrected,
      factorized_scales,
      factorized_fields,
      factorized_channels,
      factorized_time);
  if (format8.empty() ||
      format8.front() != kFactorizedPackedCorrectionFormat ||
      factorized_fields.exact_values.empty())
    throw std::runtime_error(
        "factorized packed self-test did not create format 8 with exact escapes");
  const auto factorized_maximum_raw =
      factorized_channels * 25U + factorized_values * 10U +
      factorized_fields.codes.size() * 15U;
  const auto factorized_raw = decompress_direct_zstd_frame(
      format8.data() + 1, format8.size() - 1, factorized_maximum_raw);
  const auto expected_descriptors =
      std::array<unsigned char, factorized_channels>{0U, 9U, 11U, 12U, 34U, 25U};
  if (factorized_raw.size() < factorized_channels ||
      !std::equal(
          expected_descriptors.begin(),
          expected_descriptors.end(),
          factorized_raw.begin()))
    throw std::runtime_error(
        "factorized packed self-test did not select every canonical mode fixture");

  // Frozen raw golden #1: QPC v14/tag 8 stores all active scales, then all
  // position fields, then all value fields.  Two partially active channels
  // distinguish this from a per-channel scale/position/value interleaving.
  // The literal is independent of the encoder so both sides cannot drift
  // together without failing the compatibility test.
  const auto legacy_v14_layout_raw = std::vector<unsigned char>{
      0x0aU, 0x0aU,                    // two bitmap+constant channels
      0x00U, 0x00U, 0x00U, 0x3fU,    // channel-0 scale = 0.5f
      0x00U, 0x00U, 0x80U, 0x3eU,    // channel-1 scale = 0.25f
      0x0aU, 0x05U,                    // ch0 positions, then ch1 positions
      0x04U, 0x02U,                    // ch0 code +2, then ch1 code +1
  };
  const auto legacy_v14_layout_base = std::vector<float>(8, 0.0F);
  auto legacy_v14_layout_corrected = legacy_v14_layout_base;
  legacy_v14_layout_corrected[1] = 1.0F;
  legacy_v14_layout_corrected[3] = 1.0F;
  legacy_v14_layout_corrected[4] = 0.25F;
  legacy_v14_layout_corrected[6] = 0.25F;
  const auto legacy_v14_layout_scales = std::vector<float>{0.5F, 0.25F};
  const auto legacy_v14_layout_fields = make_quantized_correction_fields(
      legacy_v14_layout_base,
      legacy_v14_layout_corrected,
      legacy_v14_layout_scales,
      2,
      4);
  const auto legacy_v14_layout_payload =
      encode_factorized_packed_correction_format8(
          legacy_v14_layout_base,
          legacy_v14_layout_corrected,
          legacy_v14_layout_scales,
          legacy_v14_layout_fields,
          2,
          4);
  const auto legacy_v14_layout_encoded_raw = decompress_direct_zstd_frame(
      legacy_v14_layout_payload.data() + 1,
      legacy_v14_layout_payload.size() - 1,
      64);
  if (legacy_v14_layout_encoded_raw != legacy_v14_layout_raw)
    throw std::runtime_error(
        "format 8 encoder changed the frozen QPC v14 column layout");
  auto legacy_v14_layout_decoded = legacy_v14_layout_base;
  apply_correction_payload(
      legacy_v14_layout_decoded,
      frame_factorized_packed_correction_raw(legacy_v14_layout_raw),
      static_cast<std::uint32_t>(legacy_v14_layout_fields.codes.size()),
      2,
      4,
      kFactorizedPackedVersion);
  if (!bitwise_equal_floats(
          legacy_v14_layout_decoded, legacy_v14_layout_corrected))
    throw std::runtime_error(
        "format 8 decoder changed the frozen QPC v14 column layout");

  // Frozen raw golden #2: the v14 modal is the most frequent ZigZag code
  // (ties use the smallest byte), not the default giving the smallest patch
  // payload.  Here code +1 occurs 17 times and +127 occurs 15 times; choosing
  // +127 would be shorter, so this fixture distinguishes the two semantics.
  constexpr std::size_t legacy_v14_modal_time = 32;
  const auto legacy_v14_modal_base =
      std::vector<float>(legacy_v14_modal_time, 0.0F);
  auto legacy_v14_modal_corrected = legacy_v14_modal_base;
  for (std::size_t index = 0; index < legacy_v14_modal_time; ++index) {
    const auto code = index < 30 && index % 2U != 0 ? 127 : 1;
    legacy_v14_modal_corrected[index] =
        static_cast<float>(code) * 0.125F;
  }
  const auto legacy_v14_modal_scales = std::vector<float>{0.125F};
  const auto legacy_v14_modal_fields = make_quantized_correction_fields(
      legacy_v14_modal_base,
      legacy_v14_modal_corrected,
      legacy_v14_modal_scales,
      1,
      legacy_v14_modal_time);
  auto legacy_v14_modal_raw = std::vector<unsigned char>{
      0x21U,                          // all + modal-patched
      0x00U, 0x00U, 0x00U, 0x3eU,  // active scale = 0.125f
      0x02U,                          // modal code +1 (ZigZag 2)
      0xaaU, 0xaaU, 0xaaU, 0x2aU,  // +127 at odd positions below 30
      0x08U,                          // exception width
  };
  legacy_v14_modal_raw.insert(legacy_v14_modal_raw.end(), 15U, 0xfeU);
  const auto legacy_v14_modal_payload =
      encode_factorized_packed_correction_format8(
          legacy_v14_modal_base,
          legacy_v14_modal_corrected,
          legacy_v14_modal_scales,
          legacy_v14_modal_fields,
          1,
          legacy_v14_modal_time);
  const auto legacy_v14_modal_encoded_raw = decompress_direct_zstd_frame(
      legacy_v14_modal_payload.data() + 1,
      legacy_v14_modal_payload.size() - 1,
      128);
  if (legacy_v14_modal_encoded_raw != legacy_v14_modal_raw)
    throw std::runtime_error(
        "format 8 encoder changed the frozen QPC v14 modal rule");
  auto legacy_v14_modal_decoded = legacy_v14_modal_base;
  apply_correction_payload(
      legacy_v14_modal_decoded,
      frame_factorized_packed_correction_raw(legacy_v14_modal_raw),
      static_cast<std::uint32_t>(legacy_v14_modal_fields.codes.size()),
      1,
      legacy_v14_modal_time,
      kFactorizedPackedVersion);
  if (!bitwise_equal_floats(
          legacy_v14_modal_decoded, legacy_v14_modal_corrected))
    throw std::runtime_error(
        "format 8 decoder changed the frozen QPC v14 modal rule");

  auto format8_decoded = factorized_base;
  apply_correction_payload(
      format8_decoded,
      format8,
      static_cast<std::uint32_t>(factorized_fields.codes.size()),
      factorized_channels,
      factorized_time,
      kFactorizedPackedVersion);
  if (!bitwise_equal_floats(format8_decoded, factorized_corrected))
    throw std::runtime_error("format 8 changed correction semantics");
  const auto factorized_format9 =
      encode_subset_rank_packed_correction_format9(
          factorized_base,
          factorized_corrected,
          factorized_scales,
          factorized_fields,
          factorized_channels,
          factorized_time);
  if (factorized_format9.empty() ||
      factorized_format9.front() != kSubsetRankPackedCorrectionFormat)
    throw std::runtime_error(
        "subset-rank packed self-test did not create format 9");
  auto factorized_format9_decoded = factorized_base;
  apply_correction_payload(
      factorized_format9_decoded,
      factorized_format9,
      static_cast<std::uint32_t>(factorized_fields.codes.size()),
      factorized_channels,
      factorized_time,
      kSubsetRankPackedVersion);
  if (!bitwise_equal_floats(
          factorized_format9_decoded, factorized_corrected))
    throw std::runtime_error("format 9 changed correction semantics");

  auto delta_bitpack_codes = std::vector<std::int8_t>();
  delta_bitpack_codes.reserve(128U);
  auto delta_bitpack_code = 10;
  delta_bitpack_codes.push_back(static_cast<std::int8_t>(delta_bitpack_code));
  for (std::size_t index = 1; index < 128U; ++index) {
    const auto phase = (index - 1U) % 3U;
    delta_bitpack_code += phase == 0U ? 0 : (phase == 1U ? -1 : 1);
    delta_bitpack_codes.push_back(
        static_cast<std::int8_t>(delta_bitpack_code));
  }
  if (encode_predictive_values(delta_bitpack_codes).mode !=
      static_cast<std::uint8_t>(FactorizedValueMode::delta_bitpack))
    throw std::runtime_error(
        "predictive value fixture did not select delta-bitpack mode");
  auto delta_modal_codes = std::vector<std::int8_t>();
  delta_modal_codes.reserve(512U);
  for (std::size_t index = 0; index < 512U; ++index)
    delta_modal_codes.push_back(
        static_cast<std::int8_t>(1U + index % 127U));
  if (encode_predictive_values(delta_modal_codes).mode !=
      static_cast<std::uint8_t>(FactorizedValueMode::delta_modal_patched))
    throw std::runtime_error(
        "predictive value fixture did not select delta-modal mode");

  // Fixed-reconstruction transcode fixture: tag 9 and tag 10 receive exactly
  // the same positions, scales, codes and exact-XOR semantics.  Only the
  // per-channel value representation may differ, and the comparison is made
  // on the final serialized payload bytes (including the Zstd frame).
  constexpr std::size_t predictive_channels = 4;
  constexpr std::size_t predictive_time = 512;
  const auto predictive_base = std::vector<float>(
      predictive_channels * predictive_time, 0.0F);
  auto predictive_corrected = predictive_base;
  const auto predictive_scales =
      std::vector<float>(predictive_channels, 0.03125F);
  for (std::size_t channel = 0; channel < predictive_channels; ++channel) {
    for (std::size_t index = 0; index < predictive_time; ++index) {
      const auto code = static_cast<int>(
          1U + (index + channel * 17U) % 127U);
      predictive_corrected[channel * predictive_time + index] =
          static_cast<float>(code) * predictive_scales[channel];
    }
  }
  const auto predictive_fields = make_quantized_correction_fields(
      predictive_base,
      predictive_corrected,
      predictive_scales,
      predictive_channels,
      predictive_time);
  const auto predictive_tag9 = encode_subset_rank_packed_correction_format9(
      predictive_base,
      predictive_corrected,
      predictive_scales,
      predictive_fields,
      predictive_channels,
      predictive_time);
  const auto predictive_tag10 = encode_predictive_packed_correction_format10(
      predictive_base,
      predictive_corrected,
      predictive_scales,
      predictive_fields,
      predictive_channels,
      predictive_time);
  if (predictive_tag9.empty() || predictive_tag10.empty() ||
      predictive_tag9.front() != kSubsetRankPackedCorrectionFormat ||
      predictive_tag10.front() != kPredictivePackedCorrectionFormat ||
      predictive_tag10.size() >= predictive_tag9.size())
    throw std::runtime_error(
        "predictive fixed transcode did not strictly reduce serialized bytes");
  auto predictive_tag9_decoded = predictive_base;
  apply_correction_payload(
      predictive_tag9_decoded,
      predictive_tag9,
      static_cast<std::uint32_t>(predictive_fields.codes.size()),
      predictive_channels,
      predictive_time,
      kSubsetRankPackedVersion);
  auto predictive_tag10_decoded = predictive_base;
  apply_correction_payload(
      predictive_tag10_decoded,
      predictive_tag10,
      static_cast<std::uint32_t>(predictive_fields.codes.size()),
      predictive_channels,
      predictive_time,
      kPredictivePackedVersion);
  if (!bitwise_equal_floats(
          predictive_tag9_decoded, predictive_tag10_decoded) ||
      !bitwise_equal_floats(
          predictive_tag10_decoded, predictive_corrected))
    throw std::runtime_error(
        "predictive fixed transcode changed reconstructed float32 bits");
  const auto predictive_report = qpet_correlation::verify_constraints(
      predictive_corrected,
      predictive_tag10_decoded,
      predictive_channels,
      predictive_time,
      1e-12,
      1e-12,
      0.999999,
      8);
  if (!predictive_report.all_ok())
    throw std::runtime_error(
        "predictive fixed transcode changed the hard-constraint outcome");
  auto predictive_v15_rejected = false;
  try {
    auto invalid = predictive_base;
    apply_correction_payload(
        invalid,
        predictive_tag10,
        static_cast<std::uint32_t>(predictive_fields.codes.size()),
        predictive_channels,
        predictive_time,
        kSubsetRankPackedVersion);
  }
  catch (const std::runtime_error&) {
    predictive_v15_rejected = true;
  }
  if (!predictive_v15_rejected)
    throw std::runtime_error("QPC v15 accepted correction format 10");

  // Frozen raw golden for QPC v15/tag 9.  Channel 0 selects {1,3,6} from
  // T=8 and channel 1 selects every point.  The partial channel stores
  // k-1=2 in three bits followed by combinadic rank 24 in six bits, producing
  // the canonical MSB-first position bytes 0x4c,0x00 (seven zero pad bits).
  constexpr std::size_t subset_golden_channels = 2;
  constexpr std::size_t subset_golden_time = 8;
  auto subset_golden_base =
      std::vector<float>(subset_golden_channels * subset_golden_time, 0.0F);
  auto subset_golden_corrected = subset_golden_base;
  for (const auto index : {std::size_t{1}, std::size_t{3}, std::size_t{6}})
    subset_golden_corrected[index] = 0.5F;
  for (std::size_t index = 0; index < subset_golden_time; ++index)
    subset_golden_corrected[subset_golden_time + index] = -0.25F;
  const auto subset_golden_scales = std::vector<float>{0.5F, 0.25F};
  const auto subset_golden_fields = make_quantized_correction_fields(
      subset_golden_base,
      subset_golden_corrected,
      subset_golden_scales,
      subset_golden_channels,
      subset_golden_time);
  const auto subset_golden_payload =
      encode_subset_rank_packed_correction_format9(
          subset_golden_base,
          subset_golden_corrected,
          subset_golden_scales,
          subset_golden_fields,
          subset_golden_channels,
          subset_golden_time);
  const auto subset_golden_encoded_raw = decompress_direct_zstd_frame(
      subset_golden_payload.data() + 1,
      subset_golden_payload.size() - 1,
      64);
  const auto subset_golden_raw = std::vector<unsigned char>{
      0x0dU, 0x09U,                  // subset+constant, all+constant
      0x00U, 0x00U, 0x00U, 0x3fU,  // channel-0 scale = 0.5f
      0x00U, 0x00U, 0x80U, 0x3eU,  // channel-1 scale = 0.25f
      0x4cU, 0x00U,                  // interleaved k/rank position stream
      0x02U, 0x01U,                  // constant codes +1 and -1
  };
  if (subset_golden_encoded_raw != subset_golden_raw)
    throw std::runtime_error(
        "format 9 encoder changed the frozen QPC v15 combinadic layout");
  auto subset_golden_decoded = subset_golden_base;
  apply_correction_payload(
      subset_golden_decoded,
      frame_subset_rank_packed_correction_raw(subset_golden_raw),
      static_cast<std::uint32_t>(subset_golden_fields.codes.size()),
      subset_golden_channels,
      subset_golden_time,
      kSubsetRankPackedVersion);
  if (!bitwise_equal_floats(
          subset_golden_decoded, subset_golden_corrected))
    throw std::runtime_error(
        "format 9 decoder changed the frozen QPC v15 combinadic layout");

  const auto subset_exact_base = std::vector<float>(8, 0.0F);
  auto subset_exact_corrected = subset_exact_base;
  subset_exact_corrected[1] = 0.1F;
  subset_exact_corrected[6] = -0.1F;
  const auto subset_exact_scales = std::vector<float>{0.125F};
  const auto subset_exact_fields = make_quantized_correction_fields(
      subset_exact_base,
      subset_exact_corrected,
      subset_exact_scales,
      1,
      8);
  const auto subset_exact_payload =
      encode_subset_rank_packed_correction_format9(
          subset_exact_base,
          subset_exact_corrected,
          subset_exact_scales,
          subset_exact_fields,
          1,
          8);
  const auto subset_exact_raw = decompress_direct_zstd_frame(
      subset_exact_payload.data() + 1,
      subset_exact_payload.size() - 1,
      64);
  if (subset_exact_fields.codes !=
          std::vector<std::int8_t>({std::int8_t{0}, std::int8_t{0}}) ||
      subset_exact_raw.size() < 1U + sizeof(float) ||
      std::any_of(
          subset_exact_raw.begin() + 1,
          subset_exact_raw.begin() + 1 +
              static_cast<std::ptrdiff_t>(sizeof(float)),
          [](unsigned char byte) { return byte != 0U; }))
    throw std::runtime_error(
        "format 9 exact-only channel did not store canonical +0 scale");
  auto subset_exact_decoded = subset_exact_base;
  apply_correction_payload(
      subset_exact_decoded,
      subset_exact_payload,
      static_cast<std::uint32_t>(subset_exact_fields.codes.size()),
      1,
      8,
      kSubsetRankPackedVersion);
  if (!bitwise_equal_floats(subset_exact_decoded, subset_exact_corrected))
    throw std::runtime_error(
        "format 9 exact-only canonical scale changed reconstruction");

  // A deterministic, domain-agnostic serializer fixture in which every
  // channel has a different sparse subset.  The reconstruction is frozen
  // before either serializer runs, so this is a pure tag-8 versus tag-9 byte
  // comparison rather than a search-quality comparison.
  constexpr std::size_t subset_win_channels = 32;
  constexpr std::size_t subset_win_time = 256;
  constexpr std::size_t subset_win_count = 32;
  auto subset_win_base =
      std::vector<float>(subset_win_channels * subset_win_time, 0.0F);
  auto subset_win_corrected = subset_win_base;
  const auto subset_win_scales =
      std::vector<float>(subset_win_channels, 0.125F);
  for (std::size_t channel = 0; channel < subset_win_channels; ++channel) {
    auto occupied = std::vector<bool>(subset_win_time, false);
    auto selected = std::size_t{0};
    auto subset_state = static_cast<std::uint32_t>(
        0x6d2b79f5U ^ static_cast<std::uint32_t>(channel * 0x9e3779b9U));
    while (selected < subset_win_count) {
      subset_state = subset_state * 1664525U + 1013904223U;
      const auto index = static_cast<std::size_t>(
          (subset_state >> 16U) % subset_win_time);
      if (occupied[index])
        continue;
      occupied[index] = true;
      ++selected;
      subset_win_corrected[channel * subset_win_time + index] = 0.125F;
    }
  }
  const auto subset_win_fields = make_quantized_correction_fields(
      subset_win_base,
      subset_win_corrected,
      subset_win_scales,
      subset_win_channels,
      subset_win_time);
  const auto subset_win_format8 = encode_factorized_packed_correction_format8(
      subset_win_base,
      subset_win_corrected,
      subset_win_scales,
      subset_win_fields,
      subset_win_channels,
      subset_win_time);
  const auto subset_win_format9 =
      encode_subset_rank_packed_correction_format9(
          subset_win_base,
          subset_win_corrected,
          subset_win_scales,
          subset_win_fields,
          subset_win_channels,
          subset_win_time);
  if (subset_win_format9.empty() ||
      subset_win_format9.front() != kSubsetRankPackedCorrectionFormat ||
      subset_win_format9.size() >= subset_win_format8.size())
    throw std::runtime_error(
        "fixed-reconstruction subset-rank fixture did not strictly beat format 8");
  auto subset_win_decoded = subset_win_base;
  apply_correction_payload(
      subset_win_decoded,
      subset_win_format9,
      static_cast<std::uint32_t>(subset_win_fields.codes.size()),
      subset_win_channels,
      subset_win_time,
      kSubsetRankPackedVersion);
  if (!bitwise_equal_floats(subset_win_decoded, subset_win_corrected))
    throw std::runtime_error(
        "fixed-reconstruction format 9 winner failed bitwise replay");
  const auto factorized_format7 = encode_adaptive_packed_correction_format7(
      factorized_base,
      factorized_corrected,
      factorized_scales,
      factorized_fields,
      factorized_channels,
      factorized_time);
  if (format8.size() >= factorized_format7.size())
    throw std::runtime_error(
        "factorized packed fixture did not strictly improve format 7 bytes");

  // The ablation switch defines nested *real-byte* portfolios.  Compute the
  // expected winner independently from the public portfolio helper so a
  // dispatch regression cannot make both the test and implementation drift.
  const auto factorized_legacy = compress_correction_payload(
      factorized_base,
      factorized_corrected,
      &factorized_scales,
      factorized_channels,
      factorized_time);
  auto factorized_expected7 = factorized_legacy;
  if (factorized_format7.size() < factorized_expected7.size())
    factorized_expected7 = factorized_format7;
  auto factorized_expected8 = factorized_expected7;
  if (format8.size() < factorized_expected8.size())
    factorized_expected8 = format8;
  auto factorized_expected9 = factorized_expected8;
  if (factorized_format9.size() < factorized_expected9.size())
    factorized_expected9 = factorized_format9;
  const auto factorized_format10 = encode_predictive_packed_correction_format10(
      factorized_base,
      factorized_corrected,
      factorized_scales,
      factorized_fields,
      factorized_channels,
      factorized_time);
  auto factorized_expected10 = factorized_expected9;
  if (factorized_format10.size() < factorized_expected10.size())
    factorized_expected10 = factorized_format10;
  const auto factorized_ceiling6 = compress_adaptive_correction_payload(
      factorized_base,
      factorized_corrected,
      factorized_scales,
      factorized_channels,
      factorized_time,
      kPackedQuantizedCorrectionFormat);
  const auto factorized_ceiling7 = compress_adaptive_correction_payload(
      factorized_base,
      factorized_corrected,
      factorized_scales,
      factorized_channels,
      factorized_time,
      kAdaptivePackedCorrectionFormat);
  const auto factorized_ceiling8 = compress_adaptive_correction_payload(
      factorized_base,
      factorized_corrected,
      factorized_scales,
      factorized_channels,
      factorized_time,
      kFactorizedPackedCorrectionFormat);
  const auto factorized_ceiling9 = compress_adaptive_correction_payload(
      factorized_base,
      factorized_corrected,
      factorized_scales,
      factorized_channels,
      factorized_time,
      kSubsetRankPackedCorrectionFormat);
  const auto factorized_ceiling10 = compress_adaptive_correction_payload(
      factorized_base,
      factorized_corrected,
      factorized_scales,
      factorized_channels,
      factorized_time,
      kPredictivePackedCorrectionFormat);
  if (factorized_ceiling6 != factorized_legacy ||
      factorized_ceiling7 != factorized_expected7 ||
      factorized_ceiling8 != factorized_expected8 ||
      factorized_ceiling9 != factorized_expected9 ||
      factorized_ceiling10 != factorized_expected10 ||
      factorized_ceiling6.empty() || factorized_ceiling7.empty() ||
      factorized_ceiling8.empty() || factorized_ceiling9.empty() ||
      factorized_ceiling10.empty() ||
      factorized_ceiling6.front() > kPackedQuantizedCorrectionFormat ||
      factorized_ceiling7.front() > kAdaptivePackedCorrectionFormat ||
      factorized_ceiling8.front() > kFactorizedPackedCorrectionFormat ||
      factorized_ceiling9.front() > kSubsetRankPackedCorrectionFormat ||
      factorized_ceiling10.front() > kPredictivePackedCorrectionFormat ||
      factorized_ceiling7.size() > factorized_ceiling6.size() ||
      factorized_ceiling8.size() > factorized_ceiling7.size() ||
      factorized_ceiling9.size() > factorized_ceiling8.size() ||
      factorized_ceiling10.size() > factorized_ceiling9.size() ||
      factorized_ceiling8.front() != kFactorizedPackedCorrectionFormat)
    throw std::runtime_error(
        "nested correction format ceilings changed real-byte selection");
  for (const auto& payload :
       {factorized_ceiling6,
        factorized_ceiling7,
        factorized_ceiling8,
        factorized_ceiling9,
        factorized_ceiling10}) {
    auto decoded = factorized_base;
    apply_correction_payload(
        decoded,
        payload,
        static_cast<std::uint32_t>(factorized_fields.codes.size()),
        factorized_channels,
        factorized_time,
        correction_payload_required_stream_version(payload));
    if (!bitwise_equal_floats(decoded, factorized_corrected))
      throw std::runtime_error(
          "a correction format ceiling winner failed bitwise replay");
  }
  auto invalid_ceiling_rejected = false;
  try {
    (void)compress_adaptive_correction_payload(
        factorized_base,
        factorized_corrected,
        factorized_scales,
        factorized_channels,
        factorized_time,
        5U);
  }
  catch (const std::invalid_argument&) {
    invalid_ceiling_rejected = true;
  }
  if (!invalid_ceiling_rejected)
    throw std::runtime_error(
        "adaptive correction portfolio accepted an invalid ceiling");

  auto expect_rejected = [&](const std::string& label,
                             const std::vector<float>& initial,
                             const std::vector<unsigned char>& payload,
                             std::uint32_t corrections,
                             std::size_t test_channels,
                             std::size_t test_time,
                             std::uint32_t version) {
    auto candidate = initial;
    bool rejected = false;
    try {
      apply_correction_payload(
          candidate,
          payload,
          corrections,
          test_channels,
          test_time,
          version);
    }
    catch (const std::exception&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("packed self-test accepted " + label);
    if (!bitwise_equal_floats(candidate, initial))
      throw std::runtime_error("packed self-test rejection was non-transactional: " + label);
  };

  expect_rejected(
      "format 9 under QPC v14",
      subset_golden_base,
      subset_golden_payload,
      static_cast<std::uint32_t>(subset_golden_fields.codes.size()),
      subset_golden_channels,
      subset_golden_time,
      kFactorizedPackedVersion);
  auto trailing_subset_rank_frame = subset_golden_payload;
  trailing_subset_rank_frame.push_back(0U);
  expect_rejected(
      "format 9 with a trailing ZSTD frame byte",
      subset_golden_base,
      trailing_subset_rank_frame,
      static_cast<std::uint32_t>(subset_golden_fields.codes.size()),
      subset_golden_channels,
      subset_golden_time,
      kSubsetRankPackedVersion);
  auto truncated_subset_rank_frame = subset_golden_payload;
  truncated_subset_rank_frame.pop_back();
  expect_rejected(
      "a truncated format 9 ZSTD frame",
      subset_golden_base,
      truncated_subset_rank_frame,
      static_cast<std::uint32_t>(subset_golden_fields.codes.size()),
      subset_golden_channels,
      subset_golden_time,
      kSubsetRankPackedVersion);
  auto reserved_subset_rank_raw = subset_golden_raw;
  reserved_subset_rank_raw[0] |= 0x40U;
  expect_rejected(
      "reserved format 9 descriptor bits",
      subset_golden_base,
      frame_subset_rank_packed_correction_raw(reserved_subset_rank_raw),
      static_cast<std::uint32_t>(subset_golden_fields.codes.size()),
      subset_golden_channels,
      subset_golden_time,
      kSubsetRankPackedVersion);
  auto unknown_subset_rank_mode_raw = subset_golden_raw;
  unknown_subset_rank_mode_raw[0] = static_cast<unsigned char>(
      (unknown_subset_rank_mode_raw[0] & 0xf8U) | 0x02U);
  expect_rejected(
      "an unknown format 9 position mode",
      subset_golden_base,
      frame_subset_rank_packed_correction_raw(unknown_subset_rank_mode_raw),
      static_cast<std::uint32_t>(subset_golden_fields.codes.size()),
      subset_golden_channels,
      subset_golden_time,
      kSubsetRankPackedVersion);
  auto unknown_subset_rank_value_raw = subset_golden_raw;
  unknown_subset_rank_value_raw[0] = static_cast<unsigned char>(
      (unknown_subset_rank_value_raw[0] & 0x07U) | (5U << 3U));
  expect_rejected(
      "an unknown format 9 value mode",
      subset_golden_base,
      frame_subset_rank_packed_correction_raw(unknown_subset_rank_value_raw),
      static_cast<std::uint32_t>(subset_golden_fields.codes.size()),
      subset_golden_channels,
      subset_golden_time,
      kSubsetRankPackedVersion);
  auto zero_nonexact_scale_raw = subset_golden_raw;
  std::fill(
      zero_nonexact_scale_raw.begin() + 2,
      zero_nonexact_scale_raw.begin() + 2 +
          static_cast<std::ptrdiff_t>(sizeof(float)),
      0U);
  expect_rejected(
      "a zero format 9 non-exact scale",
      subset_golden_base,
      frame_subset_rank_packed_correction_raw(zero_nonexact_scale_raw),
      static_cast<std::uint32_t>(subset_golden_fields.codes.size()),
      subset_golden_channels,
      subset_golden_time,
      kSubsetRankPackedVersion);
  auto nan_subset_rank_scale_raw = subset_golden_raw;
  const auto nan_subset_rank_scale =
      std::numeric_limits<float>::quiet_NaN();
  std::memcpy(
      nan_subset_rank_scale_raw.data() + 2,
      &nan_subset_rank_scale,
      sizeof(nan_subset_rank_scale));
  expect_rejected(
      "a NaN format 9 scale",
      subset_golden_base,
      frame_subset_rank_packed_correction_raw(nan_subset_rank_scale_raw),
      static_cast<std::uint32_t>(subset_golden_fields.codes.size()),
      subset_golden_channels,
      subset_golden_time,
      kSubsetRankPackedVersion);
  auto truncated_subset_rank_scale_raw = subset_golden_raw;
  truncated_subset_rank_scale_raw.resize(2U + sizeof(float) - 1U);
  expect_rejected(
      "a structurally truncated format 9 scale",
      subset_golden_base,
      frame_subset_rank_packed_correction_raw(
          truncated_subset_rank_scale_raw),
      static_cast<std::uint32_t>(subset_golden_fields.codes.size()),
      subset_golden_channels,
      subset_golden_time,
      kSubsetRankPackedVersion);
  auto out_of_range_subset_rank_raw = subset_golden_raw;
  out_of_range_subset_rank_raw[10] = 0x5fU;
  out_of_range_subset_rank_raw[11] = 0x80U;
  expect_rejected(
      "an out-of-range format 9 combinadic rank",
      subset_golden_base,
      frame_subset_rank_packed_correction_raw(out_of_range_subset_rank_raw),
      static_cast<std::uint32_t>(subset_golden_fields.codes.size()),
      subset_golden_channels,
      subset_golden_time,
      kSubsetRankPackedVersion);
  auto nonzero_subset_rank_padding_raw = subset_golden_raw;
  nonzero_subset_rank_padding_raw[11] |= 0x01U;
  expect_rejected(
      "nonzero format 9 position padding",
      subset_golden_base,
      frame_subset_rank_packed_correction_raw(nonzero_subset_rank_padding_raw),
      static_cast<std::uint32_t>(subset_golden_fields.codes.size()),
      subset_golden_channels,
      subset_golden_time,
      kSubsetRankPackedVersion);
  auto oversized_subset_rank_raw = subset_golden_raw;
  oversized_subset_rank_raw.insert(oversized_subset_rank_raw.end(), 128U, 0U);
  expect_rejected(
      "a format 9 raw frame above its decoded-size bound",
      subset_golden_base,
      frame_subset_rank_packed_correction_raw(oversized_subset_rank_raw),
      static_cast<std::uint32_t>(subset_golden_fields.codes.size()),
      subset_golden_channels,
      subset_golden_time,
      kSubsetRankPackedVersion);
  expect_rejected(
      "format 9 with a mismatched correction count",
      subset_golden_base,
      subset_golden_payload,
      static_cast<std::uint32_t>(subset_golden_fields.codes.size() - 1U),
      subset_golden_channels,
      subset_golden_time,
      kSubsetRankPackedVersion);
  auto negative_zero_exact_scale_raw = subset_exact_raw;
  const auto negative_zero_bits = std::uint32_t{0x80000000U};
  std::memcpy(
      negative_zero_exact_scale_raw.data() + 1,
      &negative_zero_bits,
      sizeof(negative_zero_bits));
  expect_rejected(
      "a negative-zero format 9 exact-only scale",
      subset_exact_base,
      frame_subset_rank_packed_correction_raw(
          negative_zero_exact_scale_raw),
      static_cast<std::uint32_t>(subset_exact_fields.codes.size()),
      1,
      8,
      kSubsetRankPackedVersion);
  auto nonzero_exact_scale_raw = subset_exact_raw;
  const auto nonzero_exact_scale = 1.0F;
  std::memcpy(
      nonzero_exact_scale_raw.data() + 1,
      &nonzero_exact_scale,
      sizeof(nonzero_exact_scale));
  expect_rejected(
      "a nonzero format 9 exact-only scale",
      subset_exact_base,
      frame_subset_rank_packed_correction_raw(nonzero_exact_scale_raw),
      static_cast<std::uint32_t>(subset_exact_fields.codes.size()),
      1,
      8,
      kSubsetRankPackedVersion);
  const auto over_cap_time =
      qpet_correlation::subset_rank::kMaximumTime + 1U;
  const auto over_cap_base = std::vector<float>(over_cap_time, 0.0F);
  expect_rejected(
      "format 9 time above its production safety cap",
      over_cap_base,
      subset_golden_payload,
      static_cast<std::uint32_t>(subset_golden_fields.codes.size()),
      1,
      over_cap_time,
      kSubsetRankPackedVersion);
  const auto factorized_format9_maximum_raw =
      factorized_channels * 25U + factorized_values * 10U +
      factorized_fields.codes.size() * 15U;
  auto zero_exact_subset_rank_raw = decompress_direct_zstd_frame(
      factorized_format9.data() + 1,
      factorized_format9.size() - 1,
      factorized_format9_maximum_raw);
  if (zero_exact_subset_rank_raw.size() < sizeof(std::uint32_t))
    throw std::runtime_error("format 9 exact fixture is unexpectedly truncated");
  std::fill(
      zero_exact_subset_rank_raw.end() -
          static_cast<std::ptrdiff_t>(sizeof(std::uint32_t)),
      zero_exact_subset_rank_raw.end(),
      0U);
  expect_rejected(
      "a zero format 9 exact XOR",
      factorized_base,
      frame_subset_rank_packed_correction_raw(zero_exact_subset_rank_raw),
      static_cast<std::uint32_t>(factorized_fields.codes.size()),
      factorized_channels,
      factorized_time,
      kSubsetRankPackedVersion);

  expect_rejected(
      "format 6 under QPC v11",
      base,
      format6,
      static_cast<std::uint32_t>(fields.codes.size()),
      channels,
      time,
      11);
  expect_rejected(
      "format 7 under QPC v12",
      adaptive_base,
      format7,
      static_cast<std::uint32_t>(adaptive_fields.codes.size()),
      adaptive_channels,
      adaptive_time,
      12);
  auto trailing_adaptive_frame = format7;
  trailing_adaptive_frame.push_back(0U);
  expect_rejected(
      "format 7 with a trailing ZSTD frame byte",
      adaptive_base,
      trailing_adaptive_frame,
      static_cast<std::uint32_t>(adaptive_fields.codes.size()),
      adaptive_channels,
      adaptive_time,
      kAdaptivePackedVersion);
  auto truncated_adaptive_frame = format7;
  truncated_adaptive_frame.pop_back();
  expect_rejected(
      "a truncated format 7 ZSTD frame",
      adaptive_base,
      truncated_adaptive_frame,
      static_cast<std::uint32_t>(adaptive_fields.codes.size()),
      adaptive_channels,
      adaptive_time,
      kAdaptivePackedVersion);
  auto nonminimal_adaptive_raw = adaptive_raw;
  nonminimal_adaptive_raw[adaptive_width_offset] = 3U;
  expect_rejected(
      "a nonminimal format 7 channel width",
      adaptive_base,
      frame_adaptive_packed_correction_raw(nonminimal_adaptive_raw),
      static_cast<std::uint32_t>(adaptive_fields.codes.size()),
      adaptive_channels,
      adaptive_time,
      kAdaptivePackedVersion);
  auto truncated_adaptive_raw = adaptive_raw;
  truncated_adaptive_raw.pop_back();
  expect_rejected(
      "a truncated format 7 raw field",
      adaptive_base,
      frame_adaptive_packed_correction_raw(truncated_adaptive_raw),
      static_cast<std::uint32_t>(adaptive_fields.codes.size()),
      adaptive_channels,
      adaptive_time,
      kAdaptivePackedVersion);
  expect_rejected(
      "format 8 under QPC v13",
      factorized_base,
      format8,
      static_cast<std::uint32_t>(factorized_fields.codes.size()),
      factorized_channels,
      factorized_time,
      kAdaptivePackedVersion);
  auto trailing_factorized_frame = format8;
  trailing_factorized_frame.push_back(0U);
  expect_rejected(
      "format 8 with a trailing ZSTD frame byte",
      factorized_base,
      trailing_factorized_frame,
      static_cast<std::uint32_t>(factorized_fields.codes.size()),
      factorized_channels,
      factorized_time,
      kFactorizedPackedVersion);
  auto truncated_factorized_frame = format8;
  truncated_factorized_frame.pop_back();
  expect_rejected(
      "a truncated format 8 ZSTD frame",
      factorized_base,
      truncated_factorized_frame,
      static_cast<std::uint32_t>(factorized_fields.codes.size()),
      factorized_channels,
      factorized_time,
      kFactorizedPackedVersion);
  auto reserved_factorized_raw = factorized_raw;
  reserved_factorized_raw.front() |= 0xc0U;
  expect_rejected(
      "reserved format 8 descriptor bits",
      factorized_base,
      frame_factorized_packed_correction_raw(reserved_factorized_raw),
      static_cast<std::uint32_t>(factorized_fields.codes.size()),
      factorized_channels,
      factorized_time,
      kFactorizedPackedVersion);
  auto mismatched_factorized_raw = factorized_raw;
  mismatched_factorized_raw.front() = static_cast<unsigned char>(
      static_cast<std::uint8_t>(FactorizedValueMode::constant) << 3U);
  expect_rejected(
      "mismatched format 8 activity modes",
      factorized_base,
      frame_factorized_packed_correction_raw(mismatched_factorized_raw),
      static_cast<std::uint32_t>(factorized_fields.codes.size()),
      factorized_channels,
      factorized_time,
      kFactorizedPackedVersion);
  auto overlong_factorized_raw = factorized_raw;
  // descriptors followed by the five-active-channel scale column
  const auto factorized_gap_offset = factorized_channels +
      5U * sizeof(float);
  if (overlong_factorized_raw.at(factorized_gap_offset) != 2U)
    throw std::runtime_error(
        "factorized packed self-test gap offset is not deterministic");
  overlong_factorized_raw[factorized_gap_offset] = 0x82U;
  overlong_factorized_raw.insert(
      overlong_factorized_raw.begin() +
          static_cast<std::ptrdiff_t>(factorized_gap_offset + 1U),
      0U);
  expect_rejected(
      "an overlong format 8 gap varint",
      factorized_base,
      frame_factorized_packed_correction_raw(overlong_factorized_raw),
      static_cast<std::uint32_t>(factorized_fields.codes.size()),
      factorized_channels,
      factorized_time,
      kFactorizedPackedVersion);
  auto zero_xor_factorized_raw = factorized_raw;
  const auto factorized_exact_bytes =
      factorized_fields.exact_values.size() * sizeof(std::uint32_t);
  if (factorized_exact_bytes == 0 ||
      factorized_exact_bytes > zero_xor_factorized_raw.size())
    throw std::runtime_error(
        "factorized packed self-test exact offset is invalid");
  std::fill(
      zero_xor_factorized_raw.end() -
          static_cast<std::ptrdiff_t>(factorized_exact_bytes),
      zero_xor_factorized_raw.end() -
          static_cast<std::ptrdiff_t>(factorized_exact_bytes) +
          static_cast<std::ptrdiff_t>(sizeof(std::uint32_t)),
      0U);
  expect_rejected(
      "a zero format 8 exact XOR",
      factorized_base,
      frame_factorized_packed_correction_raw(zero_xor_factorized_raw),
      static_cast<std::uint32_t>(factorized_fields.codes.size()),
      factorized_channels,
      factorized_time,
      kFactorizedPackedVersion);
  auto truncated_factorized_raw = factorized_raw;
  truncated_factorized_raw.pop_back();
  expect_rejected(
      "a truncated format 8 raw field",
      factorized_base,
      frame_factorized_packed_correction_raw(truncated_factorized_raw),
      static_cast<std::uint32_t>(factorized_fields.codes.size()),
      factorized_channels,
      factorized_time,
      kFactorizedPackedVersion);
  auto trailing_frame = format6;
  trailing_frame.push_back(0U);
  expect_rejected(
      "a trailing ZSTD frame byte",
      base,
      trailing_frame,
      static_cast<std::uint32_t>(fields.codes.size()),
      channels,
      time,
      12);
  auto truncated_frame = format6;
  truncated_frame.pop_back();
  expect_rejected(
      "a truncated ZSTD frame",
      base,
      truncated_frame,
      static_cast<std::uint32_t>(fields.codes.size()),
      channels,
      time,
      12);

  const auto small_base = std::vector<float>{10, 11, 12, 13, 14};
  auto small_corrected = small_base;
  small_corrected[0] += 1.0F;
  small_corrected[2] -= 1.0F;
  small_corrected[4] += 1.0F;
  const auto small_scales = std::vector<float>{1.0F};
  const auto small_fields = make_quantized_correction_fields(
      small_base, small_corrected, small_scales, 1, small_base.size());
  const auto small_format6 = encode_quantized_correction_format6(
      small_scales, small_fields);
  const auto small_maximum = packed_correction_maximum_raw_size(
      small_base.size(), 1, small_fields.codes.size());
  const auto small_raw = decompress_direct_zstd_frame(
      small_format6.data() + 1, small_format6.size() - 1, small_maximum);
  if (small_raw.front() != 2U)
    throw std::runtime_error("packed self-test expected a two-bit code width");
  const auto small_mask_offset = 1 + sizeof(float);
  const auto small_code_offset = small_mask_offset + 1;
  auto malformed_raw = small_raw;
  malformed_raw[small_code_offset] |= 0x80U;
  expect_rejected(
      "nonzero packed padding",
      small_base,
      frame_packed_correction_raw(malformed_raw),
      3,
      1,
      small_base.size(),
      12);
  malformed_raw = small_raw;
  malformed_raw[small_mask_offset] |= 1U << 1U;
  expect_rejected(
      "a mask popcount mismatch",
      small_base,
      frame_packed_correction_raw(malformed_raw),
      3,
      1,
      small_base.size(),
      12);
  malformed_raw = small_raw;
  malformed_raw[small_mask_offset] |= 1U << 7U;
  expect_rejected(
      "an out-of-range mask bit",
      small_base,
      frame_packed_correction_raw(malformed_raw),
      3,
      1,
      small_base.size(),
      12);
  malformed_raw = small_raw;
  malformed_raw.front() = 0;
  expect_rejected(
      "bit width zero",
      small_base,
      frame_packed_correction_raw(malformed_raw),
      3,
      1,
      small_base.size(),
      12);
  malformed_raw = small_raw;
  malformed_raw.front() = 9;
  expect_rejected(
      "bit width nine",
      small_base,
      frame_packed_correction_raw(malformed_raw),
      3,
      1,
      small_base.size(),
      12);
  auto nonminimal_raw = std::vector<unsigned char>(1 + sizeof(float) + 1 + 3);
  nonminimal_raw.front() = 8U;
  std::memcpy(nonminimal_raw.data() + 1, small_scales.data(), sizeof(float));
  nonminimal_raw[small_mask_offset] = small_fields.mask.front();
  for (std::size_t index = 0; index < small_fields.codes.size(); ++index)
    nonminimal_raw[small_code_offset + index] = zigzag_int8(small_fields.codes[index]);
  expect_rejected(
      "a nonminimal bit width",
      small_base,
      frame_packed_correction_raw(nonminimal_raw),
      3,
      1,
      small_base.size(),
      12);
  malformed_raw = small_raw;
  const auto zero_scale = 0.0F;
  std::memcpy(malformed_raw.data() + 1, &zero_scale, sizeof(float));
  expect_rejected(
      "a zero scale for nonzero codes",
      small_base,
      frame_packed_correction_raw(malformed_raw),
      3,
      1,
      small_base.size(),
      12);
  malformed_raw = small_raw;
  const auto nan_scale = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(malformed_raw.data() + 1, &nan_scale, sizeof(float));
  expect_rejected(
      "a nonfinite scale",
      small_base,
      frame_packed_correction_raw(malformed_raw),
      3,
      1,
      small_base.size(),
      12);
  malformed_raw = small_raw;
  const auto tiny_scale = std::numeric_limits<float>::denorm_min();
  std::memcpy(malformed_raw.data() + 1, &tiny_scale, sizeof(float));
  expect_rejected(
      "a numerically unchanged result",
      small_base,
      frame_packed_correction_raw(malformed_raw),
      3,
      1,
      small_base.size(),
      12);
  malformed_raw = small_raw;
  malformed_raw.push_back(0U);
  expect_rejected(
      "an invalid raw byte count",
      small_base,
      frame_packed_correction_raw(malformed_raw),
      3,
      1,
      small_base.size(),
      12);

  auto escape_corrected = small_base;
  escape_corrected[0] += 0.25F;
  escape_corrected[2] += 0.25F;
  escape_corrected[4] += 0.25F;
  const auto escape_scales = std::vector<float>{0.0F};
  const auto escape_fields = make_quantized_correction_fields(
      small_base, escape_corrected, escape_scales, 1, small_base.size());
  const auto escape_format6 = encode_quantized_correction_format6(
      escape_scales, escape_fields);
  const auto escape_maximum = packed_correction_maximum_raw_size(
      small_base.size(), 1, escape_fields.codes.size());
  const auto escape_raw = decompress_direct_zstd_frame(
      escape_format6.data() + 1, escape_format6.size() - 1, escape_maximum);
  if (escape_raw.front() != 1U)
    throw std::runtime_error("all-escape correction did not select one-bit codes");
  auto escape_decoded = small_base;
  apply_correction_payload(escape_decoded, escape_format6, 3, 1, small_base.size(), 12);
  if (!bitwise_equal_floats(escape_decoded, escape_corrected))
    throw std::runtime_error("all-escape packed correction changed exact values");
  auto invalid_escape_raw = escape_raw;
  invalid_escape_raw.resize(invalid_escape_raw.size() - sizeof(float));
  expect_rejected(
      "too few exact escapes",
      small_base,
      frame_packed_correction_raw(invalid_escape_raw),
      3,
      1,
      small_base.size(),
      12);
  invalid_escape_raw = escape_raw;
  const auto escape_exact_offset = 1 + sizeof(float) + 1 + 1;
  const auto nan_exact = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(
      invalid_escape_raw.data() + escape_exact_offset, &nan_exact, sizeof(float));
  expect_rejected(
      "a nonfinite exact escape",
      small_base,
      frame_packed_correction_raw(invalid_escape_raw),
      3,
      1,
      small_base.size(),
      12);

  constexpr std::size_t nested_channels = 2;
  constexpr std::size_t nested_time = 16;
  auto nested_base = std::vector<float>(nested_channels * nested_time);
  auto nested_corrected = nested_base;
  auto overridden = nested_base;
  for (std::size_t index = 0; index < nested_time; ++index) {
    nested_base[index] = static_cast<float>(index) / 8.0F;
    nested_corrected[index] = 100.0F;
    overridden[index] = 100.0F;
    const auto second = nested_time + index;
    nested_base[second] = 10.0F + static_cast<float>(index) / 2.0F;
    nested_corrected[second] = nested_base[second] + 0.25F;
    overridden[second] = nested_base[second];
  }
  auto legacy5_corrected = nested_base;
  for (std::size_t index = 0; index < nested_time; ++index)
    legacy5_corrected[index] = 100.0F;
  auto legacy5 = std::vector<unsigned char>{5U};
  append_varint(legacy5, nested_time);
  append_varint(legacy5, 0U);
  legacy5.push_back(0x01U);
  const auto legacy5_constant = 100.0F;
  const auto* legacy5_constant_bytes =
      reinterpret_cast<const unsigned char*>(&legacy5_constant);
  legacy5.insert(
      legacy5.end(),
      legacy5_constant_bytes,
      legacy5_constant_bytes + sizeof(float));
  for (const auto version : {10U, 11U, 12U}) {
    auto decoded = nested_base;
    apply_correction_payload(
        decoded,
        legacy5,
        static_cast<std::uint32_t>(nested_time),
        nested_channels,
        nested_time,
        version);
    if (!bitwise_equal_floats(decoded, legacy5_corrected))
      throw std::runtime_error(
          "format 5 changed semantics under a supported QPC version");
  }
  expect_rejected(
      "format 5 under QPC v9",
      nested_base,
      legacy5,
      static_cast<std::uint32_t>(nested_time),
      nested_channels,
      nested_time,
      9);
  const auto nested_scales = std::vector<float>{0.0F, 0.25F};
  const auto nested_fields = make_quantized_correction_fields(
      overridden,
      nested_corrected,
      nested_scales,
      nested_channels,
      nested_time);
  const auto inner6 = encode_quantized_correction_format6(
      nested_scales, nested_fields);
  auto nested5 = std::vector<unsigned char>{5U};
  append_varint(nested5, nested_time);
  append_varint(nested5, nested_fields.codes.size());
  nested5.push_back(0x01U);
  const auto constant = 100.0F;
  const auto* constant_bytes = reinterpret_cast<const unsigned char*>(&constant);
  nested5.insert(nested5.end(), constant_bytes, constant_bytes + sizeof(float));
  nested5.insert(nested5.end(), inner6.begin(), inner6.end());
  auto nested_decoded = nested_base;
  apply_correction_payload(
      nested_decoded,
      nested5,
      static_cast<std::uint32_t>(nested_time + nested_fields.codes.size()),
      nested_channels,
      nested_time,
      12);
  if (!bitwise_equal_floats(nested_decoded, nested_corrected))
    throw std::runtime_error("format 5 with nested format 6 failed roundtrip");
  expect_rejected(
      "format 5 with nested format 6 under QPC v11",
      nested_base,
      nested5,
      static_cast<std::uint32_t>(nested_time + nested_fields.codes.size()),
      nested_channels,
      nested_time,
      11);
  auto nested5_recursive = nested5;
  nested5_recursive.resize(
      1 + varint_size(nested_time) + varint_size(nested_fields.codes.size()) +
      1 + sizeof(float));
  nested5_recursive.push_back(5U);
  expect_rejected(
      "a recursively nested format 5",
      nested_base,
      nested5_recursive,
      static_cast<std::uint32_t>(nested_time + nested_fields.codes.size()),
      nested_channels,
      nested_time,
      12);

  Settings settings;
  settings.batch = 1;
  settings.channels = channels;
  settings.time = time;
  settings.pointwise_tolerance = 2.0;
  settings.acf_tolerance = -1.0;
  settings.minimum_correlation = 0.5;
  settings.max_lag = 0;
  settings.qoz_level = 1;
  settings.correction_mode = 2;
  settings.estimator_scales = {1.0};
  settings.qpet_confidence = 0.999;
  settings.qpet_k = 2.0;
  settings.group_size = 1;
  settings.repair_denominators = {7};
  validate_settings(settings);
  auto expect_invalid_settings = [&](Settings invalid, const std::string& label) {
    try {
      validate_settings(invalid);
    }
    catch (const std::invalid_argument&) {
      return;
    }
    throw std::runtime_error("settings validation accepted " + label);
  };
  auto invalid_settings = settings;
  invalid_settings.pointwise_tolerance =
      std::numeric_limits<double>::quiet_NaN();
  expect_invalid_settings(invalid_settings, "a NaN pointwise tolerance");
  invalid_settings = settings;
  invalid_settings.acf_tolerance = std::numeric_limits<double>::infinity();
  expect_invalid_settings(invalid_settings, "an infinite ACF tolerance");
  invalid_settings = settings;
  invalid_settings.minimum_correlation =
      -std::numeric_limits<double>::infinity();
  expect_invalid_settings(invalid_settings, "an infinite Pearson sentinel");
  invalid_settings = settings;
  invalid_settings.qpet_confidence =
      std::numeric_limits<double>::quiet_NaN();
  expect_invalid_settings(invalid_settings, "a NaN QPET confidence");
  invalid_settings = settings;
  invalid_settings.qpet_k = std::numeric_limits<double>::infinity();
  expect_invalid_settings(invalid_settings, "an infinite QPET k");
  invalid_settings = settings;
  invalid_settings.estimator_scales = {
      std::numeric_limits<double>::quiet_NaN()};
  expect_invalid_settings(invalid_settings, "a NaN estimator scale");
  invalid_settings = settings;
  invalid_settings.acf_tolerance = -0.5;
  expect_invalid_settings(invalid_settings, "a noncanonical ACF sentinel");
  invalid_settings = settings;
  invalid_settings.minimum_correlation = -0.5;
  invalid_settings.acf_tolerance = 0.1;
  invalid_settings.max_lag = 1;
  expect_invalid_settings(invalid_settings, "a noncanonical Pearson sentinel");
  invalid_settings = settings;
  invalid_settings.acf_tolerance = 2.01;
  invalid_settings.max_lag = 1;
  expect_invalid_settings(invalid_settings, "an out-of-range ACF tolerance");
  invalid_settings = settings;
  invalid_settings.correction_format_ceiling =
      kAdaptivePackedCorrectionFormat;
  expect_invalid_settings(
      invalid_settings, "a non-SSBB adaptive correction ceiling");
  invalid_settings = settings;
  invalid_settings.correction_format_ceiling = 5U;
  expect_invalid_settings(invalid_settings, "correction ceiling 5");
  invalid_settings = settings;
  invalid_settings.correction_format_ceiling = 9U;
  expect_invalid_settings(invalid_settings, "correction ceiling 9");
  invalid_settings = settings;
  invalid_settings.batch =
      static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()) + 1U;
  invalid_settings.group_size = invalid_settings.batch;
  expect_invalid_settings(invalid_settings, "a uint16-overflowing compact group");
  auto maximum_group_settings = settings;
  maximum_group_settings.batch = std::numeric_limits<std::uint16_t>::max();
  maximum_group_settings.group_size = maximum_group_settings.batch;
  validate_settings(maximum_group_settings);
  EncodedWindow packed_window;
  packed_window.base_mode = 4;
  packed_window.selected_global_bound = settings.pointwise_tolerance;
  packed_window.payload = compress_lossless_zstd_payload(base);
  packed_window.corrections = static_cast<std::uint32_t>(fields.codes.size());
  packed_window.correction_payload = format6;
  const auto prefix = std::string(argv[2]);
  const auto v12_path = prefix + "-format6-v12.qpc";
  write_stream(v12_path, settings, {packed_window});
  const auto [v12_settings, v12_decoded] = decode_stream(v12_path);
  (void)v12_settings;
  if (!bitwise_equal_floats(v12_decoded, corrected))
    throw std::runtime_error("QPC v12 format 6 outer roundtrip changed values");

  auto adaptive_settings = settings;
  adaptive_settings.channels = adaptive_channels;
  adaptive_settings.time = adaptive_time;
  adaptive_settings.repair_optimizer = RepairOptimizer::ssbb_value;
  adaptive_settings.repair_denominators = {7, 15, 31, 63, 127};
  // A ceiling of 8 must not force QPC v14: the actual selected tag 7 keeps
  // this stream at v13.
  adaptive_settings.correction_format_ceiling =
      kFactorizedPackedCorrectionFormat;
  validate_settings(adaptive_settings);
  EncodedWindow adaptive_window;
  adaptive_window.base_mode = 4;
  adaptive_window.selected_global_bound = adaptive_settings.pointwise_tolerance;
  adaptive_window.payload = compress_lossless_zstd_payload(adaptive_base);
  adaptive_window.corrections =
      static_cast<std::uint32_t>(adaptive_fields.codes.size());
  adaptive_window.correction_payload = format7;
  const auto v13_path = prefix + "-format7-v13.qpc";
  write_stream(v13_path, adaptive_settings, {adaptive_window});
  const auto v13_bytes = read_bytes(v13_path);
  std::uint32_t observed_v13 = 0;
  if (v13_bytes.size() < kMagic.size() + sizeof(observed_v13))
    throw std::runtime_error("QPC v13 format 7 outer stream is truncated");
  std::memcpy(
      &observed_v13, v13_bytes.data() + kMagic.size(), sizeof(observed_v13));
  if (observed_v13 != kAdaptivePackedVersion)
    throw std::runtime_error("format 7 outer writer did not emit QPC v13");
  const auto [v13_settings, v13_decoded] = decode_stream(v13_path);
  (void)v13_settings;
  if (!bitwise_equal_floats(v13_decoded, adaptive_corrected))
    throw std::runtime_error("QPC v13 format 7 outer roundtrip changed values");
  auto invalid_v12_format7 = v13_bytes;
  const auto forced_v12 = kVersion;
  std::memcpy(
      invalid_v12_format7.data() + kMagic.size(),
      &forced_v12,
      sizeof(forced_v12));
  const auto invalid_v12_format7_path = prefix + "-format7-v12-invalid.qpc";
  write_bytes(invalid_v12_format7_path, invalid_v12_format7);
  bool rejected_v12_format7 = false;
  try {
    (void)decode_stream(invalid_v12_format7_path);
  }
  catch (const std::runtime_error&) {
    rejected_v12_format7 = true;
  }
  if (!rejected_v12_format7)
    throw std::runtime_error("QPC v12 accepted an outer format 7 payload");

  auto factorized_settings = settings;
  factorized_settings.channels = factorized_channels;
  factorized_settings.time = factorized_time;
  factorized_settings.repair_optimizer = RepairOptimizer::ssbb_value;
  factorized_settings.repair_denominators = {7, 15, 31, 63, 127};
  factorized_settings.correction_format_ceiling =
      kFactorizedPackedCorrectionFormat;
  validate_settings(factorized_settings);
  EncodedWindow factorized_window;
  factorized_window.base_mode = 4;
  factorized_window.selected_global_bound =
      factorized_settings.pointwise_tolerance;
  factorized_window.payload = compress_lossless_zstd_payload(factorized_base);
  factorized_window.corrections =
      static_cast<std::uint32_t>(factorized_fields.codes.size());
  factorized_window.correction_payload = format8;
  const auto v14_path = prefix + "-format8-v14.qpc";
  write_stream(v14_path, factorized_settings, {factorized_window});
  const auto v14_bytes = read_bytes(v14_path);
  std::uint32_t observed_v14 = 0;
  if (v14_bytes.size() < kMagic.size() + sizeof(observed_v14))
    throw std::runtime_error("QPC v14 format 8 outer stream is truncated");
  std::memcpy(
      &observed_v14, v14_bytes.data() + kMagic.size(), sizeof(observed_v14));
  if (observed_v14 != kFactorizedPackedVersion)
    throw std::runtime_error("format 8 outer writer did not emit QPC v14");
  const auto [v14_settings, v14_decoded] = decode_stream(v14_path);
  (void)v14_settings;
  if (!bitwise_equal_floats(v14_decoded, factorized_corrected))
    throw std::runtime_error("QPC v14 format 8 outer roundtrip changed values");
  const auto fixed_transcode_path = prefix + "-format9-fixed-transcode-v15.qpc";
  auto fixed_transcode_mode_name = std::string("fixed-serializer-transcode");
  auto fixed_transcode_source = v14_path;
  auto fixed_transcode_output = fixed_transcode_path;
  auto fixed_transcode_argv = std::array<char*, 4>{
      argv[0],
      fixed_transcode_mode_name.data(),
      fixed_transcode_source.data(),
      fixed_transcode_output.data()};
  fixed_serializer_transcode_mode(
      static_cast<int>(fixed_transcode_argv.size()),
      fixed_transcode_argv.data());
  auto fixed_transcode_records = std::vector<DecodedWindowRecord>();
  auto fixed_transcode_version = std::uint32_t{0};
  const auto [fixed_transcode_settings, fixed_transcode_decoded] =
      decode_stream(
          fixed_transcode_path,
          &fixed_transcode_records,
          &fixed_transcode_version);
  (void)fixed_transcode_settings;
  if (fixed_transcode_version != kSubsetRankPackedVersion ||
      fixed_transcode_records.size() != 1 ||
      fixed_transcode_records.front().correction_payload.empty() ||
      fixed_transcode_records.front().correction_payload.front() !=
          kSubsetRankPackedCorrectionFormat ||
      !bitwise_equal_floats(
          fixed_transcode_decoded, factorized_corrected))
    throw std::runtime_error(
        "fixed serializer transcode self-test changed framing or reconstruction");
  auto invalid_v13_format8 = v14_bytes;
  const auto forced_v13 = kAdaptivePackedVersion;
  std::memcpy(
      invalid_v13_format8.data() + kMagic.size(),
      &forced_v13,
      sizeof(forced_v13));
  const auto invalid_v13_format8_path = prefix + "-format8-v13-invalid.qpc";
  write_bytes(invalid_v13_format8_path, invalid_v13_format8);
  bool rejected_v13_format8 = false;
  try {
    (void)decode_stream(invalid_v13_format8_path);
  }
  catch (const std::runtime_error&) {
    rejected_v13_format8 = true;
  }
  if (!rejected_v13_format8)
    throw std::runtime_error("QPC v13 accepted an outer format 8 payload");

  auto subset_rank_settings = factorized_settings;
  subset_rank_settings.channels = subset_golden_channels;
  subset_rank_settings.time = subset_golden_time;
  subset_rank_settings.correction_format_ceiling =
      kSubsetRankPackedCorrectionFormat;
  validate_settings(subset_rank_settings);
  EncodedWindow subset_rank_window;
  subset_rank_window.base_mode = 4;
  subset_rank_window.selected_global_bound =
      subset_rank_settings.pointwise_tolerance;
  subset_rank_window.payload =
      compress_lossless_zstd_payload(subset_golden_base);
  subset_rank_window.corrections =
      static_cast<std::uint32_t>(subset_golden_fields.codes.size());
  subset_rank_window.correction_payload = subset_golden_payload;
  const auto v15_path = prefix + "-format9-v15.qpc";
  write_stream(v15_path, subset_rank_settings, {subset_rank_window});
  const auto v15_bytes = read_bytes(v15_path);
  std::uint32_t observed_v15 = 0;
  if (v15_bytes.size() < kMagic.size() + sizeof(observed_v15))
    throw std::runtime_error("QPC v15 format 9 outer stream is truncated");
  std::memcpy(
      &observed_v15, v15_bytes.data() + kMagic.size(), sizeof(observed_v15));
  if (observed_v15 != kSubsetRankPackedVersion)
    throw std::runtime_error("format 9 outer writer did not emit QPC v15");
  const auto [v15_settings, v15_decoded] = decode_stream(v15_path);
  (void)v15_settings;
  if (!bitwise_equal_floats(v15_decoded, subset_golden_corrected))
    throw std::runtime_error("QPC v15 format 9 outer roundtrip changed values");
  auto invalid_v14_format9 = v15_bytes;
  const auto forced_v14 = kFactorizedPackedVersion;
  std::memcpy(
      invalid_v14_format9.data() + kMagic.size(),
      &forced_v14,
      sizeof(forced_v14));
  const auto invalid_v14_format9_path = prefix + "-format9-v14-invalid.qpc";
  write_bytes(invalid_v14_format9_path, invalid_v14_format9);
  auto rejected_v14_format9 = false;
  try {
    (void)decode_stream(invalid_v14_format9_path);
  }
  catch (const std::runtime_error&) {
    rejected_v14_format9 = true;
  }
  if (!rejected_v14_format9)
    throw std::runtime_error("QPC v14 accepted an outer format 9 payload");

  auto predictive_settings = settings;
  predictive_settings.channels = predictive_channels;
  predictive_settings.time = predictive_time;
  predictive_settings.repair_optimizer = RepairOptimizer::ssbb_value;
  predictive_settings.repair_denominators = {7, 15, 31, 63, 127};
  predictive_settings.correction_format_ceiling =
      kPredictivePackedCorrectionFormat;
  validate_settings(predictive_settings);
  EncodedWindow predictive_window;
  predictive_window.base_mode = 4;
  predictive_window.selected_global_bound =
      predictive_settings.pointwise_tolerance;
  predictive_window.payload =
      compress_lossless_zstd_payload(predictive_base);
  predictive_window.corrections =
      static_cast<std::uint32_t>(predictive_fields.codes.size());
  predictive_window.correction_payload = predictive_tag10;
  const auto v16_path = prefix + "-format10-v16.qpc";
  write_stream(v16_path, predictive_settings, {predictive_window});
  const auto v16_bytes = read_bytes(v16_path);
  auto observed_v16 = std::uint32_t{0};
  if (v16_bytes.size() < kMagic.size() + sizeof(observed_v16))
    throw std::runtime_error("QPC v16 format 10 outer stream is truncated");
  std::memcpy(
      &observed_v16, v16_bytes.data() + kMagic.size(), sizeof(observed_v16));
  auto v16_records = std::vector<DecodedWindowRecord>();
  const auto [v16_settings, v16_decoded] = decode_stream(
      v16_path, &v16_records);
  (void)v16_settings;
  if (observed_v16 != kPredictivePackedVersion ||
      v16_records.size() != 1U ||
      v16_records.front().correction_payload != predictive_tag10 ||
      !bitwise_equal_floats(v16_decoded, predictive_corrected))
    throw std::runtime_error(
        "QPC v16 format 10 outer roundtrip changed bytes or reconstruction");
  auto invalid_v15_format10 = v16_bytes;
  const auto forced_predictive_v15 = kSubsetRankPackedVersion;
  std::memcpy(
      invalid_v15_format10.data() + kMagic.size(),
      &forced_predictive_v15,
      sizeof(forced_predictive_v15));
  const auto invalid_v15_format10_path =
      prefix + "-format10-v15-invalid.qpc";
  write_bytes(invalid_v15_format10_path, invalid_v15_format10);
  auto rejected_v15_format10 = false;
  try {
    (void)decode_stream(invalid_v15_format10_path);
  }
  catch (const std::runtime_error&) {
    rejected_v15_format10 = true;
  }
  if (!rejected_v15_format10)
    throw std::runtime_error("QPC v15 accepted an outer format 10 payload");

  // Versions 12--16 are canonical declarations of the maximum correction
  // tag actually present, not feature flags that may be over-declared.
  auto overdeclared_v15_format8 = v14_bytes;
  const auto forced_v15 = kSubsetRankPackedVersion;
  std::memcpy(
      overdeclared_v15_format8.data() + kMagic.size(),
      &forced_v15,
      sizeof(forced_v15));
  const auto overdeclared_v15_format8_path =
      prefix + "-format8-v15-overdeclared-invalid.qpc";
  write_bytes(overdeclared_v15_format8_path, overdeclared_v15_format8);
  auto rejected_overdeclared_v15_format8 = false;
  try {
    (void)decode_stream(overdeclared_v15_format8_path);
  }
  catch (const std::runtime_error&) {
    rejected_overdeclared_v15_format8 = true;
  }
  if (!rejected_overdeclared_v15_format8)
    throw std::runtime_error(
        "outer decoder accepted QPC v15 without a format 9 payload");
  auto overdeclared_v16_format9 = v15_bytes;
  const auto forced_v16 = kPredictivePackedVersion;
  std::memcpy(
      overdeclared_v16_format9.data() + kMagic.size(),
      &forced_v16,
      sizeof(forced_v16));
  const auto overdeclared_v16_format9_path =
      prefix + "-format9-v16-overdeclared-invalid.qpc";
  write_bytes(overdeclared_v16_format9_path, overdeclared_v16_format9);
  auto rejected_overdeclared_v16_format9 = false;
  try {
    (void)decode_stream(overdeclared_v16_format9_path);
  }
  catch (const std::runtime_error&) {
    rejected_overdeclared_v16_format9 = true;
  }
  if (!rejected_overdeclared_v16_format9)
    throw std::runtime_error(
        "outer decoder accepted QPC v16 without a format 10 payload");

  auto mixed_version_settings = factorized_settings;
  mixed_version_settings.batch = 2;
  mixed_version_settings.correction_format_ceiling =
      kSubsetRankPackedCorrectionFormat;
  validate_settings(mixed_version_settings);
  auto mixed_format9_window = factorized_window;
  mixed_format9_window.correction_payload = factorized_format9;
  const auto mixed_v15_path = prefix + "-mixed-format8-format9-v15.qpc";
  write_stream(
      mixed_v15_path,
      mixed_version_settings,
      {factorized_window, mixed_format9_window});
  auto mixed_version = std::uint32_t{0};
  const auto [mixed_decoded_settings, mixed_decoded] = decode_stream(
      mixed_v15_path, nullptr, &mixed_version);
  (void)mixed_decoded_settings;
  auto mixed_expected = factorized_corrected;
  mixed_expected.insert(
      mixed_expected.end(),
      factorized_corrected.begin(),
      factorized_corrected.end());
  if (mixed_version != kSubsetRankPackedVersion ||
      !bitwise_equal_floats(mixed_decoded, mixed_expected))
    throw std::runtime_error(
        "mixed tag8/tag9 QPC v15 stream changed version or reconstruction");

  auto writer_ceiling6 = adaptive_settings;
  writer_ceiling6.correction_format_ceiling =
      kPackedQuantizedCorrectionFormat;
  auto writer_rejected_format7 = false;
  try {
    write_stream(
        prefix + "-format7-ceiling6-invalid.qpc",
        writer_ceiling6,
        {adaptive_window});
  }
  catch (const std::runtime_error&) {
    writer_rejected_format7 = true;
  }
  if (!writer_rejected_format7)
    throw std::runtime_error(
        "QPET writer accepted format 7 above correction ceiling 6");
  auto writer_ceiling7 = factorized_settings;
  writer_ceiling7.correction_format_ceiling =
      kAdaptivePackedCorrectionFormat;
  auto writer_rejected_format8 = false;
  try {
    write_stream(
        prefix + "-format8-ceiling7-invalid.qpc",
        writer_ceiling7,
        {factorized_window});
  }
  catch (const std::runtime_error&) {
    writer_rejected_format8 = true;
  }
  if (!writer_rejected_format8)
    throw std::runtime_error(
        "QPET writer accepted format 8 above correction ceiling 7");
  auto writer_ceiling8 = subset_rank_settings;
  writer_ceiling8.correction_format_ceiling =
      kFactorizedPackedCorrectionFormat;
  auto writer_rejected_format9 = false;
  try {
    write_stream(
        prefix + "-format9-ceiling8-invalid.qpc",
        writer_ceiling8,
        {subset_rank_window});
  }
  catch (const std::runtime_error&) {
    writer_rejected_format9 = true;
  }
  if (!writer_rejected_format9)
    throw std::runtime_error(
        "QPET writer accepted format 9 above correction ceiling 8");

  // QPET_META must describe the independently decoded float32 tensor, never
  // an encoder-side diagnostic cached in EncodedWindow.
  packed_window.report.max_abs_error = 123.0;
  packed_window.report.max_acf_error = 124.0;
  packed_window.report.min_series_correlation = -0.75;
  const auto outer_report = summarize_outer_constraints(base, v12_decoded, settings);
  const auto expected_outer_report = qpet_correlation::verify_constraints(
      base,
      v12_decoded,
      settings.channels,
      settings.time,
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag);
  if (outer_report.max_abs_error != expected_outer_report.max_abs_error ||
      outer_report.max_acf_error != expected_outer_report.max_acf_error ||
      outer_report.min_series_correlation !=
          expected_outer_report.min_series_correlation ||
      outer_report.max_abs_error == packed_window.report.max_abs_error ||
      outer_report.max_acf_error == packed_window.report.max_acf_error ||
      outer_report.min_series_correlation ==
          packed_window.report.min_series_correlation)
    throw std::runtime_error(
        "QPET metadata self-test did not use outer-decoded constraint metrics");

  // max_digits10 (17 for binary64) is required so every QPET_META double can
  // be parsed back without losing a bit.
  auto precision_report = qpet_correlation::ConstraintReport();
  precision_report.max_abs_error = std::nextafter(0.1, 1.0);
  precision_report.max_acf_error = std::nextafter(0.2, 1.0);
  precision_report.min_series_correlation = std::nextafter(1.0, 0.0);
  auto formatted_metrics = std::ostringstream();
  write_qpet_constraint_metrics(formatted_metrics, precision_report);
  const auto formatted_text = formatted_metrics.str();
  auto parse_metric = [&](const std::string& key) {
    const auto begin = formatted_text.find(key);
    if (begin == std::string::npos)
      throw std::runtime_error("QPET metadata self-test omitted " + key);
    const auto value_begin = begin + key.size();
    const auto end = formatted_text.find(' ', value_begin);
    return std::stod(formatted_text.substr(value_begin, end - value_begin));
  };
  if (parse_metric("max_abs=") != precision_report.max_abs_error ||
      parse_metric("max_acf=") != precision_report.max_acf_error ||
      parse_metric("min_corr=") != precision_report.min_series_correlation)
    throw std::runtime_error(
        "QPET metadata self-test lost binary64 precision");
  auto v11_invalid = read_bytes(v12_path);
  const auto legacy_version = std::uint32_t{11};
  std::memcpy(
      v11_invalid.data() + kMagic.size(), &legacy_version, sizeof(legacy_version));
  const auto v11_invalid_path = prefix + "-format6-v11-invalid.qpc";
  write_bytes(v11_invalid_path, v11_invalid);
  bool rejected_v11_stream = false;
  try {
    (void)decode_stream(v11_invalid_path);
  }
  catch (const std::runtime_error& error) {
    rejected_v11_stream = std::string(error.what()).find(
        "format 6 requires QPC version 12") != std::string::npos;
    if (!rejected_v11_stream)
      throw;
  }
  if (!rejected_v11_stream)
    throw std::runtime_error("QPC v11 accepted an outer format 6 payload");

  EncodedWindow legacy_window;
  legacy_window.base_mode = 4;
  legacy_window.selected_global_bound = settings.pointwise_tolerance;
  legacy_window.payload = compress_lossless_zstd_payload(base);
  auto zero_logical_window = legacy_window;
  zero_logical_window.logical_windows = 0;
  bool writer_rejected_zero_logical = false;
  try {
    write_stream(
        prefix + "-invalid-writer-zero-logical.qpc",
        settings,
        {zero_logical_window});
  }
  catch (const std::runtime_error&) {
    writer_rejected_zero_logical = true;
  }
  if (!writer_rejected_zero_logical)
    throw std::runtime_error("compact writer accepted zero logical windows");
  for (const auto version : {8U, 9U}) {
    const auto path = prefix + "-legacy-v" + std::to_string(version) + ".qpc";
    write_legacy_selftest_stream(path, version, settings, legacy_window);
    const auto [legacy_settings, decoded] = decode_stream(path);
    (void)legacy_settings;
    if (!bitwise_equal_floats(decoded, base))
      throw std::runtime_error("legacy QPC stream changed its lossless payload");
  }
  const auto compact_path = prefix + "-compact-v12.qpc";
  write_stream(compact_path, settings, {legacy_window});
  const auto compact_bytes = read_bytes(compact_path);
  auto expect_stream_rejected = [&](const std::string& label,
                                    const std::vector<unsigned char>& bytes) {
    const auto path = prefix + "-invalid-" + label + ".qpc";
    write_bytes(path, bytes);
    try {
      (void)decode_stream(path);
    }
    catch (const std::runtime_error&) {
      return;
    }
    catch (const std::invalid_argument&) {
      return;
    }
    throw std::runtime_error("outer decoder accepted " + label);
  };
  constexpr auto compact_record_offset =
      kMagic.size() + 9 * sizeof(std::uint32_t) + 5 * sizeof(double);
  constexpr auto compact_global_bound_offset = compact_record_offset +
      sizeof(std::uint16_t) + 2 * sizeof(std::uint8_t);
  constexpr auto compact_payload_size_offset = compact_global_bound_offset +
      sizeof(double) + sizeof(std::uint32_t);
  auto oversized_compact = compact_bytes;
  const auto maximum_u32 = std::numeric_limits<std::uint32_t>::max();
  std::memcpy(
      oversized_compact.data() + compact_payload_size_offset,
      &maximum_u32,
      sizeof(maximum_u32));
  expect_stream_rejected("oversized-compact-payload", oversized_compact);
  auto truncated_compact = compact_bytes;
  truncated_compact.pop_back();
  expect_stream_rejected("truncated-compact-payload", truncated_compact);
  auto nan_bound_compact = compact_bytes;
  const auto nan_bound = std::numeric_limits<double>::quiet_NaN();
  std::memcpy(
      nan_bound_compact.data() + compact_global_bound_offset,
      &nan_bound,
      sizeof(nan_bound));
  expect_stream_rejected("nan-global-bound", nan_bound_compact);
  auto oversized_group_compact = compact_bytes;
  constexpr auto group_size_offset =
      kMagic.size() + 8 * sizeof(std::uint32_t);
  const auto oversized_group =
      static_cast<std::uint32_t>(std::numeric_limits<std::uint16_t>::max()) + 1U;
  std::memcpy(
      oversized_group_compact.data() + group_size_offset,
      &oversized_group,
      sizeof(oversized_group));
  expect_stream_rejected("oversized-compact-group", oversized_group_compact);
  auto oversized_tensor_compact = compact_bytes;
  constexpr auto compact_channels_offset =
      kMagic.size() + 2U * sizeof(std::uint32_t);
  const auto oversized_channels = static_cast<std::uint32_t>(
      kMaximumDecodedValues / settings.time + 1U);
  std::memcpy(
      oversized_tensor_compact.data() + compact_channels_offset,
      &oversized_channels,
      sizeof(oversized_channels));
  expect_stream_rejected(
      "decoded-tensor-above-safety-policy", oversized_tensor_compact);
  auto zero_logical_compact = compact_bytes;
  const auto zero_logical = std::uint16_t{0};
  std::memcpy(
      zero_logical_compact.data() + compact_record_offset,
      &zero_logical,
      sizeof(zero_logical));
  expect_stream_rejected("zero-logical-windows", zero_logical_compact);

  auto oversized_legacy = read_bytes(prefix + "-legacy-v9.qpc");
  constexpr auto legacy_payload_size_offset = compact_record_offset +
      2 * sizeof(std::uint32_t) + 2 * sizeof(double) +
      2 * sizeof(std::uint32_t);
  const auto maximum_u64 = std::numeric_limits<std::uint64_t>::max();
  const auto one_u64 = std::uint64_t{1};
  std::memcpy(
      oversized_legacy.data() + legacy_payload_size_offset,
      &maximum_u64,
      sizeof(maximum_u64));
  std::memcpy(
      oversized_legacy.data() + legacy_payload_size_offset + sizeof(maximum_u64),
      &one_u64,
      sizeof(one_u64));
  expect_stream_rejected("overflowing-legacy-size-sum", oversized_legacy);
  for (const auto version : {10U, 11U}) {
    auto legacy = compact_bytes;
    std::memcpy(legacy.data() + kMagic.size(), &version, sizeof(version));
    const auto path = prefix + "-legacy-v" + std::to_string(version) + ".qpc";
    write_bytes(path, legacy);
    const auto [legacy_settings, decoded] = decode_stream(path);
    (void)legacy_settings;
    if (!bitwise_equal_floats(decoded, base))
      throw std::runtime_error("compact legacy QPC stream changed its payload");
  }

  std::cout << "QPET_PACKED_SELFTEST format4_bytes=" << format4.size()
            << " format6_bytes=" << format6.size()
            << " format7_bytes=" << format7.size()
            << " format8_bytes=" << format8.size()
            << " factorized_format9_bytes=" << factorized_format9.size()
            << " subset_win_format8_bytes=" << subset_win_format8.size()
            << " subset_win_format9_bytes=" << subset_win_format9.size()
            << " factorized_ceiling6_bytes="
            << factorized_ceiling6.size()
            << " factorized_ceiling7_bytes="
            << factorized_ceiling7.size()
            << " factorized_ceiling8_bytes="
            << factorized_ceiling8.size()
            << " factorized_ceiling9_bytes="
            << factorized_ceiling9.size()
            << " bit_width=" << static_cast<unsigned int>(packed_codes.bit_width)
            << " corrections=" << fields.codes.size()
            << " outer_bytes=" << read_bytes(v12_path).size()
            << " adaptive_outer_bytes=" << v13_bytes.size()
            << " factorized_outer_bytes=" << v14_bytes.size()
            << " subset_rank_outer_bytes=" << v15_bytes.size() << '\n';
}

struct ZrcarOracleResult {
  std::size_t bytes = std::numeric_limits<std::size_t>::max();
  std::size_t feasible_states = 0;
  std::vector<std::size_t> blocked;
};

ZrcarOracleResult exhaustive_zrcar_oracle(
    const std::vector<float>& original,
    const std::vector<float>& base,
    const LegacyRepairSeed& seed,
    const Settings& settings)
{
  auto eligible = std::vector<std::size_t>();
  for (std::size_t index = 0; index < base.size(); ++index) {
    if (seed.targets[index] != base[index] &&
        std::abs(static_cast<double>(original[index]) - base[index]) <=
            settings.pointwise_tolerance)
      eligible.push_back(index);
  }
  if (eligible.size() > 10)
    throw std::invalid_argument("small-T ZRCAR oracle supports at most ten decisions");
  ZrcarOracleResult result;
  const auto subsets = std::size_t{1} << eligible.size();
  for (std::size_t mask = 0; mask < subsets; ++mask) {
    auto blocked = std::vector<std::size_t>();
    for (std::size_t bit = 0; bit < eligible.size(); ++bit) {
      if ((mask & (std::size_t{1} << bit)) != 0)
        blocked.push_back(eligible[bit]);
    }
    auto candidate = evaluate_zrcar_state(
        original, base, seed, blocked, settings);
    if (!candidate.feasible)
      continue;
    ++result.feasible_states;
    if (candidate.correction_payload.size() < result.bytes) {
      result.bytes = candidate.correction_payload.size();
      result.blocked = std::move(blocked);
    }
  }
  if (result.feasible_states == 0)
    throw std::runtime_error("small-T ZRCAR oracle found no feasible state");
  return result;
}

void zrcar_selftest_mode(int argc, char** argv)
{
  if (argc != 2)
    throw std::invalid_argument("usage: qpet_correlation_batch zrcar-selftest");
  constexpr std::size_t time = 10;
  Settings settings;
  settings.batch = 1;
  settings.channels = 1;
  settings.time = time;
  settings.pointwise_tolerance = 0.4;
  settings.acf_tolerance = 0.03;
  settings.minimum_correlation = 0.99;
  settings.max_lag = 3;
  settings.qoz_level = 1;
  settings.correction_mode = 2;
  settings.estimator_scales = {1.0};
  settings.qpet_confidence = 0.999;
  settings.qpet_k = 2.0;
  settings.group_size = 1;
  settings.repair_denominators = {};
  settings.repair_optimizer = RepairOptimizer::zrcar;
  validate_settings(settings);
  auto invalid_zrcar = settings;
  invalid_zrcar.correction_mode = 0;
  auto rejected_online_zrcar = false;
  try {
    validate_settings(invalid_zrcar);
  }
  catch (const std::invalid_argument&) {
    rejected_online_zrcar = true;
  }
  if (!rejected_online_zrcar)
    throw std::runtime_error("ZRCAR self-test accepted online correction mode");
  invalid_zrcar = settings;
  invalid_zrcar.enable_vehicle_semantic = true;
  auto rejected_semantic_zrcar = false;
  try {
    validate_settings(invalid_zrcar);
  }
  catch (const std::invalid_argument&) {
    rejected_semantic_zrcar = true;
  }
  if (!rejected_semantic_zrcar)
    throw std::runtime_error("ZRCAR self-test accepted the semantic profile");
  invalid_zrcar = settings;
  invalid_zrcar.batch = 2;
  invalid_zrcar.group_size = 2;
  auto rejected_grouped_zrcar = false;
  try {
    validate_settings(invalid_zrcar);
  }
  catch (const std::invalid_argument&) {
    rejected_grouped_zrcar = true;
  }
  if (!rejected_grouped_zrcar)
    throw std::runtime_error("ZRCAR self-test accepted group_size greater than one");

  // Exact stored-base identities must consume one top-K slot even when an
  // estimator-scale candidate generated the same decoder inputs. The safe
  // base-byte lower bound is applied only after the complete legacy search,
  // so filtered records cannot leave a previously discarded vacancy.
  auto retained_fixture = std::vector<RetainedLegacyRecord>();
  auto make_retained_fixture = [](std::size_t base_bytes, unsigned char marker) {
    EncodedWindow encoded;
    encoded.base_mode = 0;
    encoded.selected_global_bound = 0.25;
    encoded.selected_estimator_scale = 1.0;
    encoded.corrections = 1;
    encoded.payload.assign(base_bytes, marker);
    encoded.correction_payload.assign(5, 0x7fU);
    encoded.reconstruction = {1.0F, 2.0F};
    return encoded;
  };
  auto retained_first = make_retained_fixture(8, 0x01U);
  retain_unique_stored_base(retained_fixture, retained_first, 0);
  auto retained_duplicate = retained_first;
  retained_duplicate.selected_estimator_scale = 9.0;
  retain_unique_stored_base(retained_fixture, retained_duplicate, 1);
  for (std::size_t index = 0; index < 5; ++index) {
    retain_unique_stored_base(
        retained_fixture,
        make_retained_fixture(
            9 + index, static_cast<unsigned char>(0x10U + index)),
        index + 2);
  }
  retain_unique_stored_base(
      retained_fixture, make_retained_fixture(20, 0x30U), 7);
  if (retained_fixture.size() != 7)
    throw std::runtime_error("ZRCAR stored-base identity de-duplication failed");
  select_zrcar_base_records(retained_fixture, 20);
  if (retained_fixture.size() != kZrcarBaseRecordCount ||
      retained_fixture.front().encoded.payload.size() != 8 ||
      retained_fixture.back().encoded.payload.size() != 11 ||
      std::any_of(
          retained_fixture.begin(),
          retained_fixture.end(),
          [](const RetainedLegacyRecord& record) {
            return record.encoded.payload.size() >= 20;
          }))
    throw std::runtime_error(
        "ZRCAR top-K ranking or safe base-byte lower bound failed");

  LegacyRepairSeed seed_identity_first;
  seed_identity_first.ordinal = 0;
  seed_identity_first.targets = {1.0F, 2.0F};
  seed_identity_first.channel_scales = {0.25F};
  seed_identity_first.corrections = 1;
  seed_identity_first.correction_payload = {0x01U, 0x02U};
  seed_identity_first.reconstruction = {1.0F, 2.0F};
  auto seed_identity_duplicate = seed_identity_first;
  seed_identity_duplicate.ordinal = 1;
  auto seed_identity_second = seed_identity_first;
  seed_identity_second.ordinal = 2;
  seed_identity_second.targets[1] = 3.0F;
  seed_identity_second.correction_payload.push_back(0x03U);
  seed_identity_second.reconstruction[1] = 3.0F;
  const auto seed_identity_order = select_zrcar_seed_order(
      {seed_identity_first, seed_identity_duplicate, seed_identity_second});
  if (seed_identity_order != std::vector<std::size_t>{0, 2})
    throw std::runtime_error("ZRCAR seed identity de-duplication hid a unique seed");
  auto seed_identity_third = seed_identity_first;
  seed_identity_third.ordinal = 3;
  seed_identity_third.targets[0] = 4.0F;
  seed_identity_third.correction_payload = {0x04U, 0x05U, 0x06U, 0x07U};
  seed_identity_third.reconstruction[0] = 4.0F;
  auto seed_identity_fvro_best = seed_identity_first;
  seed_identity_fvro_best.ordinal = 4;
  seed_identity_fvro_best.targets = {5.0F, 2.0F};
  seed_identity_fvro_best.correction_payload = {0x08U};
  seed_identity_fvro_best.reconstruction = seed_identity_fvro_best.targets;
  seed_identity_fvro_best.fvro_derived = true;
  auto seed_identity_fvro_second = seed_identity_first;
  seed_identity_fvro_second.ordinal = 5;
  seed_identity_fvro_second.targets = {6.0F, 2.0F};
  seed_identity_fvro_second.correction_payload = {0x09U, 0x0aU};
  seed_identity_fvro_second.reconstruction = seed_identity_fvro_second.targets;
  seed_identity_fvro_second.fvro_derived = true;
  const auto mixed_seeds = std::vector<LegacyRepairSeed>{
      seed_identity_first,
      seed_identity_duplicate,
      seed_identity_second,
      seed_identity_third,
      seed_identity_fvro_best,
      seed_identity_fvro_second};
  const auto standalone_order = select_zrcar_seed_order(
      mixed_seeds, RepairOptimizer::zrcar);
  const auto full_order = select_zrcar_seed_order(
      mixed_seeds, RepairOptimizer::fvro);
  if (standalone_order != std::vector<std::size_t>{0, 2} ||
      full_order != std::vector<std::size_t>{0, 2, 4})
    throw std::runtime_error(
        "full FVRO did not freeze standalone ZRCAR seeds before its extra seed");

  auto boundary_settings = settings;
  boundary_settings.pointwise_tolerance = std::nextafter(1.0, 0.0);
  auto boundary_original = std::vector<float>(time, 0.0F);
  auto boundary_base = boundary_original;
  boundary_base[3] = 1.0F;
  const auto below_boundary = rank_zrcar_rollbacks(
      boundary_original,
      boundary_base,
      boundary_original,
      {},
      boundary_settings);
  if (!below_boundary.empty())
    throw std::runtime_error(
        "ZRCAR point-wise ranking admitted a target above the strict tolerance");
  boundary_settings.pointwise_tolerance = 1.0;
  const auto at_boundary = rank_zrcar_rollbacks(
      boundary_original,
      boundary_base,
      boundary_original,
      {},
      boundary_settings);
  if (at_boundary.size() != 1 || at_boundary.front().index != 3)
    throw std::runtime_error(
        "ZRCAR point-wise ranking rejected a target exactly at tolerance");

  auto evaluate_fixture = [&](std::size_t fixture) {
    auto original = std::vector<float>(time);
    auto base = std::vector<float>(time);
    auto state = static_cast<std::uint32_t>(0x9e3779b9U + fixture * 2654435761U);
    for (std::size_t index = 0; index < time; ++index) {
      original[index] = static_cast<float>(
          0.8 * std::sin(0.43 * static_cast<double>(index)) +
          0.07 * static_cast<double>(index));
      state = state * 1664525U + 1013904223U;
      const auto centered = static_cast<int>((state >> 24U) & 0xffU) - 128;
      const auto error = 0.34 * static_cast<double>(centered) / 128.0;
      base[index] = static_cast<float>(original[index] + error);
    }
    EncodedWindow legacy;
    legacy.reconstruction = base;
    auto seeds = std::vector<LegacyRepairSeed>();
    finalize_posthoc_repair(original, settings, legacy, &seeds);
    if (seeds.size() != 1 || legacy.corrections < 2)
      throw std::runtime_error("fixed ZRCAR fixture did not produce one sparse seed");
    auto optimized = legacy;
    run_zrcar_search(original, base, settings, seeds, optimized);
    auto seed = std::move(seeds.front());
    auto oracle = exhaustive_zrcar_oracle(original, base, seed, settings);
    return std::make_tuple(
        std::move(original),
        std::move(base),
        std::move(legacy),
        std::move(optimized),
        std::move(seed),
        std::move(oracle));
  };

  // Fixture 2 is a fixed strict-win case on which the bounded search reaches
  // the exact actual-serializer oracle. Fixture 666 is a fixed, disclosed gap
  // case demonstrating that ZRCAR remains a bounded heuristic. Fixture 48
  // proves that a retained exhausted-infeasible parent can produce a fully
  // serialized and verified feasible child.
  constexpr std::size_t found_fixture = 2;
  auto [found_original,
        found_base,
        found_legacy,
        found_optimized,
        found_seed,
        oracle] = evaluate_fixture(found_fixture);
  constexpr std::size_t gap_fixture = 666;
  auto [gap_original,
        gap_base,
        gap_legacy,
        gap_optimized,
        gap_seed,
        gap_oracle] = evaluate_fixture(gap_fixture);
  constexpr std::size_t recovery_fixture = 48;
  auto [recovery_original,
        recovery_base,
        recovery_legacy,
        recovery_optimized,
        recovery_seed,
        recovery_oracle] = evaluate_fixture(recovery_fixture);
  (void)gap_original;
  (void)gap_base;
  (void)gap_seed;
  (void)recovery_original;
  (void)recovery_base;
  (void)recovery_legacy;
  (void)recovery_seed;
  (void)recovery_oracle;
  if (!found_optimized.report.all_ok() ||
      found_optimized.correction_payload.size() >=
          found_legacy.correction_payload.size() ||
      found_optimized.correction_payload.size() < oracle.bytes ||
      found_optimized.zrcar_legacy_seeds_selected > kZrcarSeedCount ||
      found_optimized.zrcar_fvro_seeds_selected != 0 ||
      found_optimized.zrcar_candidates_tested > kZrcarBudgetPerSeed ||
      found_optimized.zrcar_prefix_candidates_tested >
          kZrcarPrefixBudgetPerSeed ||
      found_optimized.zrcar_states_generated >
          kZrcarMaximumGeneratedPerSeed ||
      found_optimized.zrcar_infeasible_frontier_expansions >
          kZrcarInfeasibleBeamWidth * kZrcarMaximumDepth ||
      found_optimized.zrcar_candidates_tested !=
          found_optimized.zrcar_prefix_candidates_tested +
              found_optimized.zrcar_local_candidates_tested)
    throw std::runtime_error("ZRCAR bounded search/oracle accounting is invalid");
  if (found_optimized.correction_payload.size() != oracle.bytes)
    throw std::runtime_error("ZRCAR strict-win fixture did not hit the exact oracle");
  if (gap_optimized.correction_payload.size() >=
          gap_legacy.correction_payload.size() ||
      gap_optimized.correction_payload.size() <= gap_oracle.bytes)
    throw std::runtime_error(
        "ZRCAR disclosed-gap fixture no longer records a positive heuristic gap: " +
        std::to_string(gap_legacy.correction_payload.size()) + "/" +
        std::to_string(gap_optimized.correction_payload.size()) + "/" +
        std::to_string(gap_oracle.bytes));
  if (recovery_optimized.zrcar_infeasible_frontier_expansions == 0 ||
      recovery_optimized.zrcar_infeasible_frontier_recoveries == 0 ||
      !recovery_optimized.report.all_ok())
    throw std::runtime_error(
        "ZRCAR infeasible-frontier fixture did not recover a verified child");

  auto repeated = found_legacy;
  run_zrcar_search(
      found_original,
      found_base,
      settings,
      std::vector<LegacyRepairSeed>{found_seed},
      repeated);
  if (repeated.corrections != found_optimized.corrections ||
      repeated.correction_payload != found_optimized.correction_payload)
    throw std::runtime_error("ZRCAR repeated search was not deterministic");
  require_float32_bitwise_equal(
      repeated.reconstruction,
      found_optimized.reconstruction,
      "ZRCAR repeated reconstruction was not bitwise deterministic");

  auto legacy_settings = settings;
  legacy_settings.repair_optimizer = RepairOptimizer::legacy;
  auto legacy_untouched = found_legacy;
  run_zrcar_search(
      found_original,
      found_base,
      legacy_settings,
      std::vector<LegacyRepairSeed>{found_seed},
      legacy_untouched);
  if (legacy_untouched.correction_payload != found_legacy.correction_payload ||
      legacy_untouched.zrcar_candidates_tested != 0 ||
      legacy_untouched.zrcar_selected)
    throw std::runtime_error("legacy optimizer executed or selected ZRCAR work");

  EncodedWindow tie_legacy;
  tie_legacy.selected_global_bound = 0.1;
  tie_legacy.payload.assign(20, 0x11U);
  tie_legacy.correction_payload.assign(10, 0x12U);
  tie_legacy.corrections = 2;
  tie_legacy.reconstruction = {0.0F, -0.0F};
  const auto tie_snapshot = tie_legacy;
  EncodedWindow tie_zrcar;
  tie_zrcar.selected_global_bound = 999.0;
  tie_zrcar.payload.assign(25, 0x21U);
  tie_zrcar.correction_payload.assign(5, 0x22U);
  tie_zrcar.corrections = 1;
  tie_zrcar.reconstruction = {1.0F, 2.0F};
  if (retain_strict_zrcar_whole_window(std::move(tie_zrcar), tie_legacy) ||
      tie_legacy.payload != tie_snapshot.payload ||
      tie_legacy.correction_payload != tie_snapshot.correction_payload ||
      tie_legacy.corrections != tie_snapshot.corrections)
    throw std::runtime_error("whole-window ZRCAR tie replaced the legacy incumbent");
  require_float32_bitwise_equal(
      tie_legacy.reconstruction,
      tie_snapshot.reconstruction,
      "whole-window ZRCAR tie changed the legacy reconstruction");

  auto rejected_signed_zero = false;
  try {
    require_float32_bitwise_equal(
        std::vector<float>{0.0F},
        std::vector<float>{-0.0F},
        "intentional signed-zero serializer replay mismatch");
  }
  catch (const std::runtime_error&) {
    rejected_signed_zero = true;
  }
  if (!rejected_signed_zero)
    throw std::runtime_error("ZRCAR bitwise replay accepted a signed-zero mismatch");

  // Exercise the production two-stage base portfolio on a bounded synthetic
  // window. This catches regressions where the ZRCAR winner accidentally
  // influences the independent legacy Algorithm-4 trajectory or a whole-
  // window tie uses the legacy larger-bound tie breaker.
  auto production_settings = settings;
  production_settings.time = 32;
  production_settings.max_lag = 4;
  production_settings.pointwise_tolerance = 0.25;
  production_settings.acf_tolerance = 0.05;
  production_settings.minimum_correlation = 0.99;
  production_settings.repair_denominators = {7};
  auto production_source = std::vector<float>(production_settings.time);
  for (std::size_t index = 0; index < production_source.size(); ++index) {
    production_source[index] = static_cast<float>(
        0.6 * std::sin(0.21 * static_cast<double>(index)) +
        0.15 * std::cos(0.07 * static_cast<double>(index)) +
        0.003 * static_cast<double>(index));
  }
  const auto production_zrcar = compress_window(
      production_source, production_settings);
  auto production_legacy_settings = production_settings;
  production_legacy_settings.repair_optimizer = RepairOptimizer::legacy;
  const auto production_legacy = compress_window(
      production_source, production_legacy_settings);
  if (production_zrcar.encoded_size() > production_legacy.encoded_size() ||
      production_zrcar.zrcar_base_records_tested > kZrcarBaseRecordCount ||
      production_zrcar.zrcar_legacy_seeds_selected >
          kZrcarBaseRecordCount * kZrcarSeedCount ||
      production_zrcar.zrcar_fvro_seeds_selected != 0 ||
      production_zrcar.zrcar_candidates_tested >
          kZrcarBaseRecordCount * kZrcarSeedCount * kZrcarBudgetPerSeed ||
      production_zrcar.zrcar_prefix_candidates_tested >
          kZrcarBaseRecordCount * kZrcarSeedCount *
              kZrcarPrefixBudgetPerSeed ||
      production_zrcar.zrcar_states_generated >
          kZrcarBaseRecordCount * kZrcarSeedCount *
              kZrcarMaximumGeneratedPerSeed ||
      production_zrcar.zrcar_infeasible_frontier_expansions >
          kZrcarBaseRecordCount * kZrcarSeedCount *
              kZrcarInfeasibleBeamWidth * kZrcarMaximumDepth ||
      production_zrcar.zrcar_candidates_tested !=
          production_zrcar.zrcar_prefix_candidates_tested +
              production_zrcar.zrcar_local_candidates_tested ||
      production_zrcar.zrcar_states_generated !=
          production_zrcar.zrcar_candidates_tested +
              production_zrcar.zrcar_infeasible_states)
    throw std::runtime_error(
        "production ZRCAR portfolio violated zero-regression or budget accounting");
  if (production_zrcar.encoded_size() == production_legacy.encoded_size()) {
    if (production_zrcar.payload != production_legacy.payload ||
        production_zrcar.correction_payload !=
            production_legacy.correction_payload ||
        production_zrcar.corrections != production_legacy.corrections)
      throw std::runtime_error(
          "production whole-window tie did not retain legacy bytes/count");
    require_float32_bitwise_equal(
        production_zrcar.reconstruction,
        production_legacy.reconstruction,
        "production whole-window tie changed the legacy reconstruction");
  }

  std::cout << "QPET_ZRCAR_SELFTEST fixture=" << found_fixture
            << " legacy_bytes=" << found_legacy.correction_payload.size()
            << " zrcar_bytes=" << found_optimized.correction_payload.size()
            << " oracle_bytes=" << oracle.bytes
            << " oracle_feasible=" << oracle.feasible_states
            << " prefix_evaluated="
            << found_optimized.zrcar_prefix_candidates_tested
            << " local_evaluated="
            << found_optimized.zrcar_local_candidates_tested
            << " generated=" << found_optimized.zrcar_states_generated
            << " infeasible=" << found_optimized.zrcar_infeasible_states
            << " frontier_expansions="
            << found_optimized.zrcar_infeasible_frontier_expansions
            << " frontier_recoveries="
            << found_optimized.zrcar_infeasible_frontier_recoveries
            << " gap_fixture=" << gap_fixture
            << " gap_heuristic=" << gap_optimized.correction_payload.size()
            << " gap_oracle=" << gap_oracle.bytes
            << " recovery_fixture=" << recovery_fixture
            << " recovery_expansions="
            << recovery_optimized.zrcar_infeasible_frontier_expansions
            << " recovery_children="
            << recovery_optimized.zrcar_infeasible_frontier_recoveries
            << " production_legacy_bytes=" << production_legacy.encoded_size()
            << " production_zrcar_bytes=" << production_zrcar.encoded_size()
            << " production_base_records="
            << production_zrcar.zrcar_base_records_tested << '\n';
}

void fvro_selftest_mode(int argc, char** argv)
{
  if (argc != 2)
    throw std::invalid_argument("usage: qpet_correlation_batch fvro-selftest");
  if (parse_repair_optimizer("legacy") != RepairOptimizer::legacy ||
      parse_repair_optimizer("zrcar") != RepairOptimizer::zrcar ||
      parse_repair_optimizer("fvro_value") != RepairOptimizer::fvro_value ||
      parse_repair_optimizer("fvro") != RepairOptimizer::fvro ||
      std::string(repair_optimizer_name(RepairOptimizer::fvro_value)) !=
          "fvro_value" ||
      std::string(repair_optimizer_name(RepairOptimizer::fvro)) != "fvro")
    throw std::runtime_error("FVRO optimizer mode parsing/naming is inconsistent");

  constexpr std::size_t channels = 2;
  constexpr std::size_t time = 256;
  Settings settings;
  settings.batch = 1;
  settings.channels = channels;
  settings.time = time;
  settings.pointwise_tolerance = 0.03;
  settings.acf_tolerance = 0.001;
  settings.minimum_correlation = 0.99999;
  settings.max_lag = 16;
  settings.qoz_level = 1;
  settings.correction_mode = 2;
  settings.estimator_scales = {1.0};
  settings.qpet_confidence = 0.999;
  settings.qpet_k = 2.0;
  settings.group_size = 1;
  settings.repair_denominators.assign(
      kFvroLatticeDenominators.begin(), kFvroLatticeDenominators.end());
  settings.repair_optimizer = RepairOptimizer::fvro_value;
  validate_settings(settings);

  auto require_invalid = [](const Settings& candidate, const char* message) {
    try {
      validate_settings(candidate);
    }
    catch (const std::invalid_argument&) {
      return;
    }
    throw std::runtime_error(message);
  };
  auto invalid = settings;
  invalid.repair_denominators = {15, 7, 31, 63, 127};
  require_invalid(
      invalid,
      "FVRO accepted a reordered denominator bank that changes protocol v1");
  invalid = settings;
  invalid.repair_denominators = {7, 15, 31, 63};
  require_invalid(invalid, "FVRO accepted an incomplete denominator bank");
  invalid = settings;
  invalid.correction_mode = 0;
  require_invalid(invalid, "FVRO accepted online correction mode");
  invalid = settings;
  invalid.enable_vehicle_semantic = true;
  require_invalid(invalid, "FVRO accepted the vehicle-semantic profile");
  invalid = settings;
  invalid.batch = 2;
  invalid.group_size = 2;
  require_invalid(invalid, "FVRO accepted group_size greater than one");
  invalid = settings;
  invalid.repair_optimizer = RepairOptimizer::fvro;
  invalid.repair_denominators = {7, 15, 31, 63, 126};
  require_invalid(invalid, "full FVRO accepted a different lattice bank");

  // Each channel has a largest residual of exactly seven. The frozen D=7
  // lattice therefore has scale one. All other residuals have a fractional
  // part in [0.2,0.8], making exact escapes mandatory at epsilon=0.03. The
  // D=127 lattice has step 7/127 and hence a pointwise-feasible grid value for
  // every point. Pseudorandom fractional bits prevent this from degenerating
  // into an artificially compressible constant-residual fixture.
  auto original = std::vector<float>(channels * time);
  auto base = std::vector<float>(channels * time);
  auto state = std::uint32_t{0x31415926U};
  for (std::size_t channel = 0; channel < channels; ++channel) {
    for (std::size_t index = 0; index < time; ++index) {
      const auto offset = channel * time + index;
      const auto phase = 0.031 * static_cast<double>(index) + 0.37 * channel;
      original[offset] = static_cast<float>(
          900.0 * std::sin(phase) + 130.0 * std::cos(0.43 * phase) +
          0.75 * static_cast<double>(index) + 50.0 * channel);
      state = state * 1664525U + 1013904223U;
      const auto integer = 1U + ((state >> 27U) % 5U);
      const auto fraction_bits = state & 0x00ffffffU;
      const auto fraction = 0.2 +
          0.6 * static_cast<double>(fraction_bits) / 16777215.0;
      const auto residual = index == 0
          ? 7.0
          : static_cast<double>(integer) + fraction;
      base[offset] = static_cast<float>(original[offset] - residual);
    }
  }

  const auto coarse_scales = make_fvro_channel_scales(
      original, base, channels, time, 7);
  for (const auto scale : coarse_scales) {
    if (float32_bits(scale) != float32_bits(1.0F))
      throw std::runtime_error("FVRO fixed fixture lost its exact D=7 scale");
  }
  auto legacy_seed = serialize_fvro_reconstruction(
      original,
      base,
      original,
      coarse_scales,
      original,
      settings,
      0);
  legacy_seed.fvro_derived = false;
  EncodedWindow legacy;
  legacy.corrections = legacy_seed.corrections;
  legacy.correction_payload = legacy_seed.correction_payload;
  legacy.reconstruction = legacy_seed.reconstruction;
  legacy.report = legacy_seed.report;
  legacy.repair_candidates_tested = 1;
  if (legacy.correction_payload.empty() || !legacy.report.all_ok())
    throw std::runtime_error("FVRO fixed legacy fixture is empty or infeasible");

  auto duplicate_provenance = std::vector<LegacyRepairSeed>{legacy_seed};
  auto fvro_rediscovery = legacy_seed;
  fvro_rediscovery.ordinal = 1;
  fvro_rediscovery.fvro_derived = true;
  retain_unique_fvro_seed(
      duplicate_provenance, std::move(fvro_rediscovery));
  if (duplicate_provenance.size() != 1 ||
      duplicate_provenance.front().fvro_derived)
    throw std::runtime_error(
        "FVRO rediscovery falsely changed frozen-legacy seed provenance");

  // The production strict comparator must not replace bytes, count, or
  // reconstruction on a tie, even if a later-search seed is retained.
  auto tied_incumbent = legacy;
  const auto tied_snapshot = tied_incumbent;
  auto tied_candidate = legacy_seed;
  tied_candidate.correction_payload.assign(
      legacy.correction_payload.size(), 0x5aU);
  tied_candidate.corrections = 1;
  tied_candidate.reconstruction = base;
  retain_fvro_if_strictly_smaller(tied_candidate, tied_incumbent);
  if (tied_incumbent.correction_payload != tied_snapshot.correction_payload ||
      tied_incumbent.corrections != tied_snapshot.corrections ||
      tied_incumbent.fvro_selected)
    throw std::runtime_error("FVRO correction-byte tie replaced the incumbent");
  require_float32_bitwise_equal(
      tied_incumbent.reconstruction,
      tied_snapshot.reconstruction,
      "FVRO correction-byte tie changed the incumbent reconstruction");

  auto disabled_settings = settings;
  disabled_settings.repair_optimizer = RepairOptimizer::legacy;
  auto disabled = legacy;
  auto disabled_seeds = std::vector<LegacyRepairSeed>{legacy_seed};
  run_fvro_value_search(
      original, base, disabled_settings, disabled_seeds, disabled);
  if (disabled.correction_payload != legacy.correction_payload ||
      disabled.corrections != legacy.corrections ||
      disabled.fvro_global_candidates_generated != 0 ||
      disabled.fvro_channel_candidates_generated != 0 ||
      disabled.fvro_local_proposals_generated != 0 ||
      disabled.fvro_selected_channels != 0 ||
      disabled.fvro_strict_channel_steps != 0 ||
      disabled.fvro_selected)
    throw std::runtime_error("legacy optimizer executed FVRO work");
  require_float32_bitwise_equal(
      disabled.reconstruction,
      legacy.reconstruction,
      "legacy optimizer changed bytes through the FVRO entry point");

  // Independently prove that the fine lattice is an exact-serializer feasible
  // strict-win candidate. This is not a proxy assertion: evaluate performs
  // repair, formats-0--6 selection, QPC-v12 decode, bitwise replay, and direct
  // three-constraint verification.
  const auto fine_scales = make_fvro_channel_scales(
      original, base, channels, time, 127);
  qpet_correlation::feasible_value::TargetBuilder fine_builder(
      original,
      base,
      fine_scales,
      channels,
      time,
      settings.pointwise_tolerance);
  const auto fine_minimum = fine_builder.minimum_candidate();
  auto fine = evaluate_fvro_targets(
      original,
      base,
      fine_minimum.values,
      fine_scales,
      settings,
      1);
  if (!fine.feasible || !fine.seed.report.all_ok() ||
      fine.seed.correction_payload.size() >= legacy.correction_payload.size())
    throw std::runtime_error(
        "FVRO fixed fine-lattice candidate is not a verified strict byte win");
  const auto fine_escape_counts = quantized_escape_counts_by_channel(
      base, fine.seed.reconstruction, fine_scales, channels, time);
  if (std::accumulate(
          fine_escape_counts.begin(),
          fine_escape_counts.end(),
          std::size_t{0}) != 0)
    throw std::runtime_error("FVRO fine-lattice fixture retained an exact escape");

  auto optimized = legacy;
  auto seeds = std::vector<LegacyRepairSeed>{legacy_seed};
  run_fvro_value_search(original, base, settings, seeds, optimized);
  if (!optimized.report.all_ok() || !optimized.fvro_selected ||
      optimized.correction_payload.size() >= legacy.correction_payload.size())
    throw std::runtime_error("FVRO fixed bounded search lost its strict byte win");
  if (optimized.fvro_global_candidates_generated !=
          optimized.fvro_global_candidates_serialized +
              optimized.fvro_global_candidates_infeasible ||
      optimized.fvro_channel_candidates_generated !=
          optimized.fvro_channel_candidates_serialized +
              optimized.fvro_channel_candidates_infeasible ||
      optimized.fvro_local_proposals_infeasible >
          optimized.fvro_local_proposals_generated)
    throw std::runtime_error("FVRO generated/serialized/infeasible counts diverged");
  if (optimized.fvro_global_candidates_serialized >
          kFvroGlobalSerializerBudget ||
      optimized.fvro_channel_candidates_serialized >
          kFvroChannelSerializerBudget ||
      optimized.fvro_local_proposals_generated >
          kFvroLocalProposalBudgetPerBase ||
      optimized.fvro_global_candidates_serialized +
              optimized.fvro_channel_candidates_serialized >
          kFvroFullSerializerBudgetPerBase ||
      optimized.fvro_selected_channels > kFvroMaximumDifficultChannels ||
      optimized.fvro_selected_channels >
          optimized.fvro_strict_channel_steps ||
      optimized.fvro_strict_channel_steps >
          optimized.fvro_channel_candidates_serialized ||
      ((optimized.fvro_selected_channels == 0) !=
       (optimized.fvro_strict_channel_steps == 0)) ||
      optimized.fvro_seeds_retained > kFvroFullSerializerBudgetPerBase)
    throw std::runtime_error("FVRO fixed search exceeded a v1 budget");
  if (optimized.fvro_channel_candidates_generated == 0 ||
      optimized.fvro_local_proposals_generated == 0 ||
      optimized.zrcar_candidates_tested != 0 || optimized.zrcar_selected)
    throw std::runtime_error(
        "FVRO value-only fixture skipped channel search or executed ZRCAR");
  if (optimized.repair_candidates_tested !=
      1U + optimized.fvro_global_candidates_serialized +
          optimized.fvro_channel_candidates_serialized)
    throw std::runtime_error("FVRO repair-candidate accounting is inconsistent");
  if (seeds.size() <= 1 ||
      optimized.fvro_seeds_retained != seeds.size() - 1)
    throw std::runtime_error("FVRO did not retain its unique feasible seed bank");

  // Fixed fixture 13 from the disclosed T=10 deterministic generator. Unlike
  // the portfolio smoke fixture below, it exercises the genuinely composed
  // path: value search contributes the third seed and ZRCAR from that seed is
  // the strict final winner. This is a fixed regression, not a runtime search.
  constexpr std::size_t composed_fixture = 13;
  auto composed_settings = settings;
  composed_settings.channels = 1;
  composed_settings.time = 10;
  composed_settings.max_lag = 3;
  composed_settings.pointwise_tolerance = 0.4;
  composed_settings.acf_tolerance = 0.03;
  composed_settings.minimum_correlation = 0.99;
  composed_settings.repair_optimizer = RepairOptimizer::fvro;
  validate_settings(composed_settings);
  auto composed_original = std::vector<float>(composed_settings.time);
  auto composed_base = std::vector<float>(composed_settings.time);
  auto composed_state = static_cast<std::uint32_t>(
      0x9e3779b9U + composed_fixture * 2654435761U);
  for (std::size_t index = 0; index < composed_settings.time; ++index) {
    composed_original[index] = static_cast<float>(
        0.8 * std::sin(0.43 * static_cast<double>(index)) +
        0.07 * static_cast<double>(index));
    composed_state = composed_state * 1664525U + 1013904223U;
    const auto centered =
        static_cast<int>((composed_state >> 24U) & 0xffU) - 128;
    const auto error = 0.34 * static_cast<double>(centered) / 128.0;
    composed_base[index] =
        static_cast<float>(composed_original[index] + error);
  }
  EncodedWindow composed_legacy;
  composed_legacy.selected_global_bound = 0.25;
  composed_legacy.payload.assign(17, 0xa5U);
  composed_legacy.reconstruction = composed_base;
  auto composed_frozen_seeds = std::vector<LegacyRepairSeed>();
  finalize_posthoc_repair(
      composed_original,
      composed_settings,
      composed_legacy,
      &composed_frozen_seeds);

  auto composed_value = composed_legacy;
  auto composed_value_seeds = composed_frozen_seeds;
  auto composed_value_settings = composed_settings;
  composed_value_settings.repair_optimizer = RepairOptimizer::fvro_value;
  run_fvro_value_search(
      composed_original,
      composed_base,
      composed_value_settings,
      composed_value_seeds,
      composed_value);

  auto composed_zrcar = composed_legacy;
  auto composed_zrcar_settings = composed_settings;
  composed_zrcar_settings.repair_optimizer = RepairOptimizer::zrcar;
  run_zrcar_search(
      composed_original,
      composed_base,
      composed_zrcar_settings,
      composed_frozen_seeds,
      composed_zrcar);

  auto composed_full = composed_legacy;
  auto composed_full_seeds = composed_frozen_seeds;
  run_fvro_value_search(
      composed_original,
      composed_base,
      composed_settings,
      composed_full_seeds,
      composed_full);
  const auto composed_full_value = composed_full;
  const auto standalone_seed_order = select_zrcar_seed_order(
      composed_frozen_seeds, RepairOptimizer::zrcar);
  const auto full_seed_order = select_zrcar_seed_order(
      composed_full_seeds, RepairOptimizer::fvro);
  if (standalone_seed_order.size() != kZrcarSeedCount ||
      full_seed_order.size() != kFullFvroZrcarSeedCount ||
      !composed_full_seeds[full_seed_order.back()].fvro_derived)
    throw std::runtime_error(
        "fixed composed fixture did not select its third FVRO-derived seed");
  for (std::size_t index = 0; index < standalone_seed_order.size(); ++index) {
    const auto& standalone_seed =
        composed_frozen_seeds[standalone_seed_order[index]];
    const auto& full_seed = composed_full_seeds[full_seed_order[index]];
    if (full_seed.fvro_derived ||
        !same_float32_tensor_bits(standalone_seed.targets, full_seed.targets) ||
        !same_float32_tensor_bits(
            standalone_seed.channel_scales, full_seed.channel_scales) ||
        standalone_seed.correction_payload != full_seed.correction_payload)
      throw std::runtime_error(
          "full composed fixture did not preserve standalone ZRCAR seeds");
  }
  if (composed_full_value.correction_payload !=
          composed_value.correction_payload ||
      composed_full_value.corrections != composed_value.corrections)
    throw std::runtime_error(
        "full composed fixture did not preserve the value-only incumbent");
  require_float32_bitwise_equal(
      composed_full_value.reconstruction,
      composed_value.reconstruction,
      "full composed value stage changed the value-only reconstruction");

  run_zrcar_search(
      composed_original,
      composed_base,
      composed_settings,
      composed_full_seeds,
      composed_full);
  if (!composed_legacy.report.all_ok() || !composed_value.report.all_ok() ||
      !composed_zrcar.report.all_ok() || !composed_full.report.all_ok() ||
      !composed_full.zrcar_selected ||
      !composed_full.zrcar_from_fvro_seed || composed_full.fvro_selected ||
      composed_full.zrcar_legacy_seeds_selected != kZrcarSeedCount ||
      composed_full.zrcar_fvro_seeds_selected !=
          kFvroAdditionalZrcarSeedCount ||
      composed_full.zrcar_candidates_tested == 0 ||
      composed_full.encoded_size() >= composed_full_value.encoded_size() ||
      composed_full.encoded_size() >= composed_value.encoded_size() ||
      composed_full.encoded_size() >= composed_zrcar.encoded_size() ||
      composed_full.encoded_size() >= composed_legacy.encoded_size())
    throw std::runtime_error(
        "fixed composed fixture lost its strict FVRO-seed ZRCAR winner or dominance");
  auto composed_replayed = composed_base;
  apply_correction_payload(
      composed_replayed,
      composed_full.correction_payload,
      composed_full.corrections,
      composed_settings.channels,
      composed_settings.time,
      kVersion);
  require_float32_bitwise_equal(
      composed_replayed,
      composed_full.reconstruction,
      "fixed composed fixture failed independent bitwise replay");
  const auto composed_report = qpet_correlation::verify_constraints(
      composed_original,
      composed_replayed,
      composed_settings.channels,
      composed_settings.time,
      composed_settings.pointwise_tolerance,
      composed_settings.acf_tolerance,
      composed_settings.minimum_correlation,
      composed_settings.max_lag);
  if (!composed_report.all_ok())
    throw std::runtime_error(
        "fixed composed fixture failed independent constraint verification");
  auto composed_repeated = composed_legacy;
  auto composed_repeated_seeds = composed_frozen_seeds;
  run_fvro_value_search(
      composed_original,
      composed_base,
      composed_settings,
      composed_repeated_seeds,
      composed_repeated);
  run_zrcar_search(
      composed_original,
      composed_base,
      composed_settings,
      composed_repeated_seeds,
      composed_repeated);
  if (composed_repeated.corrections != composed_full.corrections ||
      composed_repeated.correction_payload != composed_full.correction_payload ||
      composed_repeated.zrcar_fvro_seeds_selected !=
          composed_full.zrcar_fvro_seeds_selected ||
      composed_repeated.zrcar_from_fvro_seed !=
          composed_full.zrcar_from_fvro_seed)
    throw std::runtime_error(
        "fixed composed FVRO/ZRCAR fixture was not deterministic");
  require_float32_bitwise_equal(
      composed_repeated.reconstruction,
      composed_full.reconstruction,
      "fixed composed repeated reconstruction changed float32 bits");

  auto replayed = base;
  apply_correction_payload(
      replayed,
      optimized.correction_payload,
      optimized.corrections,
      channels,
      time,
      kVersion);
  require_float32_bitwise_equal(
      replayed,
      optimized.reconstruction,
      "FVRO final independent replay changed float32 bits");
  const auto replay_report = qpet_correlation::verify_constraints(
      original,
      replayed,
      channels,
      time,
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag);
  if (!replay_report.all_ok())
    throw std::runtime_error("FVRO final independent replay failed a constraint");

  auto repeated = legacy;
  auto repeated_seeds = std::vector<LegacyRepairSeed>{legacy_seed};
  run_fvro_value_search(
      original, base, settings, repeated_seeds, repeated);
  if (repeated.corrections != optimized.corrections ||
      repeated.correction_payload != optimized.correction_payload ||
      repeated.fvro_global_candidates_generated !=
          optimized.fvro_global_candidates_generated ||
      repeated.fvro_global_candidates_serialized !=
          optimized.fvro_global_candidates_serialized ||
      repeated.fvro_global_candidates_infeasible !=
          optimized.fvro_global_candidates_infeasible ||
      repeated.fvro_channel_candidates_generated !=
          optimized.fvro_channel_candidates_generated ||
      repeated.fvro_channel_candidates_serialized !=
          optimized.fvro_channel_candidates_serialized ||
      repeated.fvro_channel_candidates_infeasible !=
          optimized.fvro_channel_candidates_infeasible ||
      repeated.fvro_selected_channels != optimized.fvro_selected_channels ||
      repeated.fvro_strict_channel_steps !=
          optimized.fvro_strict_channel_steps)
    throw std::runtime_error("FVRO repeated search was not deterministic");
  require_float32_bitwise_equal(
      repeated.reconstruction,
      optimized.reconstruction,
      "FVRO repeated reconstruction was not bitwise deterministic");

  // Exercise the complete stored-base portfolio and its window-level
  // diagnostic aggregation for both public FVRO modes. This invokes official
  // base codecs, but writes no artifact and never touches the canonical bin.
  auto production_settings = settings;
  production_settings.channels = 1;
  production_settings.time = 32;
  production_settings.max_lag = 4;
  production_settings.pointwise_tolerance = 0.25;
  production_settings.acf_tolerance = 0.05;
  production_settings.minimum_correlation = 0.99;
  auto production_source = std::vector<float>(production_settings.time);
  for (std::size_t index = 0; index < production_source.size(); ++index) {
    production_source[index] = static_cast<float>(
        0.6 * std::sin(0.21 * static_cast<double>(index)) +
        0.15 * std::cos(0.07 * static_cast<double>(index)) +
        0.003 * static_cast<double>(index));
  }
  const auto production_fvro_value = compress_window(
      production_source, production_settings);
  auto production_legacy_settings = production_settings;
  production_legacy_settings.repair_optimizer = RepairOptimizer::legacy;
  const auto production_legacy = compress_window(
      production_source, production_legacy_settings);
  if (!production_fvro_value.report.all_ok() ||
      production_fvro_value.encoded_size() > production_legacy.encoded_size() ||
      production_fvro_value.fvro_base_records_tested > kZrcarBaseRecordCount ||
      production_fvro_value.fvro_global_candidates_serialized >
          production_fvro_value.fvro_base_records_tested *
              kFvroGlobalSerializerBudget ||
      production_fvro_value.fvro_channel_candidates_serialized >
          production_fvro_value.fvro_base_records_tested *
              kFvroChannelSerializerBudget ||
      production_fvro_value.fvro_selected_channels >
          production_fvro_value.fvro_strict_channel_steps ||
      production_fvro_value.fvro_strict_channel_steps >
          production_fvro_value.fvro_channel_candidates_serialized ||
      ((production_fvro_value.fvro_selected_channels == 0) !=
       (production_fvro_value.fvro_strict_channel_steps == 0)) ||
      production_fvro_value.fvro_local_proposals_generated >
          production_fvro_value.fvro_base_records_tested *
              kFvroLocalProposalBudgetPerBase ||
      production_fvro_value.fvro_global_candidates_generated !=
          production_fvro_value.fvro_global_candidates_serialized +
              production_fvro_value.fvro_global_candidates_infeasible ||
      production_fvro_value.fvro_channel_candidates_generated !=
          production_fvro_value.fvro_channel_candidates_serialized +
              production_fvro_value.fvro_channel_candidates_infeasible ||
      production_fvro_value.zrcar_base_records_tested != 0 ||
      production_fvro_value.zrcar_legacy_seeds_selected != 0 ||
      production_fvro_value.zrcar_fvro_seeds_selected != 0 ||
      production_fvro_value.zrcar_candidates_tested != 0 ||
      production_fvro_value.zrcar_selected ||
      production_fvro_value.fvro_legacy_reference_bytes !=
          production_legacy.encoded_size() ||
      production_fvro_value.fvro_selected !=
          (production_fvro_value.encoded_size() <
           production_legacy.encoded_size()))
    throw std::runtime_error(
        "production FVRO-value portfolio violated mode, budget, or no-regression rules");
  if (production_fvro_value.encoded_size() ==
      production_legacy.encoded_size()) {
    if (production_fvro_value.payload != production_legacy.payload ||
        production_fvro_value.correction_payload !=
            production_legacy.correction_payload ||
        production_fvro_value.corrections != production_legacy.corrections)
      throw std::runtime_error(
          "production FVRO-value whole-window tie changed legacy bytes/count");
    require_float32_bitwise_equal(
        production_fvro_value.reconstruction,
        production_legacy.reconstruction,
        "production FVRO-value whole-window tie changed reconstruction");
  }

  auto production_zrcar_settings = production_settings;
  production_zrcar_settings.repair_optimizer = RepairOptimizer::zrcar;
  const auto production_zrcar = compress_window(
      production_source, production_zrcar_settings);
  if (!production_zrcar.report.all_ok() ||
      production_zrcar.encoded_size() > production_legacy.encoded_size() ||
      production_zrcar.zrcar_legacy_seeds_selected >
          production_zrcar.zrcar_base_records_tested * kZrcarSeedCount ||
      production_zrcar.zrcar_fvro_seeds_selected != 0 ||
      production_zrcar.fvro_base_records_tested != 0 ||
      production_zrcar.fvro_global_candidates_generated != 0)
    throw std::runtime_error(
        "production standalone ZRCAR violated its frozen-seed ablation");

  production_settings.repair_optimizer = RepairOptimizer::fvro;
  const auto production_fvro = compress_window(
      production_source, production_settings);
  if (!production_fvro.report.all_ok() ||
      production_fvro.encoded_size() > production_fvro_value.encoded_size() ||
      production_fvro.encoded_size() > production_zrcar.encoded_size() ||
      production_fvro.fvro_base_records_tested > kZrcarBaseRecordCount ||
      production_fvro.zrcar_base_records_tested > kZrcarBaseRecordCount ||
      production_fvro.zrcar_legacy_seeds_selected >
          production_fvro.zrcar_base_records_tested * kZrcarSeedCount ||
      production_fvro.zrcar_fvro_seeds_selected >
          production_fvro.zrcar_base_records_tested *
              kFvroAdditionalZrcarSeedCount ||
      production_fvro.zrcar_candidates_tested >
          kZrcarBaseRecordCount * kFullFvroZrcarSeedCount *
              kZrcarBudgetPerSeed ||
      production_fvro.fvro_selected_channels >
          production_fvro.fvro_strict_channel_steps ||
      production_fvro.fvro_strict_channel_steps >
          production_fvro.fvro_channel_candidates_serialized ||
      ((production_fvro.fvro_selected_channels == 0) !=
       (production_fvro.fvro_strict_channel_steps == 0)) ||
      production_fvro.fvro_legacy_reference_bytes !=
          production_legacy.encoded_size() ||
      production_fvro.zrcar_legacy_reference_bytes !=
          production_legacy.encoded_size())
    throw std::runtime_error(
        "production full-FVRO portfolio violated composition or no-regression rules");

  std::cout << "QPET_FVRO_SELFTEST legacy_bytes="
            << legacy.correction_payload.size()
            << " fine_bytes=" << fine.seed.correction_payload.size()
            << " fvro_bytes=" << optimized.correction_payload.size()
            << " global_generated="
            << optimized.fvro_global_candidates_generated
            << " global_infeasible="
            << optimized.fvro_global_candidates_infeasible
            << " global_serialized="
            << optimized.fvro_global_candidates_serialized
            << " local_generated="
            << optimized.fvro_local_proposals_generated
            << " local_infeasible="
            << optimized.fvro_local_proposals_infeasible
            << " channel_generated="
            << optimized.fvro_channel_candidates_generated
            << " channel_serialized="
            << optimized.fvro_channel_candidates_serialized
            << " selected_channels=" << optimized.fvro_selected_channels
            << " strict_channel_steps="
            << optimized.fvro_strict_channel_steps
            << " retained_seeds=" << optimized.fvro_seeds_retained
            << " composed_fixture=" << composed_fixture
            << " composed_legacy_bytes=" << composed_legacy.encoded_size()
            << " composed_value_bytes=" << composed_value.encoded_size()
            << " composed_zrcar_bytes=" << composed_zrcar.encoded_size()
            << " composed_full_bytes=" << composed_full.encoded_size()
            << " production_legacy_bytes=" << production_legacy.encoded_size()
            << " production_value_bytes="
            << production_fvro_value.encoded_size()
            << " production_zrcar_bytes=" << production_zrcar.encoded_size()
            << " production_full_bytes=" << production_fvro.encoded_size()
            << " production_base_records="
            << production_fvro.fvro_base_records_tested << '\n';
}

void ssbb_selftest_mode(int argc, char** argv)
{
  if (argc != 2)
    throw std::invalid_argument("usage: qpet_correlation_batch ssbb-selftest");
  if (parse_repair_optimizer("ssbb_value") != RepairOptimizer::ssbb_value ||
      parse_repair_optimizer("ssbb") != RepairOptimizer::ssbb ||
      std::string(repair_optimizer_name(RepairOptimizer::ssbb_value)) !=
          "ssbb_value" ||
      std::string(repair_optimizer_name(RepairOptimizer::ssbb)) != "ssbb")
    throw std::runtime_error("SSBB optimizer parser/name round trip failed");
  Settings settings;
  settings.batch = 1;
  settings.channels = 1;
  settings.time = 5;
  settings.pointwise_tolerance = 0.35;
  settings.acf_tolerance = 0.35;
  settings.minimum_correlation = 0.90;
  settings.max_lag = 2;
  settings.qoz_level = 1;
  settings.correction_mode = 2;
  settings.estimator_scales = {1.0};
  settings.group_size = 1;
  settings.repair_denominators = {7, 15, 31, 63, 127};
  settings.repair_optimizer = RepairOptimizer::ssbb_value;
  settings.correction_format_ceiling =
      kSubsetRankPackedCorrectionFormat;
  validate_settings(settings);
  LegacyRepairSeed tag9_composition_slot_fixture;
  tag9_composition_slot_fixture.correction_payload = {
      kSubsetRankPackedCorrectionFormat};
  const auto tag9_composition_slot =
      ssbb_correction_format(tag9_composition_slot_fixture);
  if (!tag9_composition_slot.has_value() ||
      *tag9_composition_slot != kSubsetRankPackedCorrectionFormat)
    throw std::runtime_error(
        "SSBB composition did not reserve a frontier slot for format 9");

  const auto original = std::vector<float>{
      0.10F, 0.82F, 0.24F, -0.43F, 0.31F};
  const auto base = std::vector<float>{
      -0.15F, 1.02F, -0.06F, -0.18F, 0.11F};
  const auto start = original;
  const auto scales = std::vector<float>{0.0625F};
  auto legacy_seed = serialize_fvro_reconstruction(
      original, base, start, scales, start, settings, 0);
  legacy_seed.fvro_derived = false;
  EncodedWindow incumbent;
  incumbent.payload.assign(11, 0x5aU);
  incumbent.corrections = legacy_seed.corrections;
  incumbent.correction_payload = legacy_seed.correction_payload;
  incumbent.reconstruction = legacy_seed.reconstruction;
  incumbent.report = legacy_seed.report;
  incumbent.correction_channel_scales = scales;
  incumbent.repair_candidates_tested = 1;

  const qpet_correlation::ssbb::Constraints constraints{
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag};
  const auto plan = qpet_correlation::ssbb::make_position_plan(
      original,
      base,
      start,
      scales.front(),
      settings.pointwise_tolerance);
  auto combinations = std::size_t{1};
  for (const auto& position : plan) {
    if (position.actions.empty() ||
        combinations >
            qpet_correlation::ssbb::kExhaustiveOracleCombinationLimit /
                position.actions.size())
      throw std::runtime_error("SSBB actual-byte oracle fixture exceeds its cap");
    combinations *= position.actions.size();
  }
  if (combinations > qpet_correlation::ssbb::kExhaustiveOracleCombinationLimit)
    throw std::runtime_error("SSBB actual-byte oracle fixture exceeds its cap");

  const auto normalized_oracle_seed = serialize_fvro_reconstruction(
      original, base, start, scales, start, settings, 1, true);
  auto oracle_bytes = std::min(
      incumbent.correction_payload.size(),
      normalized_oracle_seed.correction_payload.size());
  auto oracle_feasible = std::size_t{0};
  auto oracle_serialized = std::size_t{0};
  for (std::size_t combination = 0; combination < combinations; ++combination) {
    auto reconstruction = start;
    auto cursor = combination;
    for (const auto& position : plan) {
      const auto choice = cursor % position.actions.size();
      cursor /= position.actions.size();
      reconstruction[position.index] = position.actions[choice].value;
    }
    const auto report = qpet_correlation::verify_constraints(
        original,
        reconstruction,
        settings.channels,
        settings.time,
        settings.pointwise_tolerance,
        settings.acf_tolerance,
        settings.minimum_correlation,
        settings.max_lag);
    if (!report.all_ok())
      continue;
    ++oracle_feasible;
    auto candidate = serialize_fvro_reconstruction(
        original,
        base,
        reconstruction,
        scales,
        reconstruction,
        settings,
        combination + 2,
        true);
    ++oracle_serialized;
    oracle_bytes = std::min(
        oracle_bytes, candidate.correction_payload.size());
  }
  if (oracle_feasible == 0 || oracle_serialized != oracle_feasible)
    throw std::runtime_error("SSBB actual-byte oracle lost a feasible leaf");

  auto optimized = incumbent;
  auto seeds = std::vector<LegacyRepairSeed>{legacy_seed};
  run_ssbb_value_search(original, base, settings, seeds, optimized);
  const auto ceiling_paths = correction_format_ceiling_path_count(
      settings.correction_format_ceiling);
  if (!optimized.report.all_ok() ||
      optimized.correction_payload.size() > incumbent.correction_payload.size() ||
      optimized.correction_payload.size() < oracle_bytes ||
      optimized.ssbb_base_records_tested != 1 ||
      optimized.ssbb_channels_searched != ceiling_paths ||
      optimized.ssbb_positions_processed !=
          ceiling_paths * settings.time ||
      optimized.ssbb_actions_generated <
          optimized.ssbb_positions_processed ||
      optimized.ssbb_actions_generated >
          optimized.ssbb_positions_processed *
              qpet_correlation::ssbb::kMaximumActionsPerPosition ||
      optimized.ssbb_candidates_serialized == 0 ||
      optimized.ssbb_candidates_serialized >
          ceiling_paths * kSsbbMaximumSerializedPerChannel ||
      optimized.ssbb_outputs_returned !=
          optimized.ssbb_candidates_serialized ||
      optimized.ssbb_states_generated !=
          optimized.ssbb_states_deduplicated +
              optimized.ssbb_statistics_feasible_states +
              optimized.ssbb_statistics_infeasible_states ||
      optimized.ssbb_statistics_feasible_states +
              optimized.ssbb_statistics_infeasible_states !=
          optimized.ssbb_feasible_states_retained +
              optimized.ssbb_infeasible_states_retained +
              optimized.ssbb_states_pruned_by_beam ||
      optimized.ssbb_terminal_states !=
          optimized.ssbb_terminal_direct_feasible +
              optimized.ssbb_terminal_direct_infeasible)
    throw std::runtime_error(
        "SSBB production search violated actual-byte, constraint, or budget invariants");
  auto replayed = base;
  const auto replay_version =
      correction_payload_required_stream_version(
          optimized.correction_payload);
  apply_correction_payload(
      replayed,
      optimized.correction_payload,
      optimized.corrections,
      settings.channels,
      settings.time,
      replay_version);
  require_float32_bitwise_equal(
      replayed,
      optimized.reconstruction,
      "SSBB self-test actual serializer failed bitwise replay");
  const auto replay_report = qpet_correlation::verify_constraints(
      original,
      replayed,
      settings.channels,
      settings.time,
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag);
  if (!replay_report.all_ok())
    throw std::runtime_error("SSBB self-test replay failed a hard constraint");

  auto repeated = incumbent;
  auto repeated_seeds = std::vector<LegacyRepairSeed>{legacy_seed};
  run_ssbb_value_search(original, base, settings, repeated_seeds, repeated);
  if (repeated.correction_payload != optimized.correction_payload ||
      repeated.corrections != optimized.corrections ||
      repeated.ssbb_states_generated != optimized.ssbb_states_generated ||
      repeated.ssbb_candidates_serialized !=
          optimized.ssbb_candidates_serialized)
    throw std::runtime_error("SSBB production search is not deterministic");
  require_float32_bitwise_equal(
      repeated.reconstruction,
      optimized.reconstruction,
      "SSBB repeated reconstruction changed float32 bits");

  auto single_channel_ablation_bytes =
      std::array<std::size_t, kSsbbMaximumCeilingPaths>{};
  auto ceiling_ordinal = std::size_t{0};
  for (const auto ceiling :
       {kPackedQuantizedCorrectionFormat,
        kAdaptivePackedCorrectionFormat,
        kFactorizedPackedCorrectionFormat,
        kSubsetRankPackedCorrectionFormat}) {
    auto ablation_settings = settings;
    ablation_settings.correction_format_ceiling = ceiling;
    validate_settings(ablation_settings);
    auto ablated = incumbent;
    auto ablated_seeds = std::vector<LegacyRepairSeed>{legacy_seed};
    run_ssbb_value_search(
        original, base, ablation_settings, ablated_seeds, ablated);
    if (!ablated.report.all_ok() ||
        (!ablated.correction_payload.empty() &&
         ablated.correction_payload.front() > ceiling))
      throw std::runtime_error(
          "SSBB correction ceiling ablation violated nesting or constraints");
    auto ablated_replay = base;
    const auto ablated_version =
        correction_payload_required_stream_version(
            ablated.correction_payload);
    apply_correction_payload(
        ablated_replay,
        ablated.correction_payload,
        ablated.corrections,
        ablation_settings.channels,
        ablation_settings.time,
        ablated_version);
    require_float32_bitwise_equal(
        ablated_replay,
        ablated.reconstruction,
        "SSBB correction ceiling ablation failed bitwise replay");
    single_channel_ablation_bytes[ceiling_ordinal++] = ablated.encoded_size();
  }
  if (single_channel_ablation_bytes[1] > single_channel_ablation_bytes[0] ||
      single_channel_ablation_bytes[2] > single_channel_ablation_bytes[1] ||
      single_channel_ablation_bytes[3] > single_channel_ablation_bytes[2])
    throw std::runtime_error(
        "SSBB complete correction-ceiling candidates are not nested");

  // A two-channel search exercises trajectory/ranking changes that a
  // serializer-only or one-channel fixture cannot expose.  Each larger
  // ceiling reruns and immutably retains every complete lower-ceiling path.
  auto multi_settings = settings;
  multi_settings.channels = 2;
  auto multi_original = original;
  auto multi_base = base;
  for (std::size_t index = 0; index < settings.time; ++index) {
    multi_original.push_back(static_cast<float>(
        -0.3 + 0.8 * static_cast<double>(original[settings.time - 1 - index])));
    multi_base.push_back(static_cast<float>(
        -0.3 + 0.8 * static_cast<double>(base[settings.time - 1 - index])));
  }
  const auto multi_scales = std::vector<float>{0.0625F, 0.03125F};
  auto multi_legacy_seed = serialize_fvro_reconstruction(
      multi_original,
      multi_base,
      multi_original,
      multi_scales,
      multi_original,
      multi_settings,
      0);
  multi_legacy_seed.fvro_derived = false;
  EncodedWindow multi_incumbent;
  multi_incumbent.payload.assign(13, 0xa5U);
  multi_incumbent.corrections = multi_legacy_seed.corrections;
  multi_incumbent.correction_payload = multi_legacy_seed.correction_payload;
  multi_incumbent.reconstruction = multi_legacy_seed.reconstruction;
  multi_incumbent.report = multi_legacy_seed.report;
  multi_incumbent.correction_channel_scales = multi_scales;
  multi_incumbent.repair_candidates_tested = 1;
  auto multi_ablation_bytes =
      std::array<std::size_t, kSsbbMaximumCeilingPaths>{};
  ceiling_ordinal = 0;
  for (const auto ceiling :
       {kPackedQuantizedCorrectionFormat,
        kAdaptivePackedCorrectionFormat,
        kFactorizedPackedCorrectionFormat,
        kSubsetRankPackedCorrectionFormat}) {
    auto ablation_settings = multi_settings;
    ablation_settings.correction_format_ceiling = ceiling;
    validate_settings(ablation_settings);
    auto ablated = multi_incumbent;
    auto ablated_seeds =
        std::vector<LegacyRepairSeed>{multi_legacy_seed};
    run_ssbb_value_search(
        multi_original,
        multi_base,
        ablation_settings,
        ablated_seeds,
        ablated);
    if (!ablated.report.all_ok() ||
        ablated.ssbb_composition_starts !=
            correction_format_ceiling_path_count(ceiling) ||
        (!ablated.correction_payload.empty() &&
         ablated.correction_payload.front() > ceiling))
      throw std::runtime_error(
          "multi-channel SSBB ceiling path violated its invariants");
    auto decoded = multi_base;
    const auto version = correction_payload_required_stream_version(
        ablated.correction_payload);
    apply_correction_payload(
        decoded,
        ablated.correction_payload,
        ablated.corrections,
        multi_settings.channels,
        multi_settings.time,
        version);
    require_float32_bitwise_equal(
        decoded,
        ablated.reconstruction,
        "multi-channel SSBB ceiling winner failed bitwise replay");
    multi_ablation_bytes[ceiling_ordinal++] = ablated.encoded_size();
  }
  if (multi_ablation_bytes[1] > multi_ablation_bytes[0] ||
      multi_ablation_bytes[2] > multi_ablation_bytes[1] ||
      multi_ablation_bytes[3] > multi_ablation_bytes[2])
    throw std::runtime_error(
        "multi-channel SSBB final bytes violated ceiling nesting");

  // A stored base can start above the frozen/global incumbent and still win
  // after SSBB.  Its local saving is then larger than the realized global
  // saving.  Rebase only the selected provenance envelope; all search-work
  // diagnostics remain the aggregate work that was actually performed.  The
  // byte counts reproduce the Tesla medium/cf6 regression that originally
  // exposed this cross-base accounting error.
  EncodedWindow cross_base_winner;
  cross_base_winner.payload.assign(1346, 0x6bU);
  cross_base_winner.ssbb_selected = true;
  cross_base_winner.ssbb_reference_bytes = 1425;
  cross_base_winner.ssbb_search_reference_bytes = 1425;
  cross_base_winner.ssbb_composition_selected = true;
  cross_base_winner.ssbb_composition_selected_channels = 2;
  cross_base_winner.ssbb_composition_reference_bytes = 1425;
  cross_base_winner.ssbb_representation_candidates_serialized = 7;
  cross_base_winner.ssbb_representation_improved_starts = 3;
  cross_base_winner.ssbb_representation_gross_saved_bytes = 91;
  cross_base_winner.ssbb_candidates_serialized = 19;
  cross_base_winner.repair_candidates_tested = 31;
  realize_ssbb_winner_against_incumbent(
      cross_base_winner, 1389, false);
  if (cross_base_winner.ssbb_reference_bytes != 1389 ||
      cross_base_winner.ssbb_search_reference_bytes != 1389 ||
      cross_base_winner.ssbb_composition_reference_bytes != 1389 ||
      cross_base_winner.ssbb_reference_bytes -
              cross_base_winner.encoded_size() !=
          43 ||
      cross_base_winner.ssbb_representation_candidates_serialized != 7 ||
      cross_base_winner.ssbb_representation_improved_starts != 3 ||
      cross_base_winner.ssbb_representation_gross_saved_bytes != 91 ||
      cross_base_winner.ssbb_candidates_serialized != 19 ||
      cross_base_winner.repair_candidates_tested != 31)
    throw std::runtime_error(
        "SSBB cross-base winner did not rebase provenance without changing work diagnostics");

  // The alternate full-SSBB start is a distinct search trajectory.  Even
  // when its local reference is already below the primary start, a strict
  // alternate win is realized against the primary incumbent so the merged
  // result never attributes cross-start selection to representation work.
  EncodedWindow alternate_start_winner;
  alternate_start_winner.payload.assign(1300, 0x7cU);
  alternate_start_winner.ssbb_selected = true;
  alternate_start_winner.ssbb_reference_bytes = 1320;
  alternate_start_winner.ssbb_search_reference_bytes = 1310;
  alternate_start_winner.ssbb_representation_gross_saved_bytes = 17;
  alternate_start_winner.ssbb_candidates_serialized = 11;
  realize_ssbb_winner_against_incumbent(
      alternate_start_winner, 1350, true);
  if (alternate_start_winner.ssbb_reference_bytes != 1350 ||
      alternate_start_winner.ssbb_search_reference_bytes != 1350 ||
      alternate_start_winner.ssbb_reference_bytes -
              alternate_start_winner.ssbb_search_reference_bytes !=
          0 ||
      alternate_start_winner.ssbb_search_reference_bytes -
              alternate_start_winner.encoded_size() !=
          50 ||
      alternate_start_winner.ssbb_representation_gross_saved_bytes != 17 ||
      alternate_start_winner.ssbb_candidates_serialized != 11)
    throw std::runtime_error(
        "SSBB alternate-start winner did not preserve realized/work provenance separation");

  EncodedWindow local_provenance_winner;
  local_provenance_winner.payload.assign(1300, 0x4dU);
  local_provenance_winner.ssbb_selected = true;
  local_provenance_winner.ssbb_reference_bytes = 1340;
  local_provenance_winner.ssbb_search_reference_bytes = 1320;
  realize_ssbb_winner_against_incumbent(
      local_provenance_winner, 1350, false);
  if (local_provenance_winner.ssbb_reference_bytes != 1340 ||
      local_provenance_winner.ssbb_search_reference_bytes != 1320)
    throw std::runtime_error(
        "SSBB in-branch winner unnecessarily changed valid local provenance");

  EncodedWindow non_ssbb_winner;
  non_ssbb_winner.payload.assign(1340, 0x5eU);
  non_ssbb_winner.ssbb_reference_bytes = 1340;
  non_ssbb_winner.ssbb_search_reference_bytes = 1340;
  realize_ssbb_winner_against_incumbent(non_ssbb_winner, 1350, false);
  if (non_ssbb_winner.ssbb_selected ||
      non_ssbb_winner.ssbb_reference_bytes != 1340 ||
      non_ssbb_winner.ssbb_search_reference_bytes != 1340)
    throw std::runtime_error(
        "non-SSBB winner changed through SSBB provenance rebasing");

  auto disabled_settings = settings;
  disabled_settings.repair_optimizer = RepairOptimizer::fvro_value;
  disabled_settings.correction_format_ceiling =
      kPackedQuantizedCorrectionFormat;
  auto disabled = incumbent;
  auto disabled_seeds = std::vector<LegacyRepairSeed>{legacy_seed};
  run_ssbb_value_search(
      original, base, disabled_settings, disabled_seeds, disabled);
  if (disabled.correction_payload != incumbent.correction_payload ||
      disabled.ssbb_base_records_tested != 0 ||
      disabled.ssbb_candidates_serialized != 0 || disabled.ssbb_selected)
    throw std::runtime_error("FVRO v1 mode executed SSBB work");
  require_float32_bitwise_equal(
      disabled.reconstruction,
      incumbent.reconstruction,
      "FVRO v1 mode changed through the SSBB entry point");

  std::cout << "QPET_SSBB_SELFTEST incumbent_bytes="
            << incumbent.correction_payload.size()
            << " beam_bytes=" << optimized.correction_payload.size()
            << " oracle_bytes=" << oracle_bytes
            << " oracle_gap="
            << optimized.correction_payload.size() - oracle_bytes
            << " oracle_combinations=" << combinations
            << " oracle_feasible=" << oracle_feasible
            << " states_generated=" << optimized.ssbb_states_generated
            << " candidates_serialized="
            << optimized.ssbb_candidates_serialized
            << " multi_ceiling6_bytes=" << multi_ablation_bytes[0]
            << " multi_ceiling7_bytes=" << multi_ablation_bytes[1]
            << " multi_ceiling8_bytes=" << multi_ablation_bytes[2]
            << " multi_ceiling9_bytes=" << multi_ablation_bytes[3] << '\n';
}

void cjrf_selftest_mode(int argc, char** argv)
{
  if (argc != 3)
    throw std::invalid_argument(
        "usage: qpet_correlation_batch cjrf-selftest output-prefix");

  const auto expect_invalid = [](auto&& action, const char* message) {
    auto rejected = false;
    try {
      action();
    }
    catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error(message);
  };
  const auto expect_overflow = [](auto&& action, const char* message) {
    auto rejected = false;
    try {
      action();
    }
    catch (const std::overflow_error&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error(message);
  };

  if (kCjrfRelaxedBoundRatios !=
          std::array<double, 3>{1.25, 1.5, 2.0} ||
      kCjrfRelaxedBaseModes !=
          std::array<std::uint32_t, 4>{1U, 3U, 5U, 6U})
    throw std::runtime_error("CJRF fixed shell or official-arm schedule changed");
  if (parse_base_frontier("incumbent_only") !=
          BaseFrontier::incumbent_only ||
      parse_base_frontier("cjrf_v1") != BaseFrontier::cjrf_v1 ||
      parse_cjrf_ratio_mask("none") != 0U ||
      parse_cjrf_ratio_mask("1.25") != 1U ||
      parse_cjrf_ratio_mask("1.5") != 3U ||
      parse_cjrf_ratio_mask("2") != 7U ||
      parse_cjrf_ratio_mask("2.0") != 7U)
    throw std::runtime_error("CJRF parser changed its canonical prefix mapping");
  for (const auto* malformed :
       {"", "CJRF", "cjrf", "incumbent", "unknown"})
    expect_invalid(
        [&] { (void)parse_base_frontier(malformed); },
        "CJRF base-frontier parser accepted a noncanonical token");
  for (const auto* malformed :
       {"0", "1", "1.0", "1.2500001", "1.75", "3", "7", "nan", "inf"})
    expect_invalid(
        [&] { (void)parse_cjrf_ratio_mask(malformed); },
        "CJRF ratio parser accepted a noncanonical or nonnested token");

  Settings incumbent_settings;
  incumbent_settings.batch = 1;
  incumbent_settings.channels = 1;
  incumbent_settings.time = 5;
  incumbent_settings.pointwise_tolerance = 0.25;
  incumbent_settings.acf_tolerance = 0.25;
  incumbent_settings.minimum_correlation = 0.90;
  incumbent_settings.max_lag = 2;
  incumbent_settings.qoz_level = 1;
  incumbent_settings.correction_mode = 2;
  incumbent_settings.estimator_scales = {1.0};
  incumbent_settings.group_size = 1;
  incumbent_settings.repair_denominators = {7, 15, 31, 63, 127};
  incumbent_settings.repair_optimizer = RepairOptimizer::ssbb;
  incumbent_settings.correction_format_ceiling =
      kSubsetRankPackedCorrectionFormat;
  incumbent_settings.base_frontier = BaseFrontier::incumbent_only;
  incumbent_settings.cjrf_ratio_mask = 0U;
  validate_settings(incumbent_settings);

  auto cjrf_settings = incumbent_settings;
  cjrf_settings.base_frontier = BaseFrontier::cjrf_v1;
  for (const auto mask : {std::uint8_t{1}, std::uint8_t{3}, std::uint8_t{7}}) {
    cjrf_settings.cjrf_ratio_mask = mask;
    validate_settings(cjrf_settings);
  }
  auto invalid_settings = incumbent_settings;
  invalid_settings.cjrf_ratio_mask = 1U;
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "incumbent-only settings accepted a CJRF ratio mask");
  invalid_settings = cjrf_settings;
  for (const auto mask :
       {std::uint8_t{0}, std::uint8_t{2}, std::uint8_t{4},
        std::uint8_t{5}, std::uint8_t{6}, std::uint8_t{8},
        std::numeric_limits<std::uint8_t>::max()}) {
    invalid_settings.cjrf_ratio_mask = mask;
    expect_invalid(
        [&] { validate_settings(invalid_settings); },
        "CJRF settings accepted a nonempty nonprefix ratio mask");
  }
  invalid_settings = cjrf_settings;
  invalid_settings.repair_optimizer = RepairOptimizer::fvro;
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CJRF settings accepted a non-SSBB repair optimizer");
  invalid_settings = cjrf_settings;
  invalid_settings.correction_mode = 1U;
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CJRF settings accepted a non-greedy correction mode");
  invalid_settings = cjrf_settings;
  invalid_settings.correction_format_ceiling =
      kFactorizedPackedCorrectionFormat;
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CJRF settings accepted a correction ceiling below format 9");
  invalid_settings = cjrf_settings;
  invalid_settings.enable_vehicle_semantic = true;
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CJRF settings accepted a domain-specific candidate profile");
  invalid_settings = cjrf_settings;
  invalid_settings.batch = 2;
  invalid_settings.group_size = 2;
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CJRF settings accepted a grouped-window search");
  invalid_settings = incumbent_settings;
  invalid_settings.base_frontier = static_cast<BaseFrontier>(99U);
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "QPET settings accepted an unknown base-frontier enum");
  invalid_settings = cjrf_settings;
  invalid_settings.pointwise_tolerance =
      std::numeric_limits<double>::denorm_min();
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CJRF settings accepted a collapsed subnormal relaxed bound");
  invalid_settings = cjrf_settings;
  invalid_settings.pointwise_tolerance =
      std::numeric_limits<double>::max();
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CJRF settings accepted a nonfinite relaxed bound");
  invalid_settings = cjrf_settings;
  invalid_settings.pointwise_tolerance = 2.0e38;
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CJRF settings accepted a relaxed bound outside the float backend range");

  auto make_window = [](std::size_t payload_bytes,
                        std::size_t correction_bytes,
                        double bound,
                        unsigned char marker) {
    EncodedWindow window;
    window.base_mode = 1U;
    window.selected_global_bound = bound;
    window.selected_estimator_scale = 0.0;
    window.corrections = correction_bytes == 0 ? 0U : 1U;
    window.payload.assign(payload_bytes, marker);
    window.correction_payload.assign(correction_bytes, marker);
    if (!window.correction_payload.empty())
      window.correction_payload.front() = 0U;
    window.reconstruction = {
        static_cast<float>(marker), -static_cast<float>(marker)};
    return window;
  };

  auto incumbent = make_window(11U, 7U, 0.25, 0x11U);
  const auto protected_incumbent = incumbent;
  if (incumbent.encoded_size() != 18U ||
      compact_record_bytes(incumbent) != 42U ||
      compact_base_record_lower_bound_bytes(incumbent) != 35U)
    throw std::runtime_error("CJRF compact-record metadata accounting changed");

  auto equal_lower_bound = make_window(18U, 0U, 0.5, 0x22U);
  auto larger_lower_bound = make_window(19U, 0U, 0.5, 0x23U);
  auto smaller_lower_bound = make_window(17U, 0U, 0.5, 0x24U);
  const auto protected_bytes = compact_record_bytes(incumbent);
  if (compact_base_record_lower_bound_bytes(equal_lower_bound) !=
          protected_bytes ||
      compact_base_record_lower_bound_bytes(larger_lower_bound) <=
          protected_bytes ||
      compact_base_record_lower_bound_bytes(smaller_lower_bound) >=
          protected_bytes)
    throw std::runtime_error("CJRF safe-prune strict boundary changed");
  expect_overflow(
      [] {
        (void)checked_add_size(
            std::numeric_limits<std::size_t>::max(),
            kCompactRecordHeaderBytes,
            "CJRF test overflow");
      },
      "CJRF compact-record overflow was not rejected");

  auto worse_total = make_window(10U, 9U, 0.5, 0x31U);
  if (retain_strict_zrcar_whole_window(std::move(worse_total), incumbent) ||
      !same_persisted_record_and_reconstruction(
          incumbent, protected_incumbent))
    throw std::runtime_error(
        "CJRF total-byte selection preferred a smaller base with larger total bytes");
  auto byte_tie = make_window(12U, 6U, 0.5, 0x32U);
  if (retain_strict_zrcar_whole_window(std::move(byte_tie), incumbent) ||
      !same_persisted_record_and_reconstruction(
          incumbent, protected_incumbent))
    throw std::runtime_error(
        "CJRF byte tie replaced the protected incumbent");
  auto one_byte_win = make_window(12U, 5U, 0.5, 0x33U);
  if (!retain_strict_zrcar_whole_window(one_byte_win, incumbent) ||
      compact_record_bytes(incumbent) + 1U != protected_bytes ||
      !same_persisted_record_and_reconstruction(incumbent, one_byte_win))
    throw std::runtime_error("CJRF strict one-byte win was not selected");

  auto nested = protected_incumbent;
  const auto shell0_bytes = compact_record_bytes(nested);
  auto shell125 = make_window(10U, 7U, 0.3125, 0x41U);
  if (!retain_strict_zrcar_whole_window(shell125, nested))
    throw std::runtime_error("CJRF 1.25 shell lost a strict synthetic win");
  const auto shell125_winner = nested;
  const auto shell125_bytes = compact_record_bytes(nested);
  auto shell150_tie = make_window(11U, 6U, 0.375, 0x42U);
  if (retain_strict_zrcar_whole_window(shell150_tie, nested) ||
      !same_persisted_record_and_reconstruction(nested, shell125_winner))
    throw std::runtime_error("CJRF 1.5 shell did not retain a lower-shell tie");
  const auto shell150_bytes = compact_record_bytes(nested);
  auto shell200 = make_window(10U, 6U, 0.5, 0x43U);
  if (!retain_strict_zrcar_whole_window(shell200, nested))
    throw std::runtime_error("CJRF 2.0 shell lost a strict synthetic win");
  const auto shell200_bytes = compact_record_bytes(nested);
  if (shell125_bytes > shell0_bytes || shell150_bytes > shell125_bytes ||
      shell200_bytes > shell150_bytes ||
      shell150_bytes != shell125_bytes)
    throw std::runtime_error("CJRF nested shell bytes are not monotone");

  const auto incumbent_path = std::string(argv[2]) + "-incumbent.qpc";
  const auto tied_path = std::string(argv[2]) + "-tie.qpc";
  const auto cjrf_policy_path = std::string(argv[2]) + "-cjrf-policy.qpc";
  const auto persisted_source = std::vector<float>{
      0.125F, -0.75F, 1.5F, 0.375F, -1.0F};
  auto persisted_incumbent = compress_candidate(
      persisted_source,
      incumbent_settings,
      incumbent_settings.pointwise_tolerance,
      0.0,
      4U);
  if (!persisted_incumbent.report.all_ok() ||
      !persisted_incumbent.correction_payload.empty())
    throw std::runtime_error(
        "CJRF persisted incumbent fixture is not an exact valid base");
  write_stream(
      incumbent_path, incumbent_settings, {persisted_incumbent});
  write_stream(tied_path, incumbent_settings, {persisted_incumbent});
  cjrf_settings.cjrf_ratio_mask = 1U;
  write_stream(
      cjrf_policy_path, cjrf_settings, {persisted_incumbent});
  const auto incumbent_bytes = read_bytes(incumbent_path);
  const auto tied_bytes = read_bytes(tied_path);
  const auto cjrf_policy_bytes = read_bytes(cjrf_policy_path);
  const auto [incumbent_decoded_settings, incumbent_decoded] =
      decode_stream(incumbent_path);
  const auto [cjrf_decoded_settings, cjrf_decoded] =
      decode_stream(cjrf_policy_path);
  constexpr auto compact_stream_header_bytes =
      kMagic.size() + sizeof(std::uint32_t) +
      8U * sizeof(std::uint32_t) + 5U * sizeof(double);
  static_assert(compact_stream_header_bytes == 80U);
  if (incumbent_bytes.size() !=
          compact_stream_header_bytes +
              compact_record_bytes(persisted_incumbent) ||
      incumbent_bytes != tied_bytes ||
      incumbent_bytes != cjrf_policy_bytes ||
      incumbent_decoded_settings.base_frontier !=
          BaseFrontier::incumbent_only ||
      cjrf_decoded_settings.base_frontier !=
          BaseFrontier::incumbent_only ||
      !same_float32_tensor_bits(incumbent_decoded, persisted_source) ||
      !same_float32_tensor_bits(cjrf_decoded, persisted_source))
    throw std::runtime_error(
        "CJRF incumbent-only or encoder-policy stream identity changed");

  std::cout << "QPET_CJRF_SELFTEST record_header_bytes="
            << kCompactRecordHeaderBytes
            << " incumbent_record_bytes=" << protected_bytes
            << " strict_win_record_bytes=" << compact_record_bytes(incumbent)
            << " shell125_bytes=" << shell125_bytes
            << " shell150_bytes=" << shell150_bytes
            << " shell200_bytes=" << shell200_bytes
            << " persisted_policy_identity=1\n";
}

void cfsr_selftest_mode(int argc, char** argv)
{
  if (argc != 3)
    throw std::invalid_argument(
        "usage: qpet_correlation_batch cfsr-selftest output-prefix");

  const auto expect_invalid = [](auto&& action, const char* message) {
    auto rejected = false;
    try {
      action();
    }
    catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error(message);
  };
  const auto expect_overflow = [](auto&& action, const char* message) {
    auto rejected = false;
    try {
      action();
    }
    catch (const std::overflow_error&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error(message);
  };

  if (kCfsrSurrogateBoundRatios !=
          std::array<double, 3>{0.25, 0.5, 1.0} ||
      kCfsrSurrogateBaseModes !=
          std::array<std::uint32_t, 4>{1U, 3U, 5U, 6U})
    throw std::runtime_error(
        "CFSR fixed surrogate shell or official-arm schedule changed");
  if (parse_surrogate_frontier("none") != SurrogateFrontier::none ||
      parse_surrogate_frontier("cfsr_v1") !=
          SurrogateFrontier::cfsr_v1 ||
      parse_cfsr_ratio_mask("none") != 0U ||
      parse_cfsr_ratio_mask("0.25") != 1U ||
      parse_cfsr_ratio_mask("0.5") != 3U ||
      parse_cfsr_ratio_mask("1") != 7U ||
      parse_cfsr_ratio_mask("1.0") != 7U)
    throw std::runtime_error("CFSR parser changed its canonical prefix mapping");
  for (const auto* malformed :
       {"", "CFSR", "cfsr", "enabled", "incumbent_only", "unknown"})
    expect_invalid(
        [&] { (void)parse_surrogate_frontier(malformed); },
        "CFSR frontier parser accepted a noncanonical token");
  for (const auto* malformed :
       {"0", "0.0", "0.125", "0.2500001", "0.75", "2", "3", "7",
        "nan", "inf"})
    expect_invalid(
        [&] { (void)parse_cfsr_ratio_mask(malformed); },
        "CFSR ratio parser accepted a noncanonical or nonnested token");

  Settings settings;
  settings.batch = 1;
  settings.channels = 2;
  settings.time = 32;
  settings.pointwise_tolerance = 0.25;
  settings.acf_tolerance = 0.25;
  settings.minimum_correlation = 0.90;
  settings.max_lag = 4;
  settings.qoz_level = 1;
  settings.correction_mode = 2;
  settings.estimator_scales = {1.0};
  settings.group_size = 1;
  settings.repair_denominators = {7, 15, 31, 63, 127};
  settings.repair_optimizer = RepairOptimizer::ssbb;
  settings.correction_format_ceiling =
      kSubsetRankPackedCorrectionFormat;
  settings.base_frontier = BaseFrontier::incumbent_only;
  settings.cjrf_ratio_mask = 0U;
  settings.surrogate_frontier = SurrogateFrontier::none;
  settings.cfsr_ratio_mask = 0U;
  validate_settings(settings);

  auto invalid_single_channel_online = settings;
  invalid_single_channel_online.channels = 1U;
  invalid_single_channel_online.correction_mode = 0U;
  invalid_single_channel_online.repair_optimizer = RepairOptimizer::legacy;
  expect_invalid(
      [&] { validate_settings(invalid_single_channel_online); },
      "online QPET accepted the unsupported single-channel 2-D HPEZ shape");

  auto cfsr_settings = settings;
  cfsr_settings.surrogate_frontier = SurrogateFrontier::cfsr_v1;
  for (const auto mask : {std::uint8_t{1}, std::uint8_t{3}, std::uint8_t{7}}) {
    cfsr_settings.cfsr_ratio_mask = mask;
    validate_settings(cfsr_settings);
  }
  auto invalid_settings = settings;
  invalid_settings.cfsr_ratio_mask = 1U;
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "disabled surrogate frontier accepted a CFSR ratio mask");
  invalid_settings = cfsr_settings;
  for (const auto mask :
       {std::uint8_t{0}, std::uint8_t{2}, std::uint8_t{4},
        std::uint8_t{5}, std::uint8_t{6}, std::uint8_t{8},
        std::numeric_limits<std::uint8_t>::max()}) {
    invalid_settings.cfsr_ratio_mask = mask;
    expect_invalid(
        [&] { validate_settings(invalid_settings); },
        "CFSR settings accepted a nonempty nonprefix ratio mask");
  }
  invalid_settings = cfsr_settings;
  invalid_settings.base_frontier = BaseFrontier::cjrf_v1;
  invalid_settings.cjrf_ratio_mask = 1U;
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CFSR v1 stacked with active CJRF");
  invalid_settings = cfsr_settings;
  invalid_settings.repair_optimizer = RepairOptimizer::fvro;
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CFSR settings accepted a non-SSBB repair optimizer");
  invalid_settings = cfsr_settings;
  invalid_settings.correction_mode = 1U;
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CFSR settings accepted a non-greedy correction mode");
  invalid_settings = cfsr_settings;
  invalid_settings.correction_format_ceiling =
      kFactorizedPackedCorrectionFormat;
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CFSR settings accepted a correction ceiling below format 9");
  invalid_settings = cfsr_settings;
  invalid_settings.enable_vehicle_semantic = true;
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CFSR settings accepted a domain-specific profile");
  invalid_settings = cfsr_settings;
  invalid_settings.batch = 2;
  invalid_settings.group_size = 2;
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CFSR settings accepted grouped windows");
  invalid_settings = settings;
  invalid_settings.surrogate_frontier =
      static_cast<SurrogateFrontier>(99U);
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "QPET settings accepted an unknown surrogate-frontier enum");
  invalid_settings = cfsr_settings;
  invalid_settings.pointwise_tolerance =
      std::numeric_limits<double>::denorm_min();
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CFSR settings accepted a float-collapsed surrogate bound");
  invalid_settings = cfsr_settings;
  invalid_settings.pointwise_tolerance =
      std::numeric_limits<double>::max();
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CFSR settings accepted a nonfinite surrogate bound");
  invalid_settings = cfsr_settings;
  invalid_settings.pointwise_tolerance = 4.0e38;
  expect_invalid(
      [&] { validate_settings(invalid_settings); },
      "CFSR settings accepted a bound outside the float backend range");

  auto make_window = [](std::size_t payload_bytes,
                        std::size_t correction_bytes,
                        double bound,
                        unsigned char marker) {
    EncodedWindow window;
    window.base_mode = 1U;
    window.selected_global_bound = bound;
    window.selected_estimator_scale = 0.0;
    window.corrections = correction_bytes == 0 ? 0U : 1U;
    window.payload.assign(payload_bytes, marker);
    window.correction_payload.assign(correction_bytes, marker);
    if (!window.correction_payload.empty())
      window.correction_payload.front() = 0U;
    window.reconstruction = {
        static_cast<float>(marker), -static_cast<float>(marker)};
    return window;
  };
  auto incumbent = make_window(11U, 7U, 0.25, 0x51U);
  const auto protected_incumbent = incumbent;
  const auto protected_bytes = compact_record_bytes(incumbent);
  auto equal_lower_bound = make_window(18U, 0U, 0.5, 0x52U);
  auto larger_lower_bound = make_window(19U, 0U, 0.5, 0x53U);
  auto smaller_lower_bound = make_window(17U, 0U, 0.5, 0x54U);
  if (protected_bytes != 42U ||
      compact_base_record_lower_bound_bytes(equal_lower_bound) !=
          protected_bytes ||
      compact_base_record_lower_bound_bytes(larger_lower_bound) <=
          protected_bytes ||
      compact_base_record_lower_bound_bytes(smaller_lower_bound) >=
          protected_bytes)
    throw std::runtime_error("CFSR safe-prune strict boundary changed");
  auto byte_tie = make_window(12U, 6U, 0.5, 0x55U);
  if (retain_strict_zrcar_whole_window(std::move(byte_tie), incumbent) ||
      !same_persisted_record_and_reconstruction(
          incumbent, protected_incumbent))
    throw std::runtime_error("CFSR byte tie replaced the protected incumbent");
  auto one_byte_win = make_window(12U, 5U, 0.5, 0x56U);
  if (!retain_strict_zrcar_whole_window(one_byte_win, incumbent) ||
      compact_record_bytes(incumbent) + 1U != protected_bytes ||
      !same_persisted_record_and_reconstruction(incumbent, one_byte_win))
    throw std::runtime_error("CFSR strict one-byte win was not selected");
  auto exact_identity = make_window(8U, 3U, 0.125, 0x57U);
  auto same_base_different_repair = exact_identity;
  same_base_different_repair.corrections = 2U;
  same_base_different_repair.correction_payload = {0U, 0x58U};
  auto adjacent_bound = exact_identity;
  adjacent_bound.selected_global_bound = std::nextafter(
      exact_identity.selected_global_bound,
      std::numeric_limits<double>::infinity());
  auto adjacent_payload = exact_identity;
  adjacent_payload.payload.back() ^= 1U;
  if (!same_stored_base_identity(
          exact_identity, same_base_different_repair) ||
      same_stored_base_identity(exact_identity, adjacent_bound) ||
      same_stored_base_identity(exact_identity, adjacent_payload))
    throw std::runtime_error(
        "CFSR stored-base identity is not exact in mode/bound/payload bytes");
  expect_overflow(
      [] {
        (void)checked_add_size(
            std::numeric_limits<std::size_t>::max(),
            kCompactRecordHeaderBytes,
            "CFSR test overflow");
      },
      "CFSR compact-record overflow was not rejected");

  auto original = std::vector<float>(settings.channels * settings.time);
  auto surrogate = std::vector<float>(original.size());
  for (std::size_t channel = 0; channel < settings.channels; ++channel) {
    for (std::size_t index = 0; index < settings.time; ++index) {
      const auto offset = channel * settings.time + index;
      original[offset] = static_cast<float>(
          0.4 * static_cast<double>(channel) +
          std::sin(0.19 * static_cast<double>(index) + 0.13 * channel));
      surrogate[offset] = static_cast<float>(
          original[offset] + (index % 2 == 0 ? 0.01 : -0.01));
    }
  }
  const auto surrogate_report = qpet_correlation::verify_constraints(
      original,
      surrogate,
      settings.channels,
      settings.time,
      settings.pointwise_tolerance,
      settings.acf_tolerance,
      settings.minimum_correlation,
      settings.max_lag);
  if (!surrogate_report.all_ok() ||
      same_float32_tensor_bits(original, surrogate))
    throw std::runtime_error(
        "CFSR self-test surrogate is not a distinct feasible tensor");

  // Force a tag-9 incumbent so the canonical replay path cannot accidentally
  // use the QPC-v12 default correction decoder.
  auto tag9_base = std::vector<float>(surrogate.size(), 0.0F);
  auto tag9_scales = std::vector<float>(settings.channels, 0.0F);
  const auto tag9_fields = make_quantized_correction_fields(
      tag9_base,
      surrogate,
      tag9_scales,
      settings.channels,
      settings.time);
  auto tag9_payload = encode_subset_rank_packed_correction_format9(
      tag9_base,
      surrogate,
      tag9_scales,
      tag9_fields,
      settings.channels,
      settings.time);
  if (tag9_payload.empty() ||
      tag9_payload.front() != kSubsetRankPackedCorrectionFormat ||
      tag9_fields.codes.empty() ||
      tag9_fields.codes.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("CFSR tag-9 canonical replay fixture is invalid");
  EncodedWindow tag9_incumbent;
  tag9_incumbent.base_mode = 4U;
  tag9_incumbent.selected_global_bound = settings.pointwise_tolerance;
  tag9_incumbent.selected_estimator_scale = 0.0;
  tag9_incumbent.corrections =
      static_cast<std::uint32_t>(tag9_fields.codes.size());
  tag9_incumbent.payload = compress_lossless_zstd_payload(tag9_base);
  tag9_incumbent.correction_payload = std::move(tag9_payload);
  tag9_incumbent.reconstruction = surrogate;
  cfsr_settings.cfsr_ratio_mask = 1U;
  const auto canonical = canonical_cfsr_surrogate_replay(
      original, cfsr_settings, tag9_incumbent);
  require_float32_bitwise_equal(
      canonical,
      surrogate,
      "CFSR tag-9 canonical replay changed the feasible surrogate");

  auto cached_mismatch = tag9_incumbent;
  cached_mismatch.reconstruction.front() = std::nextafter(
      cached_mismatch.reconstruction.front(),
      std::numeric_limits<float>::infinity());
  auto cached_mismatch_rejected = false;
  try {
    (void)canonical_cfsr_surrogate_replay(
        original, cfsr_settings, cached_mismatch);
  }
  catch (const std::runtime_error&) {
    cached_mismatch_rejected = true;
  }
  if (!cached_mismatch_rejected)
    throw std::runtime_error(
        "CFSR trusted a malicious cached incumbent reconstruction");

  auto nonfinite_values = surrogate;
  nonfinite_values.front() = std::numeric_limits<float>::quiet_NaN();
  auto nonfinite_incumbent = EncodedWindow();
  nonfinite_incumbent.base_mode = 4U;
  nonfinite_incumbent.selected_global_bound = settings.pointwise_tolerance;
  nonfinite_incumbent.selected_estimator_scale = 0.0;
  nonfinite_incumbent.payload =
      compress_lossless_zstd_payload(nonfinite_values);
  nonfinite_incumbent.reconstruction = nonfinite_values;
  auto nonfinite_rejected = false;
  try {
    (void)canonical_cfsr_surrogate_replay(
        original, cfsr_settings, nonfinite_incumbent);
  }
  catch (const std::runtime_error&) {
    nonfinite_rejected = true;
  }
  if (!nonfinite_rejected)
    throw std::runtime_error(
        "CFSR accepted a non-finite canonical surrogate");

  auto infeasible_values = surrogate;
  infeasible_values.front() += 4.0F *
      static_cast<float>(settings.pointwise_tolerance);
  auto malicious_report_incumbent = EncodedWindow();
  malicious_report_incumbent.base_mode = 4U;
  malicious_report_incumbent.selected_global_bound =
      settings.pointwise_tolerance;
  malicious_report_incumbent.selected_estimator_scale = 0.0;
  malicious_report_incumbent.payload =
      compress_lossless_zstd_payload(infeasible_values);
  malicious_report_incumbent.reconstruction = infeasible_values;
  malicious_report_incumbent.report = {};
  auto malicious_report_rejected = false;
  try {
    (void)canonical_cfsr_surrogate_replay(
        original, cfsr_settings, malicious_report_incumbent);
  }
  catch (const std::runtime_error&) {
    malicious_report_rejected = true;
  }
  if (!malicious_report_rejected)
    throw std::runtime_error(
        "CFSR trusted a malicious cached feasible report");

  // Exercise every official CFSR arm with codec source Y0 while repair and
  // direct constraints remain relative to the distinct original X.
  auto official_attempts = std::size_t{0};
  auto official_successes = std::size_t{0};
  auto official_typed_failures = std::size_t{0};
  auto successful_mode = std::array<bool, 7>{};
  for (const auto base_mode :
       std::array<std::uint32_t, 5>{4U, 1U, 3U, 5U, 6U}) {
    ++official_attempts;
    const auto bound = base_mode == 4U
        ? settings.pointwise_tolerance
        : settings.pointwise_tolerance * kCfsrSurrogateBoundRatios.front();
    auto decoded = DecodedBaseCandidate();
    try {
      decoded = encode_decode_base_candidate(
          surrogate,
          cfsr_settings,
          bound,
          0.0,
          base_mode,
          base_mode == 1U);
    }
    catch (const BackendCandidateFailure&) {
      // A backend may reject one arm at its existing byte-stream safety gate.
      // This is a tested, typed candidate outcome, not permission to weaken
      // that gate or substitute an in-house codec.  The real frontier below
      // must account for the same outcome in its failures partition.
      ++official_typed_failures;
      continue;
    }
    ++official_successes;
    successful_mode[base_mode] = true;
    if (decoded.encoded.payload.empty() ||
        decoded.encoded.reconstruction.size() != surrogate.size())
      throw std::runtime_error("CFSR official arm returned an invalid base");
    if (base_mode == 4U) {
      require_float32_bitwise_equal(
          decoded.encoded.reconstruction,
          surrogate,
          "CFSR lossless arm encoded X instead of Y0");
      if (same_float32_tensor_bits(decoded.encoded.reconstruction, original))
        throw std::runtime_error("CFSR X/Y0 confusion fixture collapsed");
    }
    const auto repaired = finalize_decoded_base_candidate(
        original, cfsr_settings, std::move(decoded));
    if (!repaired.report.all_ok())
      throw std::runtime_error(
          "CFSR official arm failed original-data final constraints");
    if (base_mode == 4U &&
        (repaired.corrections != 0U ||
         !repaired.correction_payload.empty() ||
         !same_float32_tensor_bits(repaired.reconstruction, surrogate)))
      throw std::runtime_error(
          "CFSR lossless arm was not a zero-correction Y0 anchor");
  }
  if (official_attempts != 5U ||
      official_successes + official_typed_failures != official_attempts ||
      !successful_mode[4U] || !successful_mode[3U] ||
      !successful_mode[5U] || !successful_mode[6U])
    throw std::runtime_error(
        "CFSR did not exercise every official arm or lost a required "
        "non-QPET backend witness");

  // The oscillating fixture above intentionally reaches HPEZ's fail-closed
  // symbol-count gate on this pinned upstream revision.  A separate defined
  // two-dimensional source must nevertheless complete the real mode-1 encode,
  // bounded official payload decode, X-relative repair, and certification.
  auto mode1_settings = cfsr_settings;
  mode1_settings.time = 128U;
  mode1_settings.max_lag = 16U;
  auto mode1_original = std::vector<float>(
      mode1_settings.channels * mode1_settings.time);
  auto mode1_state = std::uint32_t{0x6d2b79f5U};
  for (std::size_t channel = 0; channel < mode1_settings.channels; ++channel) {
    for (std::size_t index = 0; index < mode1_settings.time; ++index) {
      mode1_state = mode1_state * 1664525U + 1013904223U;
      const auto unit = static_cast<double>(mode1_state >> 8U) /
          static_cast<double>(1U << 24U);
      mode1_original[channel * mode1_settings.time + index] =
          static_cast<float>(
              2.0 * unit - 1.0 + 0.125 * static_cast<double>(channel));
    }
  }
  auto mode1_decoded = encode_decode_base_candidate(
      mode1_original,
      mode1_settings,
      settings.pointwise_tolerance * kCfsrSurrogateBoundRatios.front(),
      0.0,
      1U,
      true);
  const auto mode1_repeated = encode_decode_base_candidate(
      mode1_original,
      mode1_settings,
      settings.pointwise_tolerance * kCfsrSurrogateBoundRatios.front(),
      0.0,
      1U,
      true);
  if (mode1_decoded.encoded.payload != mode1_repeated.encoded.payload ||
      !same_float32_tensor_bits(
          mode1_decoded.encoded.reconstruction,
          mode1_repeated.encoded.reconstruction))
    throw std::runtime_error(
        "CFSR repeated mode-1 arm changed official payload or reconstruction");
  const auto mode1_repaired = finalize_decoded_base_candidate(
      mode1_original, mode1_settings, std::move(mode1_decoded));
  if (mode1_repaired.payload.empty() || !mode1_repaired.report.all_ok())
    throw std::runtime_error(
        "CFSR did not complete a successful real official mode-1 path");

  // The lossless anchor and persisted-record equality are bitwise, including
  // IEEE-754 signed zero; numerical equality is intentionally insufficient.
  auto signed_zero_settings = cfsr_settings;
  signed_zero_settings.channels = 1U;
  signed_zero_settings.time = 4U;
  signed_zero_settings.max_lag = 2U;
  const auto signed_zero_source = std::vector<float>{
      0.0F, -0.0F, 1.0F, -1.0F};
  const auto signed_zero_anchor = encode_decode_base_candidate(
      signed_zero_source,
      signed_zero_settings,
      signed_zero_settings.pointwise_tolerance,
      0.0,
      4U);
  require_float32_bitwise_equal(
      signed_zero_anchor.encoded.reconstruction,
      signed_zero_source,
      "CFSR lossless anchor collapsed signed-zero bits");

  auto nested_records =
      std::array<std::vector<RetainedLegacyRecord>, 3>();
  auto nested_work = std::array<CfsrSurrogateWork, 3>();
  const auto masks = std::array<std::uint8_t, 3>{1U, 3U, 7U};
  const auto nested_original = mode1_original;
  const auto nested_surrogate = nested_original;
  for (std::size_t level = 0; level < masks.size(); ++level) {
    auto level_settings = mode1_settings;
    level_settings.cfsr_ratio_mask = masks[level];
    auto ordinal = std::size_t{0};
    append_cfsr_surrogate_base_records(
        nested_original,
        nested_surrogate,
        level_settings,
        std::numeric_limits<std::size_t>::max(),
        nested_records[level],
        ordinal,
        nested_work[level]);
    if (nested_work[level].attempts !=
            expected_cfsr_base_attempts(level_settings) ||
        nested_work[level].sources != 1U ||
        nested_work[level].failures != 0U ||
        nested_work[level].pruned != 0U ||
        nested_work[level].records !=
            nested_work[level].legacy_repaired ||
        nested_work[level].legacy_repair_candidates <
            nested_work[level].legacy_repaired ||
        nested_work[level].outside_original_pointwise >
            nested_work[level].records ||
        nested_work[level].failures + nested_work[level].pruned +
                nested_work[level].duplicates + nested_work[level].records !=
            nested_work[level].attempts ||
        nested_records[level].empty())
      throw std::runtime_error(
          "CFSR real official-arm accounting did not partition attempts");
    if (level != 0) {
      for (const auto& lower : nested_records[level - 1]) {
        const auto retained = std::find_if(
            nested_records[level].begin(),
            nested_records[level].end(),
            [&](const RetainedLegacyRecord& upper) {
              return same_stored_base_identity(
                  lower.encoded, upper.encoded);
            });
        if (retained == nested_records[level].end()) {
          auto message = std::ostringstream();
          message << "CFSR larger ratio prefix lost a lower-prefix official "
                  << "base: mode=" << lower.encoded.base_mode
                  << ", bound=" << std::setprecision(17)
                  << lower.encoded.selected_global_bound
                  << ", payload_bytes=" << lower.encoded.payload.size();
          throw std::runtime_error(message.str());
        }
      }
    }
  }
  const auto representative = optimize_stored_legacy_candidate(
      nested_original,
      mode1_settings,
      nested_records.front().front().encoded);
  if (!representative.report.all_ok() ||
      representative.encoded_size() >
          nested_records.front().front().encoded.encoded_size() ||
      !representative.cfsr_surrogate_base)
    throw std::runtime_error(
        "CFSR retained base did not complete the full repair optimizer");

  auto nonbinary_settings = mode1_settings;
  nonbinary_settings.pointwise_tolerance = 0.1;
  validate_settings(nonbinary_settings);
  const auto nonbinary_registered_bound =
      nonbinary_settings.pointwise_tolerance *
      kCfsrSurrogateBoundRatios.front();
  const auto nearest_float_bound =
      static_cast<float>(nonbinary_registered_bound);
  const auto narrowed_float_bound =
      cfsr_float_bound_not_above(nonbinary_registered_bound);
  if (!(static_cast<double>(nearest_float_bound) >
            nonbinary_registered_bound) ||
      !(narrowed_float_bound > 0.0F) ||
      static_cast<double>(narrowed_float_bound) >
          nonbinary_registered_bound)
    throw std::runtime_error(
        "CFSR non-binary bound fixture did not exercise downward float rounding");
  auto nonbinary_records = std::vector<RetainedLegacyRecord>();
  auto nonbinary_ordinal = std::size_t{0};
  auto nonbinary_work = CfsrSurrogateWork();
  append_cfsr_surrogate_base_records(
      nested_original,
      nested_surrogate,
      nonbinary_settings,
      std::numeric_limits<std::size_t>::max(),
      nonbinary_records,
      nonbinary_ordinal,
      nonbinary_work);
  for (const auto base_mode : {5U, 6U}) {
    const auto witness = std::find_if(
        nonbinary_records.begin(),
        nonbinary_records.end(),
        [&](const RetainedLegacyRecord& record) {
          return record.encoded.base_mode == base_mode &&
              same_binary64_bits(
                  record.encoded.selected_global_bound,
                  static_cast<double>(narrowed_float_bound));
        });
    if (witness == nonbinary_records.end() ||
        base_exceeds_final_pointwise_tolerance(
            nested_surrogate,
            witness->encoded.reconstruction,
            static_cast<double>(narrowed_float_bound)) ||
        witness->encoded.selected_global_bound >
            nonbinary_registered_bound)
      throw std::runtime_error(
          "CFSR Serf arm lost its exact registered non-binary bound");
  }
  if (nonbinary_work.failures != 0U ||
      nonbinary_work.attempts != expected_cfsr_base_attempts(
          nonbinary_settings))
    throw std::runtime_error(
        "CFSR non-binary official-bound witness did not complete");

  // Seed the real append path with the exact lossless base identity.  Only
  // that arm may enter the duplicate bucket; repairs are deliberately not
  // part of stored-base identity.
  auto duplicate_decoded = encode_decode_base_candidate(
      canonical,
      cfsr_settings,
      cfsr_settings.pointwise_tolerance,
      0.0,
      4U);
  auto duplicate_anchor = finalize_decoded_base_candidate(
      original, cfsr_settings, std::move(duplicate_decoded));
  auto duplicate_records = std::vector<RetainedLegacyRecord>{
      RetainedLegacyRecord{0U, duplicate_anchor}};
  auto duplicate_ordinal = std::size_t{1};
  auto duplicate_work = CfsrSurrogateWork();
  append_cfsr_surrogate_base_records(
      original,
      canonical,
      cfsr_settings,
      std::numeric_limits<std::size_t>::max(),
      duplicate_records,
      duplicate_ordinal,
      duplicate_work);
  if (duplicate_work.sources != 1U || duplicate_work.duplicates != 1U ||
      duplicate_work.attempts != expected_cfsr_base_attempts(cfsr_settings) ||
      duplicate_work.failures + duplicate_work.pruned +
              duplicate_work.duplicates + duplicate_work.records !=
          duplicate_work.attempts ||
      !same_persisted_record_and_reconstruction(
          duplicate_records.front().encoded, duplicate_anchor))
    throw std::runtime_error(
        "CFSR exact stored-base duplicate path changed identity or accounting");

  auto equality_prune_records = std::vector<RetainedLegacyRecord>();
  auto equality_prune_ordinal = std::size_t{0};
  auto equality_prune_work = CfsrSurrogateWork();
  const auto equality_lossless_probe = std::find_if(
      nested_records.front().begin(),
      nested_records.front().end(),
      [](const RetainedLegacyRecord& record) {
        return record.encoded.base_mode == 4U;
      });
  if (equality_lossless_probe == nested_records.front().end())
    throw std::runtime_error(
        "CFSR equality-prune test lost its lossless probe");
  append_cfsr_surrogate_base_records(
      nested_original,
      nested_surrogate,
      mode1_settings,
      compact_base_record_lower_bound_bytes(
          equality_lossless_probe->encoded),
      equality_prune_records,
      equality_prune_ordinal,
      equality_prune_work);
  const auto retained_lossless = std::any_of(
      equality_prune_records.begin(),
      equality_prune_records.end(),
      [](const RetainedLegacyRecord& record) {
        return record.encoded.base_mode == 4U;
      });
  if (equality_prune_work.sources != 1U ||
      equality_prune_work.pruned == 0U || retained_lossless ||
      equality_prune_work.failures + equality_prune_work.pruned +
              equality_prune_work.duplicates + equality_prune_work.records !=
          equality_prune_work.attempts)
    throw std::runtime_error(
        "CFSR actual equality lower-bound prune was not safe and strict");

  std::size_t isolated_failures = 0;
  const auto typed_was_retained = try_retain_candidate(
      []() -> EncodedWindow {
        throw BackendCandidateFailure("intentional CFSR backend failure");
      },
      [](EncodedWindow) {},
      isolated_failures);
  if (typed_was_retained || isolated_failures != 1U)
    throw std::runtime_error("CFSR typed backend failure was not isolated");
  auto generic_runtime_propagated = false;
  try {
    (void)try_retain_candidate(
        []() -> EncodedWindow {
          throw std::runtime_error("intentional CFSR replay failure");
        },
        [](EncodedWindow) {},
        isolated_failures);
  }
  catch (const std::runtime_error&) {
    generic_runtime_propagated = true;
  }
  if (!generic_runtime_propagated || isolated_failures != 1U)
    throw std::runtime_error("CFSR isolated a generic runtime failure");

  auto ordinal_overflow_records = std::vector<RetainedLegacyRecord>();
  auto ordinal_overflow_work = CfsrSurrogateWork();
  auto maximum_ordinal = std::numeric_limits<std::size_t>::max();
  expect_overflow(
      [&] {
        append_cfsr_surrogate_base_records(
            original,
            canonical,
            cfsr_settings,
            std::numeric_limits<std::size_t>::max(),
            ordinal_overflow_records,
            maximum_ordinal,
            ordinal_overflow_work);
      },
      "CFSR accepted a retained-record ordinal overflow");

  // Encoder-only policy must not change QPC framing or decoder settings.
  const auto prefix = std::string(argv[2]);
  const auto incumbent_path = prefix + "-incumbent.qpc";
  const auto cfsr_policy_path = prefix + "-cfsr-policy.qpc";
  const auto mode1_path = prefix + "-mode1.qpc";
  write_stream(incumbent_path, settings, {tag9_incumbent});
  write_stream(cfsr_policy_path, cfsr_settings, {tag9_incumbent});
  write_stream(mode1_path, mode1_settings, {mode1_repaired});
  auto decoded_version = std::uint32_t{0};
  const auto [decoded_settings, decoded] = decode_stream(
      cfsr_policy_path, nullptr, &decoded_version);
  if (read_bytes(incumbent_path) != read_bytes(cfsr_policy_path) ||
      decoded_version != kSubsetRankPackedVersion ||
      decoded_settings.surrogate_frontier != SurrogateFrontier::none ||
      decoded_settings.cfsr_ratio_mask != 0U ||
      !same_float32_tensor_bits(decoded, surrogate))
    throw std::runtime_error(
        "CFSR encoder policy changed QPC bytes, version, or decoded values");

  // A small production call exercises the exact depth-zero boundary, the
  // one-shot suffix, full optimizer, final attribution, and tie retention.
  auto production_settings = cfsr_settings;
  production_settings.channels = 1;
  production_settings.time = 5;
  production_settings.max_lag = 2;
  production_settings.cfsr_ratio_mask = 1U;
  const auto production_source = std::vector<float>{
      -0.75F, -0.125F, 0.5F, 1.0F, 0.25F};
  auto baseline_settings = production_settings;
  baseline_settings.surrogate_frontier = SurrogateFrontier::none;
  baseline_settings.cfsr_ratio_mask = 0U;
  auto one_channel_hpez_isolated = std::array<bool, 3>{};
  auto persisted_one_channel_hpez_is_fatal = std::array<bool, 3>{};
  for (const auto base_mode : std::array<std::uint32_t, 3>{0U, 1U, 2U}) {
    try {
      (void)encode_decode_base_candidate(
          production_source,
          production_settings,
          production_settings.pointwise_tolerance,
          base_mode == 0U ? 1.0 : 0.0,
          base_mode);
    }
    catch (const BackendCandidateFailure&) {
      one_channel_hpez_isolated[base_mode] = true;
    }
    try {
      (void)decode_payload(
          std::vector<unsigned char>{0U},
          production_settings,
          production_settings.pointwise_tolerance,
          base_mode);
    }
    catch (const QoZ::CodecFormatError&) {
      persisted_one_channel_hpez_is_fatal[base_mode] = true;
    }
  }
  const auto production_baseline = compress_window(
      production_source, baseline_settings);
  const auto production_cfsr = compress_window(
      production_source, production_settings);
  if (!production_baseline.report.all_ok() ||
      !production_cfsr.report.all_ok() ||
      compact_record_bytes(production_cfsr) >
          compact_record_bytes(production_baseline) ||
      production_baseline.cfsr_surrogate_sources != 0U ||
      production_cfsr.cfsr_surrogate_sources != 1U ||
      expected_cfsr_base_attempts(production_settings) != 3U ||
      production_cfsr.cfsr_surrogate_base_attempts != 3U ||
      !std::all_of(
          one_channel_hpez_isolated.begin(),
          one_channel_hpez_isolated.end(),
          [](bool isolated) { return isolated; }) ||
      !std::all_of(
          persisted_one_channel_hpez_is_fatal.begin(),
          persisted_one_channel_hpez_is_fatal.end(),
          [](bool fatal) { return fatal; }) ||
      production_cfsr.cfsr_surrogate_base_attempts !=
          production_cfsr.cfsr_surrogate_base_failures +
              production_cfsr.cfsr_surrogate_base_pruned +
              production_cfsr.cfsr_surrogate_base_duplicates +
              production_cfsr.cfsr_surrogate_base_records ||
      production_cfsr.cfsr_surrogate_legacy_repaired !=
          production_cfsr.cfsr_surrogate_base_records ||
      production_cfsr.cfsr_selected !=
          (compact_record_bytes(production_cfsr) <
           compact_record_bytes(production_baseline)))
    throw std::runtime_error(
        "CFSR production depth-one integration violated accounting or no-regression");
  if (production_cfsr.cfsr_selected &&
      production_cfsr.base_mode != 4U && production_cfsr.base_mode != 5U &&
      production_cfsr.base_mode != 6U)
    throw std::runtime_error(
        "single-channel CFSR production selected an unregistered base mode");
  if (!production_cfsr.cfsr_selected &&
      !same_persisted_record_and_reconstruction(
          production_cfsr, production_baseline))
    throw std::runtime_error(
        "CFSR production tie changed protected incumbent bits");

  // A larger two-channel production fixture must retain and fully optimize
  // at least one depth-1 record inside compress_window itself.  This covers
  // the exact suffix boundary and guards against a vacuous all-pruned test.
  auto retained_cfsr_settings = cfsr_settings;
  retained_cfsr_settings.time = 128U;
  retained_cfsr_settings.pointwise_tolerance = 0.1;
  retained_cfsr_settings.acf_tolerance = 1.0e-7;
  retained_cfsr_settings.minimum_correlation = 0.9999999;
  retained_cfsr_settings.max_lag = 16U;
  validate_settings(retained_cfsr_settings);
  auto retained_source = std::vector<float>(
      retained_cfsr_settings.channels * retained_cfsr_settings.time);
  auto retained_state = std::uint32_t{0x6d2b79f5U};
  for (std::size_t channel = 0;
       channel < retained_cfsr_settings.channels;
       ++channel) {
    for (std::size_t index = 0;
         index < retained_cfsr_settings.time;
         ++index) {
      retained_state = retained_state * 1664525U + 1013904223U;
      const auto unit = static_cast<double>(retained_state >> 8U) /
          static_cast<double>(1U << 24U);
      retained_source[
          channel * retained_cfsr_settings.time + index] =
          static_cast<float>(
              2.0 * unit - 1.0 + 0.125 * static_cast<double>(channel));
    }
  }
  auto retained_baseline_settings = retained_cfsr_settings;
  retained_baseline_settings.surrogate_frontier = SurrogateFrontier::none;
  retained_baseline_settings.cfsr_ratio_mask = 0U;
  const auto retained_baseline = compress_window(
      retained_source, retained_baseline_settings);
  const auto retained_cfsr = compress_window(
      retained_source, retained_cfsr_settings);
  if (!retained_baseline.report.all_ok() || !retained_cfsr.report.all_ok() ||
      retained_cfsr.cfsr_surrogate_sources != 1U ||
      retained_cfsr.cfsr_surrogate_base_records == 0U ||
      retained_cfsr.cfsr_surrogate_legacy_repaired !=
          retained_cfsr.cfsr_surrogate_base_records ||
      retained_cfsr.cfsr_surrogate_base_attempts != 5U ||
      compact_record_bytes(retained_cfsr) >
          compact_record_bytes(retained_baseline) ||
      retained_cfsr.cfsr_selected !=
          (compact_record_bytes(retained_cfsr) <
           compact_record_bytes(retained_baseline)))
  {
    auto message = std::ostringstream();
    message << "CFSR retained-record production integration is vacuous or "
            << "regressed: attempts="
            << retained_cfsr.cfsr_surrogate_base_attempts
            << ", failures="
            << retained_cfsr.cfsr_surrogate_base_failures
            << ", pruned=" << retained_cfsr.cfsr_surrogate_base_pruned
            << ", duplicates="
            << retained_cfsr.cfsr_surrogate_base_duplicates
            << ", records=" << retained_cfsr.cfsr_surrogate_base_records
            << ", baseline_bytes=" << compact_record_bytes(retained_baseline)
            << ", final_bytes=" << compact_record_bytes(retained_cfsr)
            << ", selected=" << retained_cfsr.cfsr_selected;
    throw std::runtime_error(message.str());
  }
  if (!retained_cfsr.cfsr_selected &&
      !same_persisted_record_and_reconstruction(
          retained_cfsr, retained_baseline))
    throw std::runtime_error(
        "CFSR retained-record production tie changed incumbent bits");
  const auto production_path = prefix + "-production.qpc";
  write_stream(
      production_path, retained_cfsr_settings, {retained_cfsr});
  const auto [production_decoded_settings, production_replayed] =
      decode_stream(production_path);
  if (production_decoded_settings.surrogate_frontier !=
          SurrogateFrontier::none ||
      production_decoded_settings.cfsr_ratio_mask != 0U ||
      !same_float32_tensor_bits(
          production_replayed, retained_cfsr.reconstruction))
    throw std::runtime_error(
        "CFSR production QPC replay changed policy neutrality or float32 bits");

  std::cout << "QPET_CFSR_SELFTEST record_header_bytes="
            << kCompactRecordHeaderBytes
            << " mask1_attempts=" << nested_work[0].attempts
            << " mask3_attempts=" << nested_work[1].attempts
            << " mask7_attempts=" << nested_work[2].attempts
            << " production_attempts="
            << production_cfsr.cfsr_surrogate_base_attempts
            << " surrogate_sources="
            << production_cfsr.cfsr_surrogate_sources
            << " production_records="
            << production_cfsr.cfsr_surrogate_base_records
            << " retained_production_records="
            << retained_cfsr.cfsr_surrogate_base_records
            << " production_selected=" << production_cfsr.cfsr_selected
            << " tag9_canonical_replay=1 official_modes=5"
            << " official_successes=" << official_successes
            << " official_typed_failures=" << official_typed_failures
            << " mask1_failures=" << nested_work[0].failures
            << " mask3_failures=" << nested_work[1].failures
            << " mask7_failures=" << nested_work[2].failures
            << " exact_duplicates=" << duplicate_work.duplicates
            << " signed_zero_preserved=1 mode1_success_witness=1"
            << " cached_mismatch_rejected=1 nonfinite_rejected=1"
            << " malicious_report_rejected=1 equality_prune=1"
            << " nonbinary_bound_witness=1"
            << " persisted_policy_identity=1\n";
}

void semantic_selftest_mode(int argc, char** argv)
{
  if (argc != 3)
    throw std::invalid_argument(
        "usage: qpet_correlation_batch semantic-selftest output-prefix");
  constexpr std::size_t channels = 16;
  constexpr std::size_t time = 32;
  auto source = std::vector<float>(channels * time);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    for (std::size_t index = 0; index < time; ++index) {
      source[channel * time + index] = static_cast<float>(
          0.125 * static_cast<double>(channel) +
          std::sin(0.17 * static_cast<double>(index) + 0.11 * channel));
    }
  }
  for (std::size_t index = 0; index < time; ++index) {
    source[0 * time + index] = 1.5F;
    source[8 * time + index] = -0.0F;
    source[10 * time + index] = 4.0F;
    source[11 * time + index] = 5.0F;
    source[12 * time + index] = 6.0F;
  }

  // Bitwise detection must not collapse a numerically equal +0/-0 series.
  auto signed_zero = std::vector<float>(2 * time, -0.0F);
  for (std::size_t index = 0; index < time; ++index)
    signed_zero[index] = index == time / 2 ? -0.0F : 0.0F;
  const auto signed_zero_layout =
      split_strict_constant_channels(signed_zero, 2, time);
  if (signed_zero_layout.active_count != 1 ||
      signed_zero_layout.constant_values.size() != 1 ||
      float32_bits(signed_zero_layout.constant_values.front()) != 0x80000000U)
    throw std::runtime_error("semantic self-test collapsed signed-zero bit patterns");
  auto signed_zero_base = std::vector<float>(time, 1.0F);
  auto signed_zero_corrected = std::vector<float>(time, 0.0F);
  for (std::size_t index = 0; index < time; ++index) {
    if (index % 2 != 0)
      signed_zero_corrected[index] = -0.0F;
  }
  const auto signed_zero_correction = compress_correction_payload(
      signed_zero_base, signed_zero_corrected, nullptr, 1, time);
  if (signed_zero_correction.empty() || signed_zero_correction.front() == 5U)
    throw std::runtime_error("format 5 treated mixed signed zeros as one bitwise constant");

  // Reject malformed masks and an invalid one-active-channel SPERR wrapper.
  bool rejected_mask = false;
  try {
    (void)parse_strict_constant_block_payload(
        std::vector<unsigned char>{0x80U}, 3, kSemanticConstantOnlyMode);
  }
  catch (const std::runtime_error&) {
    rejected_mask = true;
  }
  if (!rejected_mask)
    throw std::runtime_error("semantic self-test accepted out-of-range mask bits");
  bool rejected_single_sperr = false;
  try {
    auto malformed = std::vector<unsigned char>{0x01U};
    const auto constant = 3.0F;
    const auto* bytes = reinterpret_cast<const unsigned char*>(&constant);
    malformed.insert(malformed.end(), bytes, bytes + sizeof(float));
    malformed.push_back(0U);
    (void)parse_strict_constant_block_payload(malformed, 2, 3);
  }
  catch (const std::runtime_error&) {
    rejected_single_sperr = true;
  }
  if (!rejected_single_sperr)
    throw std::runtime_error("semantic self-test accepted one-channel SPERR");

  // Only a typed, single-candidate official backend failure may be isolated.
  // The next time-32 fallback must still run.  Generic runtime, retain,
  // integrity, constraint, and overflow failures remain fatal.
  std::size_t isolated_failures = 0;
  bool retained_after_failure = false;
  auto observe_candidate = [&](EncodedWindow, std::uint8_t) {
    retained_after_failure = true;
  };
  const auto failed_candidate_was_retained = try_retain_semantic_candidate(
      []() -> EncodedWindow {
        throw BackendCandidateFailure("intentional time-32 backend failure");
      },
      0,
      observe_candidate,
      isolated_failures);
  if (failed_candidate_was_retained || retained_after_failure || isolated_failures != 1)
    throw std::runtime_error("semantic self-test did not isolate one backend failure");
  const auto fallback_was_retained = try_retain_semantic_candidate(
      [] {
        auto fallback = EncodedWindow();
        fallback.base_mode = 4;
        return fallback;
      },
      0xffU,
      observe_candidate,
      isolated_failures);
  if (!fallback_was_retained || !retained_after_failure || isolated_failures != 1)
    throw std::runtime_error("semantic time-32 fallback did not survive a backend failure");

  bool runtime_propagated = false;
  try {
    (void)try_retain_semantic_candidate(
        []() -> EncodedWindow {
          throw std::runtime_error("intentional hard-constraint/integrity failure");
        },
        0,
        observe_candidate,
        isolated_failures);
  }
  catch (const std::runtime_error&) {
    runtime_propagated = true;
  }
  if (!runtime_propagated || isolated_failures != 1)
    throw std::runtime_error("semantic self-test isolated a generic runtime failure");

  bool retain_failure_propagated = false;
  try {
    (void)try_retain_candidate(
        [] { return EncodedWindow(); },
        [](EncodedWindow) {
          throw std::runtime_error("intentional retain/accounting failure");
        },
        isolated_failures);
  }
  catch (const std::runtime_error&) {
    retain_failure_propagated = true;
  }
  if (!retain_failure_propagated || isolated_failures != 1)
    throw std::runtime_error("semantic self-test isolated a retain failure");

  bool overflow_propagated = false;
  try {
    (void)try_retain_semantic_candidate(
        []() -> EncodedWindow {
          throw std::overflow_error("intentional correction-count overflow");
        },
        0,
        observe_candidate,
        isolated_failures);
  }
  catch (const std::overflow_error&) {
    overflow_propagated = true;
  }
  if (!overflow_propagated || isolated_failures != 1)
    throw std::runtime_error("semantic self-test swallowed a framing overflow");

  // The independent/local objective and the shared/base-only objective must
  // be tracked independently from the same already encoded candidate stream.
  SemanticBlockWinners synthetic_winners;
  std::size_t synthetic_attempts = 0;
  EncodedWindow local_total_winner;
  local_total_winner.base_mode = 6;
  local_total_winner.selected_global_bound = 0.125;
  local_total_winner.repair_candidates_tested = 2;
  local_total_winner.payload.assign(100, 0x11U);
  retain_semantic_block_candidate(
      std::move(local_total_winner),
      4,
      synthetic_winners,
      synthetic_attempts,
      "synthetic semantic attempt overflow");
  EncodedWindow base_only_winner;
  base_only_winner.base_mode = 3;
  base_only_winner.selected_global_bound = 0.0625;
  base_only_winner.corrections = 1;
  base_only_winner.repair_candidates_tested = 3;
  base_only_winner.payload.assign(60, 0x22U);
  base_only_winner.correction_payload.assign(80, 0x33U);
  retain_semantic_block_candidate(
      std::move(base_only_winner),
      5,
      synthetic_winners,
      synthetic_attempts,
      "synthetic semantic attempt overflow");
  if (!synthetic_winners.has_local || !synthetic_winners.has_base ||
      synthetic_winners.local.encoded.payload.size() != 100 ||
      synthetic_winners.base.payload.size() != 60 || synthetic_attempts != 5)
    throw std::runtime_error(
        "semantic self-test did not separate local-total and base-only winners");

  EncodedWindow synthetic_local;
  synthetic_local.payload.assign(100, 0x44U);
  synthetic_local.correction_payload.assign(30, 0x45U);
  synthetic_local.repair_candidates_tested = 2;
  EncodedWindow synthetic_base;
  synthetic_base.payload.assign(110, 0x46U);
  synthetic_base.correction_payload.assign(10, 0x47U);
  synthetic_base.repair_candidates_tested = 3;
  auto synthetic_selected = select_dual_shared_candidate(
      std::move(synthetic_local), std::move(synthetic_base));
  if (synthetic_selected.encoded_size() != 120 ||
      synthetic_selected.shared_local_reference_bytes != 130 ||
      synthetic_selected.shared_base_search_groups != 1 ||
      synthetic_selected.repair_candidates_tested != 5)
    throw std::runtime_error(
        "semantic self-test did not select a strict final-byte base-only win");

  EncodedWindow synthetic_tie_local;
  synthetic_tie_local.payload.assign(100, 0x48U);
  synthetic_tie_local.correction_payload.assign(20, 0x49U);
  synthetic_tie_local.repair_candidates_tested = 4;
  EncodedWindow synthetic_tie_base;
  synthetic_tie_base.payload.assign(110, 0x4aU);
  synthetic_tie_base.correction_payload.assign(10, 0x4bU);
  synthetic_tie_base.repair_candidates_tested = 5;
  auto synthetic_tie = select_dual_shared_candidate(
      std::move(synthetic_tie_local), std::move(synthetic_tie_base));
  if (synthetic_tie.payload.size() != 100 ||
      synthetic_tie.shared_base_search_groups != 0 ||
      synthetic_tie.repair_candidates_tested != 9)
    throw std::runtime_error(
        "semantic self-test did not retain the old shared candidate on a tie");

  const auto duplicate_choice = SemanticBaseChoice{6, 2, 0.125, {1U, 2U, 3U}};
  auto duplicate_left = std::vector<SemanticBaseChoice>{duplicate_choice};
  auto duplicate_right = duplicate_left;
  duplicate_right.front().selected_global_bound = 999.0;
  if (!same_semantic_base_choices(duplicate_left, duplicate_right))
    throw std::runtime_error(
        "semantic self-test failed to deduplicate identical serialized bases");
  duplicate_right.front().bound_index ^= 1U;
  if (same_semantic_base_choices(duplicate_left, duplicate_right))
    throw std::runtime_error(
        "semantic self-test deduplicated different serialized descriptors");

  bool repair_count_overflow_rejected = false;
  try {
    (void)checked_add_u32(
        std::numeric_limits<std::uint32_t>::max(),
        1U,
        "intentional shared repair count overflow");
  }
  catch (const std::overflow_error&) {
    repair_count_overflow_rejected = true;
  }
  if (!repair_count_overflow_rejected)
    throw std::runtime_error(
        "semantic self-test accepted a shared repair-count overflow");

  Settings settings;
  settings.batch = 1;
  settings.channels = channels;
  settings.time = time;
  settings.pointwise_tolerance = 0.25;
  settings.acf_tolerance = 0.05;
  settings.minimum_correlation = 0.99;
  settings.max_lag = 4;
  settings.qoz_level = 1;
  settings.correction_mode = 2;
  settings.estimator_scales = {1.0};
  settings.qpet_confidence = 0.999;
  settings.qpet_k = 2.0;
  settings.group_size = 1;
  settings.repair_denominators = {7};
  settings.enable_vehicle_semantic = true;
  validate_settings(settings);

  auto schema1 = compress_semantic_candidates(source, settings);
  auto schema2 = compress_strict_constant_semantic_candidates(source, settings);
  if (schema1.second.shared_local_reference_bytes == 0 ||
      schema1.second.encoded_size() >
          schema1.second.shared_local_reference_bytes ||
      schema1.second.shared_base_search_groups > 1)
    throw std::runtime_error(
        "schema-1 dual shared search did not preserve the old byte result");
  if (schema2.second.shared_local_reference_bytes == 0 ||
      schema2.second.encoded_size() >
          schema2.second.shared_local_reference_bytes ||
      schema2.second.shared_base_search_groups > 1)
    throw std::runtime_error(
        "schema-2 dual shared search did not preserve the old byte result");
  const auto portfolio = compress_window(source, settings);
  const auto semantic_best_bytes = std::min(
      {schema1.first.encoded_size(),
       schema1.second.encoded_size(),
       schema2.first.encoded_size(),
       schema2.second.encoded_size()});
  if (!portfolio.report.all_ok() ||
      portfolio.encoded_size() > semantic_best_bytes ||
      portfolio.candidates_tested == 0 ||
      portfolio.candidate_failures > portfolio.candidates_tested)
    throw std::runtime_error(
        "semantic self-test time-32 portfolio did not retain a valid byte-minimum");
  auto generic_settings = settings;
  generic_settings.enable_vehicle_semantic = false;
  const auto generic_portfolio = compress_window(source, generic_settings);
  if (!generic_portfolio.report.all_ok() || generic_portfolio.base_mode >= 8 ||
      generic_portfolio.candidates_tested == 0)
    throw std::runtime_error(
        "generic candidate profile selected a vehicle-semantic mode");
  auto verify_outer = [&](const std::string& path,
                          const EncodedWindow& encoded,
                          std::uint32_t version,
                          bool expect_success) {
    write_stream(path, settings, std::vector<EncodedWindow>{encoded});
    if (version != kVersion) {
      auto bytes = read_bytes(path);
      if (bytes.size() < kMagic.size() + sizeof(version))
        throw std::runtime_error("semantic self-test stream is truncated");
      std::memcpy(bytes.data() + kMagic.size(), &version, sizeof(version));
      write_bytes(path, bytes);
    }
    if (!expect_success) {
      try {
        (void) decode_stream(path);
      }
      catch (const std::runtime_error&) {
        return;
      }
      throw std::runtime_error("semantic self-test accepted schema 2 under QPC v10");
    }
    const auto [decoded_settings, reconstructed] = decode_stream(path);
    {
      if (decoded_settings.channels != channels || decoded_settings.time != time)
        throw std::runtime_error("semantic self-test changed the decoded shape");
      const auto report = qpet_correlation::verify_constraints(
          source,
          reconstructed,
          channels,
          time,
          settings.pointwise_tolerance,
          settings.acf_tolerance,
          settings.minimum_correlation,
          settings.max_lag);
      if (!report.all_ok())
        throw std::runtime_error("semantic self-test outer roundtrip violated a hard constraint");
    }
  };

  const auto prefix = std::string(argv[2]);
  bool generic_writer_rejected_semantic = false;
  try {
    write_stream(
        prefix + "-generic-semantic-invalid.qpc",
        generic_settings,
        std::vector<EncodedWindow>{schema1.first});
  }
  catch (const std::runtime_error& error) {
    generic_writer_rejected_semantic = std::string(error.what()).find(
        "generic candidate profile") != std::string::npos;
    if (!generic_writer_rejected_semantic)
      throw;
  }
  if (!generic_writer_rejected_semantic)
    throw std::runtime_error(
        "generic writer accepted a vehicle-semantic encoded window");
  verify_outer(prefix + "-schema1-v10.qpc", schema1.first, 10, true);
  verify_outer(prefix + "-schema2-independent-v11.qpc", schema2.first, 11, true);
  verify_outer(prefix + "-schema2-shared-v11.qpc", schema2.second, 11, true);
  verify_outer(prefix + "-schema2-v10-invalid.qpc", schema2.first, 10, false);

  const auto valid_independent_path = prefix + "-schema2-independent-v11.qpc";
  const auto invalid_count_path = prefix + "-schema2-count-invalid.qpc";
  auto invalid_count_bytes = read_bytes(valid_independent_path);
  constexpr auto compact_correction_count_offset =
      kMagic.size() + 9 * sizeof(std::uint32_t) + 5 * sizeof(double) +
      sizeof(std::uint16_t) + 2 * sizeof(std::uint8_t) + sizeof(double);
  if (invalid_count_bytes.size() <
      compact_correction_count_offset + sizeof(std::uint32_t))
    throw std::runtime_error("semantic self-test stream lacks outer correction metadata");
  std::uint32_t invalid_outer_count = 0;
  std::memcpy(
      &invalid_outer_count,
      invalid_count_bytes.data() + compact_correction_count_offset,
      sizeof(invalid_outer_count));
  invalid_outer_count ^= 1U;
  std::memcpy(
      invalid_count_bytes.data() + compact_correction_count_offset,
      &invalid_outer_count,
      sizeof(invalid_outer_count));
  write_bytes(invalid_count_path, invalid_count_bytes);
  bool rejected_outer_count = false;
  try {
    (void)decode_stream(invalid_count_path);
  }
  catch (const std::runtime_error& error) {
    rejected_outer_count = std::string(error.what()).find(
        "semantic inner correction count disagrees") != std::string::npos;
    if (!rejected_outer_count)
      throw;
  }
  if (!rejected_outer_count)
    throw std::runtime_error("semantic self-test accepted a mismatched outer count");
  std::cout << "QPET_SEMANTIC_SELFTEST schema1_bytes="
            << schema1.first.encoded_size()
            << " schema1_shared_local_bytes="
            << schema1.second.shared_local_reference_bytes
            << " schema1_shared_selected_bytes=" << schema1.second.encoded_size()
            << " schema1_shared_base_selected="
            << schema1.second.shared_base_search_groups
            << " schema2_independent_bytes=" << schema2.first.encoded_size()
            << " schema2_shared_local_bytes="
            << schema2.second.shared_local_reference_bytes
            << " schema2_shared_selected_bytes=" << schema2.second.encoded_size()
            << " schema2_shared_base_selected="
            << schema2.second.shared_base_search_groups
            << " portfolio_bytes=" << portfolio.encoded_size()
            << " portfolio_candidates=" << portfolio.candidates_tested
            << " candidate_failures=" << portfolio.candidate_failures
            << " strict_constants=5 active0_blocks=1 active1_blocks=1\n";
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    if (argc < 2)
      throw std::invalid_argument(
          "expected compress, decompress, fixed-serializer-transcode, "
          "repair-external, apply-external, or "
          "semantic-selftest/packed-selftest/zrcar-selftest/fvro-selftest/"
          "ssbb-selftest/cjrf-selftest/cfsr-selftest mode");
    const auto mode = std::string(argv[1]);
    if (mode == "compress")
      compress_mode(argc, argv);
    else if (mode == "decompress")
      decompress_mode(argc, argv);
    else if (mode == "fixed-serializer-transcode")
      fixed_serializer_transcode_mode(argc, argv);
    else if (mode == "repair-external")
      repair_external_mode(argc, argv);
    else if (mode == "apply-external")
      apply_external_mode(argc, argv);
    else if (mode == "semantic-selftest")
      semantic_selftest_mode(argc, argv);
    else if (mode == "packed-selftest")
      packed_correction_selftest_mode(argc, argv);
    else if (mode == "zrcar-selftest")
      zrcar_selftest_mode(argc, argv);
    else if (mode == "fvro-selftest")
      fvro_selftest_mode(argc, argv);
    else if (mode == "ssbb-selftest")
      ssbb_selftest_mode(argc, argv);
    else if (mode == "cjrf-selftest")
      cjrf_selftest_mode(argc, argv);
    else if (mode == "cfsr-selftest")
      cfsr_selftest_mode(argc, argv);
    else
      throw std::invalid_argument(
          "expected compress, decompress, fixed-serializer-transcode, "
          "repair-external, apply-external, or "
          "semantic-selftest/packed-selftest/zrcar-selftest/fvro-selftest/"
          "ssbb-selftest/cjrf-selftest/cfsr-selftest mode");
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
