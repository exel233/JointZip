#include "SubsetRankCodec.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sr = qpet_correlation::subset_rank;

namespace {

void require(bool condition, const std::string& message)
{
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function>
void require_throw(Function&& function, const std::string& message)
{
  auto threw = false;
  try {
    function();
  }
  catch (const std::exception&) {
    threw = true;
  }
  require(threw, message);
}

std::string hex(const std::vector<std::uint8_t>& bytes)
{
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (const auto byte : bytes) {
    result.push_back(digits[byte >> 4U]);
    result.push_back(digits[byte & 0x0fU]);
  }
  return result;
}

std::vector<std::size_t> first_subset(std::size_t k)
{
  std::vector<std::size_t> result(k);
  for (std::size_t index = 0; index < k; ++index)
    result[index] = index;
  return result;
}

std::vector<std::size_t> last_subset(std::size_t time, std::size_t k)
{
  std::vector<std::size_t> result(k);
  for (std::size_t index = 0; index < k; ++index)
    result[index] = time - k + index;
  return result;
}

void enumerate_subsets(
    std::size_t time,
    std::size_t k,
    std::size_t next,
    std::vector<std::size_t>& current,
    std::vector<std::vector<std::size_t>>& output)
{
  if (current.size() == k) {
    output.push_back(current);
    return;
  }
  const auto needed = k - current.size();
  for (std::size_t value = next; value + needed <= time; ++value) {
    current.push_back(value);
    enumerate_subsets(time, k, value + 1U, current, output);
    current.pop_back();
  }
}

void test_binomial_and_boundaries()
{
  require(sr::checked_binomial(128, 1) == 128, "C(128,1) differs");
  require(sr::checked_binomial(128, 2) == 8128, "C(128,2) differs");
  require(
      sr::checked_binomial(128, 64).get_str() ==
          "23951146041928082866135587776380551750",
      "C(128,64) differs");
  require_throw(
      [] { (void)sr::checked_binomial(4, 5); },
      "binomial accepted k > n");
  require_throw(
      [] { (void)sr::checked_binomial(sr::kMaximumTime + 1U, 1); },
      "binomial accepted n above cap");

  for (const auto time : {3U, 8U, 9U, 127U, 128U}) {
    for (const auto k : {std::size_t(1), static_cast<std::size_t>(time - 1U)}) {
      const auto first = first_subset(k);
      const auto last = last_subset(time, k);
      const auto combinations = sr::checked_binomial(time, k);
      require(sr::rank_positions(time, first) == 0, "first subset rank is not zero");
      require(
          sr::rank_positions(time, last) == combinations - 1,
          "last subset rank is not C(T,k)-1");
      require(
          sr::unrank_positions(time, k, 0) == first,
          "first subset unrank differs");
      require(
          sr::unrank_positions(time, k, combinations - 1) == last,
          "last subset unrank differs");
    }
  }
  const auto maximum_central = first_subset(sr::kMaximumTime / 2U);
  const auto maximum_central_rank =
      sr::rank_positions(sr::kMaximumTime, maximum_central);
  require(
      sr::unrank_positions(
          sr::kMaximumTime,
          maximum_central.size(),
          maximum_central_rank) == maximum_central,
      "maximum-time central subset roundtrip differs");
}

void test_exhaustive_small_reference()
{
  for (std::size_t time = 3; time <= 9; ++time) {
    for (std::size_t k = 0; k <= time; ++k) {
      auto subsets = std::vector<std::vector<std::size_t>>();
      auto current = std::vector<std::size_t>();
      enumerate_subsets(time, k, 0, current, subsets);
      auto observed = std::vector<bool>(subsets.size(), false);
      for (const auto& subset : subsets) {
        const auto rank = sr::rank_positions(time, subset);
        require(rank >= 0 && rank < subsets.size(), "small reference rank is invalid");
        require(rank.fits_ulong_p(), "small reference rank does not fit ulong");
        const auto index = static_cast<std::size_t>(rank.get_ui());
        require(!observed[index], "small reference rank is not injective");
        observed[index] = true;
        require(
            sr::unrank_positions(time, k, rank) == subset,
            "small exhaustive rank/unrank differs");
      }
      require(
          std::all_of(observed.begin(), observed.end(), [](bool value) { return value; }),
          "small reference ranks do not cover the binomial range");
    }
  }
}

void test_random_roundtrips()
{
  std::mt19937_64 random(0x5a17c0decafef00dULL);
  for (const auto time : {3U, 8U, 9U, 127U, 128U}) {
    for (std::size_t iteration = 0; iteration < 200; ++iteration) {
      auto all = first_subset(time);
      std::shuffle(all.begin(), all.end(), random);
      const auto k = 1U + static_cast<std::size_t>(random() % (time - 1U));
      all.resize(k);
      std::sort(all.begin(), all.end());
      const auto rank = sr::rank_positions(time, all);
      require(
          sr::unrank_positions(time, k, rank) == all,
          "random subset rank/unrank differs");
    }
  }
}

void test_protocol_and_golden()
{
  const auto descriptors = std::vector<std::uint8_t>{
      sr::kPositionModeNone,
      sr::kPositionModeSubsetRank,
      sr::kPositionModeAll,
      sr::kPositionModeSubsetRank,
  };
  const auto positions = std::vector<std::vector<std::size_t>>{
      {},
      {0, 2, 5},
      {0, 1, 2, 3, 4, 5, 6, 7},
      {1, 6},
  };
  const auto encoded =
      sr::encode_interleaved_position_stream(8, descriptors, positions);
  require(hex(encoded) == "459800", "golden interleaved stream differs");
  const auto decoded =
      sr::decode_interleaved_position_prefix(8, descriptors, encoded, 0, 13);
  require(decoded.positions == positions, "golden protocol roundtrip differs");
  require(decoded.bytes_consumed == encoded.size(), "golden byte count differs");

  auto framed = std::vector<std::uint8_t>{0xfeU};
  framed.insert(framed.end(), encoded.begin(), encoded.end());
  framed.push_back(0xa5U);
  framed.push_back(0x5aU);
  const auto prefix =
      sr::decode_interleaved_position_prefix(8, descriptors, framed, 1, 13);
  require(prefix.positions == positions, "prefixed protocol roundtrip differs");
  require(prefix.bytes_consumed == encoded.size(), "prefix consumed sentinel bytes");
  require(
      framed[1U + prefix.bytes_consumed] == 0xa5U,
      "prefix decoder did not stop before the nonzero sentinel");

  for (const auto time : {3U, 8U, 9U, 127U, 128U}) {
    const auto protocol_descriptors = std::vector<std::uint8_t>{
        sr::kPositionModeNone,
        sr::kPositionModeSubsetRank,
        sr::kPositionModeSubsetRank,
        sr::kPositionModeAll,
    };
    const auto protocol_positions = std::vector<std::vector<std::size_t>>{
        {}, first_subset(1), last_subset(time, time - 1U), first_subset(time)};
    const auto stream = sr::encode_interleaved_position_stream(
        time, protocol_descriptors, protocol_positions);
    require(
        sr::decode_interleaved_position_prefix(
            time,
            protocol_descriptors,
            stream,
            0,
            2U * time).positions == protocol_positions,
        "boundary protocol roundtrip differs");
  }
}

void test_rejections_and_transactionality()
{
  require_throw(
      [] {
        (void)sr::encode_interleaved_position_stream(2, {}, {});
      },
      "protocol accepted T < 3");
  require_throw(
      [] {
        (void)sr::decode_interleaved_position_prefix(
            sr::kMaximumTime + 1U, {}, {}, 0, 0);
      },
      "protocol accepted T above cap");
  require_throw(
      [] {
        (void)sr::rank_positions(8, {2, 2});
      },
      "rank accepted duplicate positions");
  require_throw(
      [] {
        (void)sr::rank_positions(8, {3, 2});
      },
      "rank accepted decreasing positions");
  require_throw(
      [] {
        (void)sr::rank_positions(8, {8});
      },
      "rank accepted an out-of-range position");
  require_throw(
      [] {
        (void)sr::unrank_positions(8, 3, sr::checked_binomial(8, 3));
      },
      "unrank accepted rank == C(T,k)");
  require_throw(
      [] {
        (void)sr::encode_interleaved_position_stream(
            8, {sr::kPositionModeNone}, {{1}});
      },
      "none mode accepted positions");
  require_throw(
      [] {
        (void)sr::encode_interleaved_position_stream(
            8, {sr::kPositionModeSubsetRank}, {{}});
      },
      "partial mode accepted an empty subset");
  require_throw(
      [] {
        (void)sr::encode_interleaved_position_stream(8, {2}, {{1}});
      },
      "encoder accepted an unknown mode");

  const auto descriptors =
      std::vector<std::uint8_t>{sr::kPositionModeSubsetRank};
  const auto valid =
      sr::encode_interleaved_position_stream(8, descriptors, {{0, 2, 5}});

  require_throw(
      [&] {
        auto corrupt = valid;
        corrupt.clear();
        (void)sr::decode_interleaved_position_prefix(
            8, descriptors, corrupt, 0, 3);
      },
      "decoder accepted a truncated interleaved stream");
  require_throw(
      [&] {
        (void)sr::decode_interleaved_position_prefix(
            8, descriptors, valid, 0, 2);
      },
      "decoder performed rank work beyond the correction-count budget");
  require_throw(
      [&] {
        auto corrupt = valid;
        corrupt.back() |= 1U;
        (void)sr::decode_interleaved_position_prefix(
            8, descriptors, corrupt, 0, 3);
      },
      "decoder accepted nonzero interleaved padding");
  require_throw(
      [&] {
        auto corrupt = std::vector<std::uint8_t>{0xe0U};
        (void)sr::decode_interleaved_position_prefix(
            8, descriptors, corrupt, 0, 3);
      },
      "decoder accepted an invalid partial count");
  require_throw(
      [&] {
        // count=2 (k=3), rank=56 == C(8,3), then zero padding.
        auto corrupt = std::vector<std::uint8_t>{0x5cU, 0x00U};
        (void)sr::decode_interleaved_position_prefix(
            8, descriptors, corrupt, 0, 3);
      },
      "decoder accepted rank == C(T,k)");
  require_throw(
      [&] {
        (void)sr::decode_interleaved_position_prefix(8, {7}, {}, 0, 0);
      },
      "decoder accepted an unknown mode");
  require_throw(
      [&] {
        (void)sr::decode_interleaved_position_prefix(
            8, descriptors, valid, valid.size() + 1U, 3);
      },
      "decoder accepted an out-of-range offset");

  const auto work_channels = sr::kMaximumCombinadicWork / 4U + 1U;
  const auto work_descriptors = std::vector<std::uint8_t>(
      work_channels, sr::kPositionModeSubsetRank);
  const auto work_positions = std::vector<std::vector<std::size_t>>(
      work_channels, std::vector<std::size_t>{0, 1, 2, 3});
  require_throw(
      [&] {
        (void)sr::encode_interleaved_position_stream(
            8, work_descriptors, work_positions);
      },
      "encoder exceeded the cumulative combinadic-work budget");
  sr::MsbBitWriter malicious_work_stream;
  const auto count_width = sr::count_bit_width(8);
  const auto rank_width = sr::bit_width(sr::checked_binomial(8, 4) - 1);
  for (std::size_t channel = 0; channel < work_channels; ++channel) {
    malicious_work_stream.write(mpz_class(3), count_width);
    malicious_work_stream.write(mpz_class(0), rank_width);
  }
  require_throw(
      [&] {
        (void)sr::decode_interleaved_position_prefix(
            8,
            work_descriptors,
            malicious_work_stream.bytes(),
            0,
            work_channels * 4U);
      },
      "decoder exceeded the cumulative combinadic-work budget");
}

}  // namespace

int main()
{
  try {
    test_binomial_and_boundaries();
    test_exhaustive_small_reference();
    test_random_roundtrips();
    test_protocol_and_golden();
    test_rejections_and_transactionality();
    std::cout << "subset-rank codec tests passed; golden interleaved=459800\n";
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << "subset-rank codec test failed: " << error.what() << '\n';
    return 1;
  }
}
