cmake_minimum_required(VERSION 3.20)
project(qpet_correlation_extension LANGUAGES CXX)

include(CTest)
find_package(SymEngine CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(ZSTD REQUIRED IMPORTED_TARGET libzstd)
find_package(GSL REQUIRED)
find_library(GMPXX_LIBRARY NAMES gmpxx REQUIRED)

if(NOT DEFINED QPET_SOURCE OR NOT EXISTS "${QPET_SOURCE}/include/QoZ")
  message(FATAL_ERROR "Set QPET_SOURCE to the exact official QPET szfamily revision checkout")
endif()
if(NOT DEFINED SPERR_SOURCE OR NOT EXISTS "${SPERR_SOURCE}/include/SPERR_C_API.h")
  message(FATAL_ERROR "Set SPERR_SOURCE to the exact official NCAR/SPERR checkout")
endif()
if(NOT DEFINED SPERR_BUILD OR NOT EXISTS "${SPERR_BUILD}/src/libSPERR.a")
  message(FATAL_ERROR "Set SPERR_BUILD to the matching official static-library build")
endif()
if(NOT DEFINED SERF_SOURCE OR NOT EXISTS "${SERF_SOURCE}/src/compressor_32/serf_xor_compressor_32.h")
  message(FATAL_ERROR "Set SERF_SOURCE to the pinned official Serf checkout")
endif()
if(NOT DEFINED SERF_BUILD OR NOT EXISTS "${SERF_BUILD}/libserf.so")
  message(FATAL_ERROR "Set SERF_BUILD to the matching official Serf library build")
endif()

add_library(qpet_official_headers INTERFACE)
target_compile_features(qpet_official_headers INTERFACE cxx_std_17)
target_include_directories(qpet_official_headers INTERFACE "${QPET_SOURCE}/include")
target_compile_definitions(qpet_official_headers INTERFACE QoZ_DEBUG_TIMINGS=0 ENABLE_GSL="TRUE")
target_link_libraries(
  qpet_official_headers
  INTERFACE symengine ${GMPXX_LIBRARY} gmp OpenMP::OpenMP_CXX PkgConfig::ZSTD GSL::gsl)

add_executable(qpet_correlation_batch qpet_correlation_batch.cpp)
add_library(sperr_official STATIC IMPORTED)
set_target_properties(
  sperr_official
  PROPERTIES IMPORTED_LOCATION "${SPERR_BUILD}/src/libSPERR.a")
add_library(serf_official SHARED IMPORTED)
set_target_properties(
  serf_official
  PROPERTIES IMPORTED_LOCATION "${SERF_BUILD}/libserf.so")
target_include_directories(
  qpet_correlation_batch PRIVATE
  "${SPERR_SOURCE}/include" "${SPERR_BUILD}" "${SERF_SOURCE}/src")
target_link_libraries(
  qpet_correlation_batch PRIVATE qpet_official_headers sperr_official serf_official)
set_target_properties(qpet_correlation_batch PROPERTIES BUILD_RPATH "${SERF_BUILD}")
add_test(
  NAME semantic_schema2_dual_shared_zero_regression
  COMMAND qpet_correlation_batch
          semantic-selftest
          "${CMAKE_CURRENT_BINARY_DIR}/semantic-schema2-selftest")
set_tests_properties(
  semantic_schema2_dual_shared_zero_regression
  PROPERTIES LABELS "semantic;dual_shared;zero_regression")
add_test(
  NAME packed_quantized_correction_round_trip
  COMMAND qpet_correlation_batch
          packed-selftest
          "${CMAKE_CURRENT_BINARY_DIR}/packed-correction-selftest")
add_test(
  NAME zrcar_bounded_byte_search_zero_regression
  COMMAND qpet_correlation_batch zrcar-selftest)
set_tests_properties(
  zrcar_bounded_byte_search_zero_regression
  PROPERTIES LABELS "zrcar;byte_search;zero_regression;exact_oracle")
add_test(
  NAME fvro_feasible_value_true_byte_zero_regression
  COMMAND qpet_correlation_batch fvro-selftest)
set_tests_properties(
  fvro_feasible_value_true_byte_zero_regression
  PROPERTIES
    LABELS "fvro;feasible_value;true_bytes;bitwise_replay;composed_seed;strict_steps;zero_regression")
add_test(
  NAME ssbb_true_serializer_replay_zero_regression
  COMMAND qpet_correlation_batch ssbb-selftest)
set_tests_properties(
  ssbb_true_serializer_replay_zero_regression
  PROPERTIES
    LABELS "ssbb;true_bytes;formats_0_9;bitwise_replay;finite_oracle;zero_regression")
add_test(
  NAME cjrf_certified_joint_rate_repair_frontier_zero_regression
  COMMAND qpet_correlation_batch
          cjrf-selftest
          "${CMAKE_CURRENT_BINARY_DIR}/cjrf-selftest")
set_tests_properties(
  cjrf_certified_joint_rate_repair_frontier_zero_regression
  PROPERTIES
    LABELS "cjrf;true_record_bytes;protected_incumbent;nested_shells;malicious;zero_regression")
add_test(
  NAME cfsr_certified_feasible_surrogate_reencoding_zero_regression
  COMMAND qpet_correlation_batch
          cfsr-selftest
          "${CMAKE_CURRENT_BINARY_DIR}/cfsr-selftest")
set_tests_properties(
  cfsr_certified_feasible_surrogate_reencoding_zero_regression
  PROPERTIES
    LABELS "cfsr;feasible_surrogate;official_codecs;canonical_replay;nested_prefixes;zero_regression")
add_test(
  NAME cfsr_repeated_process_witness
  COMMAND qpet_correlation_batch
          cfsr-selftest
          "${CMAKE_CURRENT_BINARY_DIR}/cfsr-selftest-repeat")
set_tests_properties(
  cfsr_repeated_process_witness
  PROPERTIES
    DEPENDS cfsr_certified_feasible_surrogate_reencoding_zero_regression
    LABELS "cfsr;deterministic;repeated_process")
add_test(
  NAME cfsr_repeated_process_qpc_identity
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${CMAKE_CURRENT_BINARY_DIR}/cfsr-selftest-production.qpc"
          "${CMAKE_CURRENT_BINARY_DIR}/cfsr-selftest-repeat-production.qpc")
set_tests_properties(
  cfsr_repeated_process_qpc_identity
  PROPERTIES
    DEPENDS "cfsr_certified_feasible_surrogate_reencoding_zero_regression;cfsr_repeated_process_witness"
    LABELS "cfsr;deterministic;repeated_process;qpc_identity;zero_regression")
add_test(
  NAME cfsr_repeated_process_mode1_identity
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${CMAKE_CURRENT_BINARY_DIR}/cfsr-selftest-mode1.qpc"
          "${CMAKE_CURRENT_BINARY_DIR}/cfsr-selftest-repeat-mode1.qpc")
set_tests_properties(
  cfsr_repeated_process_mode1_identity
  PROPERTIES
    DEPENDS "cfsr_certified_feasible_surrogate_reencoding_zero_regression;cfsr_repeated_process_witness"
    LABELS "cfsr;deterministic;official_mode1;repeated_process;qpc_identity;zero_regression")

add_executable(test_correlation_qoi test_correlation_qoi.cpp)
target_link_libraries(test_correlation_qoi PRIVATE qpet_official_headers)
add_test(NAME correlation_qoi_analytical_and_constraints COMMAND test_correlation_qoi)

add_executable(test_paper_budget test_paper_budget.cpp)
target_link_libraries(test_paper_budget PRIVATE qpet_official_headers)
add_test(NAME paper_uniform_budget_analytical_fixtures COMMAND test_paper_budget)

add_executable(test_feasible_value_targets test_feasible_value_targets.cpp)
target_compile_features(test_feasible_value_targets PRIVATE cxx_std_17)
add_test(
  NAME feasible_value_target_generation_release_safe
  COMMAND test_feasible_value_targets)
set_tests_properties(
  feasible_value_target_generation_release_safe
  PROPERTIES LABELS "feasible_value;homotopy;float32;bounded_search")

add_executable(test_sequence_beam_search test_sequence_beam_search.cpp)
target_compile_features(test_sequence_beam_search PRIVATE cxx_std_17)
target_link_libraries(test_sequence_beam_search PRIVATE qpet_official_headers)
add_test(
  NAME ssbb_sequence_beam_sufficient_statistics_and_oracle
  COMMAND test_sequence_beam_search)
set_tests_properties(
  ssbb_sequence_beam_sufficient_statistics_and_oracle
  PROPERTIES
    LABELS "ssbb;sufficient_statistics;bounded_beam;finite_oracle;deterministic")

add_executable(test_subset_rank_codec test_subset_rank_codec.cpp)
target_compile_features(test_subset_rank_codec PRIVATE cxx_std_17)
target_link_libraries(test_subset_rank_codec PRIVATE ${GMPXX_LIBRARY} gmp)
add_test(
  NAME subset_rank_position_codec_canonical_and_malicious
  COMMAND test_subset_rank_codec)
set_tests_properties(
  subset_rank_position_codec_canonical_and_malicious
  PROPERTIES
    LABELS "subset_rank;combinadic;canonical;malicious;transactional")

add_executable(test_qpet_upstream_safety test_qpet_upstream_safety.cpp)
target_link_libraries(test_qpet_upstream_safety PRIVATE qpet_official_headers)
add_test(
  NAME qpet_upstream_undefined_behavior_regression
  COMMAND test_qpet_upstream_safety)
set_tests_properties(
  qpet_upstream_undefined_behavior_regression
  PROPERTIES LABELS "qpet_upstream;safety;unaligned;overflow")

add_executable(test_serf_official test_serf_official.cpp)
target_include_directories(test_serf_official PRIVATE "${SERF_SOURCE}/src")
target_link_libraries(test_serf_official PRIVATE serf_official)
set_target_properties(test_serf_official PROPERTIES BUILD_RPATH "${SERF_BUILD}")
add_test(NAME official_serf_float32_round_trip COMMAND test_serf_official)
