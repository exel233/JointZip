// Isolated six-scale source-budget experiment. The base codecs, correction
// implementation, QPC framing, and original generic profile are unchanged.
// Included after the original framing and outer verification helpers.

struct RefinedCandidate {
  TuningCandidate tuning;
  int old_ordinal = -1;
  bool formula_member = false;
  // Membership records the proposal's origin; eligibility records whether the
  // actual decoded base honored that proposal without positive slack.
  bool formula_budget_eligible = false;
  double formula_alpha = 0.0;
  double formula_proposed = 0.0;
  std::optional<EncodedWindow> encoded;
  qpet_correlation::ConstraintReport actual;
  std::string origin() const {
    return old_ordinal >= 0 ? (formula_member ? "shared" : "old39") : "formula";
  }
};

bool refined_preferred(const EncodedWindow& current, const EncodedWindow& incumbent)
{
  // Exactly the original legacy comparison. Stable order handles complete ties.
  return current.encoded_size() < incumbent.encoded_size() ||
      (current.encoded_size() == incumbent.encoded_size() &&
       current.selected_global_bound > incumbent.selected_global_bound);
}

qpet_correlation::ConstraintReport refined_actual_report(
    const std::vector<float>& original, const std::vector<float>& decoded,
    const Settings& settings)
{
  // Compute every statistic even in E/ER ablations. The boolean flags of this
  // diagnostic call are not used for feasibility; the original verifier is.
  return qpet_correlation::verify_constraints(
      original, decoded, settings.channels, settings.time,
      settings.pointwise_tolerance, 2.0, 0.0, settings.max_lag);
}

bool refined_statistical_guard(
    const qpet_correlation::ConstraintReport& current,
    const qpet_correlation::ConstraintReport& old,
    const Settings& settings)
{
  // No numerical slack. Python independently recomputes using its original
  // float64 metric contract and can reject a native winner and fall back.
  return current.max_abs_error <= old.max_abs_error &&
      (settings.minimum_correlation < 0.0 ||
       current.min_series_correlation >= old.min_series_correlation) &&
      (!(settings.acf_tolerance > 0.0 && settings.acf_tolerance < 2.0) ||
       current.max_acf_error <= old.max_acf_error);
}

void refined_compress_mode(
    Settings settings, const std::string& input_path,
    const std::string& output_path, const std::string& reconstruction_path)
{
  const std::vector<double> required_scales{0.5, 1.0, 1.75, 2.0, 2.25, 2.5};
  if (settings.channels != 1 || settings.group_size != 1 ||
      settings.correction_mode != 2 || settings.repair_optimizer != RepairOptimizer::legacy ||
      settings.correction_format_ceiling != 6 ||
      settings.base_frontier != BaseFrontier::incumbent_only ||
      settings.surrogate_frontier != SurrogateFrontier::none ||
      settings.enable_vehicle_semantic || settings.paper_budget_profile != 0 ||
      settings.estimator_scales != required_scales ||
      settings.repair_denominators != std::vector<int>{7, 15, 31, 63, 127} ||
      settings.qpet_confidence != 0.999 || settings.qpet_k != 2.0)
    throw std::invalid_argument("refined experiment requires C=1, group=1, posthoc_greedy, "
        "legacy, cf6, alpha=.5,1,1.75,2,2.25,2.5, beta=.999, k=2, denominators=7,15,31,63,127");

  const auto source = read_floats(input_path, settings.batch * settings.time);
  std::ofstream plans(output_path + ".refined-plans.tsv");
  std::ofstream candidates_log(output_path + ".refined-candidates.tsv");
  std::ofstream winners(output_path + ".refined-winners.tsv");
  std::ofstream failures_log(output_path + ".refined-failures.tsv");
  if (!plans || !candidates_log || !winners || !failures_log)
    throw std::runtime_error("cannot open refined provenance sidecars");
  plans << std::setprecision(17)
      << "window\talpha\tproposed_bound\teffective_bound\tpearson_active\tacf_active\tusable\tduplicate\texact_required\n";
  candidates_log << std::setprecision(17)
      << "window\tcandidate_id\torigin\told_ordinal\tformula_alpha\tbase_mode\tproposed_bound\teffective_bound\tpre_max_abs\tpre_min_corr\tpre_max_acf\tpre_all_ok\tcorrections\tbase_bytes\tcorrection_bytes\trecord_bytes\tfinal_max_abs\tfinal_min_corr\tfinal_max_acf\tfinal_all_ok\tformula_member\tformula_budget_eligible\n";
  winners << std::setprecision(17)
      << "window\tvariant\tcandidate_id\torigin\tbase_mode\talpha\teffective_bound\tcorrections\tbase_bytes\tcorrection_bytes\trecord_bytes\tmax_abs\tmin_corr\tmax_acf\tall_ok\tformula_member\tformula_budget_eligible\n";
  failures_log << std::setprecision(17)
      << "window\tcandidate_id\torigin\tbase_mode\talpha\teffective_bound\terror\n";

  constexpr std::array<const char*, 4> variant_names{
      "old39", "formula_expanded", "union", "statistical_guard"};
  const std::array<std::string, 4> paths{
      output_path + ".old39.qpc", output_path + ".formula_expanded.qpc",
      output_path + ".union.qpc", output_path};
  std::array<std::vector<EncodedWindow>, 4> streams;
  std::size_t analyses = 0, plans_count = 0, attempts = 0, failures = 0;
  std::size_t successful = 0, base_passes = 0, base_bound_violations = 0;
  std::size_t formula_bound_violations = 0, shared_candidates = 0;
  std::size_t formula_candidates = 0, repair_attempts = 0;
  std::size_t formula_eligible = 0, formula_rejected = 0;
  double analysis_seconds = 0.0;
  for (std::size_t window = 0; window < settings.batch; ++window) {
    const auto begin = window * settings.time;
    const std::vector<float> original(source.begin() + begin, source.begin() + begin + settings.time);
    auto local = settings;
    local.batch = 1;
    // No PaperBudgetAudit pointer: explicitly observe the decoded base below,
    // then call the unchanged original finalizer once.
    local.paper_budget_audit = nullptr;
    local.paper_budget_profile = 0;
    std::vector<TuningCandidate> formula_tuning;
    std::vector<double> seen_bounds;
    const bool pearson_active = settings.minimum_correlation >= 0.0;
    const bool acf_active = settings.acf_tolerance > 0.0 && settings.acf_tolerance < 2.0;
    const auto analysis_started = std::chrono::steady_clock::now();
    for (const auto alpha : required_scales) {
      auto proposed = settings.pointwise_tolerance;
      if (pearson_active || acf_active) {
        auto analysis_settings = local;
        if (!acf_active) analysis_settings.acf_tolerance = -1.0;
        // Scaling must happen BEFORE clipping to epsilon. In particular .5
        // cannot multiply the alpha=1 map once that map has been capped.
        const auto analyzer = make_qoi(original, analysis_settings,
            settings.pointwise_tolerance, alpha, qpet_correlation::BoundAllocation::uniform);
        ++analyses;
        const auto& bounds = analyzer->local_bounds();
        proposed = bounds.front();
        if (!std::all_of(bounds.begin(), bounds.end(),
                        [proposed](double value) { return value == proposed; }))
          throw std::logic_error("refined C=1 uniform map is not constant");
      }
      const auto effective = downward_float_bound(proposed);
      const bool duplicate = std::find(seen_bounds.begin(), seen_bounds.end(), effective) != seen_bounds.end();
      const bool usable = effective > 0.0 && !duplicate;
      if (usable) {
        seen_bounds.push_back(effective);
        for (const auto mode : {3U, 5U, 6U})
          formula_tuning.push_back({effective, alpha, mode, proposed});
      }
      ++plans_count;
      plans << window << '\t' << alpha << '\t' << proposed << '\t' << effective
          << '\t' << pearson_active << '\t' << acf_active << '\t' << usable
          << '\t' << duplicate << '\t' << (effective == 0.0) << '\n';
    }
    analysis_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - analysis_started).count();
    formula_tuning.push_back({settings.pointwise_tolerance, 0.0, 4U, 0.0});

    // Preserve all 39 original candidates, in exactly their original order.
    auto old_settings = local;
    old_settings.estimator_scales = {1.75, 2.0, 2.25, 2.5};
    const auto old_tuning = tuning_candidates(original, old_settings);
    if (old_tuning.size() != 39)
      throw std::logic_error("refined old39 anchor changed its candidate portfolio");
    std::vector<RefinedCandidate> records;
    std::vector<std::size_t> old_ids, formula_ids;
    for (std::size_t ordinal = 0; ordinal < old_tuning.size(); ++ordinal) {
      RefinedCandidate record;
      record.tuning = old_tuning[ordinal];
      record.old_ordinal = static_cast<int>(ordinal);
      old_ids.push_back(records.size());
      records.push_back(std::move(record));
    }
    for (const auto& candidate : formula_tuning) {
      const auto found = std::find_if(records.begin(), records.end(),
          [&candidate](const RefinedCandidate& record) {
            return record.tuning.base_mode == candidate.base_mode &&
                record.tuning.global_bound == candidate.global_bound;
          });
      std::size_t id;
      if (found == records.end()) {
        id = records.size();
        RefinedCandidate record;
        record.tuning = candidate;
        records.push_back(std::move(record));
      }
      else id = static_cast<std::size_t>(found - records.begin());
      auto& record = records[id];
      record.formula_member = true;
      record.formula_alpha = candidate.estimator_scale;
      record.formula_proposed = candidate.proposed_bound;
      formula_ids.push_back(id);
    }

    for (std::size_t id = 0; id < records.size(); ++id) {
      auto& record = records[id];
      const auto& candidate = record.tuning;
      ++attempts;
      if (record.formula_member) ++formula_candidates;
      if (record.old_ordinal >= 0 && record.formula_member) ++shared_candidates;
      try {
        auto decoded = encode_decode_base_candidate(original, local,
            candidate.global_bound, candidate.estimator_scale, candidate.base_mode);
        const auto pre = refined_actual_report(original, decoded.encoded.reconstruction, local);
        if (candidate.base_mode == 4U && pre.max_abs_error != 0.0)
          throw std::runtime_error("refined lossless base is not exact");
        record.formula_budget_eligible = record.formula_member &&
            pre.max_abs_error <= candidate.global_bound;
        const auto pre_constraints = qpet_correlation::verify_constraints(original,
            decoded.encoded.reconstruction, 1, local.time, local.pointwise_tolerance,
            local.acf_tolerance, local.minimum_correlation, local.max_lag);
        if (pre_constraints.all_ok()) ++base_passes;
        if (candidate.base_mode != 4U && pre.max_abs_error > candidate.global_bound +
            64.0 * std::numeric_limits<double>::epsilon()) {
          ++base_bound_violations;
          if (record.formula_member) ++formula_bound_violations;
        }
        record.encoded = finalize_decoded_base_candidate(original, local, std::move(decoded));
        auto& encoded = *record.encoded;
        encoded.logical_windows = 1;
        record.actual = refined_actual_report(original, encoded.reconstruction, local);
        repair_attempts += encoded.repair_candidates_tested;
        ++successful;
        if (record.formula_member) {
          if (record.formula_budget_eligible) ++formula_eligible;
          else ++formula_rejected;
        }
        candidates_log << window << '\t' << id << '\t' << record.origin()
            << '\t' << record.old_ordinal << '\t' << record.formula_alpha
            << '\t' << candidate.base_mode << '\t' << record.formula_proposed
            << '\t' << candidate.global_bound << '\t' << pre.max_abs_error
            << '\t' << pre.min_series_correlation << '\t' << pre.max_acf_error
            << '\t' << pre_constraints.all_ok() << '\t' << encoded.corrections
            << '\t' << encoded.payload.size() << '\t' << encoded.correction_payload.size()
            << '\t' << compact_record_bytes(encoded) << '\t' << record.actual.max_abs_error
            << '\t' << record.actual.min_series_correlation << '\t' << record.actual.max_acf_error
            << '\t' << encoded.report.all_ok() << '\t' << record.formula_member
            << '\t' << record.formula_budget_eligible << '\n';
      }
      catch (const BackendCandidateFailure& error) {
        ++failures;
        auto message = std::string(error.what());
        std::replace(message.begin(), message.end(), '\t', ' ');
        std::replace(message.begin(), message.end(), '\n', ' ');
        std::replace(message.begin(), message.end(), '\r', ' ');
        failures_log << window << '\t' << id << '\t' << record.origin()
            << '\t' << candidate.base_mode << '\t' << record.formula_alpha
            << '\t' << candidate.global_bound << '\t' << message << '\n';
      }
    }
    const auto choose = [&records](const std::vector<std::size_t>& ids) {
      auto best = std::numeric_limits<std::size_t>::max();
      for (const auto id : ids) {
        if (!records[id].encoded) continue;
        if (best == std::numeric_limits<std::size_t>::max() ||
            refined_preferred(*records[id].encoded, *records[best].encoded)) best = id;
      }
      if (best == std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("refined candidate set has no feasible stream");
      return best;
    };
    const auto old_best = choose(old_ids);
    // Keep every legacy candidate and its repair untouched. A shared proposal
    // that misses its cap remains an old39 candidate, but has no formula role.
    auto eligible_formula_ids = std::vector<std::size_t>();
    for (const auto id : formula_ids)
      if (records[id].encoded && records[id].formula_budget_eligible)
        eligible_formula_ids.push_back(id);
    auto union_ids = old_ids;
    for (const auto id : eligible_formula_ids)
      if (records[id].old_ordinal < 0) union_ids.push_back(id);
    auto guard_ids = std::vector<std::size_t>{old_best};
    for (const auto id : union_ids) {
      if (id != old_best && records[id].encoded &&
          refined_statistical_guard(records[id].actual, records[old_best].actual, local))
        guard_ids.push_back(id);
    }
    const std::array<std::size_t, 4> selected{
        old_best, choose(eligible_formula_ids), choose(union_ids), choose(guard_ids)};
    for (std::size_t variant = 0; variant < selected.size(); ++variant) {
      const auto id = selected[variant];
      const auto& record = records[id];
      const auto& encoded = *record.encoded;
      streams[variant].push_back(encoded);
      winners << window << '\t' << variant_names[variant] << '\t' << id
          << '\t' << record.origin() << '\t' << encoded.base_mode
          << '\t' << (record.formula_member ? record.formula_alpha : encoded.selected_estimator_scale)
          << '\t' << encoded.selected_global_bound << '\t' << encoded.corrections
          << '\t' << encoded.payload.size() << '\t' << encoded.correction_payload.size()
          << '\t' << compact_record_bytes(encoded) << '\t' << record.actual.max_abs_error
          << '\t' << record.actual.min_series_correlation << '\t' << record.actual.max_acf_error
          << '\t' << encoded.report.all_ok() << '\t' << record.formula_member
          << '\t' << record.formula_budget_eligible << '\n';
    }
  }
  if (attempts != successful + failures)
    throw std::logic_error("refined candidate accounting mismatch");
  for (auto* output : {&plans, &candidates_log, &winners, &failures_log}) {
    output->flush();
    if (!*output) throw std::runtime_error("refined provenance write failed");
  }
  std::array<std::uint64_t, 4> corrections{}, base_bytes{}, correction_bytes{};
  qpet_correlation::ConstraintReport main_report;
  for (std::size_t variant = 0; variant < streams.size(); ++variant) {
    write_stream(paths[variant], settings, streams[variant]);
    auto [decoded_settings, reconstructed] = decode_stream(paths[variant]);
    if (decoded_settings.batch != settings.batch || decoded_settings.channels != 1 ||
        decoded_settings.time != settings.time)
      throw std::runtime_error("refined outer stream dimensions changed");
    const auto report = summarize_outer_constraints(source, reconstructed, decoded_settings);
    if (!report.all_ok()) throw std::runtime_error("refined outer decode violated a constraint");
    write_floats(variant == 3 ? reconstruction_path : paths[variant] + ".f32", reconstructed);
    if (variant == 3) main_report = report;
    for (const auto& encoded : streams[variant]) {
      corrections[variant] += encoded.corrections;
      base_bytes[variant] += encoded.payload.size();
      correction_bytes[variant] += encoded.correction_payload.size();
    }
    std::cout << "QPET_REFINED_VARIANT variant=" << variant_names[variant]
        << " corrections=" << corrections[variant] << " base_bytes=" << base_bytes[variant]
        << " correction_bytes=" << correction_bytes[variant];
    write_qpet_constraint_metrics(std::cout, report);
    std::cout << '\n';
  }
  std::cout << std::setprecision(17)
      << "QPET_META candidate_profile=paper_budget_refined budget_mode=paper_budget_refined"
      << " formula_budget_policy=actual_base_zero_slack_v2"
      << " repair_optimizer=legacy correction_format_ceiling=6"
      << " base_frontier=incumbent_only surrogate_frontier=none"
      << " corrections=" << corrections[3] << " online_corrections=0"
      << " candidates=" << repair_attempts << " candidate_failures=" << failures
      << " base_bytes=" << base_bytes[3] << " correction_bytes=" << correction_bytes[3]
      << " budget_analyses=" << analyses << " budget_plans=" << plans_count
      << " budget_base_attempts=" << attempts << " budget_base_failures=" << failures
      << " budget_base_candidates=" << successful << " budget_base_passes=" << base_passes
      << " budget_base_bound_violations=" << base_bound_violations
      << " formula_base_bound_violations=" << formula_bound_violations
      << " formula_budget_eligible_candidates=" << formula_eligible
      << " formula_budget_rejected_candidates=" << formula_rejected
      << " budget_analysis_seconds=" << analysis_seconds
      << " old39_memberships=" << settings.batch * 39
      << " formula_memberships=" << formula_candidates << " shared_candidates=" << shared_candidates;
  write_qpet_constraint_metrics(std::cout, main_report);
  std::cout << '\n';
}
