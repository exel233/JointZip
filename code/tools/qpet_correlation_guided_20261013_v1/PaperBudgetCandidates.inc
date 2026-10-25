// Encoder-only source analysis and audit. Included after TuningCandidate.
// This source is intentionally isolated from the frozen qpet_correlation tree.

double downward_float_bound(double proposed)
{
  if (!(proposed > 0.0) || !std::isfinite(proposed))
    return 0.0;
  auto value = static_cast<float>(proposed);
  if (static_cast<double>(value) > proposed)
    value = std::nextafter(value, 0.0F);
  return value > 0.0F ? static_cast<double>(value) : 0.0;
}

std::vector<TuningCandidate> paper_budget_candidates(
    const std::vector<float>& original,
    const Settings& settings)
{
  auto candidates = std::vector<TuningCandidate>();
  auto& audit = *settings.paper_budget_audit;
  const auto started = std::chrono::steady_clock::now();
  double unscaled_cap = settings.pointwise_tolerance;
  const bool has_pearson = settings.minimum_correlation >= 0.0;
  // |ACF(x,k)-ACF(y,k)| <= 2 identically. The frozen E/ER ablations
  // encode disabled ACF as tau=2; do not assign it a spurious budget.
  const bool has_acf = settings.acf_tolerance > 0.0 && settings.acf_tolerance < 2.0;
  if (settings.paper_budget_profile == 1 && (has_pearson || has_acf)) {
    auto analysis_settings = settings;
    if (!has_acf)
      analysis_settings.acf_tolerance = -1.0;
    const auto analyzer = make_qoi(
        original, analysis_settings, settings.pointwise_tolerance, 1.0,
        qpet_correlation::BoundAllocation::uniform);
    ++audit.analyses;
    const auto& bounds = analyzer->local_bounds();
    unscaled_cap = bounds.front();
    for (const auto value : bounds) {
      if (value != unscaled_cap)
        throw std::logic_error("paper uniform C=1 map is not constant");
    }
  }
  constexpr std::array<double, 4> fixed_ratios = {1.0, 0.5, 0.25, 0.125};
  auto seen_bounds = std::vector<double>();
  for (std::size_t index = 0; index < settings.estimator_scales.size(); ++index) {
    const auto alpha = settings.estimator_scales[index];
    const auto proposed = settings.paper_budget_profile == 1
        ? std::min(settings.pointwise_tolerance, alpha * unscaled_cap)
        : settings.pointwise_tolerance * fixed_ratios.at(index);
    const auto effective = downward_float_bound(proposed);
    const auto duplicate = std::find(seen_bounds.begin(), seen_bounds.end(), effective)
        != seen_bounds.end();
    const auto usable = effective > 0.0 && !duplicate;
    if (usable) {
      seen_bounds.push_back(effective);
      // Every lossy backend receives the same actual scalar cap. There is
      // deliberately no fallback to the historical 39-candidate grid.
      for (const auto mode : {3U, 5U, 6U})
        candidates.push_back({effective, alpha, mode, proposed});
    }
    ++audit.plans;
    audit.plans_output << settings.paper_budget_window << '\t'
        << settings.paper_budget_profile << '\t' << alpha << '\t'
        << unscaled_cap << '\t' << proposed << '\t' << effective << '\t'
        << has_pearson << '\t' << has_acf << '\t' << usable << '\t'
        << duplicate << '\t' << (effective == 0.0) << '\n';
  }
  // A zero proposal must remain exact; it is never raised to a positive floor.
  candidates.push_back({settings.pointwise_tolerance, 0.0, 4U, 0.0});
  audit.analysis_seconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  return candidates;
}
