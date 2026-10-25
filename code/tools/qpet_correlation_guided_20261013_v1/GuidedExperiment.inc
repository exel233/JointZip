// Source-guided scalar search on the unchanged scalar backends and simple repair.
// No downstream score participates. Grid and formula schedules remain independent.
struct GuidedCounts {
  std::size_t grid=0, formula=0, noformula=0, physical=0, cache=0;
  std::size_t refinement=0, failures=0, cap_rejections=0, analyses=0, repair=0;
};

struct GuidedRecord {
  std::uint32_t mode=0;
  double cap=0;
  std::optional<EncodedWindow> encoded;
  qpet_correlation::ConstraintReport pre, final;
  bool cap_eligible=false;
  std::string failure;
};

double guided_midpoint(double left, double right)
{
  if (!(left>0 && right>left)) return 0;
  double mid=downward_float_bound(std::exp((std::log(left)+std::log(right))/2));
  if (!(mid>left))
    mid=static_cast<double>(std::nextafter(static_cast<float>(left),
                                        static_cast<float>(right)));
  return mid>left && mid<right ? mid : 0;
}

void guided_compress_mode(Settings settings, const std::string& profile,
    const std::string& input_path, const std::string& output_path,
    const std::string& reconstruction_path)
{
  if (settings.channels!=1 || settings.group_size!=1 ||
      settings.correction_mode!=2 || settings.repair_optimizer!=RepairOptimizer::legacy ||
      settings.correction_format_ceiling!=6 ||
      settings.base_frontier!=BaseFrontier::incumbent_only ||
      settings.surrogate_frontier!=SurrogateFrontier::none ||
      settings.enable_vehicle_semantic || settings.paper_budget_profile!=0 ||
      settings.estimator_scales!=std::vector<double>{1.75,2.0,2.25,2.5} ||
      settings.repair_denominators!=std::vector<int>{7,15,31,63,127} ||
      settings.qpet_confidence!=.999 || settings.qpet_k!=2.0)
    throw std::invalid_argument("guided profiles require frozen C1/group1/simple/cf6 settings");
  const bool combined=profile=="guided_union" || profile=="guided_all";
  const bool need_grid=combined || profile=="guided_grid";
  const bool need_formula=combined || profile=="guided_formula";
  const bool need_control=profile=="guided_noformula";
  if (!need_grid && !need_formula && !need_control)
    throw std::invalid_argument("unknown guided profile");
  const auto source=read_floats(input_path, settings.batch*settings.time);
  if (!std::all_of(source.begin(),source.end(),[](float v){return std::isfinite(v);}))
    throw std::invalid_argument("guided source must be finite");
  std::ofstream plans(output_path+".guided-plans.tsv");
  std::ofstream candidates(output_path+".guided-candidates.tsv");
  std::ofstream winners(output_path+".guided-winners.tsv");
  std::ofstream failures(output_path+".guided-failures.tsv");
  if (!plans || !candidates || !winners || !failures)
    throw std::runtime_error("cannot create guided diagnostics");
  plans<<std::setprecision(17)
    <<"window\tstrategy\tphase\tbase_mode\tordinal\tanchor\tlower\tupper\tproposed_bound\teffective_bound\tanchor_multiplier\tcandidate_id\tcache_hit\tbudget_eligible\taccepted\tstatus\tbase_bytes\tcorrection_bytes\trecord_bytes\n";
  candidates<<std::setprecision(17)
    <<"window\tcandidate_id\tfirst_strategy\tfirst_phase\tbase_mode\teffective_bound\tpre_max_abs\tpre_min_corr\tpre_max_acf\tbudget_eligible\tcorrections\tbase_bytes\tcorrection_bytes\trecord_bytes\tfinal_max_abs\tfinal_min_corr\tfinal_max_acf\tfinal_all_ok\tstatus\n";
  winners<<std::setprecision(17)
    <<"window\tvariant\tcandidate_id\torigin\tbase_mode\teffective_bound\tbudget_eligible\tcorrections\tbase_bytes\tcorrection_bytes\trecord_bytes\tmax_abs\tmin_corr\tmax_acf\tall_ok\n";
  failures<<"window\tcandidate_id\tfirst_strategy\tbase_mode\teffective_bound\terror\n";
  GuidedCounts counts;
  std::array<std::vector<EncodedWindow>,4> streams;
  const std::array<std::string,4> variants{"grid","formula","union","noformula"};
  double analysis_seconds=0;
  for (std::size_t window=0;window<settings.batch;++window) {
    const auto begin=window*settings.time;
    const std::vector<float> original(source.begin()+begin,source.begin()+begin+settings.time);
    auto local=settings; local.batch=1; local.paper_budget_audit=nullptr;
    std::vector<GuidedRecord> records;
    std::vector<std::size_t> grid_ids,formula_ids,control_ids;
    double formula_anchor=local.pointwise_tolerance;
    if (need_formula) {
      const auto started=std::chrono::steady_clock::now();
      const bool pearson=local.minimum_correlation>=0;
      const bool acf=local.acf_tolerance>0 && local.acf_tolerance<2;
      if (pearson || acf) {
        auto analysis=local;
        if (!acf) analysis.acf_tolerance=-1;
        const auto qoi=make_qoi(original,analysis,local.pointwise_tolerance,1.0,
                               qpet_correlation::BoundAllocation::uniform);
        ++counts.analyses;
        formula_anchor=qoi->local_bounds().front();
        for (const auto value:qoi->local_bounds())
          if (value!=formula_anchor)
            throw std::logic_error("guided C1 source map is not uniform");
      }
      analysis_seconds+=std::chrono::duration<double>(
          std::chrono::steady_clock::now()-started).count();
    }
    auto physical=[&](std::uint32_t mode,double cap,const std::string& strategy,
                      const std::string& phase)->std::pair<std::size_t,bool> {
      for (std::size_t id=0;id<records.size();++id)
        if (records[id].mode==mode && records[id].cap==cap) {
          ++counts.cache; return {id,true};
        }
      const auto id=records.size();
      GuidedRecord record; record.mode=mode; record.cap=cap;
      ++counts.physical;
      try {
        auto decoded=encode_decode_base_candidate(original,local,cap,0.0,mode);
        record.pre=refined_actual_report(original,decoded.encoded.reconstruction,local);
        if (mode==4U && std::memcmp(original.data(),decoded.encoded.reconstruction.data(),
                                   original.size()*sizeof(float))!=0)
          throw std::runtime_error("guided lossless base is not bit exact");
        record.cap_eligible=mode==4U ? record.pre.max_abs_error==0 :
                                                 record.pre.max_abs_error<=cap;
        if (!record.cap_eligible) ++counts.cap_rejections;
        // The original finalizer is deliberately unchanged, including failures.
        record.encoded=finalize_decoded_base_candidate(original,local,std::move(decoded));
        record.encoded->logical_windows=1;
        record.final=refined_actual_report(original,record.encoded->reconstruction,local);
        counts.repair+=record.encoded->repair_candidates_tested;
      } catch (const BackendCandidateFailure& error) {
        ++counts.failures; record.failure=error.what();
        std::replace(record.failure.begin(),record.failure.end(),'\t',' ');
        std::replace(record.failure.begin(),record.failure.end(),'\n',' ');
        failures<<window<<'\t'<<id<<'\t'<<strategy<<'\t'<<mode<<'\t'
                <<std::setprecision(17)<<cap<<'\t'<<record.failure<<'\n';
      }
      candidates<<window<<'\t'<<id<<'\t'<<strategy<<'\t'<<phase<<'\t'<<mode<<'\t'
        <<cap<<'\t'<<record.pre.max_abs_error<<'\t'<<record.pre.min_series_correlation
        <<'\t'<<record.pre.max_acf_error<<'\t'<<record.cap_eligible<<'\t';
      if (record.encoded) {
        const auto& e=*record.encoded;
        candidates<<e.corrections<<'\t'<<e.payload.size()<<'\t'<<e.correction_payload.size()
          <<'\t'<<compact_record_bytes(e)<<'\t'<<record.final.max_abs_error<<'\t'
          <<record.final.min_series_correlation<<'\t'<<record.final.max_acf_error<<'\t'
          <<e.report.all_ok()<<'\t'<<(record.cap_eligible?"ok":"cap_rejected")<<'\n';
      } else candidates<<"0\t0\t0\t0\t0\t0\t0\t0\tbackend_failure\n";
      records.push_back(std::move(record)); return {id,false};
    };
    auto probe=[&](const std::string& strategy,const std::string& phase,
        std::uint32_t mode,std::size_t ordinal,double anchor,double lower,double upper,
        double proposed,double cap)->std::size_t {
      if (strategy=="grid") ++counts.grid;
      else if (strategy=="formula") ++counts.formula;
      else ++counts.noformula;
      if (phase=="refine") ++counts.refinement;
      const auto [id,cache_hit]=physical(mode,cap,strategy,phase);
      const auto& r=records[id];
      const bool accepted=r.encoded.has_value() && (strategy=="grid" || r.cap_eligible);
      plans<<window<<'\t'<<strategy<<'\t'<<phase<<'\t'<<mode<<'\t'<<ordinal<<'\t'
        <<anchor<<'\t'<<lower<<'\t'<<upper<<'\t'<<proposed<<'\t'<<cap<<'\t'
        <<(anchor>0 ? cap/anchor:0)<<'\t'<<id<<'\t'<<cache_hit<<'\t'
        <<r.cap_eligible<<'\t'<<accepted<<'\t'
        <<(!r.encoded?"backend_failure":r.cap_eligible?"ok":"cap_rejected")<<'\t';
      if (r.encoded)
        plans<<r.encoded->payload.size()<<'\t'<<r.encoded->correction_payload.size()
             <<'\t'<<compact_record_bytes(*r.encoded)<<'\n';
      else plans<<"0\t0\t0\n";
      return id;
    };
    if (need_grid) {
      const auto old=tuning_candidates(original,local);
      if (old.size()!=39) throw std::logic_error("guided original grid changed");
      for (std::size_t j=0;j<old.size();++j)
        grid_ids.push_back(probe("grid","grid",old[j].base_mode,j,0,0,
          local.pointwise_tolerance,old[j].global_bound,old[j].global_bound));
    }
    auto adaptive=[&](const std::string& strategy,std::vector<std::size_t>& ids) {
      for (const auto mode:{3U,5U,6U}) {
        const std::size_t limit=mode==3U?22:8, initial=mode==3U?12:5;
        const double divisor=mode==3U?512.0:64.0, upper=local.pointwise_tolerance;
        const double anchor=strategy=="formula"?formula_anchor:upper/std::sqrt(divisor);
        if (!(anchor>0) || downward_float_bound(anchor)==0) continue;
        const double lower=std::min(anchor/4,upper/divisor);
        const auto lo=downward_float_bound(lower), hi=downward_float_bound(upper);
        if (!(lo>0) || !(hi>0)) continue;
        std::vector<double> seeds;
        auto add_seed=[&](double value) {
          const auto cap=downward_float_bound(std::clamp(value,lower,upper));
          if (cap>0 && std::find(seeds.begin(),seeds.end(),cap)==seeds.end())
            seeds.push_back(cap);
        };
        for (const auto value:{lower,anchor/2,anchor,anchor*2,upper}) add_seed(value);
        // Fill the initial design by bisecting the widest logarithmic gap.
        while (seeds.size()<initial) {
          auto sorted=seeds; std::sort(sorted.begin(),sorted.end());
          double best_mid=0,best_width=-1;
          for (std::size_t j=1;j<sorted.size();++j) {
            const auto mid=guided_midpoint(sorted[j-1],sorted[j]);
            const auto width=std::log(sorted[j])-std::log(sorted[j-1]);
            if (mid>0 && width>best_width) {best_mid=mid;best_width=width;}
          }
          if (!(best_mid>0)) break;
          seeds.push_back(best_mid);
        }
        // Each strategy observes only its own proposed points, even when a
        // physical result is shared with the grid cache.
        std::vector<std::pair<double,std::size_t>> observed;
        for (const auto cap:seeds) {
          const auto id=probe(strategy,"seed",mode,observed.size(),anchor,lower,upper,cap,cap);
          observed.push_back({cap,id}); ids.push_back(id);
        }
        while (observed.size()<limit) {
          auto sorted=observed;
          std::sort(sorted.begin(),sorted.end(),[](const auto& a,const auto& b){return a.first<b.first;});
          const auto cost=[&](std::size_t id) {
            const auto& r=records[id];
            return r.encoded && r.cap_eligible ? r.encoded->encoded_size():
                                                       std::numeric_limits<std::size_t>::max();
          };
          double next=0,best_width=-1;
          auto best_cost=std::numeric_limits<std::size_t>::max();
          for (std::size_t j=1;j<sorted.size();++j) {
            const auto mid=guided_midpoint(sorted[j-1].first,sorted[j].first);
            if (!(mid>0)) continue;
            const auto score=std::min(cost(sorted[j-1].second),cost(sorted[j].second));
            const auto width=std::log(sorted[j].first)-std::log(sorted[j-1].first);
            if (next==0 || score<best_cost || (score==best_cost && width>best_width)) {
              next=mid;best_cost=score;best_width=width;
            }
          }
          if (!(next>0)) break;
          const auto id=probe(strategy,"refine",mode,observed.size(),anchor,lower,upper,next,next);
          observed.push_back({next,id});ids.push_back(id);
        }
        if (observed.size()>limit) throw std::logic_error("guided call budget exceeded");
      }
      ids.push_back(probe(strategy,"lossless",4U,0,
          strategy=="formula"?formula_anchor:0,0,local.pointwise_tolerance,
          0,local.pointwise_tolerance));
    };
    if (need_formula) adaptive("formula",formula_ids);
    if (need_control) adaptive("noformula",control_ids);
    auto choose=[&](const std::vector<std::size_t>& ids,bool require_cap) {
      auto best=std::numeric_limits<std::size_t>::max();
      for (const auto id:ids) {
        const auto& r=records[id];
        if (!r.encoded || (require_cap && !r.cap_eligible)) continue;
        if (best==std::numeric_limits<std::size_t>::max() ||
            refined_preferred(*r.encoded,*records[best].encoded)) best=id;
      }
      if (best==std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("guided strategy has no feasible candidate");
      return best;
    };
    std::array<std::size_t,4> selected;
    selected.fill(std::numeric_limits<std::size_t>::max());
    if (need_grid) selected[0]=choose(grid_ids,false);
    if (need_formula) selected[1]=choose(formula_ids,true);
    if (need_control) selected[3]=choose(control_ids,true);
    if (combined) {
      auto union_ids=grid_ids;
      for (const auto id:formula_ids)
        if (records[id].encoded && records[id].cap_eligible &&
            std::find(union_ids.begin(),union_ids.end(),id)==union_ids.end())
          union_ids.push_back(id);
      selected[2]=choose(union_ids,false);
    }
    for (std::size_t variant=0;variant<selected.size();++variant) {
      const auto id=selected[variant];
      if (id==std::numeric_limits<std::size_t>::max()) continue;
      const auto& r=records[id];const auto& e=*r.encoded;
      streams[variant].push_back(e);
      const bool old_member=std::find(grid_ids.begin(),grid_ids.end(),id)!=grid_ids.end();
      const bool formula_member=std::find(formula_ids.begin(),formula_ids.end(),id)!=formula_ids.end();
      const auto origin=old_member?(formula_member?"shared":"grid"):
                        formula_member?"formula":"noformula";
      winners<<window<<'\t'<<variants[variant]<<'\t'<<id<<'\t'<<origin<<'\t'
        <<r.mode<<'\t'<<r.cap<<'\t'<<r.cap_eligible<<'\t'<<e.corrections<<'\t'
        <<e.payload.size()<<'\t'<<e.correction_payload.size()<<'\t'<<compact_record_bytes(e)
        <<'\t'<<r.final.max_abs_error<<'\t'<<r.final.min_series_correlation<<'\t'
        <<r.final.max_acf_error<<'\t'<<e.report.all_ok()<<'\n';
    }
  }
  for (auto* stream:{&plans,&candidates,&winners,&failures}) {
    stream->flush();
    if (!*stream) throw std::runtime_error("guided diagnostics write failed");
  }
  const std::size_t main_variant=combined?2:need_grid?0:need_formula?1:3;
  qpet_correlation::ConstraintReport main_report;
  std::size_t main_base_bytes=0,main_correction_bytes=0,main_corrections=0;
  for (std::size_t variant=0;variant<streams.size();++variant) {
    if (streams[variant].empty()) continue;
    const auto path=combined?output_path+"."+variants[variant]+".qpc":output_path;
    const auto recon=combined?path+".f32":reconstruction_path;
    write_stream(path,settings,streams[variant]);
    auto [decoded_settings,reconstructed]=decode_stream(path);
    if (decoded_settings.batch!=settings.batch || decoded_settings.channels!=1 ||
        decoded_settings.time!=settings.time)
      throw std::runtime_error("guided output dimensions changed");
    const auto report=summarize_outer_constraints(source,reconstructed,decoded_settings);
    if (!report.all_ok()) throw std::runtime_error("guided output constraint failure");
    write_floats(recon,reconstructed);
    std::size_t bb=0,cb=0,cc=0;
    for (const auto& e:streams[variant]) {
      bb+=e.payload.size();cb+=e.correction_payload.size();cc+=e.corrections;
    }
    std::cout<<"QPET_GUIDED_VARIANT variant="<<variants[variant]
      <<" base_bytes="<<bb<<" correction_bytes="<<cb<<" corrections="<<cc;
    write_qpet_constraint_metrics(std::cout,report);std::cout<<'\n';
    if (variant==main_variant) {
      main_report=report;main_base_bytes=bb;main_correction_bytes=cb;main_corrections=cc;
      if (combined) {
        write_stream(output_path,settings,streams[variant]);
        write_floats(reconstruction_path,reconstructed);
      }
    }
  }
  if (counts.physical+counts.cache!=counts.grid+counts.formula+counts.noformula)
    throw std::logic_error("guided logical/physical accounting mismatch");
  std::cout<<std::setprecision(17)
    <<"QPET_META candidate_profile="<<profile<<" strategy="<<profile.substr(7)
    <<" budget_mode=source_guided_scalar_v1 search_policy=anchor_loggap_byte_refine_v1"
    <<" formula_budget_policy=actual_base_zero_slack"
    <<" repair_optimizer=legacy correction_format_ceiling=6"
    <<" base_frontier=incumbent_only surrogate_frontier=none"
    <<" attempts_grid="<<counts.grid<<" attempts_formula="<<counts.formula
    <<" attempts_noformula="<<counts.noformula<<" actual_codec_calls="<<counts.physical
    <<" cache_hits="<<counts.cache<<" refinement_attempts="<<counts.refinement
    <<" backend_failures="<<counts.failures<<" cap_rejections="<<counts.cap_rejections
    <<" budget_analyses="<<counts.analyses<<" budget_analysis_seconds="<<analysis_seconds
    <<" candidates="<<counts.repair<<" candidate_failures="<<counts.failures
    <<" corrections="<<main_corrections<<" online_corrections=0"
    <<" base_bytes="<<main_base_bytes<<" correction_bytes="<<main_correction_bytes;
  write_qpet_constraint_metrics(std::cout,main_report);std::cout<<'\n';
}
