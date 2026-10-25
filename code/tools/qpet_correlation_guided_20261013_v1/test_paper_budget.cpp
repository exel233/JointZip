#include "CorrelationQoI.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace qpet_correlation;

static void require(bool condition, const char* message)
{
  if (!condition) throw std::runtime_error(message);
}

static std::vector<double> bounds(const std::vector<float>& x, double epsilon,
                                 double rho, double tau, double alpha)
{
  CorrelationQoI q(x, 1, x.size(), epsilon, tau, rho, 2, epsilon,
                   BoundEstimator::qpet_probabilistic, BoundAllocation::uniform,
                   ComplianceMode::posthoc, alpha, .999, 2.0);
  return q.local_bounds();
}

int main()
{
  // Independent long-double fixture values, obtained directly from equations
  // (not from this implementation's gradient function).
  const std::vector<float> x{0, 1, 2, 1, 0, -1};
  const std::vector<double> expected{
      .15868850828344558, .18135829518108065,
      .20402808207871573, .2266978689763508};
  const std::vector<double> alphas{1.75, 2., 2.25, 2.5};
  const auto unit = bounds(x, .5, .99, .1, 1.).front();
  require(std::abs(unit - .09067914759054033) < 1e-14,
          "independent Eq13 unit-scale fixture mismatch");
  for (std::size_t i = 0; i < alphas.size(); ++i) {
    const auto actual = bounds(x, .5, .99, .1, alphas[i]);
    for (const auto value : actual) {
      require(std::abs(value - expected[i]) < 1e-14,
              "independent Eq13 four-alpha fixture mismatch");
      require(std::abs(value - std::min(.5, alphas[i] * unit)) < 1e-14,
              "one source pass and scaling changed the budget");
    }
  }
  const auto pearson = bounds(x, .5, .99, -1., 1.);
  require(std::abs(pearson.front() - .1350617142889379) < 1e-14,
          "independent Pearson geometry fixture mismatch");
  require(std::abs(bounds(x, .5, .99, .1, .5).front() - .045339573795270165) < 1e-14,
          "independent alpha=.5 fixture mismatch");
  // The alpha=1 map is already epsilon-clipped. Scaling that clipped map by
  // .5 would incorrectly produce .005 instead of the correct .01 here.
  require(bounds(x, .01, .99, .1, .5).front() == .01,
          "alpha=.5 incorrectly scaled an already epsilon-clipped budget");
  for (const auto alpha : alphas) {
    const auto clipped = bounds(x, .01, .99, .1, alpha);
    require(clipped.front() == .01, "epsilon cap lost after alpha scaling");
    const auto zero = bounds(x, .5, 1., .1, alpha);
    require(std::all_of(zero.begin(), zero.end(), [](double e) { return e == 0.; }),
            "rho=1 was raised to a positive error floor");
  }
  const auto constant = bounds(std::vector<float>(32, 3.25F), .5, .99, .1, 2.);
  require(std::all_of(constant.begin(), constant.end(), [](double e) { return e == 0.; }),
          "constant source did not produce an exact budget");
  std::cout << "Paper uniform Eq13 analytical fixtures passed\n";
}
