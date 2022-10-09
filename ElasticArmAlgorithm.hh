#pragma once

#include "QuasiNewton.hh"
#include <vector>


#define __DEBUG_PRINT_EAA__

namespace rse {
  template <std::size_t nPar>
  class ElasticArmAlgorithm;
}

template <std::size_t nPar>
class rse::ElasticArmAlgorithm : public rse::QuasiNewtonBase<nPar> {
  public:
  ElasticArmAlgorithm(int dim) : dim_(dim) {}
  virtual ~ElasticArmAlgorithm() {};

  inline bool ProcessEAA(const std::vector<double> &init_params, double nMaxLoop = 100, double epsilon = 1e-3);
  inline void Reserve(int n) {
    data_points.reserve(n);
  }

  template <class T>
  inline void AddPoints(const T &vec);

  inline void SetAnnealingScheme(double T_first, double T_last, int step, double lambda) {
    T_first_ = T_first;
    T_last_  = T_last;
    step_    = step;
    lambda_  = lambda;
  };

  protected:
  Eigen::Vector<double, nPar> params;

  std::vector<Eigen::Vector<double, nPar>> data_points;

  /**
   * @brief This function should be inherited.
   * Evaluate distance between a data point and curve f(x; params)
   * Here, x is n-dimensional points.
   *
   * @param data_points a n-dimensional data point.
   * @param params current parameter lists
   * @return double distance between a data point and curve f(x; params)
   * @details for example,
   * return std::hypot(data_point[0] - params[0], data_point[1] - params[1]);
   */
  virtual double Distance(const Eigen::Vector<double, nPar> &data_point, const Eigen::Vector<double, nPar> &params) = 0;

  virtual double Constraint(const Eigen::Vector<double, nPar> &params) {
    return 0;
  }

  double Func(const Eigen::Vector<double, nPar>& params);

  private:
  int dim_ = 0;
  double lambda_ = 1;
  double temperature_ = 0;
  double T_first_ = 0;
  double T_last_ = 0;
  int step_ = 0;

};

template <std::size_t nPar>
template <class T>
inline void rse::ElasticArmAlgorithm<nPar>::AddPoints(const T& vec) {
  assert(vec.size() == dim_);
  data_points.emplace_back(Eigen::Vector<double, nPar>(dim_));
  auto &point = data_points.back();
  for (int i = 0; i < dim_; ++i) {
    point[i] = vec[i];
  }
}

/**
 * @brief Calculate free energy
 *
 * @param params parameter lists.
 * @return double free energy
 */
template <std::size_t nPar>
double rse::ElasticArmAlgorithm<nPar>::Func(const Eigen::Vector<double, nPar> &params) {
  double free_energy = 0;
  double beta_ = 1./temperature_;
  free_energy += nPar * lambda_ * beta_;

  for (int ipoint = 0; ipoint < nPar; ++ipoint) {
    auto distance = Distance(data_points[ipoint], params);
    auto antilog  = 1 + std::exp(-beta_ * (distance - lambda_));
    free_energy -= std::log(antilog);
  }

  free_energy += beta_ * Constraint(params);
  return free_energy;
}

template <std::size_t nPar>
inline bool rse::ElasticArmAlgorithm<nPar>::ProcessEAA(const std::vector<double> &init_params, double nMaxLoop, double epsilon) {
  auto T = [this](double n) {
    return T_first_ * std::pow(T_last_ / T_first_, n / (step_ - 1));
  };
  std::copy(init_params.begin(), init_params.end(), params.begin());

  for (int istep = 0; istep < step_; ++istep) { // annealing step
    temperature_ = T(istep);
#ifdef __DEBUG_PRINT_EAA__
    std::cout << "========== annealing step: " << istep << " ==========\n";
    std::cout << "temperature = " << temperature_ << std::endl;
#endif

    // 1. minimization by quasi-newton
    this->SetInitalVal(params);
    bool is_converged = this->ProcMinimization(nMaxLoop, epsilon);

#ifdef __DEBUG_PRINT_EAA__
    std::cout << "Free Energy Before = " << Func(params) << std::endl;
    std::cout << "Free Energy After  = " << Func(this->GetVal()) << std::endl;
    std::cout << "Num. data points   = " << data_points.size() << std::endl;
    std::cout << "parameter : \n"
              << this->GetVal();
    std::cout << std::endl;
#endif
    // 2. Update value
    if (is_converged) {
      params = this->GetVal();
    } else {
      std::cout << "*** Error: Free Energy is not converged! " << std::endl;
      std::cout << "*** Iteration status => \n";
      std::cout << "*** -- annealing step = " << istep << "\n";
      std::cout << "*** -- temperature: " << temperature_ << "\n";
      auto result_param = this->GetVal();
      std::cout << "*** -- parameter = \n"
                << result_param;
      std::cout << std::endl;
      return is_converged;
    }
  }
  return true;
}