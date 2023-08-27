#pragma once
#include <Math/Minimizer.h>
#include <Math/Functor.h>
#include <Math/Factory.h>
#include <vector>
#include <TString.h>
#include <limits>

// #define __ERROR_PRINT_EAA__
// #define __DEBUG_PRINT_EAA__

namespace rse {
  template <class Derived>
  class EaaRootGsl;
}

template <class Derived>
class rse::EaaRootGsl {
  public:
  EaaRootGsl(int dim, int nPar);
  ~EaaRootGsl() { delete minimizer; };

  template <class T>
  inline void AddPoints(const T &vec, double lambda);
  inline void Reserve(int n) {
    data_points.reserve(n);
    lambdas_.reserve(n);
  }
  inline bool ProcessEAA(const std::vector<double> &init_params);

  inline void SetAnnealingScheme(double T_first, double T_last, int step) {
    T_first_ = T_first;
    T_last_  = T_last;
    step_    = step;
  };
  void SetPrintLevel(int level) { minimizer->SetPrintLevel(level); };

  double GetEnergy();
  inline double GetFreeEnergy() {
    return Func(params_.data());
  }

  inline double GetProb(int ipoint);

  /**
   * @brief This function should be overrided.
   * Evaluate distance between a data point and curve f(x; params)
   * Here, x is n-dimensional points.
   *
   * @param data_points a n-dimensional data point.
   * @param params current parameter lists
   * @return double distance between a data point and curve f(x; params)
   * @details for example,
   * return std::hypot(data_point[0] - params[0], data_point[1] - params[1]);
   */
  double Distance(const double *data_point, const double *params) { return 0; };
  double Constraint(const double *params) { return 0; };
  const double *GetVal() { return params_.data(); }
  double Func(const double *params);
  /**
   * @brief clear data points and lambdas
   * 
   */
  void ClearData() {
    data_points.clear();
    lambdas_.data();
  }

  protected:
  ROOT::Math::Minimizer *minimizer = nullptr;
  ROOT::Math::Functor func_;
  std::vector<double> params_;
  std::vector<double> lambdas_;
  std::vector<std::vector<double>> data_points;
  private:
  int dim_            = 0;
  int nPar_           = 0;
  double temperature_ = 0;
  double T_first_     = 0;
  double T_last_      = 0;
  int step_           = 0;
  inline void Init();
};

template <class Derived>
rse::EaaRootGsl<Derived>::EaaRootGsl(int dim, int nPar) : params_(nPar), dim_(dim), nPar_(nPar) {
  Init();
}

template <class Derived>
inline void rse::EaaRootGsl<Derived>::Init() {
  minimizer = ROOT::Math::Factory::CreateMinimizer("GSLMultiMin", "BFGS2");
  minimizer->SetMaxFunctionCalls(std::numeric_limits<unsigned int>::max());
  minimizer->SetMaxIterations(100);
  minimizer->SetTolerance(0.001);
  func_ = ROOT::Math::Functor(this, &rse::EaaRootGsl<Derived>::Func, nPar_);
  minimizer->SetFunction(func_);
  for (int i = 0; i < (int)params_.size(); ++i) {
    minimizer->SetVariable(i, Form("p%d", i), params_[i], 0.01);
  } // for <i>
}

template <class Derived>
template <class T>
inline void rse::EaaRootGsl<Derived>::AddPoints(const T &vec, double lambda) {
  assert((int)vec.size()== dim_);
  data_points.emplace_back(std::vector<double>(dim_));
  lambdas_.emplace_back(lambda);
  auto &point = data_points.back();
  for (int i = 0; i < dim_; ++i) {
    point[i] = vec[i];
  } // for <i>
}

template <class Derived>
double rse::EaaRootGsl<Derived>::Func(const double *params_) {
  double free_energy = 0;
  double beta_       = 1. / temperature_;
  int data_size      = data_points.size();

  for (int ipoint = 0; ipoint < data_size; ++ipoint) {
    auto distance = static_cast<Derived *>(this)->Distance(data_points[ipoint].data(), params_) / lambdas_[ipoint];
    auto expterm  = std::exp(-beta_ * (distance - 1));
    free_energy -= std::log1p(expterm); // log(1 + x)
  }
  free_energy += beta_ * static_cast<Derived *>(this)->Constraint(params_);
  return free_energy;
}

template <class Derived>
inline bool rse::EaaRootGsl<Derived>::ProcessEAA(const std::vector<double> &init_params) {
  auto T_ = [this](double n) {
    return T_first_ * std::pow(T_last_ / T_first_, n / (step_ - 1));
  };
  auto copy_params = [](int N, const double *min_param, std::vector<double> &param) {
    for (int i = 0; i < N; ++i) {
      param[i] = min_param[i];
    } // for <i>
  };
  std::copy(init_params.begin(), init_params.end(), params_.begin());

  for (int istep = 0; istep < step_; ++istep) { // annealing step
    // 0. Evaluate Temperature
    temperature_ = T_(istep);
#ifdef __DEBUG_PRINT_EAA__
    std::cout << "========== annealing step: " << istep << " ==========\n";
    std::cout << "temperature = " << temperature_ << std::endl;
#endif

    // 2. minimization by quasi-newton
    minimizer->SetVariableValues(params_.data());
    // std::cout << "Parameter Before: \n" << this->GetVal() << std::endl;

    bool is_converged = minimizer->Minimize();

#ifdef __DEBUG_PRINT_EAA__
    std::cout << "Energy             = " << GetEnergy() << std::endl;
    std::cout << "Free Energy Before = " << Func(params_.data()) << std::endl;
    std::cout << "Free Energy After  = " << Func(minimizer->X()) << std::endl;
    std::cout << "Num. data points   = " << data_points.size() << std::endl;
    std::cout << "parameter : \n";
    auto p = minimizer->X();
    for (int i = 0; i < nPar_; ++i) {
      std::cout << "    p[" << i << "] = " << p[i] << "\n";
    } // for <i>
    std::cout << std::endl;
#endif
    // 3. Update value
    if (is_converged) {
      auto min_param = minimizer->X();
      copy_params(nPar_, min_param, params_);
    } else {
#ifdef __ERROR_PRINT_EAA__
      std::cout << "*** Error: Free Energy is not converged! " << std::endl;
      std::cout << "*** Iteration status => \n";
      std::cout << "*** -- annealing step = " << istep << "\n";
      std::cout << "*** -- temperature: " << temperature_ << "\n";
      auto result_param = minimizer->X();
      std::cout << "*** -- parameter = \n";
      for (int i = 0; i < nPar_; ++i) {
        std::cout << "    p[" << i << "] = " << result_param[i] << "\n";
      } // for <i>
      std::cout << std::endl;
#endif
      auto min_param = minimizer->X();
      copy_params(nPar_, min_param, params_);
      return is_converged;
    }
  }
#ifdef __DEBUG_PRINT_EAA__
  std::cout << "=== End of EAA: " << std::endl;
  std::cout << "- Free Energy = " << GetFreeEnergy() << std::endl;
  std::cout << "- Parameter: \n";
  for (int i = 0; i < nPar_; ++i) {
    std::cout << "   p[" << i << "] = " << params_[i] << "\n";
  } // for <i>
#endif
  return true;
}

/**
 * @brief 計算量O(N). Nはデータ数.
 * 
 * @return double 
 */
template <class Derived>
inline double rse::EaaRootGsl<Derived>::GetEnergy() {
  double result = 0;
  int nPoints = data_points.size();
  for(int ipoint = 0; ipoint < nPoints; ++ipoint) {
    auto distance = static_cast<Derived *>(this)->Distance(data_points[ipoint].data(), params_.data());
    if(distance < lambdas_[ipoint]) {
      result += distance;
    } else {
      result += lambdas_[ipoint];
    }
  }
  return result + static_cast<Derived *>(this)->Constraint(params_.data());
}

template <class Derived>
inline double rse::EaaRootGsl<Derived>::GetProb(int ipoint) {
  auto distance = static_cast<Derived *>(this)->Distance(data_points[ipoint].data(), params_.data()) / lambdas_[ipoint];
  auto beta     = 1. / temperature_;
  return 1. / (1 + std::exp(beta * (distance - 1)));
}