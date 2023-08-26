#pragma once

#include "QuasiNewton.hh"
#include <vector>
#include <valarray>
#include <iostream>

#define __DEBUG_PRINT_EAA__
#define __ERROR_PRINT_EAA__

namespace rse {
  class ElasticArmAlgorithm;
}

class rse::ElasticArmAlgorithm : public rse::QuasiNewtonBase {
  public:
  ElasticArmAlgorithm(int dim, int nPar) : rse::QuasiNewtonBase(nPar), params(nPar), dim_(dim) {}
  virtual ~ElasticArmAlgorithm() {};

  inline bool ProcessEAA(const std::vector<double> &init_params, double nMaxLoop = 100, double epsilon = 1e-3);
  inline void Reserve(int n) {
    data_points.reserve(n);
    lambdas_.reserve(n);
  }

  template <class T>
  inline void AddPoints(const T &vec, double lambda);

  inline void SetAnnealingScheme(double T_first, double T_last, int step) {
    T_first_ = T_first;
    T_last_  = T_last;
    step_    = step;
  };
  void SetDx(double dx) { dx__ = dx; };

  inline double GetProb(int ipoint);
  inline double GetEnergy();
  inline double GetFreeEnergy() { return Func(params); }
  inline double GetDistance(int ipoint);
  inline double GetLambda(int ipoint) { return lambdas_[ipoint]; }
  inline const Eigen::VectorXd &GetDataPoints(int ipoint) const { return data_points[ipoint]; }
  double GetTemperature() { return temperature_; };

  protected:
  Eigen::VectorXd params;

  std::vector<Eigen::VectorXd> data_points;
  std::vector<double> lambdas_;

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
  virtual double Distance(const Eigen::VectorXd &data_point, const Eigen::VectorXd &params) = 0;

  virtual double Constraint(const Eigen::VectorXd &params) {
    return 0;
  }

  inline void SetTemperature(double T) { temperature_ = T; }

  inline double Func(const Eigen::VectorXd& params);
  inline Eigen::VectorXd NablaF(const Eigen::VectorXd &params);

  private:
  int dim_ = 0;
  double temperature_ = 0;
  double T_first_ = 0;
  double T_last_ = 0;
  int step_ = 0;
  double dx__ = 1e-3;
  double lambda_sum = 0;
};

template <class T>
inline void rse::ElasticArmAlgorithm::AddPoints(const T& vec, double lambda) {
  assert((int)vec.size() == dim_);
  data_points.emplace_back(Eigen::VectorXd(dim_));
  lambdas_.emplace_back(lambda);
  lambda_sum += lambda;
  auto &point = data_points.back();
  for (int i = 0; i < dim_; ++i) {
    point[i] = vec[i];
  }
}

/**
 * @brief Calculate free energy.
 * O(N).
 * @param params parameter lists.
 * @return double free energy
 */
inline double rse::ElasticArmAlgorithm::Func(const Eigen::VectorXd &params_) {
  double free_energy = 0;
  double beta_ = 1./ temperature_;
  int data_size = data_points.size();
  // free_energy +=  lambda_sum * beta_;

  for (int ipoint = 0; ipoint < data_size; ++ipoint) {
    auto distance = Distance(data_points[ipoint], params_) / lambdas_[ipoint];
    // auto antilog  = 1 + std::exp(-beta_ * (distance - lambda_));
    // auto expterm = std::exp(-beta_ * (distance - lambdas_[ipoint]));
    auto expterm = std::exp(-beta_ * (distance - 1));
    free_energy -= std::log1p(expterm); // log(1 + x)
  }
  free_energy += beta_ * Constraint(params_);
  return free_energy;
}


inline Eigen::VectorXd rse::ElasticArmAlgorithm::NablaF(const Eigen::VectorXd &params_) {
  double beta_ = 1. / temperature_;
  int data_size = data_points.size();
  std::size_t nPar_ = params_.size();

  Eigen::VectorXd result(nPar_);
  Eigen::VectorXd h_(nPar_);
  result.setZero();
  h_.setZero();
  for (std::size_t ipar = 0; ipar < nPar_; ++ipar) {
    for (int ipoint = 0; ipoint < data_size; ++ipoint) {
      double distance = Distance(data_points[ipoint], params_) / lambdas_[ipoint];
      double weight   = 1. / (1 + std::exp(beta_ * (distance - 1)));
      h_[ipar]      = dx__;
      double fw_    = Distance(data_points[ipoint], params_ + h_) / lambdas_[ipoint];
      double bw_    = Distance(data_points[ipoint], params_ - h_) / lambdas_[ipoint];
      result[ipar] += weight * (fw_ - bw_) / (2 * h_[ipar]);
      h_[ipar] = 0;
    }
  }
  for(std::size_t ipar = 0; ipar < nPar_; ++ipar) {
    h_[ipar] = dx__;
    double fw_ = Constraint(params_ + h_);
    double bw_ = Constraint(params_ - h_);
    result[ipar] += (fw_ - bw_) / (2 * h_[ipar]);
    h_[ipar] = 0;
  }
  return beta_ * result;
}

inline bool rse::ElasticArmAlgorithm::ProcessEAA(const std::vector<double> &init_params, double nMaxLoop, double epsilon) {
  auto T = [this](double n) {
    return T_first_ * std::pow(T_last_ / T_first_, n / (step_ - 1));
  };
  std::copy(init_params.begin(), init_params.end(), params.begin());

  for (int istep = 0; istep < step_; ++istep) { // annealing step
    // 0. Evaluate Temperature
    temperature_ = T(istep);
#ifdef __DEBUG_PRINT_EAA__
    std::cout << "========== annealing step: " << istep << " ==========\n";
    std::cout << "temperature = " << temperature_ << std::endl;
#endif

    // 2. minimization by quasi-newton
    this->SetInitialVal(params);
    // std::cout << "Parameter Before: \n" << this->GetVal() << std::endl;

    bool is_converged = this->ProcMinimization(nMaxLoop, epsilon);

#ifdef __DEBUG_PRINT_EAA__
    std::cout << "Free Energy Before = " << Func(params) << std::endl;
    std::cout << "Free Energy After  = " << Func(this->GetVal()) << std::endl;
    std::cout << "Num. data points   = " << data_points.size() << std::endl;
    std::cout << "parameter : \n"
              << this->GetVal();
    std::cout << std::endl;
#endif
    // 3. Update value
    if (is_converged) {
      params = this->GetVal();
    } else {
#ifdef __ERROR_PRINT_EAA__
      std::cout << "*** Error: Free Energy is not converged! " << std::endl;
      std::cout << "*** Iteration status => \n";
      std::cout << "*** -- annealing step = " << istep << "\n";
      std::cout << "*** -- temperature: " << temperature_ << "\n";
      auto result_param = this->GetVal();
      std::cout << "*** -- parameter = \n"
                << result_param;
      std::cout << std::endl;
#endif
      params = this->GetVal();
      return is_converged;
    }
  }
#ifdef __DEBUG_PRINT_EAA__
  std::cout << "=== End of EAA: " << std::endl;
  std::cout << "- Energy      = " << GetEnergy() << std::endl;
  std::cout << "- Free Energy = " << Func(params) << std::endl;
  std::cout << "- Parameter: \n" << params << std::endl;
#endif
  return true;
}

inline double rse::ElasticArmAlgorithm::GetProb(int ipoint) {
  auto distance = Distance(data_points[ipoint], params) / lambdas_[ipoint];
  auto beta     = 1. / temperature_;
  return 1. / (1 + std::exp(beta * (distance - 1)));
}

inline double rse::ElasticArmAlgorithm::GetDistance(int ipoint) {
  return Distance(data_points[ipoint], params);
}

/**
 * @brief 計算量O(N). Nはデータ数.
 * 
 * @return double 
 */
inline double rse::ElasticArmAlgorithm::GetEnergy() {
  double result = 0;
  int nPoints = data_points.size();
  for(int ipoint = 0; ipoint < nPoints; ++ipoint) {
    auto distance = Distance(data_points[ipoint], params);
    if(distance < lambdas_[ipoint]) {
      result += distance;
    } else {
      result += lambdas_[ipoint];
    }
  }
  return result + Constraint(params);
}