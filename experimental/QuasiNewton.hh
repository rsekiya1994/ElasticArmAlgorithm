#pragma once

#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

// #define _DEBUG_PRINT_QUASI_NEWTON_

#ifdef _DEBUG_PRINT_QUASI_NEWTON_
#include <iostream>
#endif

#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>

namespace rse {
  template <class T>
  class QuasiNewton;
}

template <class T>
class rse::QuasiNewton {
  public:
  QuasiNewton<T>(std::size_t nPar) : H_(nPar, nPar), x_(nPar), nPar_(nPar){};
  virtual ~QuasiNewton<T>(){};

  template <class U>
  inline void SetInitialVal(const U &init_vec);

  inline bool ProcMinimization(int nMaxLoop = 100, double epsilon = 1e-3);

  double Func(const Eigen::VectorXd &) { return 0; };
  inline Eigen::VectorXd NablaF(const Eigen::VectorXd &x);
  void SetDx(double dx) { dx_ = dx; };

  inline void SetLinearSearchParameter(double alpha, double tau, double xi, double nMaxLoop);
  inline void SetAlpha(double alpha) { alpha_init = alpha; };
  inline void SetXi(double xi) { xi_ = xi; };
  inline const Eigen::VectorXd &GetVal() const { return x_; };
  inline const Eigen::MatrixXd &GeMatrix() const { return H_; }

  protected:
  Eigen::MatrixXd H_;
  Eigen::VectorXd x_;

  private:
  std::size_t nPar_ = 0;
  // parameter for linear search
  double alpha_init  = 1.0;
  double tau_        = 0.9;
  double xi_         = 0.1;
  double max_loop_LS = 50;
  double dx_         = 1e-3;
};

template <class T>
template <class U>
inline void rse::QuasiNewton<T>::SetInitialVal(const U &init_vec) {
  for (std::size_t ipar = 0; ipar < nPar_; ++ipar) {
    x_[ipar] = init_vec[ipar];
  }
}

template <class T>
inline Eigen::VectorXd rse::QuasiNewton<T>::NablaF(const Eigen::VectorXd &x) {
  Eigen::VectorXd result(nPar_);
  auto x__ = x;
  for (std::size_t i = 0; i < nPar_; ++i) {
    x__[i] += dx_;
    double fw_ = static_cast<T *>(this)->Func(x__);
    x__[i] -= 2 * dx_;
    double bw_ = static_cast<T *>(this)->Func(x__);
    x__[i] += dx_;
    result[i] = (fw_ - bw_) / (2 * dx_);
  }
  return result;
}

/**
 * @brief 直線探索におけるパラメータを決める。
 * f(x + alpha * d) - f(x) <= xi * alpha * nablaF * d
 * となるalphaを探索する。alphaの初期値は1で、1 LOOP ごとに alpha *= tau と更新される。
 * 初期値としては tau_ = 0.9, xi = 0.1, loopの最大値は50としている。
 * @param alpha alphaの初期値。
 * @param tau １ループごとに alpha -> tau * alpha と更新する時の値。
 * @param xi f(x + alpha * d) - f(x) <= xi * alpha * nablaF * d
 * @param nMaxLoop loopの最大回数
 */
template <class T>
inline void rse::QuasiNewton<T>::SetLinearSearchParameter(double alpha, double tau, double xi, double nMaxLoop) {
  alpha_init  = alpha;
  tau_        = tau;
  xi_         = xi;
  max_loop_LS = nMaxLoop;
};

template <class T>
inline bool rse::QuasiNewton<T>::ProcMinimization(int nMaxLoop, double epsilon) {
  Eigen::MatrixXd I_(nPar_, nPar_);
  I_.setIdentity();
  H_.setIdentity();
  bool is_converged = false;
  for (int iloop = 0; iloop < nMaxLoop; ++iloop) {
    Eigen::VectorXd nablaF = static_cast<T *>(this)->NablaF(x_);
    double slope = nablaF.norm();
#ifdef _DEBUG_PRINT_QUASI_NEWTON_
    std::cout << "==== LOOP: " << iloop << " =======\n";
    std::cout << "slope = " << slope << std::endl;
    std::cout << "x_: \n"
              << x_ << std::endl;
    std::cout << "dx_ = \n"
              << nablaF << std::endl;
#endif
    // converge check
    if (slope < epsilon) {
      is_converged = true;
      break;
    }

    Eigen::VectorXd d_ = -H_ * nablaF;
    // here, Armijo rule
    double alpha = alpha_init;
    int i_itr    = 0;
    auto f_val   = static_cast<T *>(this)->Func(x_);
    for (; i_itr < max_loop_LS; ++i_itr) {
      auto f_val_shift = static_cast<T *>(this)->Func(x_ + alpha * d_);
      double threshold = xi_ * alpha * nablaF.transpose() * d_;
      if (f_val_shift - f_val < threshold) break;
      alpha *= tau_;
    }

    Eigen::VectorXd x_next = x_ + alpha * d_;
    //
    auto s_          = x_next - x_;
    auto nablaF_next = static_cast<T *>(this)->NablaF(x_next);
    auto y_          = nablaF_next - nablaF;

    // update
    double rho = y_.transpose() * s_;
    auto K_ = I_ - y_ * s_.transpose() / rho;

    H_ = K_.transpose() * H_ * K_ + s_ * s_.transpose() / rho;
    x_ = x_next;
#ifdef _DEBUG_PRINT_QUASI_NEWTON_
    std::cout << "armijo result = " << alpha << ", i_itr = " << i_itr << std::endl;
    std::cout << "x_next = \n"
              << x_next << std::endl;
    std::cout << "hessian = \n"
              << H_ << std::endl;
#endif
  }

#ifdef _DEBUG_PRINT_QUASI_NEWTON_
  std::cout << "= Result = \n";
  std::cout << x_ << std::endl;
  std::cout << "= Hessian = \n";
  std::cout << H_ << std::endl;
#endif
  return is_converged;
}
