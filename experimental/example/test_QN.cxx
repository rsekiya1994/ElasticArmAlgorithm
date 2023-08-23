#include "QuasiNewton.hh"
#include <iostream>
#include <chrono>

class MyFunc : public rse::QuasiNewton<MyFunc> {
  public:
  MyFunc(int npar) : rse::QuasiNewton<MyFunc>(npar) {};
  ~MyFunc(){};

  // Himmelblau's function
  // f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 11)^2
  double Func(const Eigen::VectorXd &x) {
    auto temp1 = x[0] * x[0] + x[1] - 11;
    auto temp2 = x[0] + x[1] * x[1] - 7;
    return temp1 * temp1 + temp2 * temp2;
  }

  // double Func(const Eigen::VectorXd &x) {
  //   return std::sin(x[0]);
  // }
};
// 10,000 回の呼び出しでvirtual関数を使った時に比べて10 ms 程度早くなる。
int main(int argc, char *argv[]) {
  using namespace std::chrono;
  auto a = std::stof(argv[1]);
  auto b = std::stof(argv[2]);
  rse::QuasiNewton<MyFunc> *minimizer = new rse::QuasiNewton<MyFunc>(2);
  // MyFunc *minimizer = new MyFunc(2);
  std::vector<double> init_param = {a, b};
  minimizer->SetInitialVal(init_param);
  minimizer->SetLinearSearchParameter(1.0, 0.9, 0.1, 100);
  int64_t N = 1000000;
  bool is_converged = false;
  auto t0 = system_clock::now();
  for (int i = 0; i < N; ++i) {
    minimizer->SetInitialVal(init_param);
    is_converged = minimizer->ProcMinimization(100, 1e-4);
  } // for <i>
  auto t1 = system_clock::now();
  std::cout << "--* " << duration_cast<milliseconds>(t1 - t0).count() << " [ms]" << std::endl;
  auto param = minimizer->GetVal();
  std::cout << "answer = \n";
  std::cout << param << std::endl;
  std::cout << minimizer->GeMatrix() << std::endl;
  if(is_converged) {
    std::cout << "Converged" << std::endl;
  } else {
    std::cout << "Not converged" << std::endl;
  }
  

  return 0;
}