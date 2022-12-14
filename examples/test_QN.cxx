#include "QuasiNewton.hh"
#include <iostream>

class MyFunc : public rse::QuasiNewtonBase<2> {
  public:
  MyFunc() {};
  ~MyFunc(){};

  // Himmelblau's function
  // f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 11)^2
  double Func(const Eigen::Vector<double, 2> &x) {
    auto temp1 = x[0] * x[0] + x[1] - 11;
    auto temp2 = x[0] + x[1] * x[1] - 7;
    return temp1 * temp1 + temp2 * temp2;
  }
};

int main() {
  MyFunc *minimizer = new MyFunc();
  std::vector<double> init_param = {0, 0};
  minimizer->SetInitalVal(init_param);
  minimizer->SetLinearSearchParameter(1.0, 0.9, 0.1, 100);
  bool is_converged = minimizer->ProcMinimization(100, 1e-4);
  auto param = minimizer->GetVal();
  std::cout << "answer = \n";
  std::cout << param << std::endl;
  if(is_converged) {
    std::cout << "Converged" << std::endl;
  } else {
    std::cout << "Not converged" << std::endl;
  }
  

  return 0;
}