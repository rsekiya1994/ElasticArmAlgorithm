#include "ElasticArmAlgorithm.hh"
#include <iostream>
#include <chrono>

// class definition

class EAATest : public rse::ElasticArmAlgorithm {
  public:
  EAATest() : rse::ElasticArmAlgorithm(1, 1){};
  ~EAATest() {};

  double Distance(const Eigen::VectorXd &data_point, const Eigen::VectorXd &params) override {
    return (data_point[0] - params[0]) * (data_point[0] - params[0]);
  }
  double GetF(double x) {
    Eigen::VectorXd vec;
    vec[0] = x;
    return this->Func(vec);
  }
};

using namespace std::chrono;

int main() {
  // データ点
  std::vector<std::vector<double>> y{{3.0}, {0.95}, {1.04}, {0.99}, {2.8}, {3.1}, {1.01}, {0.3}, {0.22}, {0.96}, {0.94}, {1.03}, {1.0}, {1}};
  EAATest eaa;
  for (auto &&r : y) {
    eaa.AddPoints(r, 0.2);
  }
  eaa.SetLinearSearchParameter(1, 0.9, 0.1, 100);
  eaa.SetAnnealingScheme(10, 0.1, 10);
  eaa.Reserve(y.size());
  std::vector<double> init_param{1};
  auto t0 = system_clock::now();
  eaa.ProcessEAA(init_param, 100, 1e-4);
  auto t1 = system_clock::now();
  std::cout << "--* " << duration_cast<microseconds>(t1 - t0).count() << " [us]" << std::endl;
  std::cout << eaa.GetVal() << std::endl;
  return 0;
}