#include "EaaRootGsl.hh"
#include <iostream>
#include <chrono>

// class definition
class EAATest : public rse::EaaRootGsl<EAATest> {
  public:
  EAATest(int dim, int npar) : rse::EaaRootGsl<EAATest>(dim, npar){};
  ~EAATest() {};

  double Distance(const double *data_point, const double *params) {
    return (data_point[0] - params[0]) * (data_point[0] - params[0]);
  }
};

using namespace std::chrono;

int main() {
  // データ点
  std::vector<std::vector<double>> y{{3.0}, {0.95}, {1.04}, {0.99}, {2.8}, {3.1}, {1.01}, {0.3}, {0.22}, {0.96}, {0.94}, {1.03}, {1.0}, {1}};
  rse::EaaRootGsl<EAATest> eaa(1, 1);
  for (auto &&r : y) {
    eaa.AddPoints(r, 0.2);
  }
  eaa.SetAnnealingScheme(10, 0.1, 10);
  eaa.Reserve(y.size());
  eaa.SetPrintLevel(3);
  std::vector<double> init_param{5};
  auto t0 = system_clock::now();
  eaa.ProcessEAA(init_param);
  auto t1  = system_clock::now();
  std::cout << "--* " << duration_cast<microseconds>(t1 - t0).count() << " [us]" << std::endl;
  auto val = eaa.GetVal();
  std::cout << val[0] << std::endl;

  return 0;
}