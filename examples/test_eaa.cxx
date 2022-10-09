#include "ElasticArmAlgorithm.hh"


// class definition
template <std::size_t nPar>
class EAATest : public rse::ElasticArmAlgorithm<nPar> {
  public:
  EAATest() : rse::ElasticArmAlgorithm<nPar>(1){};
  ~EAATest() {};

  double Distance(const Eigen::Vector<double, nPar> &data_point, const Eigen::Vector<double, nPar> &params) override {
    return (data_point[0] - params[0]) * (data_point[0] - params[0]);
  }
};


int main() {
  // データ点
  std::vector<std::vector<double>> y{{0.95}, {1.04}, {0.99}, {2.8}, {3.1}, {1.01}, {0.3}, {0.22}, {0.96}, {0.94}, {1.03}, {1.0}, {1}};
  EAATest<1> eaa;
  for (auto &&r : y) {
    eaa.AddPoints(r);
  }
  eaa.SetLinearSearchParameter(4, 0.9, 0.1, 100);
  eaa.SetAnnealingScheme(20, 0.01, 15, 0.1);
  eaa.Reserve(y.size());
  std::vector<double> init_param{1};
  eaa.ProcessEAA(init_param, 100, 1e-4);
  std::cout << eaa.GetVal() << std::endl;
  return 0;
}