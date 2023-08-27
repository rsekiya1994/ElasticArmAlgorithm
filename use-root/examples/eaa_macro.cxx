#include "../EaaRootGsl.hh"
#include <TRandom3.h>
#include <TGraphErrors.h>
#include <TF1.h>

class Pol1Fit : public rse::EaaRootGsl<Pol1Fit> {
  public:
  Pol1Fit(int dim, int npar) : rse::EaaRootGsl<Pol1Fit>(1, 1){};
  ~Pol1Fit(){};

  double Distance(const double *data, const double *p) {
    auto dist = data[1] - (p[0] + p[1] * data[0]);
    return dist * dist;
  }
};

const int N = 20;

void eaa_macro() {
  TRandom3 rand3;
  rse::EaaRootGsl<Pol1Fit> eaa(2, 2);
  std::vector<std::vector<double>> datas;
  for (int i = 0; i < N; ++i) {
    double x = i;
    double y = 0.5 * x + 1 + rand3.Gaus();
    datas.emplace_back();
    auto &data = datas.back();
    data.resize(2);
    data[0]    = x;
    data[1]    = y;
  } // for <i>
  // noise
  datas.emplace_back();
  datas.back() = {1, 10};
  datas.emplace_back();
  datas.back() = {5, 0};
  datas.emplace_back();
  datas.back() = {15.5, 4};

  // 
  TGraphErrors *g = new TGraphErrors();
  for (int i = 0; i < N + 3; ++i) {
    g->SetPoint(i, datas[i][0], datas[i][1]);
    g->SetPointError(i, 0, 1);
    eaa.AddPoints(datas[i], 2.5 * 2.5);
  } // for <i>
  g->SetMarkerStyle(20);
  g->SetMarkerSize(0.5);
  g->Draw("AP");

  eaa.SetAnnealingScheme(10, 0.5, 10);
  eaa.Reserve(100);
  eaa.SetPrintLevel(3);
  std::vector<double> init_par = {0, 0};
  eaa.ProcessEAA(init_par);
  auto val = eaa.GetVal();
  std::cout << "p[0] = " << val[0] << ", p[1] = " << val[1] << std::endl;
  TF1 *f = new TF1("pol1", "pol1", 0, 20);
  f->SetParameters(val);
  f->Draw("same");
}