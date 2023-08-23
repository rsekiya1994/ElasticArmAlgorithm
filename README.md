# ElasticArmAlgorithm
C++ Package for Elastic Arm Algorithm & Quasi-Newton Method.

Elastic Arm Algorithm 及び Quasi-Newton Method の C++ 用ライブラリ。
Elastic Arm Algorithm は Ref[1,2] を参考に実装した。
Quasi-Newton の方では、 ヘッセ行列の逆行列の update 手法として BFGS 法を使っている。

# Requirement
- Eigen3 (https://eigen.tuxfamily.org/index.php?title=Main_Page)


# Theory

最小二乗法や $\chi^2$ Fitting では通常、Fitting関数とデータ点との「距離の和」を最小にするようなパラメータを探す。しかし、全ての点を平等に扱うと、データにノイズや外れ値があった場合にFitting 結果がそれらの点に引っ張られることがある。Elastic Arm Algorithm(EAA) は、これらのノイズ/外れ値を除外しつつ Fitting を行うことができる。

$N$ 個からなる $d$ 次元上データ点 $(x_1,\cdots x_N)$ $(x_i\in \mathbb{R}^d)$ を、 $a$ 個のパラメータ $\boldsymbol{\theta} = (\theta_1,\cdots, \theta_a) \in \mathbb{R}^a$ でパラメトライズされた実数値関数 $f(\boldsymbol{x}; \boldsymbol{\theta})=0$ で Fitting することを考える。今、各データ点 $x_i$ と $f(\boldsymbol{x}; \boldsymbol{\theta})$ の距離が $d_i=d(x_i|\boldsymbol{\theta})$ として表されているとする。
EAAでは、次の「エネルギー」関数の最小化を行う。

$$ E(\boldsymbol{w};\boldsymbol{\theta}) = \sum_{i=1}^{N} \left(w_i \frac{d_i}{\lambda_i} + (1-w_i) \right) + V(\boldsymbol{\theta}) $$

ここで、 $\boldsymbol{w}=(w_1,\cdots, w_N)$ は各 $d_i$ に対する重みで、 $i=1,2,\cdots N$ に対して $w_i = 0$ or $1$ を取る。
$\lambda_i$ は $i$ 番目のデータがノイズかどうかを決めるパラメータで、$d_i>\lambda_i$ の場合に データ $i$ には $w_i=0$ が割り当てられ、第二項がエネルギーに寄与することになる。
 $V(\boldsymbol{\theta})$ はパラメータ $\boldsymbol{\theta}$ に対する制約項である。

$E(\boldsymbol{w};\boldsymbol{\theta})$ を最小化する $\boldsymbol{\theta}$ 及び $\boldsymbol{w}$ を探索するにあたり、統計力学のアナロジーから次のカノニカル分布を考える。


$$ P(\boldsymbol{w}) = \frac{\exp(-\beta E(\boldsymbol{w};\boldsymbol{\theta}))}{\sum_{\boldsymbol{w}} \exp(-\beta E(\boldsymbol{w};\boldsymbol{\theta}))}$$

ここで、 $\beta$ は逆温度パラメータで温度 $T$を用いて$\beta = 1/T$ と表される。(今はボルツマン定数を $1$ としている)。 $\sum_{\boldsymbol{w}}$ は可能な全ての $\boldsymbol{w}$ の組み合わせ(全 $2^N$ 通り)について足し合わせると言う意味で、分母の分配関数 $Z$は次のように計算ができる。

$$ Z= \sum_{w} \exp{ \left( -\beta \sum_{i=1}^{N} w_i \frac{d_i}{\lambda_i} -\beta \sum_{i=1}^{N}(1-w_i) -\beta V(\boldsymbol{\theta}) \right) } = e^{-\beta V(\boldsymbol{\theta})}\Pi_{i=1}^{N} \left(e^{-\beta} + e^{-\beta \frac{d_i}{\lambda_i}}\right) $$

従って、ヘルムホルツの自由エネルギー $F=-\frac{1}{\beta} \log{Z}$ は、

$$ F(\theta) =const. -\frac{1}{\beta}\sum_{i=1}^{N}\log \left(1 + e^{-\beta (\frac{d_i}{\lambda_i}-1)}\right)+ V(\theta)$$

となる。一般的に温度が小さい時の平衡状態 $(T\rightarrow 0)$ では、系は低いエネルギーの状態をとる(取りやすくなる)ため、低温における平衡状態を実現するような $\boldsymbol{\theta}$ が結果的にエネルギーが最小となる $\boldsymbol{\theta}$ となる。
平衡状態は自由エネルギー $F(\boldsymbol{\theta})$ が最小となるような $\boldsymbol{\theta}$ として実現するため、アルゴリズム的には温度を下げるたびに $F(\boldsymbol{\theta})$ を最小化すれば良い。高温の状態だと自由エネルギーは凸関数になるため、平衡状態となる点は局所最適解=大域的最適解となる。次の温度ではその点の近傍で最小点を探索することによって局所最適解に陥る可能性が低くなる。その結果、最終温度において大域的最適解を得られることが期待できる。

# How to Use


## Quasi-Newton

一例として、Himmelblau's function $f(x,y)= (x^2 + y - 11)^2 + (x + y^2 - 11)^2$ の最小点を探す。
`rse::QuasiNewtonBase<nPar>` (nPar はパラメータの数。今の場合、`x, y` の 2 つ) を継承し、`double Func(const Eigen::Vector<double, nPar>&)` を override する。

```c++
#include "QuasiNewton.hh"

class MyFunc : public rse::QuasiNewtonBase<2> {
  public:
  MyFunc() {};
  ~MyFunc(){};

  // Himmelblau's function
  // f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 11)^2
  double Func(const Eigen::Vector<double, 2> &x) override {
    auto temp1 = x[0] * x[0] + x[1] - 11;
    auto temp2 = x[0] + x[1] * x[1] - 7;
    return temp1 * temp1 + temp2 * temp2;
  }
};
```

あとは 初期値や直線探索のパラメータを調整して`ProcMinimization(int nMaxLoop, double epsilon)` で走らせる。
直線探索のパラメータは、

$$ F(\boldsymbol{x} + \alpha \boldsymbol{d}) \leq F(\boldsymbol{x}) + \xi \alpha \nabla F(\boldsymbol{x})^T \cdot \boldsymbol{d} $$

に対応し、1回のイタレーションで、 $\alpha \rightarrow \tau \alpha$ と更新される。

```c++
  MyFunc *minimizer = new MyFunc(); // 宣言
  std::vector<double> init_param = {0, 0}; // 初期値 (x,y)=(0,0)
  minimizer->SetInitalVal(init_param);
  minimizer->SetLinearSearchParameter(2.0, 0.9, 0.1, 100); // 直線探索のパラメータ
  bool is_converged = minimizer->ProcMinimization(100, 1e-4); // ループの上限とfの勾配のスレショールド。
```


# Reference
[1] R. Frtihwirth, A. Strandlie, Computer Physics Communications 120 (1999) 197-214, https://www.sciencedirect.com/science/article/pii/S0010465599002313

[2] N. Ueda, R. Nakano, “Deterministic Annealing -Another Type of Annealing-”,  (7/7/1997), https://www.jstage.jst.go.jp/article/jjsai/12/5/12_689/_pdf
