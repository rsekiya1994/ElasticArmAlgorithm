# ElasticArmAlgorithm
C++ Package for Elastic Arm Algorithm & Quasi-Newton Method.

Elastic Arm Algorithm 及び Quasi-Newton Method の C++ 用ライブラリです。
Elastic Arm Algorithm は Ref[1,2] を参考に実装しました。
Quasi-Newton の方では、 ヘッセ行列の逆行列の update 手法として BFGS 法を使っています。

# Requirement
- Eigen3 (https://eigen.tuxfamily.org/index.php?title=Main_Page)


# Theory

最小二乗法や $\chi^2$ Fitting では通常、Fitting関数とデータ点との「距離の和」を最小にするようなパラメータを探します。
しかし、全ての点を平等に扱うと、データにノイズや外れ値があった場合に、Fitting 結果がそれらの点に引っ張られると言う現象が多々起きます。Elastic Arm Algorithm(EAA) は、これらのノイズ/外れ値による影響を少なくしながら、シグナル成分のデータ点に絞って Fitting を行います。

$N$ 個からなる $d$ 次元上データ点 $(x_1,\cdots x_N)$ $(x_i\in \mathbb{R}^d)$ を、 $a$ 個のパラメータ $\boldsymbol{\theta} = (\theta_1,\cdots, \theta_a) \in \mathbb{R}^a$ でパラメトライズされた実数値関数 $f(\boldsymbol{x}; \boldsymbol{\theta})=0$ で Fitting することを考えます。今、各データ点 $x_i$ と $f(\boldsymbol{x}; \boldsymbol{\theta})$ の距離が $d_i=d(x_i|\boldsymbol{\theta})$ として表されているとします。
EAAでは、次の「エネルギー」関数の最小化を行います。

$$ E(\boldsymbol{w};\boldsymbol{\theta}) = \sum_{i=1}^{N} \left(w_i d_i + \lambda (1-w_i)^2 \right) + V(\boldsymbol{\theta}) $$

ここで、 $\boldsymbol{w}=(w_1,\cdots, w_N)$ は各 $d_i$ に対する重みで、 $i=1,2,\cdots N$ に対して $w_i = 0$ or $1$ を取ります。
第二項の意味は、もしこの項がなかった場合を考えるとわかりやすいです。もし、 $\sum_{i} w_id_i$ のみの最小化を考えると、全ての $i$ で $w_i=0$ となる時が最小なのは自明です。第二項はこれを防ぐためのペナルティ項です。
$d_i \leq\lambda$ の場合、 $w_i=1$ の方がエネルギーが小さくなり、逆に $d_i>\lambda$ の場合、 $w_i=0$ の方がエネルギー的に得をします。従って     $\lambda$ は、どのデータ点を採用するか除外するかの役目を担っています。 $V(\boldsymbol{\theta})$ はパラメータ $\boldsymbol{\theta}$ に対する制約項として機能します。

$E(\boldsymbol{w};\boldsymbol{\theta})$ を最小化する $\boldsymbol{\theta}$ 及び $\boldsymbol{w}$ を探索するにあたり、統計力学のアナロジーから次のカノニカル分布を考えます。


$$ P(\boldsymbol{w}) = \frac{\exp(-\beta E(\boldsymbol{w};\boldsymbol{\theta}))}{\sum_{\boldsymbol{w}} \exp(-\beta E(\boldsymbol{w};\boldsymbol{\theta}))}$$

ここで、 $\beta$ は逆温度と呼ばれるパラメータで、温度 $T$ に対して $\beta = 1/T$ の関係があります(今はボルツマン定数を $1$ としています)。 $\sum_{\boldsymbol{w}}$ は可能な全ての $\boldsymbol{w}$ の組み合わせ(全 $2^N$ 通り)について足し合わせると言う意味です。分母は分配関数 $Z$ であり、次のように計算ができます。

$$ Z= \sum_{w} \exp{ \left( -\beta \sum_{i=1}^{N} w_i d_i -\beta \lambda \sum_{i=1}^{N}(1-w_i)^2 -\beta V(\boldsymbol{\theta}) \right) } = e^{-\beta V(\boldsymbol{\theta})}\Pi_{i=1}^{N} \left(e^{-\beta\lambda} + e^{-\beta d_i}\right) $$

分配関数が計算できると、ヘルムホルツの自由エネルギー $F=-\frac{1}{\beta} \log{Z}$ が計算できます:

$$ F(\theta) =N\lambda-\frac{1}{\beta}\sum_{i=1}^{N}\log \left(1 + e^{-\beta (d_i-\lambda)}\right)+ V(\theta)$$

一般的に温度が小さい時の平衡状態 $(T\rightarrow 0)$ では、系は低いエネルギーの状態をとる(取りやすくなる)ので、平衡状態を実現するような $\boldsymbol{\theta}$ を求めながら温度を高温から低温に徐々に下げると、結果的にエネルギーが最小となる $\boldsymbol{\theta}$ が得られるだろう、と言うのがEAAの発想です。
平衡状態は自由エネルギー $F(\boldsymbol{\theta})$ が最小となるような $\boldsymbol{\theta}$ として実現できるので、アルゴリズム的には温度を下げていくごとに $F(\boldsymbol{\theta})$ の最適化問題を解けば良いことになります。高温の状態だと自由エネルギーは凸関数に(近い形に)なるので、平衡状態となる点は局所最適解=大域的最適解となり、次の温度ではその点の近傍で最小点を探索することで局所最適解に陥りにくくなります。よって、最終的に低温の状態においても求める最小エネルギーの状態が得られる、と言う戦略となっています。

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
$$ F(\boldsymbol{x} + \alpha \boldsymbol{d}) \leq F(\boldsymbol{x}) + \xi \alpha \nabla F(\boldsymbol{x})^T \cdot \boldsymbol{d}$$

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
