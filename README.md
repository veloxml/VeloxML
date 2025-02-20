# VeloxML - High-Performance Machine Learning Framework for Python (Powered by C++)

🚀 **Ultra-Fast ML, Engineered in C++, Designed for Python Users!** 🚀

**VeloxML** is a **Python-friendly machine learning framework** with a **high-performance C++ backend**.  
It combines the ease of use of Python with the **speed and efficiency of C++**, offering optimized implementations of classical ML algorithms for modern CPUs.

## Features
- ⚡ **Optimized C++ Core**: Built with BLAS/LAPACK and OpenMP/TBB for lightning-fast computations.
- 🏎️ **Pythonic API**: Seamless integration with **NumPy, pandas, and Scikit-learn**.
- 🛠️ **Simple & Powerful**: Use it like any other Python ML library (`import veloxml`).
- 🌍 **Cross-Platform**: Runs on **Linux, Windows, and macOS**.
- 📈 **Essential ML Algorithms**:
  - ✅ Linear & Logistic Regression
  - ✅ Decision Trees & Random Forests
  - ~~✅ Gradient Boosting (XGBoost-style)~~ Coming Soon! 
  - ✅ Support Vector Machines (SVM)
  - ✅ Clustering (k-means)
  - ✅ Dimensionality Reduction (PCA, ~~t-SNE~~, ~~UMAP~~) Coming Soon! 
  - ~~✅ Optimization Algorithms (SGD, Adam, RMSprop)~~ Coming Soon! 

## Installation & Usage

### Installation (MacOS only)

Currently, VeloxML is available for macOS with Apple Silicon (arm64).

#### Requirements

* macOS (Apple Silicon)
* Python 3.12+
* `numpy`, `pybind11`

```sh
pip install veloxml
```

## Example Usage

```python
import veloxml
from veloxml.linear import LinearRegression

# Create and fit a model
model = LinearRegression()
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]
model.fit(X, y)

# Make predictions
predictions = model.predict([[6]])
print(predictions)  # Expected output: [12]
```

## Planned Enhancements
- 🔄 **Distributed Learning**: Scalability for multi-node training.
- 🚀 **GPU Acceleration**: Leveraging CUDA/cuBLAS for high-speed computations.
- 🤖 **AutoML Integration**: Automated model selection and hyperparameter tuning.
- 🧠 **Deep Learning Support**: Future support for neural networks.

## License
📜 VeloxML will be released as **freeware** for unrestricted commercial and non-commercial use.

## Contributing
Contributions are welcome!  
Feel free to open issues, suggest features, or submit pull requests.

---

## 🇯🇵 VeloxML - Python向け高性能機械学習フレームワーク（C++ベース）

🚀 **超高速なML、C++のパワーをPythonで活用！** 🚀

**VeloxML** は **Python向けの機械学習フレームワーク** であり、  
**C++で実装された高性能バックエンド** により、  
シンプルなPythonの使いやすさと **C++の計算効率・速度** を両立しています。

## 特徴
- ⚡ **最適化されたC++コア**: BLAS/LAPACKとOpenMP/TBBによる高速計算。
- 🏎️ **PythonicなAPI**: **NumPy、pandas、Scikit-learn** とシームレスに統合。
- 🛠️ **シンプル & 強力**: **`import veloxml`** するだけで簡単に利用可能。
- 🌍 **クロスプラットフォーム対応**: **Linux、Windows、macOS** で動作。
- 📈 **主要な機械学習アルゴリズムを搭載**:
  - ✅ 線形回帰 & ロジスティック回帰
  - ✅ 決定木 & ランダムフォレスト
  - ~~ ✅ 勾配ブースティング（XGBoost風）~~ Coming Soon! 
  - ✅ サポートベクターマシン（SVM）
  - ✅ クラスタリング（k-means）
  - ✅ 次元削減（PCA, ~~t-SNE~~, ~~UMAP~~）Coming Soon !
  - ~~✅ 最適化アルゴリズム（SGD, Adam, RMSprop）~~ Coming Soon!

## インストール & 使い方

### インストール (MacOS のみ)

現在、VeloxML は Apple Silicon (arm64) を搭載した macOS で利用できます。

#### 環境要件

* macOS (Apple Silicon)
* Python 3.12+
* `numpy`, `pybind11`

```sh
pip install veloxml
```

## 使用例

```python
import veloxml
from veloxml.linear import LinearRegression

# Create and fit a model
model = LinearRegression()
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]
model.fit(X, y)

# Make predictions
predictions = model.predict([[6]])
print(predictions)  # Expected output: [12]
```

## 今後の拡張予定
- 🔄 **分散学習対応**: マルチノードでのスケーラブルな学習。
- 🚀 **GPUアクセラレーション**: CUDA/cuBLASを活用した高速計算。
- 🤖 **AutoML統合**: モデル選択とハイパーパラメータ最適化の自動化。
- 🧠 **ディープラーニング対応**: 将来的にニューラルネットワークをサポート。

## ライセンス
📜 VeloxMLは、**商用・非商用を問わず無料で利用可能なフリーウェア** として提供予定です。

## コントリビューション
開発への貢献は大歓迎です！  
新機能の提案、バグ報告、プルリクエストの提出など、お気軽にご参加ください。

---
