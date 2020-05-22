# ReChorus
This repository provides a general PyTorch framework for Top-K recommendation with implicit feedback, especially for research purpose. It decomposes the whole process into three modules:

- [Loader](https://github.com/THUwangcy/ReChorus/tree/master/src/helpers/BaseLoader.py): load dataset into DataFrame and append necessary information to each instance
- [Runner](https://github.com/THUwangcy/ReChorus/tree/master/src/helpers/BaseRunner.py): control the training process and model evaluation
- [Model](https://github.com/THUwangcy/ReChorus/tree/master/src/model/BaseModel.py): prepare batches according to DataFrames in the loader, and define how to generate ranking scores, which will be used in the runner



With this framework, we can easily compare different state-of-the-art models under the same experimental setting. The characteristics of our framework can be summarized as follows:

- **Light**: the framework is accomplished in less than a thousand lines of code, and a new model can be defined with dozens of lines of code
- **Efficient**: around 85-90% GPU utilization during training for deep models, and special implementations for Top-K recommendation evaluation
- **Neat**: clean codes with adequate comments, as well as beautiful training logs
- **Specializing**: concentrate on your model design in a single model file, including batch preparation, parameter definition, prediction, etc. 
- **Flexible**: implement new loaders or runners for different datasets and experimental settings, and each model can be assigned with different helpers



## Getting Started

1. Install [Anaconda](https://docs.conda.io/en/latest/miniconda.html) with Python >= 3.5
2. Clone the repository and install requirements

```bash
git clone https://github.com/THUwangcy/ReChorus.git
cd ReChorus
pip install -r requirements.txt
```

3. Run model with build-in dataset

```bash
python main.py --model_name BPR --emb_size 64 --lr 1e-3 --dataset Grocery_and_Gourmet_Food
```

4. (opt) Run jupyter notebook in `data` folder to download and build new amazon datasets, or prepare your own datasets according to [README](https://github.com/THUwangcy/ReChorus/tree/master/data/README.md) in `data`
5. (opt) Implement your own models according to [README](https://github.com/THUwangcy/ReChorus/tree/master/src/README.md) in `src`



## Models

We have implemented the following methods (still updating):

- BPR (UAI'09): [Bayesian personalized ranking from implicit feedback](https://arxiv.org/pdf/1205.2618.pdf?source=post_page)
- GMF (WWW'17): [Neural Collaborative Filtering](https://dl.acm.org/doi/pdf/10.1145/3038912.3052569)
- Tensor (RecSys'10): [N-dimensional Tensor Factorization for Context-aware Collaborative Filtering](https://dl.acm.org/doi/pdf/10.1145/1864708.1864727)
- GRU4Rec (ICLR'16): [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/pdf/1511.06939)
- NARM (CIKM'17): [Neural Attentive Session-based Recommendation](https://dl.acm.org/doi/pdf/10.1145/3132847.3132926)
- CFKG (SIGIR'18): [Learning over Knowledge-Base Embeddings for Recommendation](https://arxiv.org/pdf/1803.06540)
- SLRC (WWW'19): [Modeling Item-specific Temporal Dynamics of Repeat Consumption for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/3308558.3313594)
- Chorus (SIGIR'20): Knowledge- and Time-aware Item Modeling for Sequential Recommendation



The table below lists the results of these models in `Grocery_and_Gourmet_Food` dataset (145.8k entries). Leave-one-out is applied to split data: the most recent interaction of each user for testing, the second recent item for validation, and the remaining items for training. We randomly sample 99 negative items for each test case to rank together with the ground-truth item.  These settings are all common in Top-K sequential recommendation.

| Model                                                        |  HR@5  | NDCG@5 | Time/iter |  Sequential  |  Knowledge   |  Time-aware  |
| :----------------------------------------------------------- | :----: | :----: | :-------: | :----------: | :----------: | :----------: |
| [BPR](https://github.com/THUwangcy/ReChorus/tree/master/src/models/BPR.py) | 0.3242 | 0.2223 |   2.5s    |              |              |              |
| [GMF](https://github.com/THUwangcy/ReChorus/tree/master/src/models/GMF.py) | 0.3178 | 0.2195 |   2.9s    |              |              |              |
| [Tensor](https://github.com/THUwangcy/ReChorus/tree/master/src/models/Tensor.py) | 0.3478 | 0.2623 |   3.2s    |              |              | ✔️️ |
| [GRU4Rec](https://github.com/THUwangcy/ReChorus/tree/master/src/models/GRU4Rec.py) | 0.3560 | 0.2532 |    11s    | ✔️️ |              |              |
| [NARM](https://github.com/THUwangcy/ReChorus/tree/master/src/models/NARM.py) | 0.3590 | 0.2573 |    22s    | ✔️️ |              |              |
| [CFKG](https://github.com/THUwangcy/ReChorus/tree/master/src/models/CFKG.py) | 0.4337 | 0.3081 |    11s    |              | ✔️️ |              |
| [SLRC'](https://github.com/THUwangcy/ReChorus/tree/master/src/models/SLRC.py) | 0.4513 | 0.3329 |   6.5s    | ✔️️ | ✔️️ | ✔️️ |
| [Chorus](https://github.com/THUwangcy/ReChorus/tree/master/src/models/Chorus.py) | 0.4754 | 0.3448 |   7.6s    | ✔️️ | ✔️️ | ✔️️ |



For fair comparison, the batch size is fixed to 256, and the embedding size is set to 64. We attempt to tune all the other hyper-parameters to obtain the best performance for each model (may be not optimal now, which will be updated if better score is achieved). Current commands are listed in [run.sh](https://github.com/THUwangcy/ReChorus/tree/master/src/run.sh).  We repeat each experiment 5 times with different random seeds and report the average score (see [exp.py](https://github.com/THUwangcy/ReChorus/tree/master/src/utils/exp.py)). All experiments are conducted with a single GTX-1080Ti GPU.



## Citation

This is also our public implementation for the paper:

*Chenyang Wang, Min Zhang, Weizhi Ma, Yiqun Liu, and Shaoping Ma. [Make It a Chorus: Knowledge- and Time-aware Item Modeling for Sequential Recommendation](). In SIGIR'20.*

**Please cite this paper if you use our codes. Thanks!**



Author: Chenyang Wang (THUwangcy@gmail.com)
