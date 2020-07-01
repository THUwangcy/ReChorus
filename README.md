# ReChorus
This repository provides a general PyTorch framework for Top-K recommendation with implicit feedback, especially for research purpose. It decomposes the whole process into three modules:

- [Reader](https://github.com/THUwangcy/ReChorus/tree/master/src/helpers/BaseReader.py): read dataset into DataFrame and append necessary information to each instance
- [Runner](https://github.com/THUwangcy/ReChorus/tree/master/src/helpers/BaseRunner.py): control the training process and model evaluation
- [Model](https://github.com/THUwangcy/ReChorus/tree/master/src/models/BaseModel.py): define how to generate ranking scores and prepare batches



With this framework, we can easily compare different state-of-the-art models under the same experimental setting. The characteristics of our framework can be summarized as follows:

- **Light**: the framework is accomplished in less than a thousand lines of code, and a new model can be defined with dozens of lines
- **Efficient**: around 90% GPU utilization during training for deep models, and special implementations for the evaluation of Top-K recommendation
- **Neat**: clean codes with adequate comments, as well as beautiful training logs
- **Agile**: concentrate on your model design in a single model file and implement new models quickly
- **Flexible**: implement new readers or runners for different datasets and experimental settings, and each model can be assigned with specific helpers



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
python main.py --model_name BPR --emb_size 64 --lr 1e-3 --lr 1e-6 --dataset Grocery_and_Gourmet_Food
```

4. (optional) Run jupyter notebook in `data` folder to download and build new amazon datasets, or prepare your own datasets according to [README](https://github.com/THUwangcy/ReChorus/tree/master/data/README.md) in `data`
5. (optional) Implement your own models according to [README](https://github.com/THUwangcy/ReChorus/tree/master/src/README.md) in `src`



## Models

We have implemented the following methods (still updating):

- BPR (UAI'09): [Bayesian personalized ranking from implicit feedback](https://arxiv.org/pdf/1205.2618.pdf?source=post_page)
- NCF (WWW'17): [Neural Collaborative Filtering](https://dl.acm.org/doi/pdf/10.1145/3038912.3052569)
- Tensor (RecSys'10): [N-dimensional Tensor Factorization for Context-aware Collaborative Filtering](https://dl.acm.org/doi/pdf/10.1145/1864708.1864727)
- GRU4Rec (ICLR'16): [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/pdf/1511.06939)
- NARM (CIKM'17): [Neural Attentive Session-based Recommendation](https://dl.acm.org/doi/pdf/10.1145/3132847.3132926)
- SASRec (IEEE'18): [Self-attentive Sequential Recommendation](https://arxiv.org/pdf/1808.09781.pdf)
- TiSASRec (WSDM'20): [Time Interval Aware Self-Attention for Sequential Recommendation](https://dl.acm.org/doi/pdf/10.1145/3336191.3371786)
- CFKG (SIGIR'18): [Learning over Knowledge-Base Embeddings for Recommendation](https://arxiv.org/pdf/1803.06540)
- SLRC (WWW'19): [Modeling Item-specific Temporal Dynamics of Repeat Consumption](https://dl.acm.org/doi/pdf/10.1145/3308558.3313594)
- Chorus (SIGIR'20): [Knowledge- and Time-aware Item Modeling for Sequential Recommendation](http://www.thuir.cn/group/~mzhang/publications/SIGIR2020Wangcy.pdf)



The table below lists the results of these models in `Grocery_and_Gourmet_Food` dataset (145.8k entries). Leave-one-out is applied to split data: the most recent interaction of each user for testing, the second recent item for validation, and the remaining items for training. We randomly sample 99 negative items for each test case to rank together with the ground-truth item.  These settings are all common in Top-K sequential recommendation.

| Model                                                        |  HR@5  | NDCG@5 | Time/iter |  Sequential  |  Knowledge   |  Time-aware  |
| :----------------------------------------------------------- | :----: | :----: | :-------: | :----------: | :----------: | :----------: |
| [BPR](https://github.com/THUwangcy/ReChorus/tree/master/src/models/BPR.py) | 0.3554 | 0.2457 |   2.5s    |              |              |              |
| [NCF](https://github.com/THUwangcy/ReChorus/tree/master/src/models/NCF.py) | 0.3232 | 0.2234 |   3.4s   |              |              |              |
| [Tensor](https://github.com/THUwangcy/ReChorus/tree/master/src/models/Tensor.py) | 0.3548 | 0.2671 |   2.8s   |              |              | √ |
| [GRU4Rec](https://github.com/THUwangcy/ReChorus/tree/master/src/models/GRU4Rec.py) | 0.3646 | 0.2598 |    4.9s    | √ |              |              |
| [NARM](https://github.com/THUwangcy/ReChorus/tree/master/src/models/NARM.py) | 0.3621 | 0.2595 |    8.2s    | √ |              |              |
| [SASRec](https://github.com/THUwangcy/ReChorus/tree/master/src/models/SASRec.py) | 0.4247 | 0.3056 | 7.2s | √ | | |
| [TiSASRec](https://github.com/THUwangcy/ReChorus/tree/master/src/models/TiSASRec.py) | 0.4276 | 0.3074 | 39s | √ | | √ |
| [CFKG](https://github.com/THUwangcy/ReChorus/tree/master/src/models/CFKG.py) | 0.4239 | 0.3018 |    8.7s    |              | √ |              |
| [SLRC'](https://github.com/THUwangcy/ReChorus/tree/master/src/models/SLRC.py) | 0.4519 | 0.3335 |   4.3s   | √ | √ | √ |
| [Chorus](https://github.com/THUwangcy/ReChorus/tree/master/src/models/Chorus.py) | 0.4738 | 0.3448 |   4.9s   | √ | √ | √ |



For fair comparison, the batch size is fixed to 256, and the embedding size is set to 64. We strive to tune all the other hyper-parameters to obtain the best performance for each model (may be **not optimal now**, which will be updated if better scores are achieved). Current commands are listed in [run.sh](https://github.com/THUwangcy/ReChorus/tree/master/src/run.sh).  We repeat each experiment 5 times with different random seeds and report the average score (see [exp.py](https://github.com/THUwangcy/ReChorus/tree/master/src/utils/exp.py)). All experiments are conducted with a single GTX-1080Ti GPU.



## Citation

This is also our public implementation for the paper:

*Chenyang Wang, Min Zhang, Weizhi Ma, Yiqun Liu, and Shaoping Ma. [Make It a Chorus: Knowledge- and Time-aware Item Modeling for Sequential Recommendation](http://www.thuir.cn/group/~mzhang/publications/SIGIR2020Wangcy.pdf). In SIGIR'20.*

Checkout to `SIGIR20` branch to reproduce the results.

```
git clone -b SIGIR20 https://github.com/THUwangcy/ReChorus.git
```

**Please cite this paper if you use our codes. Thanks!**



Author: Chenyang Wang (THUwangcy@gmail.com)
