![logo](./log/_static/logo.png)
---

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

ReChorus is a general PyTorch framework for Top-K recommendation with implicit feedback, especially for research purpose. It aims to provide a fair benchmark to compare different state-of-the-art algorithms. We hope this can partially alleviate the problem that different papers adopt non-comparable experimental settings, so as to form a "Chorus" of recommendation algorithms. 

This framework is especially suitable for researchers to compare algorithms under the same experimental setting, and newcomers to get familiar with classical methods. The characteristics of our framework can be summarized as follows:

- **Swift**: concentrate on your model design ***in a single file*** and implement new models quickly.

- **Easy**: the framework is accomplished in ***less than a thousand lines of code***, which is easy to use with clean codes and adequate comments.

- **Efficient**: multi-thread batch preparation, special implementations for the evaluation, and around 90% GPU utilization during training for deep models.

- **Flexible**: implement new readers or runners for different datasets and experimental settings, and each model can be assigned with specific helpers.

## Structre

Generally, ReChorus decomposes the whole process into three modules:

- [Reader](https://github.com/THUwangcy/ReChorus/tree/master/src/helpers/BaseReader.py): read dataset into DataFrame and append necessary information to each instance
- [Runner](https://github.com/THUwangcy/ReChorus/tree/master/src/helpers/BaseRunner.py): control the training process and model evaluation
- [Model](https://github.com/THUwangcy/ReChorus/tree/master/src/models/BaseModel.py): define how to generate ranking scores and prepare batches

![logo](./log/_static/module.png)

## Getting Started

1. Install [Anaconda](https://docs.conda.io/en/latest/miniconda.html) with Python >= 3.5
2. Clone the repository

```bash
git clone https://github.com/THUwangcy/ReChorus.git
```

3. Install requirements and step into the `src` folder

```bash
cd ReChorus
pip install -r requirements.txt
cd src
```

4. Run model with the build-in dataset

```bash
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food
```

5. (optional) Run jupyter notebook in `data` folder to download and build new datasets, or prepare your own datasets according to [Guideline](https://github.com/THUwangcy/ReChorus/tree/master/data/README.md) in `data`

6. (optional) Implement your own models according to [Guideline](https://github.com/THUwangcy/ReChorus/tree/master/src/README.md) in `src`

## Arguments

The main arguments are listed below.

| Args            | Default   | Description                                                             |
| --------------- | --------- | ----------------------------------------------------------------------- |
| model_name      | 'BPRMF'   | The name of the model class.                                            |
| lr              | 1e-3      | Learning rate.                                                          |
| l2              | 0         | Weight decay in optimizer.                                              |
| test_all        | 0         | Wheter to rank all the items during evaluation.                         |
| metrics         | 'NDCG,HR' | The list of evaluation metrics (seperated by comma).                    |
| topk            | '5,10,20' | The list of K in evaluation metrics (seperated by comma).               |
| num_workers     | 5         | Number of processes when preparing batches.                             |
| batch_size      | 256       | Batch size during training.                                             |
| eval_batch_size | 256       | Batch size during inference.                                            |
| load            | 0         | Whether to load model checkpoint and continue to train.                 |
| train           | 1         | Wheter to perform model training.                                       |
| regenerate      | 0         | Wheter to regenerate intermediate files.                                |
| random_seed     | 0         | Random seed of everything.                                              |
| gpu             | '0'       | The visible GPU device (pass an empty string '' to only use CPU).       |
| buffer          | 1         | Whether to buffer batches for dev/test.                                 |
| history_max     | 20        | The maximum length of history for sequential models.                    |
| num_neg         | 1         | The number of negative items for each training instance.                |
| test_epoch      | -1        | Print test set metrics every test_epoch during training (-1: no print). |

## Models

We have implemented the following methods (still updating):

**General Recommender**

- [Bayesian personalized ranking from implicit feedback](https://arxiv.org/pdf/1205.2618.pdf?source=post_page) (BPRMF [UAI'09])
- [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf?source=post_page---------------------------) (NeuMF [WWW'17])
- [Learning over Knowledge-Base Embeddings for Recommendation](https://arxiv.org/pdf/1803.06540.pdf) (CFKG [SIGIR'18])
- [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://dl.acm.org/doi/abs/10.1145/3397271.3401063?casa_token=mMzWDMq9WxQAAAAA%3AsUQEeXtBSLjctZa7qfyOO25nOBqdHWW8ukbjZUeOmcprZcmF3QBWKBtdICrMDidOy8MJ28n3Z1zy5g) (LightGCN [SIGIR'20])
- [Bootstrapping User and Item Representations for One-Class Collaborative Filtering](https://arxiv.org/pdf/2105.06323) (BUIR [SIGIR'21])
- [Towards Representation Alignment and Uniformity in Collaborative Filtering](https://arxiv.org/pdf/2206.12811.pdf) (DirectAU [KDD'22])

**Sequential Recommender**

- [Factorizing Personalized Markov Chains for Next-Basket Recommendation](https://dl.acm.org/doi/pdf/10.1145/1772690.1772773?casa_token=hhM2wEArOQEAAAAA:r_vhs7X8VE0rJ7FF5aZ4i-P-z1mSlBABdw5O9p0cuOahTOQ8D3FVyX6_d58sbQFiV1q1vdVHB-wKqw) (FPMC [WWW'10])
- [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/pdf/1511.06939) (GRU4Rec [ICLR'16])
- [Neural Attentive Session-based Recommendation](https://arxiv.org/pdf/1711.04725.pdf) (NARM [CIKM'17])
- [Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding](https://arxiv.org/pdf/1809.07426) (Caser [WSDM'18])
- [Self-attentive Sequential Recommendation](https://arxiv.org/pdf/1808.09781.pdf) (SASRec [IEEE'18])
- [Modeling Item-specific Temporal Dynamics of Repeat Consumption for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/3308558.3313594) (SLRC [WWW'19])
- [Time Interval Aware Self-Attention for Sequential Recommendation](https://dl.acm.org/doi/pdf/10.1145/3336191.3371786) (TiSASRec [WSDM'20])
- [Make It a Chorus: Knowledge- and Time-aware Item Modeling for Sequential Recommendation](http://www.thuir.cn/group/~mzhang/publications/SIGIR2020Wangcy.pdf) (Chorus [SIGIR'20])
- [Controllable Multi-Interest Framework for Recommendation](https://dl.acm.org/doi/pdf/10.1145/3394486.3403344?casa_token=r35exDCLzSsAAAAA:hbdvRtwvH7LlbllHH7gITV_mpA5hYnAFXcpT2bW8MnbK7Gta50E60xNhC6KoQtY6AGOHaEVsK_GRVQ) (ComiRec [KDD'20])
- [Towards Dynamic User Intention: Temporal Evolutionary Effects of Item Relations in Sequential Recommendation](https://chenchongthu.github.io/files/TOIS-KDA-wcy.pdf) (KDA [TOIS'21])
- [Sequential Recommendation with Multiple Contrast Signals](https://dl.acm.org/doi/pdf/10.1145/3522673) (ContraRec [TOIS'22])
- [Target Interest Distillation for Multi-Interest Recommendation]() (TiMiRec [CIKM'22])

The table below lists the results of these models in `Grocery_and_Gourmet_Food` dataset (151.3k entries). Leave-one-out is applied to split data: the most recent interaction of each user for testing, the second recent item for validation, and the remaining items for training. We randomly sample 99 negative items for each test case to rank together with the ground-truth item (also support ranking over all the items with `--test_all 1`).

| Model                                                                                             | HR@5   | NDCG@5 | Time/iter | Sequential | Knowledge | Time-aware |
|:------------------------------------------------------------------------------------------------- |:------:|:------:|:---------:|:----------:|:---------:|:----------:|
| [MostPop](https://github.com/THUwangcy/ReChorus/tree/master/src/models/general/POP.py)            | 0.2065 | 0.1301 | -         |            |           |            |
| [BPRMF](https://github.com/THUwangcy/ReChorus/tree/master/src/models/general/BPR.py)              | 0.3549 | 0.2486 | 2.5s      |            |           |            |
| [NeuMF](https://github.com/THUwangcy/ReChorus/tree/master/src/models/general/NCF.py)              | 0.3237 | 0.2221 | 3.4s      |            |           |            |
| [LightGCN](https://github.com/THUwangcy/ReChorus/tree/master/src/models/general/LightGCN.py)      | 0.3705 | 0.2564 | 6.1s      |            |           |            |
| [BUIR](https://github.com/THUwangcy/ReChorus/tree/master/src/models/general/BUIR.py)              | 0.3701 | 0.2567 | 3.3s      |            |           |            |
| [DirectAU](https://github.com/THUwangcy/ReChorus/tree/master/src/models/general/DirectAU.py)      | 0.3911 | 0.2779 | 3.3s      |            |           |            |
| [FPMC](https://github.com/THUwangcy/ReChorus/tree/master/src/models/sequential/FPMC.py)           | 0.3594 | 0.2785 | 3.4s      | √          |           |            |
| [GRU4Rec](https://github.com/THUwangcy/ReChorus/tree/master/src/models/sequential/GRU4Rec.py)     | 0.3659 | 0.2614 | 4.9s      | √          |           |            |
| [NARM](https://github.com/THUwangcy/ReChorus/tree/master/src/models/sequential/NARM.py)           | 0.3650 | 0.2617 | 7.5s      | √          |           |            |
| [Caser](https://github.com/THUwangcy/ReChorus/tree/master/src/models/sequential/Caser.py)         | 0.3526 | 0.2499 | 7.8s      | √          |           |            |
| [SASRec](https://github.com/THUwangcy/ReChorus/tree/master/src/models/sequential/SASRec.py)       | 0.3917 | 0.2942 | 5.5s      | √          |           |            |
| [ComiRec](https://github.com/THUwangcy/ReChorus/tree/master/src/models/sequential/ComiRec.py)     | 0.3753 | 0.2675 | 4.5s      | √          |           |            |
| [TiMiRec+](https://github.com/THUwangcy/ReChorus/tree/master/src/models/sequential/TiMiRec.py)    | 0.4020 | 0.3016 | 8.8s      | √          |           |            |
| [ContraRec](https://github.com/THUwangcy/ReChorus/tree/master/src/models/sequential/ContraRec.py) | 0.4251 | 0.3285 | 5.6s      | √          |           |            |
| [TiSASRec](https://github.com/THUwangcy/ReChorus/tree/master/src/models/sequential/TiSASRec.py)   | 0.3949 | 0.2945 | 7.6s      | √          |           | √          |
| [CFKG](https://github.com/THUwangcy/ReChorus/tree/master/src/models/general/CFKG.py)              | 0.4199 | 0.2984 | 8.7s      |            | √         |            |
| [SLRC+](https://github.com/THUwangcy/ReChorus/tree/master/src/models/sequential/SLRCPlus.py)      | 0.4376 | 0.3263 | 4.3s      | √          | √         | √          |
| [Chorus](https://github.com/THUwangcy/ReChorus/tree/master/src/models/sequential/Chorus.py)       | 0.4668 | 0.3414 | 4.9s      | √          | √         | √          |
| [KDA](https://github.com/THUwangcy/ReChorus/tree/master/src/models/sequential/KDA.py)             | 0.5191 | 0.3901 | 9.9s      | √          | √         | √          |

For fair comparison, the embedding size is set to 64. We strive to tune all the other hyper-parameters to obtain the best performance for each model (may be not optimal now, which will be updated if better scores are achieved). Current commands are listed in [run.sh](https://github.com/THUwangcy/ReChorus/tree/master/src/run.sh).  We repeat each experiment 5 times with different random seeds and report the average score (see [exp.py](https://github.com/THUwangcy/ReChorus/tree/master/src/exp.py)). All experiments are conducted with a single GTX-1080Ti GPU.

## Citation

**If you find ReChorus is helpful to your research, please cite either of the following papers. Thanks!**

```
@inproceedings{wang2020make,
  title={Make it a chorus: knowledge-and time-aware item modeling for sequential recommendation},
  author={Wang, Chenyang and Zhang, Min and Ma, Weizhi and Liu, Yiqun and Ma, Shaoping},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={109--118},
  year={2020}
}
@article{王晨阳2021rechorus,
  title={ReChorus: 一个综合, 高效, 易扩展的轻量级推荐算法框架},
  author={王晨阳 and 任一 and 马为之 and 张敏 and 刘奕群 and 马少平},
  journal={软件学报},
  volume={33},
  number={4},
  pages={0--0},
  year={2021}
}
```

This is also our public implementation for the following papers (codes and datasets to reproduce the results can be found at corresponding branch):

- *Chenyang Wang, Min Zhang, Weizhi Ma, Yiqun Liu, and Shaoping Ma. [Make It a Chorus: Knowledge- and Time-aware Item Modeling for Sequential Recommendation](http://www.thuir.cn/group/~mzhang/publications/SIGIR2020Wangcy.pdf). In SIGIR'20.*

```bash
git clone -b SIGIR20 https://github.com/THUwangcy/ReChorus.git
```

- *Chenyang Wang, Weizhi Ma, Min Zhang, Chong Chen, Yiqun Liu, and Shaoping Ma. [Towards Dynamic User Intention: Temporal Evolutionary Effects of Item Relations in Sequential Recommendation](https://chenchongthu.github.io/files/TOIS-KDA-wcy.pdf). In TOIS'21.*

```bash
git clone -b TOIS21 https://github.com/THUwangcy/ReChorus.git
```

- *Chenyang Wang, Weizhi Ma, Chong, Chen, Min Zhang, Yiqun Liu, and Shaoping Ma. [Sequential Recommendation with Multiple Contrast Signals](https://dl.acm.org/doi/pdf/10.1145/3522673). In TOIS'22.*

```bash
git clone -b TOIS22 https://github.com/THUwangcy/ReChorus.git
```

- *Chenyang Wang, Zhefan Wang, Yankai Liu, Yang Ge, Weizhi Ma, Min Zhang, Yiqun Liu, Junlan Feng, Chao Deng, and Shaoping Ma. [Target Interest Distillation for Multi-Interest Recommendation](). In CIKM'22.*

```bash
git clone -b CIKM22 https://github.com/THUwangcy/ReChorus.git
```

## Contact

Chenyang Wang (THUwangcy@gmail.com)

<!-- MARKDOWN LINKS & IMAGES -->

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
