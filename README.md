# Knowledge-aware Dynamic Attention (KDA)

![model](./log/_static/model.png)
---

This is our public implementation for the paper:

*Chenyang Wang, Weizhi Ma, Min Zhang, Chong Chen, Yiqun Liu, and Shaoping Ma. [Towards Dynamic User Intention: Temporal Evolutionary Effects of Item Relations in Sequential Recommendation](). In TOIS'21.*

**Please cite this paper if you use our codes. Thanks!**



## Getting Started

1. Install [Anaconda](https://docs.conda.io/en/latest/miniconda.html) with Python >= 3.5
2. Clone the repository and install requirements

```bash
git clone -b TOIS21 https://github.com/THUwangcy/ReChorus.git
```

3. Install requirements and step into the `src` folder

```bash
cd ReChorus
pip install -r requirements.txt
cd src
```

4. Run model on the build-in dataset

```bash
python main.py --model_name KDA --emb_size 64 --include_attr 1 --freq_rand 0 --lr 1e-3 --l2 1e-6 --num_heads 4 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
```



## Arguments

The main arguments of KDA are listed below.

| Args           | Default | Help                                                         |
| -------------- | ------- | ------------------------------------------------------------ |
| emb_size       | 64      | Size of embedding vectors                                    |
| gamma          | -1      | Coefficient of KG loss (auto-determined if set to -1)        |
| include_attr   | 1       | Whether to include attribute-based item relations            |
| include_val    | 1       | Whether to enhance relation representations with relation values |
| neg_head_p     | 0.5     | Probability of sampling negative head entity during training |
| t_scalar       | 60      | Scalar of time intervals (in seconds)                        |
| n_dft          | 64      | Point of DFT                                                 |
| freq_rand      | 0       | Whether to randomly initialize frequency embeddings          |
| num_layers     | 1       | Number of self-attention layers                              |
| num_heads      | 1       | Number of self-attention heads                               |
| attention_size | 10      | Size of attention pooling hidden space                       |
| pooling        | average | Method of pooling: average / max / attention                 |
| history_max    | 20      | Maximum length of history to consider                        |
| lr             | 1e-3    | Learning rate                                                |
| l2             | 0       | Weight decay of the optimizer                                |
| regenerate     | 0       | Whether to read data again and regenerate intermediate files |
| num_workers    | 5       | Number of processors when preparing batches                  |



## Performance

The table below lists the results of these models in `Grocery_and_Gourmet_Food` dataset (145.8k entries). Leave-one-out is applied to split data: the most recent interaction of each user for testing, the second recent item for validation, and the remaining items for training. We randomly sample 99 negative items for each test case to rank together with the ground-truth item.  These settings are all common in Top-K sequential recommendation.

| Model                                                        |  HR@5  | NDCG@5 | Time/iter |  Sequential  |  Knowledge   |  Time-aware  |
| :----------------------------------------------------------- | :----: | :----: | :-------: | :----------: | :----------: | :----------: |
| [BPR](https://github.com/THUwangcy/ReChorus/tree/master/src/models/BPR.py) | 0.3574 | 0.2480 |   2.5s    |              |              |              |
| [NCF](https://github.com/THUwangcy/ReChorus/tree/master/src/models/NCF.py) | 0.3248 | 0.2235 |   3.4s   |              |              |              |
| [Tensor](https://github.com/THUwangcy/ReChorus/tree/master/src/models/Tensor.py) | 0.3547 | 0.2670 |   2.8s   |              |              | √ |
| [GRU4Rec](https://github.com/THUwangcy/ReChorus/tree/master/src/models/GRU4Rec.py) | 0.3664 | 0.2597 |    4.9s    | √ |              |              |
| [NARM](https://github.com/THUwangcy/ReChorus/tree/master/src/models/NARM.py) | 0.3621 | 0.2586 |    8.2s    | √ |              |              |
| [SASRec](https://github.com/THUwangcy/ReChorus/tree/master/src/models/SASRec.py) | 0.3888 | 0.2923 | 7.2s | √ | | |
| [TiSASRec](https://github.com/THUwangcy/ReChorus/tree/master/src/models/TiSASRec.py) | 0.3916 | 0.2922 | 35.7s | √ | | √ |
| [CFKG](https://github.com/THUwangcy/ReChorus/tree/master/src/models/CFKG.py) | 0.4228 | 0.3010 |    8.7s    |              | √ |              |
| [SLRC+](https://github.com/THUwangcy/ReChorus/tree/master/src/models/SLRCPlus.py) | 0.4514 | 0.3329 |   4.3s   | √ | √ | √ |
| [Chorus](https://github.com/THUwangcy/ReChorus/tree/master/src/models/Chorus.py) | 0.4739 | 0.3443 |   4.9s   | √ | √ | √ |
| [KDA](https://github.com/THUwangcy/ReChorus/tree/master/src/models/KDA.py) | 0.5174 | 0.3876 | 9.9s | √ | √ | √ |

We repeat each experiment 5 times with different random seeds and report the average score (see [exp.py](https://github.com/THUwangcy/ReChorus/tree/master/src/exp.py)). All experiments are conducted with a single GTX-1080Ti GPU.



Note that we have reorganized the codes for easier usage, and the final results may be a little different from that in the paper (while the order of baselines holds). Besides, we find we can achieve even higher results in `Amazon Electronics` after fine-grained parameter tuning (NDCG@5: 0.42 → 0.45), which outperforms baselines by a large margin. Current commands are listed in [run.sh](https://github.com/THUwangcy/ReChorus/tree/master/src/run.sh). 



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
