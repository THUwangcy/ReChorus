# Chorus
![model](log/_static/model.png)

This is our public implementation for the paper:

*Chenyang Wang, Min Zhang, Weizhi Ma, Yiqun Liu, and Shaoping Ma. [Make It a Chorus: Knowledge- and Time-aware Item Modeling for Sequential Recommendation](http://www.thuir.cn/group/~mzhang/publications/SIGIR2020Wangcy.pdf). In SIGIR'20.*

**Please cite this paper if you use our codes. Thanks!**

```
@inproceedings{wang2020make,
  title={Make it a chorus: knowledge-and time-aware item modeling for sequential recommendation},
  author={Wang, Chenyang and Zhang, Min and Ma, Weizhi and Liu, Yiqun and Ma, Shaoping},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={109--118},
  year={2020}
}
```



## Getting Started

1. Install [Anaconda](https://docs.conda.io/en/latest/miniconda.html) with Python >= 3.5
2. Clone the repository and install requirements

```bash
git clone -b SIGIR20 https://github.com/THUwangcy/ReChorus.git
```

3. Install requirements and step into the `src` folder

```bash
cd ReChorus
pip install -r requirements.txt
cd src
```

4. Pretrain the item and relation embeddings on the knowledge graph (stage 1)

```bash
python main.py --model_name Chorus --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --check_epoch 10 --epoch 50 --early_stop 0 --batch_size 512 --dataset Grocery_and_Gourmet_Food --stage 1
```

5. Train the recommendation task (stage 2)

```bash
python main.py --model_name Chorus --emb_size 64 --margin 1 --lr_scale 0.1 --lr 1e-3 --l2 0 --check_epoch 10 --dataset 'Grocery_and_Gourmet_Food' --base_method 'BPR' --stage 2
```



## Arguments

The main arguments of Chorus are listed below.

| Args        | Default | Help                                                         |
| ----------- | ------- | ------------------------------------------------------------ |
| emb_size    | 64      | Size of embedding vectors                                    |
| time_scalar | 864000  | Scalar of time intervals (in seconds)                        |
| stage       | 2       | Stage of training (1: KG pretrain, 2: recommendation)        |
| lr_scale    | 0.1     | Scale the learning rate for parameters in pretrained KG model. |
| base_method | BPR     | Basic method to generate recommendations: BPR / GMF          |
| history_max | 20      | Maximum length of history to consider                        |
| lr          | 1e-3    | Learning rate                                                |
| l2          | 0       | Weight decay of the optimizer                                |
| regenerate  | 0       | Whether to read data again and regenerate intermediate files |



## Performance

The table below lists the results of different models in `Grocery_and_Gourmet_Food` dataset (145.8k entries). Leave-one-out is applied to split data: the most recent interaction of each user for testing, the second recent item for validation, and the remaining items for training. We randomly sample 99 negative items for each test case to rank together with the ground-truth item.  These settings are all common in Top-K sequential recommendation.

| Model                                                        |  HR@5  | NDCG@5 | Time/iter |  Sequential  |  Knowledge   |  Time-aware  |
| :----------------------------------------------------------- | :----: | :----: | :-------: | :----------: | :----------: | :----------: |
| [BPR](https://github.com/THUwangcy/ReChorus/tree/SIGIR20/src/models/general/BPR.py) | 0.3242 | 0.2223 |   2.5s    |              |              |              |
| [GMF](https://github.com/THUwangcy/ReChorus/tree/SIGIR20/src/models/general/GMF.py) | 0.3178 | 0.2195 |   2.9s    |              |              |              |
| [Tensor](https://github.com/THUwangcy/ReChorus/tree/SIGIR20/src/models/general/Tensor.py) | 0.3478 | 0.2623 |   3.2s    |              |              | √ |
| [GRU4Rec](https://github.com/THUwangcy/ReChorus/tree/SIGIR20/src/models/sequential/GRU4Rec.py) | 0.3560 | 0.2532 |    11s    | √ |              |              |
| [NARM](https://github.com/THUwangcy/ReChorus/tree/SIGIR20/src/models/sequential/NARM.py) | 0.3590 | 0.2573 |    22s    | √ |              |              |
| [CFKG](https://github.com/THUwangcy/ReChorus/tree/SIGIR20/src/models/general/CFKG.py) | 0.4337 | 0.3081 |    11s    |              | √ |              |
| [SLRC'](https://github.com/THUwangcy/ReChorus/tree/SIGIR20/src/models/sequential/SLRC.py) | 0.4513 | 0.3329 |   6.5s    | √ | √ | √ |
| [Chorus](https://github.com/THUwangcy/ReChorus/tree/SIGIR20/src/models/sequential/Chorus.py) | 0.4754 | 0.3448 |   7.6s    | √ | √ | √ |

Current commands are listed in [run.sh](https://github.com/THUwangcy/ReChorus/tree/SIGIR20/src/run.sh).  We repeat each experiment 5 times with different random seeds and report the average score (see [exp.py](https://github.com/THUwangcy/ReChorus/tree/SIGIR20/src/utils/exp.py)). All experiments are conducted with a single GTX-1080Ti GPU.



## Contact

Chenyang Wang (THUwangcy@gmail.com)