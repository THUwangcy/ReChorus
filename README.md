# Personalized Calibration Targets (PCT)

![model](./log/_static/model.png)

This is the source code for the paper:
*Chenyang Wang, Yankai Liu, Yuanqing Yu, Weizhi Ma, Min Zhang, Yiqun Liu, Haitao Zeng, Junlan Feng and Chao Deng. [Two-sided Calibration for Quality-aware Responsible Recommendation](). In RecSys'23.*

The main modifications compared to the original ReChorus framework center in these files:

- `helpers\`
  - `GuideReader.py`: group items according to quality annotations and calculate users' historical interest distributions
  - `GuideRunner.py`: implement new evaluation metrics for two-sided calibration and call different reranking methods
- `utils\`
  - `rerankers.py`: implement all the reranking methods, including the proposed PCT and baselines

## Arguments

| Args       | Default | Description                                                                            |
| ---------- | ------- | -------------------------------------------------------------------------------------- |
| rerank     | None    | Choose a reranking method: None, Calibrated, RegExp, TFROM, PCT                 |
| target_dist | Equal     | Target group exposure distribution: Equal, AvgEqual                        |
| personal   | 1       | Whether to use the personalized calibration targets solved by the PCT-Solver |
| lambda_    | 0.5     | Tradeoff hyperparameter in MMR                                                         |

## Dataset

The QK-article dataset after pre-processing (named QK-article-1M) can be found [here](https://drive.google.com/drive/folders/1w4tZXGdxKYSuJtsPALmamspvdywXF3ZK?usp=sharing).

The other dataset CMCC-Q is a private dataset due to privacy concerns.

## Getting Started
First, train a vanilla recommender (e.g., BPRMF on the QK-article-1M dataset):

```bash
python main.py --model_name=BPRMF --emb_size=64 --lr=1e-3 --l2=1e-6 --dataset=CMCC --metric=NDCG,COV,EXP,KL --topk=10 --reader_name=GuideReader --runner_name=GuideRunner --gpu=0
```

Then, run different reranking methods based on the saved recommendation model. An example command to run PCT:

```bash
python main.py --model_name=BPRMF --emb_size=64 --lr=1e-3 --l2=1e-6 --dataset=QK-article-1M --metric=NDCG,COV,EXP,KL --topk=10 --reader_name=GuideReader --runner_name=GuideRunner --train=0 --load=1 --rerank=PCT --target_dist=Equal --personal=1 --lambda_=0.1 --gpu=0
```

## Citation

```
TBD
```

## Contact

Chenyang Wang ([THUwangcy@gmail.com](mailto:THUwangcy@gmail.com))