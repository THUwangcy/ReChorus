# Personalized Calibration Targets (PCT)

This is the source code for the paper "Two-sided Calibration for Quality-aware Responsible Recommendation".

The main modifications compared to the original ReChorus framework center in these files:

- `helpers\`
  - `GuideReader.py`: group items according to quality annotations and calculate users' historical interest distributions
  - `GuideRunner.py`: implement new evaluation metrics for two-sided calibration and call different reranking methods
- `utils\`
  - `rerankers.py`: implement all the reranking methods, including the proposed PCT and baselines

The additional hyper-parameters are explained as follows:

| Args       | Default | Description                                                                            |
| ---------- | ------- | -------------------------------------------------------------------------------------- |
| rerank     | None    | Choose a reranking method: None, Boost, Calibrated, RegExp, TFROM, PCT                 |
| exp_policy | par     | Target group exposure distribution: par (Equal), cat (AvgEqual)                        |
| personal   | 1       | Whether to use the personalized target exposure distributions solved by the PCT-Solver |
| lambda_    | 0.5     | Tradeoff hyperparameter in MMR                                                         |

We first train a vanilla recommender (e.g., SASRec on the CMCC-Q dataset):

```bash
python main.py --model_name=SASRec --emb_size=64 --lr=1e-4 --l2=1e-6 --dataset=CMCC --topk=10 --reader_name=GuideReader --runner_name=GuideRunner --gpu=0
```

Then, we run different reranking methods based on the saved recommendation model. An example command to run PCT:

```bash
python main.py --model_name=SASRec --emb_size=64 --lr=1e-4 --l2=1e-6 --dataset=CMCC --metric=NDCG,COV,EXP,KL --topk=10 --reader_name=GuideReader --runner_name=GuideRunner --train=0 --load=1 --rerank=PCT --exp_policy=par --personal=1 --lambda_=0.1
```
