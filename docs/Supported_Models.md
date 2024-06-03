## Supported Models

We have implemented the following methods (still updating):

**General Recommenders**

- [Bayesian personalized ranking from implicit feedback](https://arxiv.org/pdf/1205.2618.pdf?source=post_page) (BPRMF [UAI'09])
- [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf?source=post_page---------------------------) (NeuMF [WWW'17])
- [Learning over Knowledge-Base Embeddings for Recommendation](https://arxiv.org/pdf/1803.06540.pdf) (CFKG [SIGIR'18])
- [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://dl.acm.org/doi/abs/10.1145/3397271.3401063?casa_token=mMzWDMq9WxQAAAAA%3AsUQEeXtBSLjctZa7qfyOO25nOBqdHWW8ukbjZUeOmcprZcmF3QBWKBtdICrMDidOy8MJ28n3Z1zy5g) (LightGCN [SIGIR'20])
- [Bootstrapping User and Item Representations for One-Class Collaborative Filtering](https://arxiv.org/pdf/2105.06323) (BUIR [SIGIR'21])
- [Towards Representation Alignment and Uniformity in Collaborative Filtering](https://arxiv.org/pdf/2206.12811.pdf) (DirectAU [KDD'22])

**Sequential Recommenders**

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

**Context-aware Recommenders**

General Context-aware Recommenders
- [Factorization Machines](https://ieeexplore.ieee.org/document/5694074) (FM [ICDM'10])
- [Wide {\&} Deep Learning for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454) (WideDeep [DLRS@Recsys'16])
- [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617) (AFM [IJCAI'17])
- [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247) (DeepFM [IJCAI'17])
- [Deep & Cross Network for Ad Click Predictions](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754) (DCN [KDD'17])
- [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921) (AutoInt [CIKM'18])
- [xdeepfm: Combining explicit and implicit feature interactions for recommender systems](https://arxiv.org/pdf/1803.05170.pdf%E2%80%8B%E2%80%8B) (xDeepFM [KDD'18])
- [DCN v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems](https://arxiv.org/pdf/2008.13535) (DCNv2 [WWW'21])
- [Looking at CTR Prediction Again: Is Attention All You Need?](https://arxiv.org/pdf/2105.05563) (SAM [SIGIR'21])
- [FinalMLP: an enhanced two-stream MLP model for CTR prediction](https://ojs.aaai.org/index.php/AAAI/article/download/25577/25349) (FinalMLP [AAAI'23])

Sequential Context-aware Recommenders
- [Deep interest network for click-through rate prediction](https://arxiv.org/pdf/1706.06978) (DIN [KDD'18])
- [End-to-end user behavior retrieval in click-through rateprediction model](https://arxiv.org/pdf/2108.04468) (ETA [CoRR'18])
- [Deep interest evolution network for click-through rate prediction](https://aaai.org/ojs/index.php/AAAI/article/view/4545/4423) (DIEN [AAAI'19])
- [CAN: feature co-action network for click-through rate prediction](https://dl.acm.org/doi/abs/10.1145/3488560.3498435) (CAN [WSDM'22])
- [Sampling is all you need on modeling long-term user behaviors for CTR prediction](https://arxiv.org/pdf/2205.10249) (SDIM[CIKM'22])

**Impression-based Re-ranking Models**
- [Personalized Re-ranking for Recommendatio](https://arxiv.org/pdf/1904.06813) (PRM [RecSys'19])
- [SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval](https://arxiv.org/pdf/1912.05891) (SIGIR [SIGIR'20])
- [Multi-Level Interaction Reranking with User Behavior History](https://arxiv.org/pdf/2204.09370) (MIR[SIGIR'22])