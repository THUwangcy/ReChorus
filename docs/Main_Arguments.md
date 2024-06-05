## Main Arguments

The main arguments are listed below.

| Args            | Default   | Description                |
| --------------- | --------- | -------------------------- |
| model_name      | 'BPRMF'   | The name of the model class.                                            |
| model_mode      | ''        | The task mode for the model to implement.                               |
| lr              | 1e-3      | Learning rate.                                                          |
| l2              | 0         | Weight decay in optimizer.                                              |
| test_all        | 0         | Wheter to rank all the items during evaluation. (only work in Top-K recommendation tasks)|
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