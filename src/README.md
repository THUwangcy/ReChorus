## Source Code	

`main.py` serves as the entrance of our framework, and there are three main packages. We first describe the structure of the code and then introduce how to define a new model.

## Structure

- `helpers\`
  - `BaseLoader.py`: load dataset csv into DataFrame and append necessary information (e.g. interaction history)
  - `BaseRunner.py`: control the training and evaluating process of the model
- `models\`
  - `BaseModel.py`: basic model class, implement some common functions of a model
  - `...`: customize models inherited from *BaseModel*
- `utils\`
  - `exp.py`: repeat experiments in *run.sh* and save averaged results to csv 
  - `utils.py`: some utils function
- `main.py`: main entrance, connect all the modules
- `run.sh`: running commands for each model



## Define a New Model

Generally we should define a new class inheriting *BaseModel*, and implement at least these major functions:

```python
class NewModel(BaseModel):
    loader = 'BaseLoader'  # assign a loader class, BaseLoader by default
    runner = 'BaseRunner'  # assign a runner class, BaseRunner by default
  
    @staticmethod
    def parse_model_args(parser, model_name='NewModel'):
        # add some customized arguments in the model
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, args, corpus):
        # initialize some member variables
        # calculate some values based on corpus (loader object)
        BaseModel.__init__(self, model_path=args.model_path)

    def _define_params(self):
        # define parameters in the model

    def forward(self, feed_dict):
        # generate prediction (ranking score according to tensors in feed_dict)
        self.check_list = []
        batch_size = feed_dict['batch_size']
				prediction = ...
        out_dict = {'prediction': prediction.view(batch_size, -1), 'check': self.check_list}
        return out_dict

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
      	"""
        Generate feed dict of the given data, which will be fed into forward function.
        :param corpus: Loader object
        :param data: DataFrame in corpus.data_df (may be shuffled)
        :param batch_start: batch start index
        :param batch_size: batch size
        :param phase: 'train', 'dev' or 'test'
        """
        batch_end = min(len(data), batch_start + batch_size)
        feed_dict = {'batch_size': batch_end - batch_start}
        return feed_dict
```



If the training procedure is more complicated, you can inherit other functions in *BaseModel* (e.g. `loss`, `get_neg_items`,  `customize_parameters`...), which needs a better understanding about [BaseModel.py](https://github.com/THUwangcy/ReChorus/tree/master/src/models/BaseModel.py) and [BaseRunner.py](https://github.com/THUwangcy/ReChorus/tree/master/src/helpers/BaseRunner.py). You can also reimplement a new runner class to adapt to different experimental settings.

