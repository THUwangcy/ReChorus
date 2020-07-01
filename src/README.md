# Source Code	

`main.py` serves as the entrance of our framework, and there are three main packages. 

### Structure

- `helpers\`
  - `BaseReader.py`: read dataset csv into DataFrame and append necessary information (e.g. interaction history)
  - `BaseRunner.py`: control the training and evaluation process of a model
- `models\`
  - `BaseModel.py`: basic model class and dataset class, with some common functions of a model
  - `...`: customize models inherited from *BaseModel*
- `utils\`
  - `component.py`: common components for model definition (e.g. attention)
  - `exp.py`: repeat experiments in *run.sh* and save averaged results to csv 
  - `utils.py`: some utils functions
- `main.py`: main entrance, connect all the modules
- `run.sh`: running commands for each model



### Define a New Model

Generally we should define a new class inheriting *BaseModel*, as well as the inner class *Dataset*. The following functions need to be implement at least:

```python
class NewModel(BaseModel):
    reader = 'BaseReader'  # assign a reader class, BaseReader by default
    runner = 'BaseRunner'  # assign a runner class, BaseRunner by default

    def _define_params(self):
        # define parameters in the model

    def forward(self, feed_dict):
        # generate prediction (ranking score according to tensors in feed_dict)
        item_id = feed_dict['item_id']  # [batch_size, -1]
        user_id = feed_dict['user_id']  # [batch_size]
        prediction = (...)
        return prediction.view(feed_dict['batch_size'], -1)
    
    class Dataset(BaseModel.Dataset):
        # construct feed_dict for a single instance (called by __getitem__)
        # will be collated to a integrated feed dict for each batch
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            feed_dict['user_id'] = self.data['user_id'][index]
            return feed_dict
```



If the model definition is more complicated, you can inherit other functions in *BaseModel* (e.g. `loss`, `customize_parameters`) and *Dataset* (e.g. `_prepare`, `negative_sampling`), which needs a deeper understanding about [BaseModel.py](https://github.com/THUwangcy/ReChorus/tree/master/src/models/BaseModel.py) and [BaseRunner.py](https://github.com/THUwangcy/ReChorus/tree/master/src/helpers/BaseRunner.py). You can also implement a new runner class to accommodate different experimental settings.

