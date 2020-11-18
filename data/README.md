# Dataset

We use public [Amazon dataset](http://jmcauley.ucsd.edu/data/amazon/links.html) (*Grocery_and_Gourmet_Food* category, 5-core version with metadata) as our build-in dataset.  Other datasets after preprocessing can be downloaded online:

* Amazon Electronics: [Google Drive](https://drive.google.com/drive/folders/1F2DSMOwHQgQRmuKMjN24FcqyVh-RrlFe?usp=sharing)

* RecSys2017: [Google Drive](https://drive.google.com/drive/folders/1rhUQwTYVai4kt54GAjdFUU4CVryNPwC9?usp=sharing)

To run codes on customized datasets, we describe the format of required files below:



**train.csv**

- Format: `user_id \t item_id \t time`
- All ids begin from 1 (0 is reserved for NaN), and the followings are the same.
- Need to be sorted in time-ascending order when running sequential models.



**test.csv & dev.csv**

- Format: `user_id \t item_id \t time \t neg_items`
- The last column is the list of negative items in terms of the ground-truth item.
- The number of negative items need to be the same for a specific set, but it can be different between dev and test sets.

![dev/test data format](../log/_static/format_test.png)



**item_meta.csv** (optional)

- Format: `item_id \t i_<attribute> \t ... \t r_<relation> \t ...`
- Optional, only needed for some of the models (CFKG, SLRC+, Chorus, KDA).
- `i_<attribute>` is the attribute of an item, such as category, brand and so on. The header should start with `i_` and the values need to be discrete and finite.
- `r_<relation>` is the relations between items, and its value is a list of items (can be empty []). Assume `item_id` is `i`, if `j` appears in `r_<relation>`, then `(i, relation, j)` holds in the knowledge graph. Note that the corresponding header here must start with "r_" to be distinguished from attributes.

![meta data format](../log/_static/format_meta.png)
