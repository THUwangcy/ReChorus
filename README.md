# ReChorus
Our implementation of Chorus

```
git clone https://github.com/THUwangcy/ReChorus.git
cd ReChorus
pip install -r requirements.txt

cd data
./download_data.sh Grocery_and_Gourmet_Food
python preprocess.py --dataset Grocery_and_Gourmet_Food
(open preprocess.ipynb, run cells to observe data and build dataset)

cd ../
python main.py --model_name BPR --emb_size 64 --lr 1e-3 --dataset Grocery_and_Gourmet_Food
```

