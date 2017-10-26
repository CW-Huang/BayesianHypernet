mkdir data
cd data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip
unzip YearPredictionMSD.txt.zip
mv YearPredictionMSD.txt data.txt
python ../reformat_data.py
cp ../split_data_train_test.py data
python split_data_train_test.py

