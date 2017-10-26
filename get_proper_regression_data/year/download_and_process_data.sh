mkdir data
cd data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip
unzip YearPredictionMSD.txt.zip
cp split_data_train_test.py data
mv data
python split_data_train_test.py

