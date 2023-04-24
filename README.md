# DSC180B Quarter 2 Project: Prediction of Transaction Types using NLP analysis.
This project attempts to identify which phrases, among other features that are generated from given data from Petal are used to predict the categorization of transaction made by a user. Currently, the model in the files only represents the 'base' model in which the features are relatively basic, as well as the model that was used to generate the accuracy.

## Accessing Data
The data needs to be accessed through ``` https://drive.google.com/file/d/10JH-rN5c1cMXIEXgPkPGImWSGOzC19Kx/view?usp=share_link ```.
1) After downloading the data, replace the ``` testdata.parquet ``` file with the downloaded file.
2) In the run.py file, replace ```getData('data/testdata.parquet')``` with ```getData('data/DSC180B.parquet')```

## Viewing Results
To see the accuracy score of the model on the dataset run ``` python run.py test ```
