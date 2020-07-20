### Data visualization and transform
`explore_and_wrangling.ipynb` menghasilkan csv yang berisikan file csv yang terdiri atas field *subject, class, frame, rEar, lEar, mar, perclos, ms_rate, yw_rate*

### Data Preprocessing
`preprocessing.ipynb` menghasilkan data-data yang akan digunakan pada model *deep learning*. Untuk setiap model, ada 3 jenis data yaitu data dengan fitur utama, fitur agregasi, dan fitur gabungan.

### Training Model
* `dnn_model.ipynb` latih model DNN. data testing disimpan, model checkpoint disimpan, dan progress latih akan dicatat pada log file.
* `lstm_model.ipynb` latih model LSTM. data testing disimpan, model checkpoint disimpan, dan progress latih akan dicatat pada log file.

### Evaluation
1. Model Evaluation, `model_evaluation.ipynb`, evaluasi model dengan data testing. model terbaik dikonversi ke model tensorflow lite.
2. fps evaluation, `fps_test.csv` dan `fps_evaluation.ipynb`
3. condition evalauation, `condition.csv` dan `condition_evaluation.ipynb`
4. system evaluation, `system_evaluation.ipynb`
5. calibration evaluation, `calibration_evaluation.ipynb`

### Notes
input data untuk program ini didapatkan dari hasil program generasi aspek rasio seluruh video. Program tersebut dapat dilihat [disini](https://github.com/Arkoes07/aspect-ratio-generator)