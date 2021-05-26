# pytorch-based playground
-----

Реализовать с помощью pytorch аналог https://playground.tensorflow.org/
Фичи:
1) Генерация данных (три варианта), датасет и даталоадер (можно добавить подгрузку своих csv)
2) Класс генерирующий сеть по заданной архитектуре (архитектура задается массивом в main.py)
3) Класс обучающий сеть
4) Сделать визаулизацию (картинки промежуточных результатов обучения + .gif с полным процессом)

1) data
    * data generator  
    * preprocessing  (standard scaler, outlier detection)
    * train/test split 
    * dataset/dataloader 
    * features
    
2) nn architecture
    * n of layers
    * n of neurons
    * activations 
    
3) trainer 
    * optimizer 
    * learning rate ...
    * n_epochs
    * batch size 
    * loss
    * metrics 
    * save/load 
    
4) visualization 
    * loss
    * scatter plots
    * tensorboard
-----
## Запуск
Все параметры модели задаются в файле main.py + исполняется он же

## Results

Результаты обучения хранятся в папке /runs/{runname}: .gif файл, файл с состоянием модели, файл с логами для tensorboard и папка /pictures с промежуточными изображениями