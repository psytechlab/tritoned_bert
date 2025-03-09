# Tritoned Bert
В этом репо хранится шаблон для докер-образа модели классификации на основе BERT с помощью Trtiton Inference Server в формате ONNX.


# Предварительное наполнение содержимым

1. Поместить модель в формате onnx в папку `model_repository/model_onnx/1`.
2. Поместить файлы токенизатора в папку `model_repository/text_preprocessing/1`.
3. Поместить файл `id2label2.json` в папку `model_repository/post_processing/1`.

Файл `id2label.json` представляет собой словарь, где ключами являются индексы классов, а значениями — текстовые названия соответствующих классов.

# Сборка докер-контейнера с моделью

Положив все необходимые файлы, необходимо выполнить команду в коневой директории проекта:
```
$ docker build -t triton_cls_model:v1 .
```

Для запуска необходимо выполнить команду:
```
docker run --shm-size=256m -p8000:8000 -p8001:8001 -p8002:8002 triton_cls_model:v1
```

Для корректной работы контейнера рекомендуетяс установить размер совместной памяти в 256 Мб.

# Пошаговый запуск

1. Запустить докер с тритоном 
```bash
$ ./run_triton.sh
```
2. Внутри докера установить библиотеку `transformers`
```bash
pip install transformers
```
3. Запустить сервер командой
```
tritonserver --model-repository=/models
```

# Базовые запросы

## Проверить состояние сервера
```bash
$ curl -v is2.isa.ru:8000/v2/health/ready
```
## Сделать запрос модели
В данном примере показан запрос для антисуи-модели

```bash
$ curl -X POST http://is2.isa.ru:8000/v2/models/ensemble_model/infer -d '{"inputs":[{"name":"text_input","shape":[1,1],"datatype":"BYTES","data":["помогите мне"]}]}'
```

# Полезные ссылки

1. https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide
2. https://dev.singularitynet.io/docs/products/AIMarketplace/ForConsumers/triton-instructions/
3. https://github.com/kserve/kserve/tree/master/docs/predict-api/v2
4. https://github.com/triton-inference-server/server/tree/main/docs/protocol