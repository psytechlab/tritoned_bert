# Tritoned Bert
В этом репо хранится шаблон для докер-образа модели классификации на основе BERT с помощью Trtiton Inference Server в формате ONNX. Проетстировано с классо `AutoModelForSequenceClassification` из библиотеки Hugging Face.

# Автоматическая сборка докер-образа по шаблону

1. Убедитесь, что у вас установлены зависимости из `requirements.txt`.
2. Установите `apt-get install gettext-base`.
2. Выполните скрипт `make_triton_image.sh`, передав параметры в следующем порядке:
  1. model_path — путь до модели.
  2. tokenizer_path — путь до токенайзера.
  3. id2label_path — путь до файла с маппингом классов.
  4. num_classes — количество классов в модели.
  5. model_name — финальное имя модели, которое в том числе будет фигурировать в API.
  
## Пояснения к скриптам

* cleanup.sh — выполняет очистку шаблона от артефактов и приводит его в изначальный вид.
* process_template.sh — продуцирует конфигурационные файлы из шаблонов (имеют расшиение `.template`).
* make_triton_images.sh — основной скрипт, выполняюющий подготовку и сборку докер-образа.

# Пример команды запуска докер-контейнера

Для запуска необходимо выполнить команду:
```
docker run --shm-size=256m -p8000:8000 -p8001:8001 -p8002:8002 triton_cls_model:v1
```
Для корректной работы контейнера рекомендуетяс установить размер совместной памяти в 256 Мб. Пояснения к портам:
* 8000 — запросы по HTTP
* 8001 — запросы по gRPC (см. библиотеку `tritonclient`)
* 8002 — метрики.

# Ручная сборка

Для начала необходимо переключиться в git на коммит по тегу pretemplate.
```
$ git checkout pretemplate
```

## Предварительное наполнение содержимым

1. Поместить модель в формате onnx в папку `model_repository/model_onnx/1`.
2. Поместить файлы токенизатора в папку `model_repository/text_preprocessing/1/tokenizer`.
3. Поместить файл `id2label2.json` в папку `model_repository/post_processing/1`.

Файл `id2label.json` представляет собой словарь, где ключами являются индексы классов, а значениями — текстовые названия соответствующих классов.

## Сборка докер-образа с моделью

Положив все необходимые файлы, необходимо выполнить команду в коневой директории проекта:
```
$ docker build -t triton_cls_model:v1 .
```

## Запуск сервера на основе базового докер-обраща

1. Запустить докер с тритоном на основе официального базового образа Triton Inference Service
```bash
$ docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:22.12-py3
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
В данном примере показан запрос для модели, которая имеет общее имя "ensemble_model"

```bash
$ curl -X POST http://is2.isa.ru:8000/v2/models/ensemble_model/infer -d '{"inputs":[{"name":"text_input","shape":[1,1],"datatype":"BYTES","data":["помогите мне"]}]}'
```

# Конвертация модели в ONNX

Для быстрой конвертации модели ONNX так, чтобы всё подошло по формату, можно использовать скрипт `utils/convert_to_onnx.py`. Для его работы необходимо передать пути до самой модели и ее токенизатора, сохраненной по протоколу библиотеки transformers, а также файл с маппингом классов `id2label.json`. 

Скрипт позволяет взаимодействовать не только с путями локальными, но и с другими хранилищями, где модели и токенизаторы могут лежать. Пути могут различаться. На данный момент реализовано:
- local — путь до локальной директории
- learml — путь представляет собой id объекта в Clearml.
- hf — путь в Hugging face Hub, который имеет формат "user/model_name".

Скрипт легко расширить на другие хранилища, как, например, WandB.

# Полезные ссылки

1. https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide
2. https://dev.singularitynet.io/docs/products/AIMarketplace/ForConsumers/triton-instructions/
3. https://github.com/kserve/kserve/tree/master/docs/predict-api/v2
4. https://github.com/triton-inference-server/server/tree/main/docs/protocol