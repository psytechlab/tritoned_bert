# Tritoned Bert
В этом репо хранится шаблон для докер-образа, который содержит модель классификации на основе BERT в формате ONNX. Базовым инструментом инференса служит Trtiton Inference Server. Расчитано на класс `AutoModelForSequenceClassification` из библиотеки Hugging Face.

Важно: на данный момент сервис будет работать на CPU.

# Установка окрежния

1. Установите зависимости из `requirements.txt`.
```
$ pip install -r requirements.txt
```

2. Установите `apt-get install gettext-base cmake build-essential protobuf-compiler python3.12-dev`.

## Известные ошибки

Если при установки Python-зависимсотей возникает ошибка

```
/usr/bin/ld: /usr/lib/x86_64-linux-gnu/libprotobuf.a(arena.o): relocation R_X86_64_TPOFF32 against hidden symbol `_ZN6google8protobuf8internal15ThreadSafeArena13thread_cache_E' can not be used when making a shared object
      /usr/bin/ld: failed to set dynamic section sizes: bad value
      collect2: error: ld returned 1 exit status
      gmake[2]: *** [CMakeFiles/onnx_cpp2py_export.dir/build.make:101: onnx_cpp2py_export.cpython-312-x86_64-linux-gnu.so] Error 1
      gmake[1]: *** [CMakeFiles/Makefile2:240: CMakeFiles/onnx_cpp2py_export.dir/all] Error 2
      gmake: *** [Makefile:136: all] Error 2
```

то перед запуском установки зависимостей выполните команды

```
$ apt-get install libprotobuf-dev protobuf-compiler
$ export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
```
Ошибка обнаружена yна6.14.0-36-generic #36~24.04.1-Ubuntu

# Автоматическая сборка докер-образа по шаблону

Выполните скрипт `make_triton_image.sh`, передав параметры в следующем порядке (если есть значение по умолчанию, то можно не указывать):
  1. model_path — путь до модели.
  2. tokenizer_path — путь до токенайзера.
  3. id2label_path — путь до файла с маппингом классов.
  4. model_name — финальное имя модели, которое будет фигурировать в API и в названии образа. По умолчанию `ensemble_model`.
  5. container_tag — тег для докер-образа. По умолчанию `latest`.
  6. max_batch_size — максимальный размер батча, который будет обрабатываться Тритоном за раз.По умолчанию `4`.
  
## Пояснения к скриптам

* cleanup.sh — выполняет очистку шаблона от артефактов и приводит его в изначальный вид.
* process_template.sh — продуцирует конфигурационные файлы из шаблонов (имеют расшиение `.template`).
* make_triton_images.sh — основной скрипт, выполняюющий подготовку и сборку докер-образа.

# Пример команды запуска докер-контейнера

Для запуска необходимо выполнить команду:
```
docker run --shm-size=256m -p8000:8000 -p8001:8001 -p8002:8002 triton_cls_model:v1
```
Для корректной работы контейнера рекомендуется установить размер совместной памяти в 256 Мб. Пояснения к портам:
* 8000 — запросы по HTTP;
* 8001 — запросы по gRPC (см. библиотеку `tritonclient`);
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

## Запуск сервера на основе базового докер-образа

1. Запустить докер с тритоном на основе официального базового образа Triton Inference Service
```bash
$ docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:22.12-py3
```

Если хотите задействовать GPU, важно правильно подобрать базовый контейнер, который будет согласован с вашими драйверами для видеокарты и CUDA. Выбрать подходящий образ можно по [этой ссылке](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

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
$ curl -v 127.0.0.1:8000/v2/health/ready
```

## Сделать запрос модели
В данном примере показан запрос для модели, которая имеет общее имя "ensemble_model"

```bash
$ curl -X POST http://127.0.0.1:8000/v2/models/ensemble_model/infer -d '{"inputs":[{"name":"text_input","shape":[1,1],"datatype":"BYTES","data":["помогите мне"]}]}'
```

# Конвертация модели в ONNX

Для быстрой конвертации модели ONNX так, чтобы всё подошло по формату, можно использовать скрипт `utils/convert_to_onnx.py`. Для его работы необходимо передать пути до самой модели и ее токенизатора, сохраненной по протоколу библиотеки transformers (напимер, через класс `Trainer`), а также файл с маппингом классов `id2label.json`. 

Скрипт позволяет взаимодействовать не только с локальными путями, но и с другими хранилищами, где могут лежать модели и токенизаторы. Для токенизатора и модели надо указывать пути отдельно, что подразумевает независимость их нахождения. На данный момент реализовано:
- local — путь до локальной директории
- clearml — путь представляет собой id объекта в Clearml.
- hf — путь в Hugging face Hub, который имеет формат "user/model_name".

Скрипт легко расширить на другие хранилища, как, например, WandB.

# Полезные ссылки

1. https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide
2. https://dev.singularitynet.io/docs/products/AIMarketplace/ForConsumers/triton-instructions/
3. https://github.com/kserve/kserve/tree/master/docs/predict-api/v2
4. https://github.com/triton-inference-server/server/tree/main/docs/protocol
5. https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
