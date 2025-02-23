# Tritoned Bert
В этом репо хранится шаблон для запуска модели BERT с помощью Trtiton Inference Server.

# Запуск

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

# Запросы

## Проверить состояние сервера
```bash
$ !curl -v is2.isa.ru:8000/v2/health/ready
```
## Сделать запрос модели
В данном примере показан запрос для антисуи-модели

```bash
$ !curl -X POST http://is2.isa.ru:8000/v2/models/ensemble_model/infer -d '{"inputs":[{"name":"text_input","shape":[1,1],"datatype":"BYTES","data":["помогите мне"]}]}'
```


# Полезные ссылки

1. https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide
2. https://dev.singularitynet.io/docs/products/AIMarketplace/ForConsumers/triton-instructions/