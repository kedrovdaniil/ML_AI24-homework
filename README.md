# Демонстрация
В директории `/service/screenshots` доступна запись с демонстрацией работы сервиса.\

**Запуск сервиса локально:**
```shell
git clone git@github.com:kedrovdaniil/ML_AI24-homework.git
cd ./ML_AI24-homework/service
python3 -m venv myenv
pip install -r requirements.txt
fastapi dev
```

# Резюме к результатам работы
В рамках домашнего задания я выполнил почти полный цикл разработки (без деплоя сервиса) модели машинного обучения для задачи предсказания стоимости подержанных автомобилей начиная от очистки данных и обучения модели до создания сервиса на базе FastAPI для получения прогноза по данным с помощью API. 

## Основные этапы работы:
1. **Предобработка данных:**
   - Провёл анализ качества данных: проверил пропуски, дубликаты и типы данных.
   - Заполнил пропуски медианными значениями и преобразовал числовые признаки в соответствующий формат.
   - Удалил признаки, которые сложно обработать или незначительно влияют на целевую переменную, например, `torque`.

2. **Анализ данных и визуализация:**
   - Построил тепловую карту корреляции для поиска наиболее связанных признаков.
   - Определил ключевые зависимости, которые могут быть полезны для обучения модели.

3. **Построение моделей:**
   - Обучил базовую линейную регрессию для начального анализа.
   - Улучшил модель с помощью стандартизации данных и регуляризации (Lasso, ElasticNet, Ridge).
   - Использовал кросс-валидацию для подбора оптимальных параметров.

4. **Работа с категориальными признаками:**
   - Закодировал категориальные признаки с помощью OneHot-кодирования.
   - Устранил мультиколлинеарность путём исключения избыточных признаков.

5. **Оценка качества:**
   - Рассчитал метрики $R^2$ и MSE для каждой модели.
   - Реализовал кастомную бизнес-метрику для оценки модели с точки зрения бизнеса.
   - Наилучшие результаты показала модель ElasticNet, обеспечив показатель в 24.1% точных прогнозов по бизнес-метрике.

6. **Сохранение модели:**
   - Сохранил обученную модель в файл формата `.pkl`.
   - Написал `pipeline` в шагах которого добавил всю туже самую логику предобработки данных, которую я применял к датасету до этого момента, обучил модель ElasticNet на тех же данных и сохранил его в формате `.joblib`. Модель ElasticNet была выбрана в качестве основной, поскольку она показала наилучший результат кастомной бизнес-метрики, которая важна для бизнеса, несмотря на то, что другие модели показывали более хорошие результаты в других метриках (R2, MSE).

7. **Создание сервиса с REST API**
   - Создал 2 метода API, которые позволяют получить прогноз по запросу к API. API принимает формат JSON или CSV и при помощи сохраннёного ранее пайплайна отдаёт результат предсказания цены авто.

## Логи работ
1. **Предобработка данных.** Мы начали с анализа набора данных:
   Проверили качество данных - выявили пропуски и дубликаты.
   Построили отчёт по датасету с помощью библиотеки ydata-profiling.
   Заполнили пропуски медианными значениями, рассчитанными на основе тренировочного набора.
   Преобразовали числовые признаки, содержащие текст (`mileage`, `engine`, `max_power`), в числовой формат.
   Удалили признак `torque`.

2. **Построение базовой модели.**
   Обучили линейную регрессию на вещественных признаках для прогнозирования цены подержанных автомобилей.
   Метрики качества модели:
   - $R^2$ на тестовых данных: 0.5941
   - $R^2$ на тренировочных данных: 0.5923
   - MSE на тестовых данных: 233297548204.61
   - MSE на тренировочных данных: 116873067751.52\
     _Модель показала средние результаты без признаков переобучения._

3. **Стандартизация признаков.**
   Мы предположили, что стандартизация (с использованием `StandardScaler`) может улучшить качество модели. Однако значения метрик остались неизменными, так как линейная регрессия компенсировала масштабирование пересчётом коэффициентов.
   Стандартизация позволила интерпретировать важность признаков, наиболее значимым оказался `max_power` (максимальная мощность двигателя), а наименее значимым — `seats` (количество мест).

4. **Регуляризация и подбор параметров.**
   Для улучшения модели мы применили регуляризацию:\
   _Lasso-регрессия:_ метрики качества не изменились. Все признаки остались значимыми, коэффициенты не занулились.\
   _Метрики Lasso:_
   - Оптимальные параметры: `alpha=100`\
   - $R^2$ на тренировочных данных: 0.5722
     _ElasticNet-регрессия:_ с использованием кросс-валидации мы подобрали оптимальные параметры: `alpha=1`, `l1_ratio=0.9`.\
     _Метрики ElasticNet:_
   - $R^2$ на тестовых данных: 0.5682
   - MSE на тестовых данных: 245892672007.80

5. **Добавление категориальных признаков.**
   Закодировали категориальные признаки (`fuel`, `seller_type`, `transmission`, `owner`) и `seats` методом OneHot-кодирования с устранением мультиколлинеарности.
   Обучили Ridge-регрессию, подобрав параметр `alpha` с помощью кросс-валидации (10 фолдов).
   Качество модели незначительно улучшилось:
   - Лучший alpha: {'alpha': 100}
   - $R^2$ на тестовых данных: 0.6034

6. **Бизнес-метрика.**
   Мы реализовали кастомную метрику, оценивающую долю прогнозов, отклоняющихся от реальных значений не более чем на 10%. Результаты:
   - Linear Regression (без стандартизации): 22.7%.
   - Linear Regression (с стандартизацией): 22.7%.
   - Lasso (регуляризация): 22.7%.
   - ElasticNet (регуляризация и кросс-валидация): 24.1%.

## Метрики обученных моделей
1) **LinearRegression на вещественных признаках:**\
R^2 на тренировочных данных: 0.5923\
MSE на тренировочных данных: 116873067751.52\
R^2 на тестовых данных: 0.5941\
MSE на тестовых данных: 233297548204.61

2) **LinearRegression на всех признаках:**\
R^2 на тренировочных данных: 0.5923\
MSE на тренировочных данных: 116873067751.52\
R^2 на тестовых данных: 0.5941\
MSE на тестовых данных: 233297548204.60

3) **Lasso с L1-регуляризацией:**\
R^2 на тренировочных данных: 0.5923\
MSE на тренировочных данных: 116873067761.64\
R^2 на тестовых данных: 0.5941\
MSE на тестовых данных: 233298219170.76

4) **ElasticNet с лучшими параметрами {'alpha': 1, 'l1_ratio': 0.9}:**\
R^2 на тренировочных данных: 0.5884\
MSE на тренировочных данных: 0.57\
R^2 на тестовых данных: 0.5722\
MSE на тестовых данных: 245892672007.80

5) **Ridge с лучшими параметрами {'alpha': 100}:**\
R^2 кросс-валидация по 10-ти фолдам: 0.6034023533777344

## Итоги:
- Модель ElasticNet оказалась самой эффективной для решения задач бизнеса.
- Результаты работы показывают, что текущие данные хорошо подходят для линейных моделей, однако дальнейшие улучшения могут быть достигнуты с помощью нелинейных методов.
- Возможно, можно улучшить показатели линейных моделей обработав признак `name` вместо того, чтобы его удалять. Вероятно это сильно поднимет качество прогнозов, поскольку предположительно цена сильно зависит от марки авто.
- Результаты первых трёх моделей почти не отличаются, потому что в данных, скорее всего, уже есть хорошие связи, которые линейная регрессия смогла уловить.
Регуляризация (например, в Lasso) почти не изменила модель, так как признаки и так достаточно важны.

В целом, задание позволило мне пройти через ключевые этапы построения и оценки моделей, а также получить опыт работы с реальными данными. Работа была весьма интересной и довольно объёмной.
