# Module 6: NLP & LSTM Embeddings in PostgreSQL

Демонстрация полного пайплайна: от обучения LSTM на PyTorch до семантического поиска в PostgreSQL (pgvector).

## Что внутри
1. 01_train.py — обучение LSTM предсказывать следующее слово на текстовом корпусе.
2. model.py — архитектура WordLSTM (Embedding -> LSTM -> FC) и функция извлечения 32-мерного вектора encode().
3. 05_pg_load.py — извлечение векторов для скользящих окон текста и загрузка в PostgreSQL.
4. 06_pg_search.py — косинусный поиск по базе (оператор <=>).

## Результаты
Модель успешно обучилась (loss упал с 4.13 до 2.77) и выучила контекст. 
Поиск по запросу "каналы пропускают ионы натрия" находит фрагмент про "избирательный транспорт ионов натрия" со сходством 0.90.

## Запуск

    python3 -m venv .venv
    source .venv/bin/activate
    pip install torch numpy psycopg[binary] pgvector

    python 01_train.py
    export PG_DSN='dbname=postgres user=postgres password=postgres host=127.0.0.1'
    python 05_pg_load.py
    python 06_pg_search.py --text "Туристы посетили музей" --top 3

