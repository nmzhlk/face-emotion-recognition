# Распознавание лиц и эмоций

### 1. Быстрый запуск через Docker Compose
```
docker-compose up -d --build
```

### 2. Инициализация структуры БД
```
docker exec -it face_recognition_api python -m app.api.db.models
```

### 3. Локальная разработка (без Docker для API)

**Создание и активация виртуального окружения:**
```
python -m venv venv
# Для Windows:
venv/Scripts/activate
# Для Linux/macOS:
source venv/bin/activate
```

**Установка зависимостей:**
```
pip install -r requirements.txt
```

**Запуск сервера:**
```
uvicorn app.main:app --reload
```

### 4. Проверка результатов в базе данных
**Количество лиц в базе:**
```
docker exec -it face_recognition_api python -c "from app.api.db.db import get_connection; conn=get_connection(); cur=conn.cursor(); cur.execute('SELECT COUNT(*) FROM FACE_DETECTIONS'); print(f'\nВсего лиц в базе: {cur.fetchone()[0]}'); cur.close(); conn.close()"
```
**Детальный список:**
```
docker exec -it face_recognition_api python -c "from app.api.db.db import get_connection; conn=get_connection(); cur=conn.cursor(); cur.execute('SELECT u.ORIGINAL_FILENAME, f.EMOTION_CODE, f.CONFIDENCE FROM FACE_DETECTIONS f JOIN UPLOADED_IMAGES u ON f.SOURCE_PHOTO_ID = u.UUID'); [print(f'Файл: {r[0]} | Эмоция: {r[1]} | Уверенность: {r[2]:.2f}') for r in cur.fetchall()]; cur.close(); conn.close()"
```

### 5. Схема данных и связи
* **USERS**: Основная таблица пользователей (связь по UUID).
* **UPLOADED_IMAGES**: Метаданные загруженных файлов.
* **FACE_DETECTIONS**: Результаты работы нейросети. 

### 6. Переменные окружения (.env)
Для работы API требуются следующие параметры:

* `ORA_USER`=system
* `ORA_PASS`=admin
* `ORA_DSN`=db:1521/xepdb1

Сервис доступен по адресу: http://localhost:8000
