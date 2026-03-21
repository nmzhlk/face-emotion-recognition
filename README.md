# Распознавание лиц и эмоций

### Создание Docker-контейнера
```
docker run -d --name oracle-xe -p 1521:1521 -p 5500:5500 -e ORACLE_PASSWORD=admin gvenzl/oracle-xe
```

### Запуск контейнера
```
docker start oracle-xe
```

### Создание и активация виртуального окружения
```
python -m venv venv
venv/Scripts/activate
```

### Установка зависимостей
```
pip install -r requirements.txt
```

### Запуск сервера
```
uvicorn app.main:app --reload
```


docker network connect face-net face_api_container