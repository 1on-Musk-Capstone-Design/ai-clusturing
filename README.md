# 텍스트 클러스터링 API

텍스트 클러스터링을 위한 독립적인 FastAPI 서버입니다.

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
python main.py
```

또는

```bash
uvicorn main:app --host 0.0.0.0 --port 8002
```

## API 엔드포인트

### 1. 헬스 체크
```
GET /
GET /health
```

### 2. 텍스트 클러스터링
```
POST /v1/cluster
```

**요청 본문:**
```json
{
  "texts": ["텍스트1", "텍스트2", "텍스트3", ...],
  "n_clusters": 3,
  "model_name": "all-MiniLM-L6-v2",
  "return_visualization": false
}
```

**응답:**
```json
{
  "clusters": [
    {
      "cluster_idx": 0,
      "representative_text": "대표 텍스트",
      "texts": ["텍스트1", "텍스트2", ...],
      "distances": [0.1, 0.2, ...]
    }
  ],
  "labels": [0, 1, 2, ...],
  "n_clusters": 3,
  "visualization": null
}
```

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8002/docs
- ReDoc: http://localhost:8002/redoc

## 환경 변수

- `PORT`: 서버 포트 (기본값: 8002)
- `MODEL_NAME`: 사용할 모델 이름 (기본값: all-MiniLM-L6-v2)

## 배포

Docker를 사용한 배포 예시:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

