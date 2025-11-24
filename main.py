"""
텍스트 클러스터링 API 서버
FastAPI를 사용하여 텍스트 클러스터링 기능을 제공합니다.
"""

import os
# jax 관련 에러 방지를 위한 환경 변수 설정
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import uvicorn

app = FastAPI(
    title="Text Clustering API",
    description="텍스트 클러스터링을 위한 API 서버",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 모델 로드 (서버 시작 시 한 번만 로드)
model = None

@app.on_event("startup")
async def load_model():
    """서버 시작 시 모델 로드"""
    global model
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully!")

# 요청/응답 모델
class ClusteringRequest(BaseModel):
    texts: List[str] = Field(..., description="클러스터링할 텍스트 리스트")
    n_clusters: int = Field(3, ge=2, le=50, description="클러스터 개수 (2-50)")
    model_name: Optional[str] = Field("all-MiniLM-L6-v2", description="사용할 모델 이름")
    return_visualization: bool = Field(False, description="시각화 데이터 반환 여부")

class ClusterResult(BaseModel):
    cluster_idx: int
    representative_text: str
    texts: List[str]
    distances: List[float]

class VisualizationData(BaseModel):
    points: List[List[float]]  # [x, y] 좌표
    labels: List[int]
    centroids: List[List[float]]
    representative_indices: List[int]

class ClusteringResponse(BaseModel):
    clusters: List[ClusterResult]
    labels: List[int]
    n_clusters: int
    visualization: Optional[VisualizationData] = None

@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "ok",
        "message": "Text Clustering API is running",
        "model": "all-MiniLM-L6-v2" if model else "not loaded"
    }

@app.get("/health")
async def health():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/v1/cluster", response_model=ClusteringResponse)
async def cluster_texts(request: ClusteringRequest):
    """
    텍스트 클러스터링 수행
    
    - texts: 클러스터링할 텍스트 리스트
    - n_clusters: 클러스터 개수
    - return_visualization: 시각화 데이터 반환 여부
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if len(request.texts) < request.n_clusters:
        raise HTTPException(
            status_code=400, 
            detail=f"Number of texts ({len(request.texts)}) must be >= n_clusters ({request.n_clusters})"
        )
    
    try:
        # 1. 텍스트 임베딩
        embeddings = model.encode(request.texts)
        
        # 2. K-Means 클러스터링
        kmeans = KMeans(
            n_clusters=request.n_clusters,
            init='k-means++',
            n_init=20,
            max_iter=500,
            tol=1e-5,
            verbose=0,
            random_state=42
        )
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_
        
        # 3. 군집별 중심 단어와 나머지 단어 거리 순서대로 출력
        clusters = []
        representative_indices = []
        
        for cluster_idx in range(request.n_clusters):
            cluster_mask = labels == cluster_idx
            cluster_vecs = embeddings[cluster_mask]
            cluster_texts = np.array(request.texts)[cluster_mask]
            centroid = centroids[cluster_idx]
            
            # centroid와 각 문장 거리 계산
            distances = np.linalg.norm(cluster_vecs - centroid, axis=1)
            
            # centroid와 가장 가까운 문장 선택 (중심 단어)
            rep_idx = np.argmin(distances)
            rep_text = cluster_texts[rep_idx]
            
            # 전체 텍스트 리스트에서의 인덱스 찾기
            cluster_text_indices = np.where(cluster_mask)[0]
            global_rep_idx = cluster_text_indices[rep_idx]
            representative_indices.append(int(global_rep_idx))
            
            # 거리 순서대로 정렬
            sorted_indices = np.argsort(distances)
            sorted_texts = cluster_texts[sorted_indices].tolist()
            sorted_distances = distances[sorted_indices].tolist()
            
            clusters.append(ClusterResult(
                cluster_idx=cluster_idx,
                representative_text=rep_text,
                texts=sorted_texts,
                distances=sorted_distances
            ))
        
        # 4. 시각화 데이터 생성 (요청 시)
        visualization = None
        if request.return_visualization:
            # 차원 축소 (2D)
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(embeddings)
            reduced_centroids = pca.transform(centroids)
            
            visualization = VisualizationData(
                points=reduced.tolist(),
                labels=labels.tolist(),
                centroids=reduced_centroids.tolist(),
                representative_indices=representative_indices
            )
        
        return ClusteringResponse(
            clusters=clusters,
            labels=labels.tolist(),
            n_clusters=request.n_clusters,
            visualization=visualization
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))  # 기본값 8002로 변경
    uvicorn.run(app, host="0.0.0.0", port=port)

