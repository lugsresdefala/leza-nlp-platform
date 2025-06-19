# Plano de Implementação - Novas Features LEXA

## 1. Autenticação com Google

### 1.1 Configuração Inicial
```python
# requirements.txt
google-auth==2.22.0
google-auth-oauthlib==1.0.0
```

### 1.2 Implementação
```python
# components/auth.py
from google.oauth2 import id_token
from google.auth.transport import requests
from config import GOOGLE_CLIENT_ID

def setup_google_auth():
    """Configuração da autenticação Google."""
    flow = Flow.from_client_secrets_file(
        'client_secrets.json',
        scopes=['openid', 'email', 'profile']
    )
    return flow

def verify_google_token(token):
    """Verifica o token do Google."""
    try:
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), GOOGLE_CLIENT_ID)
        return idinfo
    except ValueError:
        return None

def handle_google_callback():
    """Manipula o callback do Google OAuth."""
    flow = setup_google_auth()
    authorization_response = st.experimental_get_query_params()
    flow.fetch_token(authorization_response=authorization_response)
    credentials = flow.credentials
    return credentials
```

### 1.3 Interface
```python
# pages/login.py
def render_login():
    st.title("Login LEXA")
    
    if st.button("Login com Google"):
        flow = setup_google_auth()
        authorization_url, state = flow.authorization_url()
        st.session_state.oauth_state = state
        st.experimental_set_query_params(
            next=authorization_url
        )
```

## 2. Motor de Análise Textual

### 2.1 Arquitetura
```python
# backend/analysis_engine.py
from typing import Dict, List
import spacy
from transformers import pipeline

class TextAnalysisEngine:
    def __init__(self):
        self.nlp = spacy.load('pt_core_news_lg')
        self.sentiment_analyzer = pipeline(
            'sentiment-analysis',
            model='neuralmind/bert-base-portuguese-cased'
        )
        
    def analyze_text(self, text: str) -> Dict:
        """Análise completa do texto."""
        doc = self.nlp(text)
        
        return {
            'cohesion': self._analyze_cohesion(doc),
            'coherence': self._analyze_coherence(doc),
            'complexity': self._analyze_complexity(doc),
            'sentiment': self._analyze_sentiment(text),
            'metrics': self._calculate_metrics(doc)
        }
        
    def _analyze_cohesion(self, doc) -> Dict:
        """Análise de coesão textual."""
        # Implementar análise de conectivos
        # Análise de referências anafóricas
        pass
        
    def _analyze_coherence(self, doc) -> Dict:
        """Análise de coerência."""
        # Implementar análise de estrutura lógica
        # Análise de consistência argumentativa
        pass
        
    def _analyze_complexity(self, doc) -> Dict:
        """Análise de complexidade linguística."""
        # Implementar índices de legibilidade
        # Análise de diversidade lexical
        pass
        
    def _analyze_sentiment(self, text: str) -> Dict:
        """Análise de sentimento."""
        return self.sentiment_analyzer(text)[0]
        
    def _calculate_metrics(self, doc) -> Dict:
        """Cálculo de métricas quantitativas."""
        return {
            'sentence_count': len(list(doc.sents)),
            'word_count': len(doc),
            'avg_word_length': sum(len(token.text) for token in doc) / len(doc),
            'unique_words': len(set(token.text.lower() for token in doc))
        }
```

## 3. API REST para Análise de Texto

### 3.1 Especificação OpenAPI
```yaml
# api/openapi.yaml
openapi: 3.0.0
info:
  title: LEXA API
  version: 1.0.0
  description: API de Análise Textual LEXA

paths:
  /analyze:
    post:
      summary: Analisa um texto
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                text:
                  type: string
                  description: Texto para análise
                options:
                  type: object
                  properties:
                    metrics:
                      type: array
                      items:
                        type: string
                        enum: [cohesion, coherence, complexity]
      responses:
        '200':
          description: Análise concluída
          content:
            application/json:
              schema:
                type: object
                properties:
                  cohesion:
                    type: object
                  coherence:
                    type: object
                  complexity:
                    type: object
                  metrics:
                    type: object
```

### 3.2 Implementação FastAPI
```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="LEXA API")

class AnalysisRequest(BaseModel):
    text: str
    options: Optional[List[str]] = None

@app.post("/analyze")
async def analyze_text(request: AnalysisRequest):
    try:
        engine = TextAnalysisEngine()
        result = engine.analyze_text(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 4. Dashboard de Métricas

### 4.1 Estrutura
```python
# pages/dashboard.py
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

def render_performance_dashboard():
    st.title("Dashboard de Performance")
    
    # Filtros de tempo
    date_range = st.date_input(
        "Período",
        [datetime.now() - timedelta(days=30), datetime.now()]
    )
    
    # Métricas principais
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Textos Analisados", "1.2k", "+12%")
    with col2:
        st.metric("Tempo Médio", "1.5s", "-8%")
    with col3:
        st.metric("Taxa de Erro", "0.1%", "-2%")
    
    # Gráficos
    fig_requests = px.line(
        get_request_data(),
        x="timestamp",
        y="requests",
        title="Requisições por Hora"
    )
    st.plotly_chart(fig_requests)
    
    fig_latency = px.box(
        get_latency_data(),
        x="endpoint",
        y="latency",
        title="Latência por Endpoint"
    )
    st.plotly_chart(fig_latency)
```

## 5. Otimização de Queries

### 5.1 Implementação de Cache
```python
# utils/cache.py
from functools import lru_cache
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(key: str, ttl: int = 3600):
    """Decorator para cache de resultados."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = f"{key}:{args}:{kwargs}"
            
            # Tenta obter do cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Executa função e armazena resultado
            result = func(*args, **kwargs)
            redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result)
            )
            return result
        return wrapper
    return decorator
```

### 5.2 Otimização de Consultas
```python
# database/queries.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

engine = create_engine('postgresql://user:pass@localhost/lexa')
Session = sessionmaker(bind=engine)

@contextmanager
def get_session():
    """Gerenciador de contexto para sessões do banco."""
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def get_analysis_results(user_id: int, start_date: datetime):
    """Query otimizada para resultados de análise."""
    with get_session() as session:
        query = text("""
            SELECT 
                date_trunc('hour', created_at) as hour,
                COUNT(*) as count,
                AVG(processing_time) as avg_time
            FROM analysis_results
            WHERE user_id = :user_id
                AND created_at >= :start_date
            GROUP BY 1
            ORDER BY 1 DESC
        """)
        
        return session.execute(
            query,
            {'user_id': user_id, 'start_date': start_date}
        ).fetchall()
```

## 6. Cronograma de Implementação

### 6.1 Fase 1 (Semana 1-2)
- Configuração da autenticação Google
- Implementação do fluxo de login
- Testes de integração

### 6.2 Fase 2 (Semana 3-4)
- Desenvolvimento do motor de análise textual
- Implementação de métricas básicas
- Testes unitários

### 6.3 Fase 3 (Semana 5-6)
- Desenvolvimento da API REST
- Documentação OpenAPI
- Testes de endpoints

### 6.4 Fase 4 (Semana 7-8)
- Implementação do dashboard
- Configuração de métricas
- Otimização de queries

## 7. Requisitos Técnicos

### 7.1 Dependências
```
google-auth==2.22.0
google-auth-oauthlib==1.0.0
spacy==3.5.0
transformers==4.30.0
fastapi==0.95.0
redis==4.5.4
sqlalchemy==2.0.0
plotly==5.13.0
```

### 7.2 Infraestrutura
- Servidor Python 3.9+
- PostgreSQL 13+
- Redis 6+
- GPU para processamento de modelos (opcional)

## 8. Considerações de Segurança

### 8.1 Autenticação
- Implementar rate limiting
- Validação de tokens JWT
- Proteção contra CSRF
- Sanitização de inputs

### 8.2 API
- Autenticação via API key
- Rate limiting por usuário
- Validação de payload
- Logging de requisições

## 9. Monitoramento

### 9.1 Métricas a serem monitoradas
- Tempo de resposta da API
- Taxa de erro
- Uso de memória
- Carga de CPU
- Cache hit ratio
- Latência de queries

### 9.2 Alertas
- Latência alta (>2s)
- Taxa de erro >1%
- Uso de memória >80%
- Falhas de autenticação

## 10. Testes

### 10.1 Testes Unitários
```python
# tests/test_analysis.py
def test_text_analysis():
    engine = TextAnalysisEngine()
    result = engine.analyze_text("Texto de exemplo")
    
    assert 'cohesion' in result
    assert 'coherence' in result
    assert 'complexity' in result
    assert 'metrics' in result

# tests/test_api.py
def test_api_endpoint():
    response = client.post(
        "/analyze",
        json={"text": "Texto de exemplo"}
    )
    
    assert response.status_code == 200
    assert 'cohesion' in response.json()
```

### 10.2 Testes de Integração
```python
# tests/integration/test_google_auth.py
def test_google_auth_flow():
    response = client.get("/auth/google")
    assert response.status_code == 302
    
    # Simula callback do Google
    response = client.get("/auth/google/callback")
    assert response.status_code == 200
    assert 'access_token' in response.json()
```

## 11. Documentação

### 11.1 Swagger UI
- Disponível em `/docs`
- Documentação interativa
- Exemplos de requisições
- Schemas de resposta

### 11.2 Postman Collection
- Endpoints documentados
- Variáveis de ambiente
- Exemplos de uso
- Testes automatizados

## 12. Próximos Passos

### 12.1 Imediatos
1. Setup do ambiente de desenvolvimento
2. Configuração do Google OAuth
3. Implementação do motor de análise

### 12.2 Médio Prazo
1. Desenvolvimento da API
2. Implementação do dashboard
3. Otimização de performance

### 12.3 Longo Prazo
1. Escalabilidade horizontal
2. Análise em tempo real
3. Machine learning avançado
