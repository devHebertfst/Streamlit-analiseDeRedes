# Análise de Redes Complexas - arXiv

Aplicação web para análise de redes de colaboração científica usando dados do arXiv (área de Astrofísica).

## Sobre

Este projeto analisa uma rede onde os nós são autores e as arestas representam colaborações (co-autorias). Os dados cobrem o período de 1992-2003.

## Funcionalidades

### Visualização da Rede
- Grafo interativo com Pyvis
- Diferentes layouts (spring, circular, etc.)
- Filtros para mostrar subconjuntos da rede
- Destaque de nós importantes

### Métricas da Rede
- Densidade e esparsidade
- Coeficiente de clustering
- Assortatividade
- Componentes conectados

### Análise de Grau
- Histograma da distribuição de grau
- Gráfico log-log
- Top nós com mais conexões

### Centralidades
- Degree centrality
- Betweenness centrality  
- Closeness centrality
- Eigenvector centrality
- Rankings e comparações

## Como executar

1. Clone o repositório
```bash
git clone https://github.com/devHebertfst/Streamlit-analiseDeRedes
cd StreamLit-analiseRedes
```

2. Instale as dependências
```bash
pip install -r requirements.txt
```

3. Baixe o dataset ca-AstroPh.txt e coloque na pasta raiz

4. Execute a aplicação
```bash
streamlit run app.py
```

## Dependências

- streamlit
- networkx
- pandas
- numpy
- plotly
- pyvis
- matplotlib
- seaborn

## Dataset

Usa o dataset CA-AstroPh da Stanford (SNAP):
- 18.000+ autores
- 396.000+ colaborações
- Formato: lista de arestas

## Estrutura

```
projeto/
├── app.py
├── requirements.txt
├── README.md
└── ca-AstroPh.txt
```
## Link streamlit
https://ms3pap8cpeirutwqulyd6a.streamlit.app/