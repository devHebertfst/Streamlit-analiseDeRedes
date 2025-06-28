import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
import tempfile
import os
from collections import Counter
from datetime import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página
st.set_page_config(
    page_title="Análise de Redes Complexas - arXiv Collaboration Network",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título e descrição
st.title("🕸️ Análise de Redes Complexas")
st.markdown("### Rede de Colaborações Científicas - arXiv Astrophysics")
st.markdown("""
Esta aplicação analisa uma rede real de colaborações científicas do repositório arXiv, 
especificamente da área de Astrofísica. A rede representa co-autorias entre pesquisadores.

**Dataset**: CA-AstroPh - Collaboration network of Arxiv Astro Physics
""")

# Informações sobre o dataset na sidebar
st.sidebar.markdown("## 📊 Sobre o Dataset")
st.sidebar.info("""
**arXiv Astro-Ph Collaboration Network**

Este dataset representa uma rede de colaboração científica onde:
- **Nós**: Autores de artigos
- **Arestas**: Co-autorias em publicações
- **Fonte**: arXiv (Astro Physics category)
- **Período**: 1992-2003
""")

@st.cache_data
def load_arxiv_data():
    """Carrega dados do arquivo ca-AstroPh.txt"""
    filename = "CA-AstroPh.txt"
    try:
        if not os.path.exists(filename):
            st.error(f"Arquivo {filename} não encontrado na raiz do projeto.")
            return []
        
        edges = []
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            node1, node2 = int(parts[0]), int(parts[1])
                            edges.append((node1, node2))
                        except ValueError:
                            continue
        
        if not edges:
            st.error("Nenhuma aresta válida encontrada no arquivo.")
            return []
            
        st.success(f"✅ Dados carregados: {len(edges):,} colaborações encontradas")
        return edges
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo {filename}: {e}")
        return []

@st.cache_data
def create_network_from_edges(edges, max_nodes=1000):
    """Cria um grafo NetworkX a partir das arestas"""
    G = nx.Graph()
    
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    # Limitar tamanho se necessário
    if G.number_of_nodes() > max_nodes:
        if nx.number_connected_components(G) > 1:
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        if G.number_of_nodes() > max_nodes:
            nodes_sample = random.sample(list(G.nodes()), max_nodes)
            G = G.subgraph(nodes_sample).copy()
    
    return G

@st.cache_data
def calculate_network_metrics(_G):
    """Calcula métricas estruturais da rede"""
    n_nodes = _G.number_of_nodes()
    n_edges = _G.number_of_edges()
    
    if n_nodes == 0:
        return {
            'nodes': 0, 'edges': 0, 'density': 0, 'sparsity': 0, 'clustering': 0,
            'assortativity': 0, 'connected_components': 0, 'is_connected': False,
            'largest_component_size': 0, 'largest_component_ratio': 0,
            'weakly_connected_components': 0, 'strongly_connected_components': 0
        }
    
    # Métricas básicas
    density = nx.density(_G)
    sparsity = 1 - density
    
    metrics = {
        'nodes': n_nodes,
        'edges': n_edges,
        'density': density,
        'sparsity': sparsity,
        'clustering': nx.average_clustering(_G),
        'connected_components': nx.number_connected_components(_G),
        'is_connected': nx.is_connected(_G)
    }
    
    # Assortatividade
    try:
        metrics['assortativity'] = nx.degree_assortativity_coefficient(_G)
    except:
        metrics['assortativity'] = 0
    
    # Componentes conectados
    components = list(nx.connected_components(_G))
    if components:
        largest_component_size = len(max(components, key=len))
        metrics['largest_component_size'] = largest_component_size
        metrics['largest_component_ratio'] = largest_component_size / n_nodes
    else:
        metrics['largest_component_size'] = 0
        metrics['largest_component_ratio'] = 0
    
    # Para grafos não-dirigidos, componentes fracamente conectados = componentes conectados
    metrics['weakly_connected_components'] = metrics['connected_components']
    
    # Componentes fortemente conectados só se aplicam a grafos dirigidos
    if _G.is_directed():
        metrics['strongly_connected_components'] = nx.number_strongly_connected_components(_G)
    else:
        metrics['strongly_connected_components'] = "N/A (Grafo não-dirigido)"
    
    return metrics

@st.cache_data
def calculate_centralities(_G, sample_size=500):
    """Calcula diferentes métricas de centralidade"""
    if _G.number_of_nodes() == 0:
        return {}, _G
    
    # Para redes grandes, usar amostra
    if _G.number_of_nodes() > sample_size:
        nodes_sample = random.sample(list(_G.nodes()), sample_size)
        G_sample = _G.subgraph(nodes_sample).copy()
    else:
        G_sample = _G
    
    centralities = {
        'degree': dict(G_sample.degree()),
        'betweenness': nx.betweenness_centrality(G_sample),
        'closeness': nx.closeness_centrality(G_sample),
    }
    
    # Eigenvector centrality
    try:
        centralities['eigenvector'] = nx.eigenvector_centrality(G_sample, max_iter=1000)
    except:
        # Fallback se não convergir
        centralities['eigenvector'] = {n: G_sample.degree(n) for n in G_sample.nodes()}
    
    return centralities, G_sample

def explain_metrics():
    """Explica o significado das métricas"""
    with st.expander("📖 Explicação das Métricas Estruturais"):
        st.markdown("""
        **🔸 Densidade**: Proporção de arestas existentes vs. arestas possíveis (0-1). 
        Redes densas têm muitas conexões, redes esparsas têm poucas.
        
        **🔸 Esparsidade**: Complemento da densidade (1 - densidade). 
        Indica quão "vazia" é a rede.
        
        **🔸 Assortatividade**: Tendência de nós similares se conectarem. 
        Positiva: nós de alto grau se conectam entre si. Negativa: nós de graus diferentes se conectam.
        
        **🔸 Coeficiente de Clustering**: Mede a tendência de formar triângulos (grupos fechados). 
        Alto clustering indica estrutura comunitária.
        
        **🔸 Componentes Conectados**: Subgrafos onde há caminho entre quaisquer dois nós.
        
        **🔸 Componentes Fortemente Conectados**: Aplicável apenas a grafos dirigidos. 
        Subgrafos onde há caminho dirigido entre quaisquer dois nós.
        
        **🔸 Componentes Fracamente Conectados**: Em grafos dirigidos, componentes conectados 
        ignorando a direção das arestas.
        """)

def explain_centralities():
    """Explica as métricas de centralidade"""
    with st.expander("📖 Explicação das Métricas de Centralidade"):
        st.markdown("""
        **🔸 Degree Centrality**: Número de conexões diretas. 
        Identifica nós com muitas conexões locais.
        
        **🔸 Betweenness Centrality**: Frequência com que um nó aparece no caminho mais curto entre outros nós. 
        Identifica "pontes" importantes na rede.
        
        **🔸 Closeness Centrality**: Inverso da distância média para todos os outros nós. 
        Identifica nós centralmente localizados.
        
        **🔸 Eigenvector Centrality**: Considera não apenas o número de conexões, mas a importância dos vizinhos. 
        Identifica nós conectados a outros nós importantes.
        """)

# Carregamento dos dados
with st.spinner("🔄 Carregando dados do arXiv..."):
    edges_data = load_arxiv_data()

if not edges_data:
    st.error("❌ Não foi possível carregar os dados. Verifique se o arquivo 'ca-AstroPh.txt' está disponível.")
    st.stop()

# Processamento da rede
with st.spinner("🔄 Processando rede..."):
    G = create_network_from_edges(edges_data)
    metrics = calculate_network_metrics(G)
    centralities, G_sample = calculate_centralities(G)

if metrics['nodes'] == 0:
    st.error("❌ Erro ao processar a rede. Verifique os dados.")
    st.stop()

# Controles da sidebar
st.sidebar.markdown("## 🎛️ Controles da Visualização")

# Filtros de subconjunto
subset_option = st.sidebar.selectbox(
    "🔍 Selecionar Subconjunto:",
    ["Rede Completa", "Componente Gigante", "Alto Grau", "Personalizado"]
)

if subset_option == "Alto Grau":
    max_degree = max([G.degree(n) for n in G.nodes()]) if G.number_of_nodes() > 0 else 20
    min_degree_threshold = st.sidebar.slider(
        "Grau mínimo:", 
        1, min(50, max_degree), 
        min(10, max_degree//2)
    )
else:
    min_degree_threshold = 1

# Layout da rede
layout_option = st.sidebar.selectbox(
    "🎨 Layout da Rede:",
    ["spring", "circular", "kamada_kawai", "shell"]
)

# Filtro de centralidade para destaque
centrality_filter = st.sidebar.selectbox(
    "⭐ Destacar por Centralidade:",
    ["Nenhum", "Degree", "Betweenness", "Closeness", "Eigenvector"]
)

top_k = st.sidebar.slider("🔝 Top K nós destacados:", 5, 50, 10)

# Controles de visualização
max_nodes_viz = st.sidebar.slider("📊 Máximo de nós na visualização:", 50, 1000, 300)

# Aplicação dos filtros
G_filtered = G.copy()

if subset_option == "Componente Gigante":
    if nx.number_connected_components(G_filtered) > 1:
        largest_cc = max(nx.connected_components(G_filtered), key=len)
        G_filtered = G_filtered.subgraph(largest_cc).copy()
elif subset_option == "Alto Grau":
    nodes_to_remove = [n for n in G_filtered.nodes() if G_filtered.degree(n) < min_degree_threshold]
    G_filtered.remove_nodes_from(nodes_to_remove)

# Limitar tamanho para visualização
if G_filtered.number_of_nodes() > max_nodes_viz:
    nodes_sample = random.sample(list(G_filtered.nodes()), max_nodes_viz)
    G_filtered = G_filtered.subgraph(nodes_sample).copy()

# Layout principal - Visualização da Rede
st.markdown("## 1. 🕸️ Visualização da Rede")

col1, col2 = st.columns([3, 1])

with col1:
    if G_filtered.number_of_nodes() == 0:
        st.warning("⚠️ Nenhum nó encontrado com os filtros aplicados. Ajuste os filtros.")
    else:
        # Criar visualização Pyvis
        net = Network(height="600px", width="100%", bgcolor="white", font_color="black")
        
        # Calcular layout
        try:
            if layout_option == "spring":
                pos = nx.spring_layout(G_filtered, k=1, iterations=50)
            elif layout_option == "circular":
                pos = nx.circular_layout(G_filtered)
            elif layout_option == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G_filtered)
            elif layout_option == "shell":
                pos = nx.shell_layout(G_filtered)
        except:
            pos = nx.spring_layout(G_filtered)
        
        # Identificar nós importantes
        top_nodes = []
        if centrality_filter != "Nenhum" and G_filtered.number_of_nodes() > 0:
            if centrality_filter == "Degree":
                cent_dict = {n: G_filtered.degree(n) for n in G_filtered.nodes()}
            else:
                try:
                    if centrality_filter == "Betweenness":
                        cent_dict = nx.betweenness_centrality(G_filtered)
                    elif centrality_filter == "Closeness":
                        cent_dict = nx.closeness_centrality(G_filtered)
                    elif centrality_filter == "Eigenvector":
                        cent_dict = nx.eigenvector_centrality(G_filtered, max_iter=1000)
                except:
                    cent_dict = {n: G_filtered.degree(n) for n in G_filtered.nodes()}
            
            top_nodes = sorted(cent_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
            top_nodes = [n[0] for n in top_nodes]
        
        # Adicionar nós
        for node in G_filtered.nodes():
            degree = G_filtered.degree(node)
            size = 10 + degree * 0.5
            color = '#3498db'  # Azul padrão
            
            if node in top_nodes:
                color = '#e74c3c'  # Vermelho para destacados
                size += 5
            
            tooltip = f"<b>Autor {node}</b><br>Colaborações: {degree}"
            
            net.add_node(
                node,
                label=str(node),
                color=color,
                size=size,
                x=pos[node][0] * 300,
                y=pos[node][1] * 300,
                title=tooltip
            )
        
        # Adicionar arestas
        for edge in G_filtered.edges():
            net.add_edge(edge[0], edge[1], color='#95a5a6', width=1)
        
        # Configurar física
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -2000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09
            }
          }
        }
        """)
        
        # Salvar e exibir
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            tmp_path = tmp.name
        
        net.save_graph(tmp_path)
        
        try:
            with open(tmp_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=650)
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

with col2:
    st.markdown("#### 📊 Info da Visualização")
    st.metric("Nós Exibidos", f"{G_filtered.number_of_nodes():,}")
    st.metric("Arestas Exibidas", f"{G_filtered.number_of_edges():,}")
    st.metric("Subconjunto", subset_option)
    
    st.markdown("#### 🎨 Legenda")
    st.markdown("🔵 Nós normais")
    st.markdown("🔴 Top K destacados")
    
    if centrality_filter != "Nenhum":
        st.markdown(f"**Destacando por**: {centrality_filter}")

# Métricas Estruturais
st.markdown("## 2. 📏 Métricas Estruturais")
explain_metrics()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🔢 Nós", f"{metrics['nodes']:,}")
    st.metric("🔗 Arestas", f"{metrics['edges']:,}")

with col2:
    st.metric("📊 Densidade", f"{metrics['density']:.6f}")
    st.metric("🕳️ Esparsidade", f"{metrics['sparsity']:.6f}")

with col3:
    st.metric("🔄 Assortatividade", f"{metrics['assortativity']:.3f}")
    st.metric("🌐 Clustering Global", f"{metrics['clustering']:.3f}")

with col4:
    st.metric("🧩 Componentes Conectados", f"{metrics['connected_components']:,}")
    if isinstance(metrics['strongly_connected_components'], str):
        st.metric("💪 Comp. Fortemente Conectados", metrics['strongly_connected_components'])
    else:
        st.metric("💪 Comp. Fortemente Conectados", f"{metrics['strongly_connected_components']:,}")

# Informações adicionais sobre componentes
st.markdown("#### 🧩 Análise de Componentes")
col1, col2 = st.columns(2)

with col1:
    st.metric("🏔️ Maior Componente", f"{metrics['largest_component_size']:,} nós")
    st.metric("📊 % do Maior Componente", f"{metrics['largest_component_ratio']:.1%}")

with col2:
    connectivity_status = "✅ Conectada" if metrics['is_connected'] else "❌ Desconectada"
    st.markdown(f"**Status da Rede**: {connectivity_status}")
    st.metric("🔗 Comp. Fracamente Conectados", f"{metrics['weakly_connected_components']:,}")

# Distribuições de Grau
st.markdown("## 3. 📈 Distribuições de Grau")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Histograma de Grau")
    degrees = [G.degree(n) for n in G.nodes()]
    
    fig_hist = px.histogram(
        x=degrees,
        nbins=min(50, len(set(degrees))),
        title="Distribuição de Grau dos Nós",
        labels={'x': 'Grau (Número de Colaborações)', 'y': 'Frequência'},
        color_discrete_sequence=['#3498db']
    )
    fig_hist.update_layout(showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Estatísticas
    st.markdown("#### 📊 Estatísticas do Grau")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Média", f"{np.mean(degrees):.2f}")
        st.metric("Mediana", f"{np.median(degrees):.0f}")
    with col_b:
        st.metric("Máximo", f"{max(degrees)}")
        st.metric("Mínimo", f"{min(degrees)}")

with col2:
    st.subheader("Distribuição Log-Log")
    degree_counts = Counter(degrees)
    degrees_unique = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in degrees_unique]
    
    fig_loglog = go.Figure()
    fig_loglog.add_trace(go.Scatter(
        x=degrees_unique,
        y=counts,
        mode='markers',
        name='Distribuição de Grau',
        marker=dict(color='#e74c3c', size=8)
    ))
    fig_loglog.update_layout(
        title="Distribuição de Grau (Escala Log-Log)",
        xaxis_title="Grau (log)",
        yaxis_title="Frequência (log)",
        xaxis_type="log",
        yaxis_type="log"
    )
    st.plotly_chart(fig_loglog, use_container_width=True)
    
    # Top nós por grau
    st.markdown("#### 🏆 Top 10 Nós por Grau")
    top_degree_nodes = sorted([(n, G.degree(n)) for n in G.nodes()], 
                             key=lambda x: x[1], reverse=True)[:10]
    
    for i, (node, degree) in enumerate(top_degree_nodes, 1):
        st.write(f"{i}. Autor {node}: {degree} colaborações")

# Análise de Centralidades
st.markdown("## 4. ⭐ Centralidade dos Nós")
explain_centralities()

if G_sample.number_of_nodes() > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlação entre Centralidades")
        
        # Criar DataFrame das centralidades
        cent_df = pd.DataFrame({
            'Node': list(centralities['degree'].keys()),
            'Degree': list(centralities['degree'].values()),
            'Betweenness': list(centralities['betweenness'].values()),
            'Closeness': list(centralities['closeness'].values()),
            'Eigenvector': list(centralities['eigenvector'].values())
        })
        
        # Matriz de correlação
        corr_cols = ['Degree', 'Betweenness', 'Closeness', 'Eigenvector']
        corr_matrix = cent_df[corr_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlação entre Métricas de Centralidade",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.subheader("Rankings por Centralidade")
        
        selected_centrality = st.selectbox(
            "Selecione a centralidade para ranking:",
            ['Degree', 'Betweenness', 'Closeness', 'Eigenvector']
        )
        
        top_central = cent_df.nlargest(15, selected_centrality)[['Node', selected_centrality]]
        
        st.markdown(f"#### 🏆 Top 15 - {selected_centrality} Centrality")
        
        for idx, (_, row) in enumerate(top_central.iterrows(), 1):
            value = row[selected_centrality]
            if selected_centrality == 'Degree':
                st.write(f"{idx}. Autor {int(row['Node'])}: {int(value)} conexões")
            else:
                st.write(f"{idx}. Autor {int(row['Node'])}: {value:.4f}")
    
    # Comparação visual das centralidades
    st.subheader("Comparação Visual das Centralidades")
    
    # Scatter plots comparando centralidades
    fig_scatter = px.scatter_matrix(
        cent_df[corr_cols], 
        title="Comparação entre Métricas de Centralidade",
        height=600
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.warning("⚠️ Não foi possível calcular centralidades para este subconjunto.")

# Análise detalhada de componentes
st.markdown("## 5. 🧩 Análise Detalhada de Componentes")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribuição de Componentes")
    components = list(nx.connected_components(G))
    component_sizes = [len(comp) for comp in components]
    
    fig_comp = px.histogram(
        x=component_sizes,
        nbins=min(50, len(component_sizes)),
        title="Distribuição de Tamanhos dos Componentes",
        labels={'x': 'Tamanho do Componente', 'y': 'Frequência'},
        color_discrete_sequence=['#9b59b6']
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Estatísticas dos componentes
    st.markdown("#### 📊 Estatísticas dos Componentes")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total de Componentes", f"{len(components):,}")
        if component_sizes:
            st.metric("Maior Componente", f"{max(component_sizes):,} nós")
    with col_b:
        if component_sizes:
            st.metric("Menor Componente", f"{min(component_sizes)} nós")
            st.metric("Tamanho Médio", f"{np.mean(component_sizes):.2f}")

with col2:
    st.subheader("Maiores Componentes")
    if component_sizes:
        largest_components = sorted(component_sizes, reverse=True)[:15]
        
        fig_bar = px.bar(
            x=range(1, len(largest_components) + 1),
            y=largest_components,
            title="15 Maiores Componentes",
            labels={'x': 'Ranking', 'y': 'Tamanho'},
            color_discrete_sequence=['#f39c12']
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Análise de isolamento
        isolated_nodes = len([comp for comp in components if len(comp) == 1])
        st.markdown("#### 🏝️ Análise de Isolamento")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Nós Isolados", f"{isolated_nodes:,}")
        with col_b:
            if G.number_of_nodes() > 0:
                isolation_pct = isolated_nodes/G.number_of_nodes()*100
                st.metric("% Isolados", f"{isolation_pct:.1f}%")

# Footer
st.markdown("---")
st.markdown("## 📚 Informações do Projeto")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🛠️ Tecnologias Utilizadas:**
    - Streamlit (Interface Web)
    - NetworkX (Análise de Redes)
    - Pyvis (Visualização Interativa)
    - Plotly (Gráficos)
    - Pandas & NumPy (Processamento)
    """)

with col2:
    st.markdown("""
    **📊 Dataset:**
    - arXiv Astro Physics Collaboration Network
    - Período: 1992-2003
    - Nós: Autores científicos
    - Arestas: Co-autorias
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>🕸️ Análise de Redes Complexas</strong></p>
    <p>Desenvolvido com Streamlit • NetworkX • Pyvis</p>
    <p>Hebert França</p>
</div>
""", unsafe_allow_html=True)