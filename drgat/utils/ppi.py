
import pandas as pd
import networkx as nx

def load_ppi_graph(ppi_df: pd.DataFrame, confidence_threshold=0.7, take_lcc=True):
    df = ppi_df.copy()
    if "score" in df.columns:
        df = df[df["score"] >= confidence_threshold]
    G = nx.Graph()
    for _, r in df.iterrows():
        G.add_edge(str(r["gene_u"]), str(r["gene_v"]))
    if take_lcc and G.number_of_nodes() > 0:
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(comps[0]).copy()
    return G
