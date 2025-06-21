import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Czy punkt leży wewnątrz kuli w $R^n$ przy użyciu różnych metryk?")

# --- Funkcje pomocnicze ---

def parse_vector(vector_str: str, expected_dim: int) -> np.ndarray:
    """Przetwarza ciąg znaków na wektor numpy, sprawdzając jego wymiar."""
    try:
        parts = [float(x.strip()) for x in vector_str.split(",")]
        if len(parts) != expected_dim:
            st.error(f"Wektor '{vector_str}' ma {len(parts)} wymiarów, a oczekiwano {expected_dim}.")
            return None
        return np.array(parts)
    except ValueError:
        st.error(f"Nie można przetworzyć wektora '{vector_str}'. Upewnij się, że współrzędne są liczbami oddzielonymi przecinkami.")
        return None

def generate_defaults(n: int) -> tuple[str, str]:
    """Generuje domyślne współrzędne dla punktu i środka w zależności od wymiaru n."""
    default_point = ",".join(["1"] * n)
    default_center = ",".join(["0"] * n)
    return default_point, default_center

def calculate_distance(p1: np.ndarray, p2: np.ndarray, metric: str) -> float:
    """Oblicza odległość między dwoma punktami zgodnie z wybraną metryką."""
    diff = p1 - p2
    if metric == 'Euklidesowa':
        # Norma L2
        return np.linalg.norm(diff, ord=2)
    if metric == 'Manhattan':
        # Norma L1
        return np.linalg.norm(diff, ord=1)
    if metric == 'Czebyszewa':
        # Norma L-nieskończoność
        return np.linalg.norm(diff, ord=np.inf)
    return 0.0

# --- Interfejs użytkownika ---
col1, col2 = st.columns(2)

with col1:
    st.header("Dane wejściowe")
    n = st.number_input("Podaj wymiar przestrzeni (n):", min_value=1, max_value=100, value=2, step=1)
    
    metric_options = ['Euklidesowa', 'Manhattan', 'Czebyszewa']
    metric_name = st.selectbox("Wybierz metrykę:", options=metric_options)
    
    default_point_str, default_center_str = generate_defaults(n)
    point_str = st.text_input(f"Współrzędne punktu w $R^{n}$", default_point_str)
    center_str = st.text_input(f"Współrzędne środka kuli w $R^{n}$", default_center_str)
    radius = st.number_input("Promień kuli", min_value=0.0, value=2.0, step=0.1)

# --- Główna logika i obliczenia ---
point = parse_vector(point_str, n)
center = parse_vector(center_str, n)

# Używamy col2 do wyświetlania wyników i wykresu
with col2:
    st.header("Wynik i wizualizacja")
    if point is not None and center is not None:
        dist = calculate_distance(point, center, metric_name)
        
        st.metric(label=f"Odległość w metryce ({metric_name})", value=f"{dist:.4f}")

        if np.isclose(dist, radius):
            st.info(f"**Punkt leży na sferze** (brzegu kuli w metryce '{metric_name}').")
        elif dist < radius:
            st.success(f"**Punkt leży wewnątrz kuli** w metryce '{metric_name}'.")
        else:
            st.warning(f"**Punkt leży poza kulą** w metryce '{metric_name}'.")

        # --- Wizualizacja ---
        if n in [1, 2, 3]:
            fig = go.Figure()
            
            # Dodaj punkt i środek (wspólne dla wszystkich wizualizacji)
            if n == 1:
                fig.add_trace(go.Scatter(x=point, y=[0], mode='markers', marker=dict(color='red', size=12), name='Punkt'))
                fig.add_trace(go.Scatter(x=center, y=[0], mode='markers', marker=dict(color='black', size=12, symbol='x'), name='Środek'))
            elif n == 2:
                fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode='markers', marker=dict(color='red', size=12), name='Punkt'))
                fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], mode='markers', marker=dict(color='black', size=12, symbol='x'), name='Środek'))
            elif n == 3:
                fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode='markers', marker=dict(color='red', size=7), name='Punkt'))
                fig.add_trace(go.Scatter3d(x=[center[0]], y=[center[1]], z=[center[2]], mode='markers', marker=dict(color='black', size=7, symbol='x'), name='Środek'))

            # Narysuj "kulę" w zależności od metryki i wymiaru
            # Kula w 1D to zawsze odcinek
            if n == 1:
                fig.add_shape(type="line", x0=center[0] - radius, y0=0, x1=center[0] + radius, y1=0,
                              line=dict(color="RoyalBlue", width=5), name='Kula')
                fig.update_yaxes(showticklabels=False, zeroline=False)
            
            # Kula w 2D
            elif n == 2:
                if metric_name == 'Euklidesowa':
                    fig.add_shape(type="circle", x0=center[0]-radius, y0=center[1]-radius, x1=center[0]+radius, y1=center[1]+radius,
                                  line_color="RoyalBlue", fillcolor="LightSkyBlue", opacity=0.5)
                elif metric_name == 'Manhattan': # Romb (obrócony kwadrat)
                    x_v = [center[0] + radius, center[0], center[0] - radius, center[0], center[0] + radius]
                    y_v = [center[1], center[1] + radius, center[1], center[1] - radius, center[1]]
                    fig.add_trace(go.Scatter(x=x_v, y=y_v, fill="toself", fillcolor="LightSkyBlue", opacity=0.5,
                                             line=dict(color='RoyalBlue'), name='Kula Manhattan'))
                elif metric_name == 'Czebyszewa': # Kwadrat
                    fig.add_shape(type="rect", x0=center[0]-radius, y0=center[1]-radius, x1=center[0]+radius, y1=center[1]+radius,
                                  line_color="RoyalBlue", fillcolor="LightSkyBlue", opacity=0.5)
                fig.update_layout(yaxis_scaleanchor="x")
            
            # Kula w 3D
            elif n == 3:
                c, r = center, radius
                if metric_name == 'Euklidesowa': # Sfera
                    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
                    xs = c[0] + r * np.cos(u) * np.sin(v)
                    ys = c[1] + r * np.sin(u) * np.sin(v)
                    zs = c[2] + r * np.cos(v)
                    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, opacity=0.4, colorscale='Blues', showscale=False, name='Sfera'))
                
                elif metric_name == 'Manhattan': # Oktaedr
                    vertices = [c+[r,0,0], c+[-r,0,0], c+[0,r,0], c+[0,-r,0], c+[0,0,r], c+[0,0,-r]]
                    x_v, y_v, z_v = zip(*vertices)
                    fig.add_trace(go.Mesh3d(x=x_v, y=y_v, z=z_v, opacity=0.4, color='LightSkyBlue',
                                            i=[0,0,0,0, 1,1,1,1], # Indeksy wierzchołków dla trójkątów
                                            j=[2,3,4,5, 2,3,4,5],
                                            k=[4,5,2,3, 4,5,3,2], name='Kula Manhattan'))
                
                elif metric_name == 'Czebyszewa': # Sześcian
                    x_v = [c[0]-r, c[0]+r, c[0]+r, c[0]-r, c[0]-r, c[0]+r, c[0]+r, c[0]-r]
                    y_v = [c[1]-r, c[1]-r, c[1]+r, c[1]+r, c[1]-r, c[1]-r, c[1]+r, c[1]+r]
                    z_v = [c[2]-r, c[2]-r, c[2]-r, c[2]-r, c[2]+r, c[2]+r, c[2]+r, c[2]+r]
                    fig.add_trace(go.Mesh3d(x=x_v, y=y_v, z=z_v, opacity=0.4, color='LightSkyBlue',
                                           # Indeksy 12 trójkątów tworzących 6 ścian sześcianu
                                           i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                                           j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                                           k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], name='Kula Czebyszewa'))

            # Aktualizacja layoutu
            fig.update_layout(
                legend_orientation="h", legend_yanchor="bottom", legend_y=1.02,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            if n==3:
                 fig.update_layout(scene=dict(aspectmode='data')) # Równe proporcje osi 3D
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Wprowadź poprawne dane wejściowe, aby zobaczyć wynik.")
