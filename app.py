import streamlit as st
import numpy as np
import plotly.graph_objects as go


# Ustawianie wyglądu strony z użyciem Streamlit
st.set_page_config(layout="wide")
st.title("Wizualizator kul w metryce Minkowskiego ($d_p$)")


# ------------------- Funkcje do przetwarzania danych wejściowych i obliczeń odległości -------------------

# funkcja do przetwarzania wektora wejściowego
def read_vector(vector_str: str, expected_dim: int) -> np.ndarray:
    # zamiana ciągu znaków na wektor NumPy
    try:
        parts = [float(x.strip()) for x in vector_str.split(",")]
        if len(parts) != expected_dim: # sprawdzenie poprawności wymiaru
            st.error(f"Wektor '{vector_str}' ma {len(parts)} wymiarów, a oczekiwano {expected_dim}.")
            return None
        return np.array(parts)
    except ValueError: # obsługa błędnych danych wejściowych
        st.error(f"Nie można odczytać wektora '{vector_str}'. Upewnij się, że współrzędne są liczbami oddzielonymi przecinkami.")
        return None

# funkcja do generowania domyślnych współrzędnych punktu i środka w zależności od wymiaru n
def generate_defaults(n: int) -> tuple[str, str]:
    default_point = ",".join(["1"] * n)
    default_center = ",".join(["0"] * n)
    return default_point, default_center

# funkcja do obliczania odległości Minkowskiego L_p
# p1, p2 - wektory punktu i środka
# p_value - wartość p dla metryki Minkowskiego
def calculate_distance(p1: np.ndarray, p2: np.ndarray, p_value: float) -> float:
    # odległości obliczana jest za pomocą funkcji normy z NumPy
    # podając p_value jako parametr ord czyli potęgę normy
    # dla p=1 mamy metrykę Manhattan, dla p=2 metrykę Euklidesową, a dla p=∞ metrykę Czebyszewa
    # norma z numpy jest odpowiednikiem obliczeń ze wzoru Minkowskiego
    if p_value == np.inf: #obługa metryki Czebyszewa
        return np.linalg.norm(p1 - p2, ord=np.inf)
    return np.linalg.norm(p1 - p2, ord=p_value)

# ------------------- Interfejs strony -------------------
with st.sidebar:
    st.header("Parametry wejściowe")
    n = st.number_input("Wymiar przestrzeni (n)", min_value=1, max_value=100, value=2, step=1,
                        help="Wizualizacja dostępna tylko dla n=1, 2, 3.")
    
    metric_type = st.radio(
        "Wybierz typ metryki",
        ('Standardowa', 'Własna (Minkowskiego)')
    )

    p_value = 2.0 # Domyślnie Euklidesowa po otworzeniu aplikacji
    if metric_type == 'Standardowa':
        metric_name = st.selectbox("Wybierz metrykę standardową:", ['Euklidesowa (p=2)', 'Manhattan (p=1)', 'Czebyszewa (p=∞)'])
        if 'p=1' in metric_name:
            p_value = 1.0
        elif 'p=2' in metric_name:
            p_value = 2.0
        elif 'p=∞' in metric_name:
            p_value = np.inf
    else:
        p_value = st.number_input("Podaj wartość p > 0", min_value=0.01, value=2.0, step=0.1)
        metric_name = f"d_{p_value:.2f}"

    # Generowanie domyślnych współrzędnych punktu i środka
    default_point_str, default_center_str = generate_defaults(n)
    point_str = st.text_input(f"Punkt w $R^{n}$", default_point_str)
    center_str = st.text_input(f"Środek kuli w $R^{n}$", default_center_str)
    promien = st.number_input("Promień kuli", min_value=0.1, value=2.0, step=0.1)

# ----- Przetwarzanie danych wejściowych i obliczenia ----
# Odczytanie i przetworzenie wektorów punktu i środka
point = read_vector(point_str, n)
center = read_vector(center_str, n)

if point is not None and center is not None: # Sprawdzenie poprawności danych wejściowych
    # Obliczenie odległości między punktem a środkiem kuli
    dist = calculate_distance(point, center, p_value)
    
    # Wyświetlenie wyników w kolumnach
    col1, col2 = st.columns([1, 3])
    with col1: # Wyświetlenie metryk i informacji
        st.metric(label=f"Odległość d_p (p={p_value})", value=f"{dist:.4f}")
        st.metric(label="Promień kuli", value=f"{promien:.4f}")

        if np.isclose(dist, promien): # Sprawdzenie, czy punkt leży na kuli
            # użyto tutaj funkcji np.isclose do porównania wartości z tolerancją, bo może wystąpić bardzo mały błąd numeryczny
            # tolerancja jest domyślnie ustawiona na 1e-9, co jest wystarczające dla większości zastosowań
            st.info(f"**Punkt leży na sferze i kuli domkniętej** w metryce {metric_name}.")
        elif dist < promien:
            st.success(f"**Punkt leży wewnątrz kuli (otwartej i domkniętej)** w metryce {metric_name}.")
        else:
            st.warning(f"**Punkt leży poza kulami i sferą** w metryce {metric_name}.")
    
    with col2: # Wykres
        if n in [1, 2, 3]:
            fig = go.Figure()
            
            # Dodawanie punktu i środka jest takie samo dla każdego wymiaru
            plot_kwargs = {'mode': 'markers', 'marker': {'size': 12}}
            if n == 1:
                fig.add_trace(go.Scatter(x=point, y=[0], name='Punkt', marker_color='red', **plot_kwargs))
                fig.add_trace(go.Scatter(x=center, y=[0], name='Środek', marker_color='black', marker_symbol='x', **plot_kwargs))
            elif n == 2:
                fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], name='Punkt', marker_color='red', **plot_kwargs))
                fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], name='Środek', marker_color='black', marker_symbol='x', **plot_kwargs))
            elif n == 3:
                fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], name='Punkt', marker_color='red', **plot_kwargs))
                fig.add_trace(go.Scatter3d(x=[center[0]], y=[center[1]], z=[center[2]], name='Środek', marker_color='black', marker_symbol='x', **plot_kwargs))

            # Rysowanie kuli
            if n == 1: # Ogólna wizualizacja 1D jako linia
                fig.add_shape(type="line", x0=center[0] - promien, y0=0, x1=center[0] + promien, y1=0, line=dict(color="RoyalBlue", width=5))
                fig.update_yaxes(showticklabels=False, zeroline=False)
            
            elif n == 2: # Ogólna wizualizacja 2D za pomocą wykresu konturowego
                grid_res = 100
                plot_range = promien * 1.5
                x_grid = np.linspace(center[0] - plot_range, center[0] + plot_range, grid_res)
                y_grid = np.linspace(center[1] - plot_range, center[1] + plot_range, grid_res)

                # Obliczanie odległości dla siatki punktów
                distances = np.zeros((grid_res, grid_res))
                for i, x in enumerate(x_grid):
                    for j, y in enumerate(y_grid):
                        distances[j, i] = calculate_distance(np.array([x, y]), center, p_value)
                
                fig.add_trace(go.Contour(
                    x=x_grid, y=y_grid, z=distances,
                    contours=dict(start=promien, end=promien, size=0), # Rysuj tylko kontur dla wartości równej 'promien'
                    showscale=False,
                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']], # Ukryj wypełnienie
                    line=dict(color='RoyalBlue', width=3)
                ))
                fig.update_layout(yaxis_scaleanchor="x")
            
            elif n == 3: # Ogólna wizualizacja 3D za pomocą izopowierzchni
                grid_res = 40
                plot_range = promien * 1.5
                X, Y, Z = np.mgrid[
                    center[0]-plot_range:center[0]+plot_range:grid_res*1j,
                    center[1]-plot_range:center[1]+plot_range:grid_res*1j,
                    center[2]-plot_range:center[2]+plot_range:grid_res*1j
                ]
                
                points_in_grid = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
                distances = np.linalg.norm(points_in_grid - center, ord=p_value, axis=1).reshape(X.shape)

                fig.add_trace(go.Isosurface(
                    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                    value=distances.flatten(),
                    isomin=promien, isomax=promien, # Rysuj powierzchnię dla wartości równej 'promien'
                    surface_count=1,
                    opacity=0.4,
                    caps=dict(x_show=False, y_show=False, z_show=False),
                    colorscale='Blues',
                    showscale=False
                ))
                fig.update_layout(scene=dict(aspectmode='data'))

            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Wprowadź poprawne dane wejściowe, aby zobaczyć wynik.")
