import streamlit as st
import numpy as np
import plotly.graph_objects as go


# Ustawianie wyglądu strony z użyciem Streamlit
st.set_page_config(layout="wide")
st.title("Czy punkt należy do kuli/sfery w (R^N)")


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
    if isinstance(p_value, str): #Obsługa metryk hamminga oraz dykretnej
        if p_value == 'hamming':
            if len(p1) != len(p2):
                raise ValueError("Wektory muszą mieć ten sam rozmiar dla metryki Hamminga.")
            return np.sum(p1 != p2)  #Hamming
        elif p_value == 'discrete':
            return 0.0 if np.array_equal(p1, p2) else 1.0
    if p_value == np.inf: #obługa metryki Czebyszewa
        return np.linalg.norm(p1 - p2, ord=np.inf)
    if p_value >= 1:
        return np.linalg.norm(p1 - p2, ord=p_value)
    # dla p < 1: NIE używamy pierwiastka (czyli "quasi-dystans")
    return np.sum(np.abs(p1-p2) ** p_value)

# ------------------- Interfejs strony -------------------
with st.sidebar:
    st.header("Parametry wejściowe")
    n = st.number_input("Wymiar przestrzeni (n)", min_value=1, max_value=100, value=2, step=1,
                        help="Wizualizacja dostępna tylko dla n=1, 2, 3.")
    
    zbior_typ = st.radio(
        "Typ zbioru",
        options=["Kula otwarta", "Kula domknięta", "Sfera"],
        help="Wybierz, do jakiego typu zbioru chcesz sprawdzić przynależność punktu."
    )

    metric_type = st.radio(
        "Wybierz typ metryki",
        ('Standardowa', 'Własna (Minkowskiego)')
    )

    p_value = 2.0 # Domyślnie Euklidesowa po otworzeniu aplikacji
    if metric_type == 'Standardowa':
        if n == 1:
            metric_name = st.selectbox("Wybierz metrykę standardową:", ['Naturalna','Dyskretna'])
            if metric_name == 'Naturalna':
                p_value = 1
            elif metric_name == 'Dyskretna':
                p_value = 'discrete'
        else:
            metric_name = st.selectbox("Wybierz metrykę standardową:", ['Euklidesowa (p=2)', 'Manhattan (p=1)', 'Czebyszewa (p=∞)', 'Hamminga', 'Dyskretna'])
            if 'p=1' in metric_name:
                p_value = 1.0
            elif 'p=2' in metric_name:
                p_value = 2.0
            elif 'p=∞' in metric_name:
                p_value = np.inf
            elif 'Hamminga' in metric_name:
                p_value = 'hamming'
            elif 'Dyskretna' in metric_name:
                p_value = 'discrete'
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
    with col1:
        # Odległość: format zależny od typu danych (float/int)
        dist_formatted = f"{dist:.4f}" if isinstance(dist, float) else str(dist)
        if metric_name in ["Hamminga","Dyskretna"]:
            st.metric(label=f"Odległość ({metric_name})", value=dist_formatted)
        else:
            st.metric(label=f"Odległość d_p (p={p_value:.4f})", value=dist_formatted)

        st.metric(label="Promień kuli", value=f"{promien:.4f}")

        # Logika przynależności do zbiorów
        # Ustal tolerancję porównania
        is_close = np.isclose(dist, promien, atol=1e-4)

        if zbior_typ == "Kula otwarta":
            if (dist < promien and not is_close):
                st.success(f"**Punkt należy do kuli otwartej** w metryce {metric_name}.")
            else:
                st.warning(f"**Punkt nie należy do kuli otwartej** w metryce {metric_name}.")

        elif zbior_typ == "Kula domknięta":
            if dist < promien or is_close:
                st.success(f"**Punkt należy do kuli domkniętej** w metryce {metric_name}.")
            else:
                st.warning(f"**Punkt nie należy do kuli domkniętej** w metryce {metric_name}.")

        elif zbior_typ == "Sfera":
            if is_close:
                st.success(f"**Punkt leży na sferze** w metryce {metric_name}.")
            else:
                st.warning(f"**Punkt nie leży na sferze** w metryce {metric_name}.")
    if metric_name in ["Hamminga","Dyskretna"]:
        with col2:
            st.info(f"Wizualizacja niedostępna dla metryki **{metric_name}** – brak sensownej reprezentacji geometrycznej.")
    else:
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

                if n == 1:  # Wizualizacja 1D jako linia

                    x0 = center[0] - promien
                    x1 = center[0] + promien

                    if zbior_typ == "Kula domknięta":
                        # Przedział domknięty [x0, x1]
                        fig.add_shape(type="line", x0=x0, y0=0, x1=x1, y1=0, line=dict(color="RoyalBlue", width=5),showlegend=False)
                        fig.add_trace(go.Scatter(x=[x0, x1], y=[0, 0], mode='markers',
                                                marker=dict(symbol='circle', color='RoyalBlue', size=10),
                                                showlegend=False))

                    elif zbior_typ == "Kula otwarta":
                        # Przedział otwarty (x0, x1) z pustymi kółkami
                        fig.add_shape(type="line", x0=x0 + 0.01, y0=0, x1=x1 - 0.01, y1=0,
                                    line=dict(color="RoyalBlue", width=5),showlegend=False)
                        fig.add_trace(go.Scatter(x=[x0, x1], y=[0, 0], mode='markers',
                                                marker=dict(symbol='circle-open', color='RoyalBlue', size=10),
                                                showlegend=False))

                    elif zbior_typ == "Sfera":
                        # Tylko dwa punkty końcowe sfery
                        fig.add_trace(go.Scatter(x=[x0, x1], y=[0, 0], mode='markers',
                                                marker=dict(symbol='circle', color='RoyalBlue', size=12),
                                                showlegend=False))

                    fig.update_yaxes(showticklabels=False, zeroline=False)
                    fig.update_xaxes(title="oś x")
                elif n == 2:  # Wizualizacja 2D
                    if p_value >=1:
                        def p_metric_circle(center, r, p, num_points=200):
                            if p == np.inf:
                                # Kwadrat z zaokrągleniem rogów poprzez interpolację narożników
                                # Prostokąt o bokach 2r, środku w center
                                x = []
                                y = []
                                steps = num_points // 4
                                # cztery boki kwadratu (startujemy od góry w prawo, przeciwnie do wskazówek zegara)
                                x += list(np.linspace(center[0] - r, center[0] + r, steps))         # góra
                                y += [center[1] + r] * steps
                                x += [center[0] + r] * steps                                        # prawa
                                y += list(np.linspace(center[1] + r, center[1] - r, steps))
                                x += list(np.linspace(center[0] + r, center[0] - r, steps))         # dół
                                y += [center[1] - r] * steps
                                x += [center[0] - r] * steps                                        # lewa
                                y += list(np.linspace(center[1] - r, center[1] + r, steps))
                                return np.array(x), np.array(y)
                            else:
                                theta = np.linspace(0, 2 * np.pi, num_points)
                                denom = (np.abs(np.cos(theta))**p + np.abs(np.sin(theta))**p)**(1/p)
                                x = center[0] + r * np.cos(theta) / denom
                                y = center[1] + r * np.sin(theta) / denom
                                return x, y


                        grid_res = 100
                        plot_range = promien * 1.5
                        x_grid = np.linspace(center[0] - plot_range, center[0] + plot_range, grid_res)
                        y_grid = np.linspace(center[1] - plot_range, center[1] + plot_range, grid_res)

                        distances = np.zeros((grid_res, grid_res))
                        for i, x in enumerate(x_grid):
                            for j, y in enumerate(y_grid):
                                distances[j, i] = calculate_distance(np.array([x, y]), center, p_value)

                        if zbior_typ == "Sfera":
                            circle_x, circle_y = p_metric_circle(center, promien, p_value)
                        

                            fig.add_trace(go.Scatter(
                                x=circle_x, y=circle_y,
                                mode='lines',
                                line=dict(color='RoyalBlue', width=3),
                                showlegend=False,
                                hoverinfo='none' # Ukryj informacje po najechaniu myszą
                            ))

                        elif zbior_typ == "Kula domknięta":
                            circle_x, circle_y = p_metric_circle(center, promien, p_value) # Wypełnienie kuli domkniętej (bez widocznego konturu na krawędzi wypełnienia)
                            fig.add_trace(go.Scatter(
                                    x=circle_x, y=circle_y,
                                    mode='lines',
                                    fill='toself', 
                                    fillcolor='rgba(65, 105, 225, 0.1)', # Ustaw przezroczysty kolor wypełnienia
                                    line=dict(width=0), 
                                    showlegend=False,
                                    hoverinfo='none'
                                ))

                                # Rysowanie granicy (okręgu) za pomocą go.Scatter
                            circle_x, circle_y = p_metric_circle(center, promien, p_value)

                            fig.add_trace(go.Scatter(
                                    x=circle_x, y=circle_y,
                                    mode='lines',
                                    line=dict(color='RoyalBlue', width=3),
                                    showlegend=False,
                                    hoverinfo='none' # Ukryj informacje po najechaniu myszą
                                ))

                        elif zbior_typ == "Kula otwarta":
                            circle_x, circle_y = p_metric_circle(center, promien, p_value)
                                    # Wypełnienie kuli domkniętej (bez widocznego konturu na krawędzi wypełnienia)
                            fig.add_trace(go.Scatter(
                                    x=circle_x, y=circle_y,
                                    mode='lines',
                                    fill='toself', 
                                    fillcolor='rgba(65, 105, 225, 0.1)', # Ustaw przezroczysty kolor wypełnienia
                                    line=dict(width=0), 
                                    showlegend=False,
                                    hoverinfo='none'
                            ))

                            fig.add_trace(go.Scatter(
                                x=circle_x, y=circle_y,
                                mode='lines',
                                line=dict(color='RoyalBlue', width=3, dash='dash'),
                                showlegend=False,
                                hoverinfo='none' # Ukryj informacje po najechaniu myszą
                            ))

                        fig.update_layout(
                        yaxis=dict(scaleanchor="x", scaleratio=1), # <-- Ta linia jest kluczowa i musi być w dict yaxis
                        xaxis=dict(constrain='domain'),
                        autosize=False,
                        width=600,
                        height=600,
                        plot_bgcolor='white',
                        title=f'{zbior_typ} dla p={p_value} z promieniem={promien}'
                        )

                        fig.update_xaxes(range=[center[0] - plot_range, center[0] + plot_range])
                        fig.update_yaxes(range=[center[1] - plot_range, center[1] + plot_range])
                    else:
                        def p_norm_boundary_points(center, r, p, num_points=500):
                            """
                            Zwraca punkty leżące na brzegu "kuli" w normie p (dla p > 0), w 2D.

                            Args:
                                center (tuple): środek (x, y)
                                r (float): promień
                                p (float): parametr normy p (> 0)
                                num_points (int): liczba punktów

                            Returns:
                                (x, y): współrzędne punktów na brzegu
                            """
                            if p == np.inf:
                                # Kwadrat jako granica normy nieskończoności
                                x = []
                                y = []
                                steps = num_points // 4
                                x += list(np.linspace(center[0] - r, center[0] + r, steps))
                                y += [center[1] + r] * steps
                                x += [center[0] + r] * steps
                                y += list(np.linspace(center[1] + r, center[1] - r, steps))
                                x += list(np.linspace(center[0] + r, center[0] - r, steps))
                                y += [center[1] - r] * steps
                                x += [center[0] - r] * steps
                                y += list(np.linspace(center[1] - r, center[1] + r, steps))
                                return np.array(x), np.array(y)
                            else:
                                theta = np.linspace(0, 2 * np.pi, num_points)
                                cos_t = np.cos(theta)
                                sin_t = np.sin(theta)

                                with np.errstate(divide='ignore', invalid='ignore'):
                                    denom = np.abs(cos_t) ** p + np.abs(sin_t) ** p
                                    rho = (r / denom) ** (1 / p)

                                rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)

                                x = center[0] + rho * cos_t
                                y = center[1] + rho * sin_t
                                return x, y

                        grid_res = 100
                        plot_range = promien * 1.5
                        x_grid = np.linspace(center[0] - plot_range, center[0] + plot_range, grid_res)
                        y_grid = np.linspace(center[1] - plot_range, center[1] + plot_range, grid_res)

                        distances = np.zeros((grid_res, grid_res))
                        for i, x in enumerate(x_grid):
                            for j, y in enumerate(y_grid):
                                distances[j, i] = calculate_distance(np.array([x, y]), center, p_value)

                        if zbior_typ == "Sfera":
                            circle_x, circle_y = p_norm_boundary_points(center, promien, p_value)
                        

                            fig.add_trace(go.Scatter(
                                x=circle_x, y=circle_y,
                                mode='lines',
                                line=dict(color='RoyalBlue', width=3),
                                showlegend=False,
                                hoverinfo='none' # Ukryj informacje po najechaniu myszą
                            ))

                        elif zbior_typ == "Kula domknięta":
                            circle_x, circle_y = p_norm_boundary_points(center, promien, p_value) # Wypełnienie kuli domkniętej (bez widocznego konturu na krawędzi wypełnienia)
                            fig.add_trace(go.Scatter(
                                    x=circle_x, y=circle_y,
                                    mode='lines',
                                    fill='toself', 
                                    fillcolor='rgba(65, 105, 225, 0.1)', # Ustaw przezroczysty kolor wypełnienia
                                    line=dict(width=0), 
                                    showlegend=False,
                                    hoverinfo='none'
                                ))

                                # Rysowanie granicy (okręgu) za pomocą go.Scatter
                            circle_x, circle_y = p_norm_boundary_points(center, promien, p_value)

                            fig.add_trace(go.Scatter(
                                    x=circle_x, y=circle_y,
                                    mode='lines',
                                    line=dict(color='RoyalBlue', width=3),
                                    showlegend=False,
                                    hoverinfo='none' # Ukryj informacje po najechaniu myszą
                                ))

                        elif zbior_typ == "Kula otwarta":
                            circle_x, circle_y = p_norm_boundary_points(center, promien, p_value)
                                    # Wypełnienie kuli domkniętej (bez widocznego konturu na krawędzi wypełnienia)
                            fig.add_trace(go.Scatter(
                                    x=circle_x, y=circle_y,
                                    mode='lines',
                                    fill='toself', 
                                    fillcolor='rgba(65, 105, 225, 0.1)', # Ustaw przezroczysty kolor wypełnienia
                                    line=dict(width=0), 
                                    showlegend=False,
                                    hoverinfo='none'
                            ))

                            fig.add_trace(go.Scatter(
                                x=circle_x, y=circle_y,
                                mode='lines',
                                line=dict(color='RoyalBlue', width=3, dash='dash'),
                                showlegend=False,
                                hoverinfo='none' # Ukryj informacje po najechaniu myszą
                            ))

                        fig.update_layout(
                        yaxis=dict(scaleanchor="x", scaleratio=1), # <-- Ta linia jest kluczowa i musi być w dict yaxis
                        xaxis=dict(constrain='domain'),
                        autosize=False,
                        width=600,
                        height=600,
                        plot_bgcolor='white',
                        title=f'{zbior_typ} dla p={p_value} z promieniem={promien}'
                        )

                        fig.update_xaxes(range=[center[0] - plot_range, center[0] + plot_range])
                        fig.update_yaxes(range=[center[1] - plot_range, center[1] + plot_range])
                elif n == 3: # Ogólna wizualizacja 3D za pomocą izopowierzchni
                    grid_res = 40
                    plot_range = promien * 1.5
                    X, Y, Z = np.mgrid[
                        center[0]-plot_range:center[0]+plot_range:grid_res*1j,
                        center[1]-plot_range:center[1]+plot_range:grid_res*1j,
                        center[2]-plot_range:center[2]+plot_range:grid_res*1j
                    ]
                    
                    points_in_grid = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
                    if p_value >= 1:
                        distances = np.linalg.norm(points_in_grid - center, ord=p_value, axis=1).reshape(X.shape)
                    else:
                        diffs = np.abs(points_in_grid - center)
                        distances = (diffs[:, 0]**p_value + diffs[:, 1]**p_value + diffs[:,2]**p_value)
                        distances = distances.reshape(X.shape)

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
