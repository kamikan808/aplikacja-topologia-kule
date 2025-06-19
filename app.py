import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Czy punkt leży wewnątrz kuli w Rⁿ?")

n = st.slider("Wybierz wymiar przestrzeni Rⁿ", 1, 10, 2)
point = st.text_input(f"Podaj współrzędne punktu (oddzielone przecinkami)", "1,2")
center = st.text_input(f"Podaj współrzędne środka kuli", "0,0")
radius = st.number_input("Promień kuli", min_value=0.0, value=1.0)

try:
    x = np.array([float(x.strip()) for x in point.split(",")])
    c = np.array([float(x.strip()) for x in center.split(",")])
    r = float(radius)

    if len(x) != n or len(c) != n:
        st.error("Wprowadzone wektory mają nieprawidłową długość!")
    else:
        dist = np.linalg.norm(x - c)
        st.write(f"Odległość punktu od środka: {dist:.3f}")
        if dist < r:
            st.success("Punkt leży **wewnątrz kuli otwartej**.")
        elif dist == r:
            st.info("Punkt leży **na sferze (brzegu)**.")
        else:
            st.warning("Punkt leży **poza kulą**.")

        if n in [1, 2, 3]:
            st.subheader("Wizualizacja")
            fig = plt.figure()
            if n == 1:
                a = np.linspace(c[0] - r - 1, c[0] + r + 1, 500)
                plt.plot(a, [0]*len(a), 'gray', alpha=0.5)
                plt.plot(x[0], 0, 'ro', label="Punkt")
                plt.plot(c[0], 0, 'bo', label="Środek")
                plt.axvline(c[0] - r, color='b', linestyle='--')
                plt.axvline(c[0] + r, color='b', linestyle='--')
                plt.legend()
                plt.yticks([])
            elif n == 2:
                ax = fig.add_subplot(111)
                circle = plt.Circle(c, r, color='blue', alpha=0.3)
                ax.add_patch(circle)
                plt.plot(x[0], x[1], 'ro', label="Punkt")
                plt.plot(c[0], c[1], 'bo', label="Środek")
                plt.axis('equal')
                plt.legend()
            elif n == 3:
                from mpl_toolkits.mplot3d import Axes3D
                ax = fig.add_subplot(111, projection='3d')
                u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
                xs = c[0] + r * np.cos(u) * np.sin(v)
                ys = c[1] + r * np.sin(u) * np.sin(v)
                zs = c[2] + r * np.cos(v)
                ax.plot_surface(xs, ys, zs, color='b', alpha=0.3)
                ax.scatter(*x, color='r', label='Punkt')
                ax.scatter(*c, color='blue', label='Środek')
                ax.legend()
            st.pyplot(fig)
except Exception as e:
    st.error(f"Błąd danych: {e}")
