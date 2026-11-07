"""Équation de la chaleur 2D : du/dt = alpha * (d2u/dx2 + d2u/dy2) sur [0,1]x[0,1].

Condition initiale : u(x, y, 0) = sin(pi x) sin(pi y), u = 0 sur le bord.
Les quatre bibliothèques de base sont utilisées :
  - sympy      : solution analytique et vérification symbolique de l'EDP
  - numpy      : grille et tableaux
  - scipy      : matrice creuse du laplacien et schéma de Crank-Nicolson
  - matplotlib : figures

Usage :  python chaleur2d.py
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
import sympy as sp
from scipy.sparse.linalg import splu

ALPHA = 0.1      # diffusivité thermique
N = 50           # points intérieurs par direction
DT = 0.005       # pas de temps
T_FIN = 0.5      # durée simulée


def solution_analytique():
    """Renvoie u(x, y, t) sous forme de fonction numpy, après vérification symbolique."""
    x, y, t, a = sp.symbols("x y t alpha", positive=True)
    u = sp.sin(sp.pi * x) * sp.sin(sp.pi * y) * sp.exp(-2 * sp.pi**2 * a * t)
    residu = sp.diff(u, t) - a * (sp.diff(u, x, 2) + sp.diff(u, y, 2))
    print("Résidu de l'EDP pour la solution exacte :", sp.simplify(residu))
    return sp.lambdify((x, y, t), u.subs(a, ALPHA), "numpy")


def resoudre(n=N, dt=DT, t_fin=T_FIN):
    """Schéma de Crank-Nicolson sur une grille n x n. Renvoie grille, états et erreurs."""
    h = 1.0 / (n + 1)
    x = np.linspace(h, 1 - h, n)
    X, Y = np.meshgrid(x, x, indexing="ij")

    # laplacien 2D creuse : L = I (x) T + T (x) I
    T = sps.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n)) / h**2
    I = sps.identity(n)
    L = sps.kron(I, T) + sps.kron(T, I)

    A = sps.identity(n * n) - 0.5 * dt * ALPHA * L
    B = sps.identity(n * n) + 0.5 * dt * ALPHA * L
    lu = splu(A.tocsc())                      # factorisation faite une seule fois

    u_exact = solution_analytique()
    u = u_exact(X, Y, 0.0).ravel()
    n_pas = round(t_fin / dt)
    etats, erreurs = {0: u.reshape(n, n).copy()}, []
    for k in range(1, n_pas + 1):
        u = lu.solve(B @ u)
        erreurs.append(np.abs(u - u_exact(X, Y, k * dt).ravel()).max())
        if k in (n_pas // 2, n_pas):
            etats[k] = u.reshape(n, n).copy()
    return X, Y, etats, np.array(erreurs), dt


def tracer(X, Y, etats, erreurs, dt, fichier="chaleur2d.png"):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    vmax = etats[0].max()
    for ax, (k, u) in zip(axes[:3], etats.items()):
        im = ax.pcolormesh(X, Y, u, cmap="inferno", vmin=0, vmax=vmax, shading="auto")
        ax.set_title(f"t = {k * dt:.2f}")
        ax.set_aspect("equal")
    fig.colorbar(im, ax=axes[:3], shrink=0.8, label="température")
    axes[3].semilogy(dt * np.arange(1, len(erreurs) + 1), erreurs)
    axes[3].set_xlabel("t")
    axes[3].set_ylabel("erreur max")
    axes[3].set_title("numérique vs analytique")
    fig.savefig(fichier, dpi=120, bbox_inches="tight")
    print("Figure enregistrée :", fichier)
    return fig


if __name__ == "__main__":
    X, Y, etats, erreurs, dt = resoudre()
    print(f"Erreur maximale sur [0, {T_FIN}] : {erreurs.max():.2e}")
    tracer(X, Y, etats, erreurs, dt)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()