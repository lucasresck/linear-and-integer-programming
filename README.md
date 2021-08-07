# linear-integer-programming

Problem sets and implementations for [Linear and Integer Programming](https://emap.fgv.br/disciplina/graduacao/programacao-linear-inteira) (2021, FGV).
Professor: [Luciano Guimarães](https://matematicamente.xyz/prof-luciano/).

## Summary

In this repository, you will find implementations to Linear and Integer Programming (LIP) algorithms, e.g., simplex and interior-point methods, and several notebooks with solutions to LIP exercises and assignments. This course covered mainly
- linear programming,
- network models, and
- integer programming.

### Implementations

- [Simplex Python implementation](https://github.com/lucasresck/lip/blob/main/simplex.py) and [examples](https://github.com/lucasresck/lip/blob/main/notebooks/simplex_examples.ipynb);
- [Interior-point method Python implementation](https://github.com/lucasresck/lip/blob/main/interior_point.py) (short-step primal-dual path-following algorithm) and [examples](https://github.com/lucasresck/lip/blob/main/notebooks/interior_point_examples.ipynb).

### Assignments

- [Mixed-Integer Linear Programming (MILP) for Local Interpretable Model-agnostic Explanations (LIME)](https://github.com/lucasresck/lip/blob/main/notebooks/lime.ipynb)

## Examples

### Interior-point method

Look how an interior-point method solves a linear programming problem, with the planes representation the restrictions and the points representing the iterations of the solution:

Small step size             |  Big step size
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/lucasresck/linear-integer-programming/main/images/example_1.png)  |  ![](https://raw.githubusercontent.com/lucasresck/linear-integer-programming/main/images/example_2.png)

### Local Interpretable Model-agnostic Explanations (LIME)

Suppose a text classification machine learning model has taken some decision and you want to understand the reasons behind it. We can apply a variation of [LIME](https://arxiv.org/abs/1602.04938), using mixed-integer programming, to solve this. For example, if we want to understand why the model thinks the following text is a positive movie review,
