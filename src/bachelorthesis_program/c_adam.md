# my Adam

## Motivation
Genauer Gradient ist nicht bekannt, da dieser zufallsabhängig ist. 
-> verwende stattdessen Erwartungswert des Gradienten.

Bei hoher Unsicherheit des Erwartungswertes = hoher Standardabweichung bzw. Varianz sollten kleinere Schritte gemacht werden, weil sie auch falsch sein könnten.
Bei hoher Sicherheit, also geringer Varianz, werden die Schritte größer.

Diese Intuition entspricht aber nicht dem, was Adam macht, weil $\sqrt{\mathbb{E}[g_t^2 ]} \neq \sigma(g_t)$ sobald $\mathbb{E}[g_t]\neq0$ . Wenn der Erwartungswert des Gradienten aber 0 ist, sind wir schon an einem kritischen Punkt.

---
## modified formulas
Adam uses an estimation $v_t$ of the second moment $\mathbb{E}[g_t^2]$ of $g_t$, defined as: (using element-wise operations)
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

As an improvement I propose using the following update instead:
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (g_t-\hat{m}_t)^2$$

That way $v_t$ should approximate the Variance of $g_t$. This changes nothing if the expected value $\mathbb{E}[g_t] \approx \hat m_t$ is 0.


---
## modified tensorflow Adam code
modified names `Adam` to `my_Adam`

added line `199`

modified old line `199`, now line `200`

---
