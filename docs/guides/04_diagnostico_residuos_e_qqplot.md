# Guia Definitivo: Diagnóstico de Resíduos, Suposições de Gauss-Markov e Análise do Q-Q Plot

Este material consolida o protocolo completo de **diagnóstico de resíduos** em modelos de regressão, detalhando a interpretação visual do **Q-Q Plot**, as causas de desvios da normalidade, as violações das premissas de Gauss-Markov e o plano de ação prático para remediar cada anomalia.

---

## 1. O que são Resíduos e por que Auditá-los?

Toda regressão gera estimativas ($\hat{y}_i$). O **resíduo** ($e_i$) é o erro cometido pelo modelo para cada observação individual:

$$e_i = y_i - \hat{y}_i \quad (\text{Valor Real} - \text{Valor Previsto})$$

Em Machine Learning, métricas agregadas como $R^2$, MAE e RMSE mostram o **tamanho médio do erro**, mas são cegas quanto à **natureza estrutural do erro**.

> **A Regra de Ouro dos Resíduos:**  
> O modelo preditivo deve absorver todo o sinal causal e previsível contido nos dados.  
> O que sobra (os resíduos) deve ser puramente **ruído branco aleatório**, desprovido de qualquer padrão sistemático, tendência ou agrupamento.

Se os resíduos exibem padrões visíveis, significa que o algoritmo falhou em capturar alguma dinâmica fundamental dos dados (assimetria, não-linearidade, heterocedasticidade ou caudas pesadas).

---

## 2. As Suposições Clássicas de Gauss-Markov (Teorema BLUE)

Para que os coeficientes da regressão linear analítica (OLS) sejam considerados os **Melhores Estimadores Lineares Não Viesados** (*Best Linear Unbiased Estimators* — **BLUE**), as seguintes premissas devem ser satisfeitas:

1. **Linearidade nos Parâmetros:** A relação matemática entre os atributos $X$ e a média condicional de $y$ é linear.
2. **Média Nula dos Erros:** A média aritmética de todos os resíduos é rigorosamente zero ($\mathbb{E}[e_i] = 0$).
3. **Homocedasticidade (Variância Constante):** A dispersão dos erros é constante para todos os níveis de previsão ($\text{Var}(e_i \mid X) = \sigma^2$).
4. **Independência / Não-Autocorrelação:** Os erros de observações distintas não compartilham dependência temporal ou sequencial ($\text{Cov}(e_i, e_j) = 0$ para $i \neq j$).
5. **Normalidade dos Erros ($e_i \sim \mathcal{N}(0, \sigma^2)$):**  
   Os resíduos seguem uma distribuição normal. Embora o OLS não exija normalidade para encontrar os coeficientes ótimos, a normalidade é **estritamente indispensável** para que os testes de hipótese (testes $t$ e $F$), $p$-valores e intervalos de confiança de 95% sejam matematicamente válidos.

---

## 3. Diagnóstico Visual 1: Resíduos vs. Valores Ajustados

É o gráfico cartesiano com os **Valores Ajustados ($\hat{y}$)** no eixo horizontal e os **Resíduos ($e$)** no eixo vertical, com uma linha de referência horizontal em zero ($e = 0$).

```text
       Resíduo (e)
          ^
     +10  |       *       *       *       *       [ PADRÃO IDEAL: CÉU ESTRELADO ]
          |   *       *       *       *       *   Dispersão uniforme e homogênea
       0 -+--------------------------------------- (Linha Zero)
          |     *       *       *       *       * Variância constante (Homocedasticidade)
     -10  |         *       *       *       *     Sem curvaturas (Linearidade confirmada)
          +---------------------------------------> Valor Previsto (ŷ)
```

### 3.1 Padrão Ideal: "Céu Estrelado"
Pontos dispersos de maneira uniforme, simétrica e puramente aleatória em uma faixa constante ao redor da linha tracejada de zero, sem formatos reconhecíveis.

### 3.2 Anomalia 1: Formato em "Funil" ou "Megafone" (Heterocedasticidade)
* **Sintoma Visual:** A nuvem de pontos começa estreita na esquerda e vai se alargando progressivamente conforme $\hat{y}$ aumenta (ou vice-versa).
* **Significado Teórico:** A variância do erro não é constante. O modelo é muito preciso para valores baixos, mas torna-se progressivamente instável para previsões altas.
* **Causas Comuns:** Dados econômicos (famílias com baixa renda têm gastos de alimentação concentrados; famílias de alta renda variam enormemente seus gastos).
* **Soluções Práticas:**
  1. **Transformação Logarítmica:** Aplicar $\log(y)$ estabiliza a variância multiplicativa.
  2. **Regressão por Mínimos Quadrados Ponderados (WLS — *Weighted Least Squares*):** Pondera cada observação pelo inverso da sua variância estimada.
  3. **Erros-Padrão Robustos de Huber-White (Sandwich Estimators):** No `statsmodels`, utilizar `cov_type='HC3'` para recalcular p-valores imunes à heterocedasticidade.

### 3.3 Anomalia 2: Formato em "U" ou Parábola (Não-Linearidade)
* **Sintoma Visual:** Os resíduos formam uma curvatura convexa ou côncava, concentrando erros negativos no meio e positivos nas extremidades.
* **Significado Teórico:** Violação da premissa de linearidade. A relação subjacente entre $X$ e $y$ é uma curva.
* **Soluções Práticas:**
  1. Inclusão de termos polinomiais ($X^2$, $X^3$) via `PolynomialFeatures`.
  2. Transformações não lineares nos preditores ($\sqrt{X}$, $\log(X)$).
  3. Adoção de modelos não lineares flexíveis (Splines, Árvores de Decisão, Gradient Boosting).

---

## 4. Diagnóstico Visual 2: O Q-Q Plot (*Quantile-Quantile Plot*)

O **Q-Q Plot** é o instrumento gráfico definitivo para avaliar a normalidade dos resíduos.

* **Eixo Horizontal (X):** Quantis Teóricos de uma distribuição Normal padronizada $\mathcal{N}(0, 1)$.
* **Eixo Vertical (Y):** Quantis Amostrais observados dos resíduos padronizados do modelo.
* **Linha Diagonal de 45º (Vermelha):** A trajetória exata que os pontos deveriam seguir caso a distribuição empírica fosse perfeitamente normal.

---

## 5. Anatomia das Anomalias no Q-Q Plot e Como Corrigi-las

Abaixo detalhamos as 3 principais anomalias visuais observadas no Q-Q Plot, suas causas estatísticas e o protocolo de correção:

```text
  [ Formato em "Banana" ]           [ Formato em "S" ]             [ Outliers Isolados ]
      Assimetria (Skew)           Caudas Pesadas (Kurtosis)         Aberrações Pontuais

        |      *                      |       *                      |          * (outlier)
        |     *                       |      *                       |       *
        |    *                        |     *                        |      *
        |   *                         |    *                         |     *
        |  *                          |   *                          |    *
        | *                           |  *                           |   *
        |*                            | *                            |  *
        +------------                 +------------                  +------------
       Curvatura em arco             Descolamento nas pontas         Pontos soltos nos extremos
```

---

### Caso 1: Formato em "Banana" (Curvatura em Arco — Assimetria / *Skewness*)

#### O Diagnóstico
Os pontos formam uma curvatura convexa ou côncava contínua, cruzando a linha diagonal em forma de arco.
* **Causa:** A distribuição da variável resposta ou dos resíduos é **fortemente assimétrica**. Há uma concentração maciça de valores moderados e uma cauda longa unilateral (ex.: salários, preços de imóveis, sinistros de seguros).

#### Protocolo de Remediação
1. **Transformação Logarítmica ($\log(y)$ ou $\log(1+y)$):**
   * É a solução padrão do mercado. A função logarítmica comprime a cauda longa à direita e expande os valores menores, simetrizando a distribuição dos erros.
   ```python
   import numpy as np

   y_train_log = np.log1p(y_train)  # log(1 + y) protege contra valores iguais a zero
   model.fit(X_train, y_train_log)
   ```
2. **Transformação Box-Cox ou Yeo-Johnson:**
   * Quando o logaritmo puro não for suficiente, o algoritmo de Box-Cox identifica analiticamente o expoente de potência ótimo ($\lambda$) para normalizar os dados:
   ```python
   from scipy import stats

   y_transformed, lmbda = stats.boxcox(y_train)
   print(f"Expoente ótimo de Box-Cox (lambda): {lmbda:.4f}")
   ```
   *(Nota: O Box-Cox exige valores estritamente positivos $y > 0$. Para dados com valores nulos ou negativos, utiliza-se a transformação de Yeo-Johnson via `PowerTransformer(method='yeo-johnson')`).*
3. **Adição de Termo Polinomial ($X^2$):**
   * Se a assimetria decorre de aceleração não-linear do alvo em função do preditor.

---

### Caso 2: Formato em "S" Pronunciado (Caudas Pesadas / Curtose Excessiva)

#### O Diagnóstico
Os pontos seguem a linha central, mas **se descolam bruscamente nas duas extremidades opostas**: a ponta esquerda curva para baixo e a ponta direita curva para cima (ou vice-versa), formando um "S".
* **Causa:** O fenômeno das **caudas pesadas** (*heavy tails*). Eventos com erros muito grandes ocorrem na amostra com frequência muito maior do que a distribuição normal prevê (excesso de curtose).

#### Por que o OLS Tradicional Sofre com Caudas Pesadas?
O algoritmo de Mínimos Quadrados minimiza a soma dos erros **elevados ao quadrado** ($\sum e_i^2$). Um único erro de magnitude $20$ tem peso de $400$ no cálculo, atuando como uma alavanca destrutiva que gira a reta inteira para tentar acomodar o ponto extremo, degradando os parâmetros de todos os outros registros.

#### Protocolo de Remediação
1. **Regressão Robusta de Huber (`HuberRegressor`):**
   * Substitui a função de perda pura do MSE por uma perda híbrida: trata erros pequenos como quadráticos ($e^2$) e erros grandes como lineares ($|e|$). Com isso, o estimador torna-se naturalmente imune ao efeito alavanca das caudas pesadas.
   ```python
   from sklearn.linear_model import HuberRegressor

   huber = HuberRegressor(epsilon=1.35)
   huber.fit(X_train, y_train)
   ```
2. **Regressão Quantílica (`QuantileRegressor`):**
   * Em vez de modelar a média condicional $\mathbb{E}[y \mid X]$, modela a **mediana condicional** (percentil 50), que é insensível a valores extremos de cauda.
3. **Regressão com Perda MAE (`SGDRegressor(loss='epsilon_insensitive')`):**
   * Penalização puramente linear das distâncias.

---

### Caso 3: Pontos Isolados e Distantes nas Extremidades (*Outliers* Pontuais)

#### O Diagnóstico
A esmagadora maioria dos pontos permanece colada em cima da linha diagonal, mas **1, 2 ou 3 pontos encontram-se isolados a quilômetros de distância** nos extremos superiores ou inferiores.
* **Causa:** Erros operacionais, falhas de medição em sensores, digitações incorretas de operadores (ex.: peso inserido como 750 kg em vez de 75 kg) ou anomalias cadastrais raras.

#### Protocolo de Remediação
1. **Auditoria Forense dos Dados:**
   * Localize os registros correspondentes no DataFrame e audite seu contexto de negócio:
   ```python
   # Identificando as observações com os maiores resíduos absolutos
   df_outliers = X_train.copy()
   df_outliers['y_real'] = y_train
   df_outliers['y_pred'] = model.predict(X_train)
   df_outliers['residuo'] = (
       df_outliers['y_real'] - df_outliers['y_pred']
   ).abs()
   piores_casos = df_outliers.sort_values(by='residuo', ascending=False).head(5)
   print(piores_casos)
   ```
2. **Cálculo da Distância de Cook (*Cook's Distance*):**
   * Avalia a alavancagem estatística do ponto. Observações com Distância de Cook $> \frac{4}{n}$ puxam a reta de regressão desproporcionalmente sozinhas.
3. **Algoritmo RANSAC (*Random Sample Consensus*):**
   * O RANSAC seleciona subconjuntos aleatórios de dados, ajusta modelos candidatos e separa automaticamente as observações entre **Inliers** (dados honestos) e **Outliers** (anomalias), descartando estes últimos do cálculo da reta final.
   ```python
   from sklearn.linear_model import RANSACRegressor

   ransac = RANSACRegressor(random_state=42)
   ransac.fit(X_train, y_train)

   inlier_mask = ransac.inlier_mask_
   print(
       f"Observações consideradas legítimas: {inlier_mask.sum()} de {len(y_train)}"
   )
   ```

---

## 6. Testes Estatísticos Numéricos Formais

Embora o exame visual do Q-Q Plot e do gráfico de dispersão de resíduos seja a ferramenta primária, os cientistas de dados validam suas conclusões por meio de **testes estatísticos de hipótese formais**:

| Teste Estatístico | Suposição Auditada | Biblioteca Python | Regra de Decisão do $p$-valor |
| :--- | :--- | :--- | :--- |
| **Shapiro-Wilk** | Normalidade dos resíduos (ideal para $n < 50$) | `scipy.stats.shapiro` | Se $p > 0.05 \implies$ **Resíduos são normais** (Não rejeita $H_0$). |
| **Jarque-Bera (JB)** | Normalidade combinada via Skewness e Kurtosis | `scipy.stats.jarque_bera` | Se $p > 0.05 \implies$ **Resíduos são normais**. |
| **Breusch-Pagan** | Homocedasticidade (variância constante) | `statsmodels.stats.diagnostic.het_breuschpagan` | Se $p > 0.05 \implies$ **Homocedasticidade confirmada** (Sem funil). |
| **Durbin-Watson** | Independência / Ausência de Autocorrelação | `statsmodels.stats.stattools.durbin_watson` | Valor próximo de **$2.0$** indica resíduos independentes. |

---

## 7. Script Completo de Diagnóstico de Resíduos em Python

Abaixo apresentamos um script modular completo para gerar o painel 2x2 padrão de auditoria de resíduos:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def plot_residual_diagnostics(y_true, y_pred, feature_name="Alvo"):
    """
    Gera o painel 2x2 com os 4 gráficos mandatórios de auditoria de resíduos:
    1. Resíduos vs. Valores Ajustados (Linearidade e Homocedasticidade)
    2. Q-Q Plot de Normalidade com reta teórica de referência
    3. Histograma com KDE dos Resíduos (Simetria e Curtose)
    4. Valores Reais vs. Previstos com linha ideal y = x
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Resíduos vs Ajustados (Homocedasticidade)
    sns.scatterplot(
        x=y_pred, y=residuals, color="#2b5c8f", s=60, alpha=0.85, ax=axes[0, 0]
    )
    axes[0, 0].axhline(0, color="crimson", linestyle="--", linewidth=1.5)
    axes[0, 0].set_title(
        "1. Resíduos vs. Valores Ajustados\n(Homocedasticidade & Linearidade)",
        fontweight="bold",
    )
    axes[0, 0].set_xlabel(f"Valores Ajustados ({feature_name})")
    axes[0, 0].set_ylabel("Resíduos (Real - Previsto)")

    # 2. Q-Q Plot de Normalidade
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].get_lines()[0].set_color("#2b5c8f")
    axes[0, 1].get_lines()[0].set_markersize(6)
    axes[0, 1].get_lines()[1].set_color("crimson")
    axes[0, 1].set_title(
        "2. Q-Q Plot dos Resíduos\n(Aderência à Distribuição Normal)",
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("Quantis Teóricos Normais")
    axes[0, 1].set_ylabel("Quantis Amostrais dos Resíduos")

    # 3. Histograma + Densidade KDE dos Resíduos
    sns.histplot(
        residuals, kde=True, color="#2b5c8f", bins=15, ax=axes[1, 0], stat="density"
    )
    # Curva normal teórica sobreposta
    mu, std = stats.norm.fit(residuals)
    x_axis = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1, 0].plot(
        x_axis,
        stats.norm.pdf(x_axis, mu, std),
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label="Normal Teórica",
    )
    axes[1, 0].set_title(
        "3. Distribuição dos Resíduos (KDE)\n(Simetria e Sino Gaussiano)",
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("Magnitude do Resíduo")
    axes[1, 0].set_ylabel("Densidade")
    axes[1, 0].legend()

    # 4. Valores Reais vs Previstos
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    sns.scatterplot(
        x=y_true, y=y_pred, color="#2b5c8f", s=60, alpha=0.85, ax=axes[1, 1]
    )
    axes[1, 1].plot(
        [min_val, max_val],
        [min_val, max_val],
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label="Previsão Perfeita (y = ŷ)",
    )
    axes[1, 1].set_title(
        "4. Real vs. Previsto\n(Alinhamento Geral do Ajuste)", fontweight="bold"
    )
    axes[1, 1].set_xlabel(f"Valor Real ({feature_name})")
    axes[1, 1].set_ylabel(f"Valor Previsto ({feature_name})")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Testes Numéricos
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print("--- Testes Estatísticos Formais de Resíduos ---")
    print(f"Média dos Resíduos:   {residuals.mean():.6e} (Esperado: ~0)")
    print(
        f"Teste de Shapiro-Wilk: Estatística = {shapiro_stat:.4f} | p-valor = {shapiro_p:.4f}"
    )
    if shapiro_p > 0.05:
        print(
            "Conclusão: Resíduos seguem distribuição normal comprovada (p > 0.05)."
        )
    else:
        print(
            "Conclusão: Resíduos violam a suposição de normalidade (p <= 0.05)."
        )
```

---

## 8. Tabela Síntese de Diagnóstico e Tomada de Decisão

| O que o Gráfico Denuncia? | Causa Raiz Estatística | Impacto no Modelo | Remédio / Ação Recomendada |
| :--- | :--- | :--- | :--- |
| **Resíduos em formato de funil** | Heterocedasticidade | $p$-valores e intervalos de confiança tornam-se inválidos. | Aplicar **$\log(y)$**, WLS ou erros robustos `HC3`. |
| **Resíduos em formato parabólico (U)** | Não-Linearidade | Alto viés (*underfitting*), previsões incorretas no centro. | Adicionar **termos polinomiais ($X^2$)** ou usar Árvores. |
| **Q-Q Plot em formato de Banana** | Forte Assimetria (*Skewness*) | Previsões enviesadas para a cauda longa. | Aplicar **transformação $\log(y)$** ou **Box-Cox**. |
| **Q-Q Plot em formato de "S"** | Caudas Pesadas / Curtose | OLS puxado por erros extremos de cauda. | Adotar **Regressão Robusta (`HuberRegressor`)**. |
| **Pontos isolados nas pontas** | *Outliers* pontuais / Erro humano | Reta inteira rotacionada pela alavancagem. | Auditar o registro ou aplicar **`RANSACRegressor`**. |
| **Pontos alinhados e nuvem uniforme** | Todas as premissas válidas | Estimador **BLUE** com garantia estatística máxima. | **Manter o modelo OLS tradicional em produção.** |
