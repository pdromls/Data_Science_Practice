# Guia Definitivo: Métricas de Regressão, Diagnóstico de Erros e Ajuste de Modelos

Este material consolida os fundamentos teóricos, intuições práticas, regras de negócio e rotinas de diagnóstico para modelos de regressão em Ciência de Dados e Machine Learning.

---

## 1. Métricas de Erro Absoluto: MAE, MSE e RMSE

Toda avaliação de regressão começa no conceito de **resíduo** (ou erro pontual): a diferença entre o valor observado real ($y_i$) e a estimativa produzida pelo modelo ($\hat{y}_i$).

$$e_i = y_i - \hat{y}_i$$

Como os erros podem ser positivos ou negativos, somá-los diretamente faria com que desvios opostos se cancelassem. Cada métrica adota uma estratégia matemática distinta para contornar esse problema.

---

### 1.1 MAE (*Mean Absolute Error* — Erro Médio Absoluto)

O MAE utiliza o **módulo (valor absoluto)** para anular os sinais negativos, calculando a média aritmética linear das distâncias:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

* **Escala/Unidade:** Permanece **exatamente na mesma unidade** da variável alvo original (reais, dias, quilogramas, metros).
* **Comportamento matemático:** Penaliza os erros de forma estritamente proporcional (*linear*). Errar por 20 tem exatamente o dobro do peso de errar por 10.
* **Vínculo estatístico:** O MAE está matematicamente associado à **mediana**. O modelo treinado para minimizar o MAE busca convergir para a mediana condicional dos dados, tornando-o altamente resistente a desvios extremos (*outliers*).

---

### 1.2 MSE (*Mean Squared Error* — Erro Quadrático Médio)

Em vez de aplicar o módulo, o MSE **eleva cada erro ao quadrado** antes de calcular a média aritmética:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

* **Escala/Unidade:** Fica na **unidade original elevada ao quadrado** ($\text{reais}^2$, $\text{dias}^2$, $\text{kg}^2$), dificultando a interpretação direta pelo negócio.
* **Comportamento matemático:** Atua como um fiscal implacável. Pequenos desvios recebem penalidades baixas, mas erros grandes são multiplicados de forma desproporcional.
* **Uso computacional:** Por ser uma parábola contínua, suave e derivável em todos os pontos, o MSE é a função de perda padrão utilizada para calibrar algoritmos de otimização (como OLS e Gradiente Descendente).

---

### 1.3 RMSE (*Root Mean Squared Error* — Raiz do Erro Quadrático Médio)

O RMSE extrai a raiz quadrada do MSE, restaurando a grandeza dimensional dos dados:

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

* **Escala/Unidade:** Retorna para a **mesma unidade** da variável original.
* **Comportamento matemático:** Mantém a sensibilidade severa do MSE aos grandes erros, mas expressa essa penalidade na unidade que os analistas e gestores compreendem.
* **Vínculo estatístico:** O RMSE está diretamente ligado à **média**. Se houver desvios gigantes na base, o RMSE se distanciará bruscamente para cima em relação ao MAE.

---

### 1.4 Exemplo Comparativo: Erros Constantes vs. Erro Extremo

Considere o aluguel real de 3 imóveis: **R$ 1.000, R$ 2.000 e R$ 3.000**.

#### Modelo A (Erros pequenos e homogêneos)
* Previsões: R$ 1.100, R$ 2.100, R$ 3.100 (resíduos de $+100$ em cada)
* $\text{MAE} = \frac{100 + 100 + 100}{3} = \mathbf{100\text{ reais}}$
* $\text{MSE} = \frac{100^2 + 100^2 + 100^2}{3} = \mathbf{10.000\text{ reais}^2}$
* $\text{RMSE} = \sqrt{10.000} = \mathbf{100\text{ reais}}$
* *Quando todos os erros têm a mesma magnitude, $\text{MAE} = \text{RMSE}$.*

#### Modelo B (Acertos perfeitos e uma catástrofe isolada)
* Previsões: R$ 1.000, R$ 2.000, R$ 3.300 (resíduos de $0, 0, +300$)
* $\text{MAE} = \frac{0 + 0 + 300}{3} = \mathbf{100\text{ reais}}$
* $\text{MSE} = \frac{0^2 + 0^2 + 300^2}{3} = \mathbf{30.000\text{ reais}^2}$
* $\text{RMSE} = \sqrt{30.000} \approx \mathbf{173{,}20\text{ reais}}$

**Conclusão prática:** Para o MAE, ambos os modelos são idênticos (erro médio de R$ 100). Para o RMSE, o Modelo B é 73% pior, denunciando o impacto do erro grave isolado de R$ 300.

---

## 2. Métricas de Erro vs. Coeficientes de Determinação ($R^2$ e $R^2$ Ajustado)

| Aspecto | MAE / MSE / RMSE | $R^2$ Tradicional | $R^2$ Ajustado |
| :--- | :--- | :--- | :--- |
| **Objetivo** | Medir a magnitude absoluta do erro | Medir a proporção explicada dos dados | Proporção explicada penalizando complexidade |
| **Escala** | Dependente dos dados (tem unidade) | Adimensional / Normalizado (0% a 100%) | Adimensional / Normalizado |
| **Compara bases diferentes?** | Não | Sim | Sim |
| **Ponto ideal** | Quanto mais próximo de $0$ | Quanto mais próximo de $1$ ($100\%$) | Quanto mais próximo de $1$ |
| **Variáveis inúteis** | Reduzem levemente o erro no treino | Aumenta ou congela (não penaliza) | **Cai** (revela *overfitting*) |

### 2.1 A Analogia do Tiro ao Alvo
* **$R^2$ (A Nota do Jogo):** Responde *"Qual foi a sua taxa de aproveitamento geral?"* (ex.: 85% de eficácia).
* **MAE / RMSE (A Distância Física):** Responde *"A quantos centímetros da mosca do alvo a flecha caiu?"* (ex.: errou por 3 cm).

---

## 3. O Dilema Prático: MAE vs. RMSE no Dia a Dia

### 3.1 O MAE ignora os outliers?
Não. O MAE trata todos os erros com **peso linear**:
* Erro padrão $= 2 \implies \text{Peso } 2$
* Outlier $= 20 \implies \text{Peso } 20$ (já pesa 10 vezes mais).

No RMSE, a ponderação é **quadrática**:
* Erro padrão $= 2 \implies 2^2 = 4$
* Outlier $= 20 \implies 20^2 = 400$ (pesa 100 vezes mais).

### 3.2 Por que o RMSE é o padrão mais adotado no mercado?
Na maioria das operações reais de negócios, o impacto financeiro de um erro cresce de forma exponencial:
1. **Erro aceitável:** Errar o aluguel em R$ 100 faz o imóvel alugar com poucos dias de atraso.
2. **Erro severo:** Errar o aluguel em R$ 1.800 faz o imóvel encalhar por um ano, gerando prejuízos com condomínio, perda do cliente e atrito judicial.

Se avaliarmos pelo MAE, esse erro de R$ 1.800 seria considerado apenas "seis erros de R$ 300". Pelo RMSE, a penalidade explode para $1.800^2$, forçando o modelo matemático a eliminar anomalias graves.

### 3.3 A Distinção entre Tipos de Outliers
* **Outlier por Falha de Coleta (O caso do motoboy):** O entregador teve o celular roubado e a entrega registrou 3 dias em vez de 30 minutos. 
  * *Solução:* Trata-se na etapa de **Limpeza de Dados (*Data Cleaning*)** por meio de regras de corte, e não pela troca de métricas.
* **Outlier por Fenômeno Real Perigoso (Carga de ponte, dosagem médica):** O evento é raro, mas real e destrutivo.
  * *Solução:* Exige o uso do **RMSE**, garantindo que o algoritmo não tolere previsões disparadas da realidade.

---

## 4. Diagnóstico de Negócio e Análise de Resíduos

Métricas escalares (como `RMSE: 620.00` e `MAE: 150.00`) são **luzes no painel do carro**: avisam que há um problema, mas não mostram qual peça quebrou. 

Para identificar falhas, os cientistas de dados realizam a **Análise de Resíduos voltada ao Negócio**:

```text
[Métricas Gerais: RMSE alto vs. MAE baixo]
                   ↓ (Sinaliza presença de anomalias)
[Passo 1: Tabela de Resíduos Ordenada]
                   ↓ (Isola as linhas com as piores discrepâncias)
[Passo 2: Gráfico Actual vs. Predicted com Faixa de Tolerância]
                   ↓ (Visualiza violações de segurança operacional)
[Passo 3: Feature Importance e Coeficientes]
                   ↓ (Identifica variáveis ausentes ou desbalanceadas)
```

### 4.1 Script em Python: Scatter Plot de Tolerância e Filtro Crítico

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    # Fallback para versões do scikit-learn anteriores à 1.4
    from sklearn.metrics import mean_squared_error
    root_mean_squared_error = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))

# 1. Simulação do DataFrame de avaliação pós-previsão
df_avaliacao = pd.DataFrame(
    {
        "imovel_id": [101, 102, 103, 104, 105, 106],
        "bairro": [
            "Centro",
            "Botafogo",
            "Tijuca",
            "Leblon",
            "Flamengo",
            "Copacabana",
        ],
        "tipo": ["Padrão", "Padrão", "Padrão", "Cobertura", "Padrão", "Padrão"],
        "preco_real": [2000, 2500, 1800, 2000, 3100, 2200],
        "preco_previsto": [2080, 2450, 1920, 3800, 3150, 2180],
    }
)

# 2. Engenharia de Resíduos e Regras de Negócio
df_avaliacao["erro_absoluto"] = (
    df_avaliacao["preco_real"] - df_avaliacao["preco_previsto"]
).abs()
df_avaliacao["erro_percentual"] = (
    df_avaliacao["erro_absoluto"] / df_avaliacao["preco_real"]
) * 100

# Limite estabelecido com a gestão: Erro máximo tolerado de R$ 500
LIMITE_TOLERANCIA_REAIS = 500
df_avaliacao["status_negocio"] = np.where(
    df_avaliacao["erro_absoluto"] > LIMITE_TOLERANCIA_REAIS,
    "Violação Crítica",
    "Aceitável",
)

# 3. Métricas Globais
mae = mean_absolute_error(
    df_avaliacao["preco_real"], df_avaliacao["preco_previsto"]
)
rmse = root_mean_squared_error(
    df_avaliacao["preco_real"], df_avaliacao["preco_previsto"]
)

print(f"--- Desempenho Global ---")
print(f"MAE:  R$ {mae:.2f}")
print(f"RMSE: R$ {rmse:.2f}  <-- Gap alto indica presença de outliers!")

# 4. Tabela de Casos Críticos para Ação Imediata
print("\n--- Casos que Violam a Regra de Negócio ---")
casos_criticos = df_avaliacao[
    df_avaliacao["status_negocio"] == "Violação Crítica"
].sort_values(by="erro_absoluto", ascending=False)
print(
    casos_criticos[
        [
            "imovel_id",
            "bairro",
            "tipo",
            "preco_real",
            "preco_previsto",
            "erro_absoluto",
        ]
    ]
)

# 5. Visualização Gráfica com Faixa de Tolerância
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df_avaliacao,
    x="preco_real",
    y="preco_previsto",
    hue="status_negocio",
    palette={"Aceitável": "forestgreen", "Violação Crítica": "crimson"},
    s=120,
    style="status_negocio",
)

# Reta de Previsão Perfeita (y = x)
min_val = min(
    df_avaliacao["preco_real"].min(), df_avaliacao["preco_previsto"].min()
)
max_val = max(
    df_avaliacao["preco_real"].max(), df_avaliacao["preco_previsto"].max()
)
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="black",
    linestyle="--",
    label="Previsão Perfeita (y = ŷ)",
)

# Linhas de Limite Operacional
plt.plot(
    [min_val, max_val],
    [min_val + LIMITE_TOLERANCIA_REAIS, max_val + LIMITE_TOLERANCIA_REAIS],
    color="gray",
    linestyle=":",
    alpha=0.7,
    label=f"Tolerância (+R$ {LIMITE_TOLERANCIA_REAIS})",
)
plt.plot(
    [min_val, max_val],
    [min_val - LIMITE_TOLERANCIA_REAIS, max_val - LIMITE_TOLERANCIA_REAIS],
    color="gray",
    linestyle=":",
    alpha=0.7,
    label=f"Tolerância (-R$ {LIMITE_TOLERANCIA_REAIS})",
)

plt.title("Diagnóstico de Previsões: Real vs. Previsto com Faixa de Segurança")
plt.xlabel("Preço Real (R$)")
plt.ylabel("Preço Previsto (R$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 5. Por que os Modelos Cometem Erros Grotescos?

### 5.1 As Quatro Falhas Estruturais mais Comuns
1. **Dados Esparsos (*Sparse Data*):** A base tem 10.000 apartamentos comuns e apenas 2 coberturas luxuosas. O modelo generaliza erroneamente que toda cobertura deve atingir valores estratosféricos.
2. **Viés da Variável Omitida (*Omitted Variable Bias*):** O modelo avalia apenas tamanho e bairro, desconhecendo fatores depreciativos graves (imóvel deteriorado, litígio de herança ou ausência de vagas).
3. **Extrapolação Linear Indevida:** A regressão assume relações fixas. Uma área externa de 300 m² é multiplicada pelo valor do metro quadrado construído tradicional, distorcendo o cálculo.
4. **Superajuste (*Overfitting*):** O algoritmo memorizou ruídos específicos dos dados de treino em vez de identificar regras gerais de mercado.

---

### 5.2 O Mecanismo de Ajuste de Pesos na Regressão Linear

Na Regressão Linear, os parâmetros são coeficientes multiplicativos contínuos ($w$):

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_k x_k$$

Esses pesos são ajustados para minimizar o somatório do erro quadrático:

#### Método Analítico: OLS (Mínimos Quadrados Ordinários)
Calcula a inclinação ideal de uma só vez através de inversão de matrizes:

$$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

* *Origem do erro:* Como eleva os desvios ao quadrado, um dado incorreto puxa a reta inteira em sua direção como uma alavanca, degradando os pesos de todas as outras previsões.

#### Método Iterativo: Gradiente Descendente
Dá passos graduais na direção oposta à derivada do MSE:

$$w_j \leftarrow w_j - \alpha \cdot \left[ -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i) x_{i,j} \right]$$

---

### 5.3 O Mecanismo de Aprendizado nas Árvores de Decisão

Árvores de Decisão de regressão **não possuem pesos multiplicativos ($w$)**. Elas aprendem por meio de partições recursivas no espaço amostral:

1. **Varredura de Divisões (*Splits*):** O algoritmo testa todos os atributos e limites numéricos possíveis (ex.: $\text{Quartos} \le 2$, $\text{Tamanho} \le 80$).
2. **Maximização da Redução da Variância:** Escolhe o ponto de corte que maximiza a queda do erro quadrático médio ponderado entre os nós resultantes.
3. **Cálculo da Folha Terminal:** A previsão atribuída a qualquer nova observação que atinja uma folha é a **média aritmética simples** dos valores históricos alocados naquele nó.

#### Por que árvores cometem erros grotescos?
* **Folhas Isoladas (Overfitting):** Se uma árvore não tiver controle de profundidade máxima (`max_depth`), ela pode isolar um imóvel atípico em uma folha individual. Qualquer dado subsequente alocado ali herdará diretamente aquele valor extremo.
* **Incapacidade de Extrapolação:** Árvores produzem degraus constantes e não traçam trajetórias lineares. Elas não conseguem prever tendências que estejam além do valor mínimo ou máximo contido nos dados de treinamento.