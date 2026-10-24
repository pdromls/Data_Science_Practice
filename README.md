# Data Science & Machine Learning Practice 🚀

Repositório dedicado ao acompanhamento prático, anotações e projetos de Data Science e Machine Learning.

O foco é ir além da reprodução das aulas: os notebooks trazem anotações críticas, comparações entre diferentes abordagens e boas práticas.

## 📌 Conteúdo Estudado
- Configuração de ambiente e boas práticas de versionamento
- **Módulo 01: Análise Exploratória de Dados (EDA)**
  - [`notebooks/01_eda/01_eda_flight_price.ipynb`](notebooks/01_eda/01_eda_flight_price.ipynb): EDA estatística e visual sobre preços de passagens aéreas, preservação de séries temporais e análise de outliers.
- **Módulo 02: Feature Engineering & Encoding**
  - [`notebooks/02_feature_engineering/02_feature_engineering_flight_price.ipynb`](notebooks/02_feature_engineering/02_feature_engineering_flight_price.ipynb): Pré-processamento modular (funções reutilizáveis), engenharia de features temporais, prevenção de vazamento de dados (split pré-encoding), ColumnTransformer e normalização do target.
- **Módulo 03: Modelos Lineares — Regressão Linear Simples**
  - [`notebooks/03_linear_regression/03_simple_linear_regression.ipynb`](notebooks/03_linear_regression/03_simple_linear_regression.ipynb): Pipeline completo de regressão linear simples, treinamento na escala original vs. padronizada, correção do intercepto no `statsmodels`, avaliação contra baseline ingênuo (`DummyRegressor`) e diagnóstico rigoroso de resíduos (Gauss-Markov).

## 📚 Guias Técnicos e Teóricos
Na pasta [`docs/guides/`](docs/guides/), encontram-se guias detalhados de aprofundamento:
- [`01_metricas_e_diagnostico_regressao.md`](docs/guides/01_metricas_e_diagnostico_regressao.md): Métricas de erro (MAE, MSE, RMSE) e interpretação de negócio.
- [`02_r2_e_r2_ajustado.md`](docs/guides/02_r2_e_r2_ajustado.md): Decomposição da variância, $R^2$ e a mecânica da penalidade no $R^2$ Ajustado.
- [`03_vies_variancia_ols_vs_gradiente.md`](docs/guides/03_vies_variancia_ols_vs_gradiente.md): Dilema Viés-Variância, catálogo prático de algoritmos e OLS vs. Gradiente Descendente.
- [`04_diagnostico_residuos_e_qqplot.md`](docs/guides/04_diagnostico_residuos_e_qqplot.md): Diagnóstico de resíduos, premissas de Gauss-Markov e guia de correção para anomalias no Q-Q Plot.

*(Novos tópicos e módulos de Machine Learning serão adicionados aqui conforme forem concluídos).*

---
> 💡 O roadmap completo e o acompanhamento das próximas etapas estão organizados nas **Issues** do repositório.
