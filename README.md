# Projeto do Final do Módulo de Aprendizado em Tempo Real - INFNET
# Sistema Avaliação de Lances

- Carregar dados dos arremessos do Kobe Bryant e separar arremessos de 2 ou 3 pontos: 2 pontos para desenvolvimento/operação e 3 pontos para novidade 
- Treinamento de modelo com regressão logística do sklearn no pyCaret
- Registros das etapas de processamento como runs
- Registro do modelo com threshold de precisão mínimo de 80% em Staging
- Aplicação Online: recomendação de vinhos
    - Consumo da base de dados de operação
    - Utilização do requests para fazer o POST e recuperar a predição do modelo
    - Propor os 5 vinhos de alta qualidade de maior nível alcólico (print data frame)
- Aplicação de Monitoramento: pipeline de monitoramento
    - Revalidacao da base de operacao para amostras de controle (simulação especialista)
    - Leitura da base de desenvolvimento (treino+teste)
    - Alarme de amostra de controle
    - Alarme com amostras de novidade
