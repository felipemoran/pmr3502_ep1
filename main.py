#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 13:42:59 2020

@author: marceloafonseca
"""

import numpy as np
from funcoes import ObtemDados, GraficoXY, ekf, menu

# Paremetros de inicio
estado_inicial = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float_).reshape((-1, 1))
covariancia_inicial = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 10.0],
    ],
    dtype=np.float_,
)

# Obtem arquivo do usuario
# arquivo = menu()

arquivo = "valores.csv"

try:
    # Obtendo dados do csv
    dados = ObtemDados(arquivo)
except:
    print("\nArquivo não encontrado")
else:
    # Passo de previsao
    estados, covariancias = ekf(
        dados, estado_inicial, covariancia_inicial, com_correcao=False
    )

    # Plotando gráfico de estado deterministico em plano XY
    print("\nPasso de Previsão - Questão 4")
    GraficoXY(estados, "Posição Determinística do Robô Plano XY")

    # Imprimindo valores de covariancia para t=1000
    cov_x = covariancias[1000][0][0]
    cov_y = covariancias[1000][1][1]
    print("")
    print("Para t=1000 temos:")
    print("Covariancia X:", cov_x)
    print("Covariancia Y:", cov_y)

    # Passo de correcao
    estados_corrigidos, covariancias_corrigidas = ekf(
        dados, estado_inicial, covariancia_inicial, com_correcao=True
    )

    # Plotando gráfico de estado corrigido em plano XY
    print("\nPasso de Correção - Questão 6")
    GraficoXY(estados_corrigidos, "Posição Corrigida do Robô Plano XY")
    cov_x = covariancias_corrigidas[1000][0][0]
    cov_y = covariancias_corrigidas[1000][1][1]
    print("")
    print("Para t=1000 temos:")
    print("Covariancia Corrigida X:", cov_x)
    print("Covariancia Corrigida Y:", cov_y)
