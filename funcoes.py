#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 16:35:51 2020

@author: marceloafonseca
"""
import csv
import sys
from collections import namedtuple
from math import tan, pi, cos, sin, sqrt, radians

import matplotlib.pyplot as plt
import numpy as np


Params = namedtuple("Params", "l, delta_t, h")

# Indices das variaveis:
#   X:
#       x: 0
#       y: 1
#       theta: 2
#       fx: 3
#       fy: 4
#   U:
#       v: 0
#       phi: 1

params = Params(l=0.3, delta_t=0.25, h=0.5)


def v(u):
    assert u.shape == (2, 1)
    return u[0, 0]


def phi(u):
    assert u.shape == (2, 1)
    return u[1, 0]


def x(mi):
    assert mi.shape == (5, 1)
    return mi[0, 0]


def y(mi):
    assert mi.shape == (5, 1)
    return mi[1, 0]


def theta(mi):
    assert mi.shape == (5, 1)
    return mi[2, 0]


def fx(mi):
    assert mi.shape == (5, 1)
    return mi[3, 0]


def fy(mi):
    assert mi.shape == (5, 1)
    return mi[4, 0]


def gama(u):
    """
    Calcula o ângulo percorrido entre 2 passos consecutivos
    :param u: parâmetros de comando
    :return: gama
    """
    v_ = v(u)
    phi_ = phi(u)

    return (v_ * params.delta_t * tan(phi_)) / params.l


def r(u):
    """
    Calcula o raio do arco a ser percorrido dados os parametros de entrada u
    :param u: parâmetros de comando
    :return: raio
    """
    phi_ = phi(u)
    if phi_ == 0:
        return 1e300

    return params.l / tan(phi_)


def F(mi, u):
    """
    Calcula o passo de atualização das médias
    :param mi: média dos estados anteriores
    :param u: parâmetros de comando
    :return: atualização das médias
    """
    r_ = r(u)
    gama_ = gama(u)

    theta_ = theta(mi)  # precisamos do ', 0' porque mi é um vetor coluna

    v_ = v(u)
    phi_ = phi(u)

    if phi_ != 0:
        F_delta = np.array(
            [
                r_ * (cos(theta_) * sin(gama_) - sin(theta_) * (1 - cos(gama_))),
                r_ * (sin(theta_) * sin(gama_) + cos(theta_) * (1 - cos(gama_))),
                phi_,
                0,
                0,
            ]
        ).reshape((-1, 1))
    else:
        F_delta = np.array(
            [
                cos(theta_) * v_ * params.delta_t,
                sin(theta_) * v_ * params.delta_t,
                gama_,
                0,
                0,
            ]
        ).reshape((-1, 1))

    return mi + F_delta


def A(mi, u):
    """
    Calcula matriz A para a atualização da matriz Sigma
    :param mi: média dos estados anteriores
    :param u: parâmetros de comando
    :return: A
    """
    gama_ = gama(u)
    r_ = r(u)

    theta_ = theta(mi)

    A = np.array(
        [
            [
                1,
                0,
                r_ * (-sin(theta_) * sin(gama_) - cos(theta_) * (1 - cos(gama_))),
                0,
                0,
            ],
            [
                0,
                1,
                r_ * (sin(theta_) * sin(gama_) - cos(theta_) * (1 - cos(gama_))),
                0,
                0,
            ],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )

    return A


def R(mi, u):
    """
    Calcula matriz R para a atualização da matriz Sigma
    :param mi: média dos estados anteriores
    :param u: parâmetros de comando
    :return: R
    """
    l = params.l
    delta_t = params.delta_t
    v_ = v(u)
    theta_ = theta(mi)

    s_l2 = (delta_t * v_ / 6) ** 2
    s_r2 = (delta_t * v_ / 12) ** 2
    s_theta2 = (delta_t * v_ / (8 * l)) ** 2

    R = np.array(
        [
            [
                cos(theta_) ** 2 * s_l2 + sin(theta_) ** 2 * s_r2,
                cos(theta_) * sin(theta_) * (s_l2 - s_r2),
                0,
                0,
                0,
            ],
            [
                cos(theta_) * sin(theta_) * (s_l2 - s_r2),
                sin(theta_) ** 2 * s_l2 + cos(theta_) ** 2 * s_r2,
                0,
                0,
                0,
            ],
            [0, 0, s_theta2, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    return R


def C(mi):
    """
    Calcula matriz C para o calculo do ganho de kalman e correção de Sigma
    :param mi: média dos estados atuais previstos
    :param u: parâmetros de comando
    :param params: parametros fixos do sistema
    :return: R
    """
    x_ = x(mi)
    y_ = y(mi)
    theta_ = theta(mi)
    fx_ = fx(mi)
    fy_ = fy(mi)

    d = sqrt(x_ ** 2 + y_ ** 2 + params.h ** 2)

    C = np.array(
        [
            [x_ / d, y_ / d, 0, 0, 0],
            [0, 0, -sin(theta_) * fx_ - cos(theta_) * fy_, cos(theta_), -sin(theta_)],
            [0, 0, -cos(theta_) * fx_ + sin(theta_) * fy_, -sin(theta_), -cos(theta_)],
        ]
    )

    return C


def Q(mi):
    """
    Calcula a matriz de covariância para o erro de medida
    :param mi: média dos estados atuais previstos
    :return: matriz de covariância do erro de medida
    """
    x_ = x(mi)
    y_ = y(mi)

    s_rho2 = (x_ ** 2 + y_ ** 2 + params.h * 2) / 400
    s_f2 = 1 / 4
    s_e2 = 1 / 4

    Q = np.array([[s_rho2, 0, 0], [0, s_f2, 0], [0, 0, s_e2]])

    return Q


def K(Sigma, C, Q):
    """
    Calcula a matriz do ganho de Kalman
    :param Sigma: matriz Sigma
    :param C: matriz C
    :param Q: matriz Q
    :return: matriz K de ganhos
    """
    K = Sigma.dot(C.transpose()).dot(
        np.linalg.inv((C.dot(Sigma).dot(C.transpose()) + Q))
    )
    return K


def G(mi):
    """
    Calcula a estimativa das observações
    :param mi: média dos estados atuais previstos
    :return: estimativa das observações no instante atual
    """
    x_ = x(mi)
    y_ = y(mi)
    theta_ = theta(mi)
    fx_ = fx(mi)
    fy_ = fy(mi)

    G = np.array(
        [
            [sqrt(x_ ** 2 + y_ ** 2 + params.h ** 2)],
            [cos(theta_) * fx_ - sin(theta_) * fy_],
            [-sin(theta_) * fx_ - cos(theta_) * fy_],
        ]
    )

    return G


def ObtemDados(arquivo):
    """
    Faz leitura de arquivo csv e retorna matriz com dados
    :param arquivo: arquivo csv com dados para analsie
    :return matriz_dados: dados do csv transformados em lista com dados tipo float
    """

    with open(arquivo) as dados:
        leitura_dados = csv.reader(dados, delimiter=",")

        matriz_dados = []
        dado = [0, 0, 0, 0, 0]

        for row in leitura_dados:
            dado[0] = float(row[0])
            dado[1] = float(row[1])
            dado[2] = float(row[2])
            dado[3] = float(row[3])
            dado[4] = float(row[4])

            matriz_dados.append(dado)

            dado = [0, 0, 0, 0, 0]

    return matriz_dados


def GraficoXY(matriz_estados, titulo):
    """
    Recebe matriz com estados e retorna gráfico com posiçoes no plano XY
    """
    matriz_estados = np.array(matriz_estados)

    X = matriz_estados[:, 0]
    Y = matriz_estados[:, 1]
    T = matriz_estados[:, 2]

    max_ = max(max(abs(X)), max(abs(Y))) * 1.05

    plt.plot(X, Y)
    plt.xlabel("Posição em X")
    plt.ylabel("Posição em Y")
    plt.xlim([-max_, max_])
    plt.ylim([-max_, max_])
    plt.title(titulo)
    plt.show()


def ekf(dados, estado_inicial, covariancia_inicial, com_correcao=True):
    """
    Recebe dados e retorna lista de vetores de estado e lista de matrizes de covariância
    :params dados: (csv na forma matricial tipo float),
    :param estado_inicial: (list),
    :param covariancia_inicial: (list)
    :param com_correcao: flag para indicar se passo de correção deve ser utilizado
    :return: estados (list), covariancias (list)
    """
    estados = [estado_inicial]
    covariancias = [covariancia_inicial]

    for t in range(1, len(dados)):
        # ======== PARTE 1: PREVISÃO ===================================================
        comando = np.array(dados[t]).reshape((-1, 1))[:2]
        observacoes = np.array(dados[t]).reshape((-1, 1))[2:]

        # Cálculo do vetor de valores médios para cada instante t
        mi_barra = F(estados[-1], comando)

        # Cálculo da matriz de covariância
        A_ = A(estados[-1], comando)
        R_ = R(estados[-1], comando)
        sigma_barra = A_.dot(covariancias[-1]).dot(A_.transpose()) + R_

        # ======== PARTE 2: CORREÇÃO ===================================================
        if not com_correcao:
            estado_corrigido = mi_barra
            covariancia_corrigida = sigma_barra

            estados += [estado_corrigido]
            covariancias += [covariancia_corrigida]
            continue

        else:
            C_ = C(mi_barra)
            Q_ = Q(mi_barra)
            K_ = K(sigma_barra, C_, Q_)

            mi = mi_barra + K_.dot(observacoes - G(mi_barra))

            sigma = (np.eye(K_.shape[0], C_.shape[1]) - K_.dot(C_)).dot(sigma_barra)

            estados += [mi]
            covariancias += [sigma]

    return estados, covariancias


def menu():
    """
    Disponibiliza menu para escolha do usuario. Pode escolher executar
    com dados ja fornecidos junto com o problema ("valores.csv") ou
    com dados novos
    :return arquivo: nome do arquivo csv com os dados
    """

    texto_menu = (
        "\nEscolhas disponíveis\n\n"
        + "1 - Executar com dados já fornecidos\n\n"
        + "2 - Executar com outros dados"
    )
    print(texto_menu)

    # Loop até fazer escolha 1 ou 2
    while True:
        try:
            escolha = int(input("Escolha um número para executar (1 ou 2): "))
            if escolha == 1:  # dados ja fornecidos
                arquivo = "valores.csv"
                break
            elif escolha == 2:  # dados novos
                arquivo = input("Digite o nome do arquivo csv com os dados: ")
                if "csv" not in arquivo:
                    arquivo += ".csv"
                break
            else:  # escolha != ( 1 ou 2 )
                print("\nNúmero errado, escolha de novo")
        except ValueError:  # escolha != int
            print("\nNúmero errado, escolha de novo")

    return arquivo
