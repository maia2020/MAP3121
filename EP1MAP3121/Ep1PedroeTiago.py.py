#Métodos Numéricos e Aplicações - MAP3121
#Pedro Maia da Silva - 11805996
#Tiago Cavalcanti - 11804891

#----------------------------------------------------------------SEÇÃO PARA MATRIZES DE MODO GERAL---------------------------------------------------------------------------------------------------------------------

"""Explicando o Algoritmo implementedo: 
Entrada: Matriz A -> Saída: Matriz L e U onde a matriz L é triangular inferior e U é triangular superior
Começaremos inicializando a matriz L como sendo a matriz identidade 
Entao, para cada ETAPA N teremos: Eliminação da coluna N e, para cada linha i será feito o calculo do fator m
entao iremos trocar o sinal do fator m e inseri-lo na matriz L e, por fim, atualizaremos a matriz A
Ou seja, a matriz U será uma atualização da A, não uma nova matriz
Encerrando com o retorno da matriz L e da matriz A modificada, no caso, a matriz U"""
#Importação das bibliotecas necessárias para a execuçaõ do algoritmo
import math
import numpy as np #Aqui abreviamos a biblioteca para facilitar sua utilização durante o código

def MatrizIdentidade(n): #Matriz responsável pela geração de uma matriz identidade
    M = np.zeros((n,n))
    for i in range(n):
        M[i][i] = 1
    return M
def DecomposicaoLU(A): #Matriz responsável pela decomposição propriamente dita
    n = len(A) 

    L = MatrizIdentidade(n) #Inicializa "L" como sendo a matriz identidade

    for k in range(0 , n-1):
        for i in range(k+1 , n):
            m = -A[i][k]/A[k][k] #Cálculo do coeficiente "m"
            L[i][k] = -m #Atualização da linha "i" da matriz "L"
            for j in range(k+1 , n):
                A[i][j] = m*A[k][j] + A[i][j] #Atualização da linha "i" da matriz "A"
            A[i][k] = 0 #Atualização da coluna "k" da matriz "A"
    return (L, A) #Retorno da matriz "L" e da matriz "A" modificada, no caso, a matriz "U"

#--------INÍCIO DO TESTE DO ALGORITMO DE DECOMPOSIÇÃO LU--------#
A = [[1 , -3 , 2],
     [-2 , 8  , -1],
     [4 , -6 , 5]]
(L , U) = DecomposicaoLU(A)
#print("Matriz L:")
#print(L)
#print("Matriz U:")
#print(U)
#--------FIM DO TESTE DO ALGORITMO DE DECOMPOSIÇÃO LU--------#

#-------RESOLUÇÃO SISTEMAS LINEARES Ly=b E Ux=y-------#

"""Para resolver o sistema linear utilizando a decomposição LU feita anteriormente, podemos separa em duas etapas:
1) Resolver Ly=b
2) Resolver Ux=y
E então retornar o vetor x como sendo a solução do nosso problema. 
Assim, resolvemos dois sitemas lineares triangulares"""

def Sub_Retroativa(A , b): 
    #Algoritmo de substituição retroativa utilizado para resolver o sistema linear triangular superior Ux=b
    #Entrada: Matriz "A" triangular superior e vetor "b"  Saída: Vetor "x"
    n = len(A)

    x = n * [0] #Inicialização do vetor x como sendo um vetor de n 0's

    for i in range( n-1 , -1 , -1):
        S = 0 
        for j in range(i+1 , n):
            S = S + A[i][j] * x[j]
        x[i] = (b[i] - S) / A[i][i]
    return x

def Sub_Sucessiva(A , b):
    #Algoritmo de substituição sucessiva utilizado para resolver o sistema linear triangular inferior Lx = b
    #Entrada: Matriz "A" triangular inferior e vetor "b"  Saída: Vetor "x"
    n = len(A)

    x= n * [0] #Inicialização do vetor x como sendo um vetor de n 0's
    for i in range(0 , n):
        S = 0 
        for j in range(0 , i):
            S = S + A[i][j] * x[j]	
        x[i] = (b[i] - S) / A[i][i] 

    return x

def ResolucaoSistema(L , U , b):

    """Função que aplica os dois algoritmos de substituição retroativa (Lx=b) e
     sucessiva (Ux=b) para resolver o sistema linear Ly=b e Ux=y"""
    #Entrada: Matriz "L" triangular inferior e Matriz "U" triangular superior e vetor "b"  Saída: Vetor "x"
    
    y = Sub_Sucessiva(L , b)
    x = Sub_Retroativa(U , y)

    return x

#-------TESTE DO ALGORITMO DE RESOLUÇÃO DE SISTEMAS-------#
A = [[1 , -3 , 2],
     [-2 , 8  , -1],
     [4 , -6 , 5]]

b = [11 , -15 , 29]
(L , U) = DecomposicaoLU(A)
x = ResolucaoSistema(L , U , b)
#print(x)

#-------TESTE DO ALGORITMO DE RESOLUÇÃO DE SISTEMAS-------#

"""
Bibliografia: 
https://www.youtube.com/watch?v=7_kottQNcX0
"""
            





#----------------------------------------------------------------SEÇÃO ESPECIFICAMENTE PARA MATRIZ TRIDIAGONAIS---------------------------------------------------------------------------------------------------------------------

#-------FUNÇÕES RESPONSÁVEIS POR GERAR A MATRIZ E VETORES DO EXEMPLO-------#
n = 20

def GeradorDaMatriz(n): #Esta função é responsável por gerar os vetores que representam as 3 diagonais da matriz A citada
    b = [] #Diagonal Principal da matriz
    c = [] #Diagonal Superior da matriz
    a = [] #Diagonal Inferior da matriz
    for i in range(1 , n):
        a.append((2*i-1)/(4*i))
        c.append(1-a[i-1])
        b.append(2)

    a.append((2*n-1)/(2*n))
    c.append(1-a[n-1])
    b.append(2)

    return(a , b , c)
#a , b , c = GeradorDaMatriz(n)

#print(a , b , c) #Linha usada apenas para testar a função

def GeradorLadoDireito(n):
    d = []
    
    for i in range(1 , n+1):
            d.append(math.cos(2*math.pi*(i**2)/(n**2)))
    #print(d) Linha usada apenas para testar a função
#GeradorLadoDireito(n) Linha usada apenas para testar a função
    return(d)
#d = GeradorLadoDireito(n)

#--------Decomposição LU para matriz tridiagonal-------#

def LUTridiagonal(a , b , c , n): #Função que de fato realiza a decomposição LU da matriz 
    li = []
    ui = []
    uii = []
    #Aqui criamos os vetores que serão os resultados da decomposição LU, a partir daqui começamos a implementação do algoritmo
    ui.append(b[0])
    for i in range(n):
        li.append(a[i]/ui[i-1])
        ui.append(b[i]-(li[i]*c[i-1]))
    for i in range(0 , n):
        uii.append(c[i])
    ui.pop() #Remove o último elemento do vetor ui por conta do for loop
    return (li , ui , uii)
#li , ui , uii = LUTridiagonal(n)

#-------Resolução Sistema utilizando os três vetores da decomposição LU-------#

def SistemaTridiagonal(li , ui , uii , d): 
    """Função que resolve um sistema tridiagonal não cíclico
    Nesse caso, os valores de a1 e cn não nos interessam, uma vez que estes serão 0"""
    #del li[0] Removemos o primeiro termo de li (a1) -> Como a função será usada para o caso de uma tridiagonal cíclica, essa linha se torna irrelevante
    #uii.pop() Removemos o último termo de uii (cn) -> Como a função será usada para o caso de uma tridiagonal cíclica, essa linha se torna irrelevante
    #Ly = d:
    nlinha = len(d) #Tamanho do vetor d 
    y = []
    x = []
    y.append(d[0])
    for i in range(1 , nlinha):
        y.append( d[i] - li[i]*y[i-1] )
    #Ux = y:
    x.append(y[nlinha-1] / ui[nlinha-1]) #Primeiro calculamos o último valor do que seria o vetor x e atribuímos ao primeiro valor, o inverteremos depois
    for i in range(nlinha-1):
        x.append((y[i] - uii[i]*x[i-1])/ui[i])
    x = x[::-1] #Inverção do vetor x

    return(x)
    

#-------Resolução Sistema Tridiagonal Cíciclico-------#

def SistemaTridiagonalCiclico(a , b , c , d):
    #Função recebe como entradas os vetores que definem a matriz A do exemplo
    """Primeiro, resolveremos a matriz T que tem dimensão (n-1)x(n-1) utilizando o algoritmo para um 
    sistema tridiagonal qualquer implementado anteriormente
    """
    v = []
    w = []
    #Definiremos os vetores v e w definidos no enunciado do exercício
    v.append(a[0])
    for i in range(1 ,n-2):
        v.append(0)
    v.append(c[n-1])

    w.append(c[n-1])
    for i in range(1 , n-2):
        w.append(0)
    w.append(a[n-1])

    #Então fazemos a decomposição LU da matriz T(n-1)x(n-1)
    li , ui , uii = LUTridiagonal(a , b , c , n-1)
    dn = d.pop() #Removemos o último elemento do vetor d para calcularmos o sistema (n-1)x(n-1) e o armazenamos
    dbarra = d

    #Agora que ja definimos os vetores que utilizaremos, começaremos a resolver o sistema

    ybarra = SistemaTridiagonal(li , ui , uii , dbarra)
    zbarra = SistemaTridiagonal(li , ui , uii , v)

    #Agora, calculamos o vetor solução X:
    X = []

    #Primeiro calculamos o xn utilizando a fórmula contida no enunciado
    xn = (dn - c[n-1]*ybarra[0] - a[n-1]*ybarra[n-2])/(b[n-1] - c[n-1]*zbarra[0] - a[n-1]*zbarra[n-2])
    #E então calcumaos o vetor xbarra que complementa o vetor x 
    xbarra = [] 
    for i in range(n-1):
        xbarra.append(ybarra[i] - zbarra[i]*xn)
    
    #Assim, juntando ambos, chegamos no vetor X que é a solução do sistema tridiagonal cíclico

    xbarra.append(xn) #Adicionamos o xn ao final do vetor xbarra
    X = xbarra

    return(X)


#-------Seção dos Inputs--------#
P = input("1: Sistema Tridiagonal , 2:Sistema Tridiagonal Cíclico\n")
T = input("1: Matriz Gerada , 2: Inserir os Dados \n")

if P == "1" and T == "1":
    a , b , c = GeradorDaMatriz(n)
    d = GeradorLadoDireito(n)
    li , ui , uii = LUTridiagonal(a , b , c ,n)
    x = SistemaTridiagonal(li , ui , uii , d)
    print("O vetor solução é: ", x)
    print("Os vetores que definem a matriz A são:", a , b , c)
    print("Os vetores da decomposição LU são, respectivamente:", li , ui , uii)
elif P == "1" and T == "2":
    n = int(input("Qual o tamanho da matriz?"))
    a = []
    b = []
    c = []
    for i in range(n):
        a.append(int(input("Digite o valor da diagonal inferior")))
    for i in range(n):
        b.append(int(input("Digite o valor da diagonal principal")))
    for i in range(n):
        c.append(int(input("Digite o valor da diagonal superior")))
    d = []
    for i in range(n):
        d.append(int(input("Digite o valor do lado direito do sistema")))
    li , ui , uii = LUTridiagonal(a , b , c , n)
    x = SistemaTridiagonal(li , ui , uii , d)
    print("O vetor solução é: ", x)
    print("Os vetores que definem a matriz A são:", a , b , c)
    print("Os vetores da decomposição LU são, respectivamente:", li , ui , uii)
elif P == "2" and T == "1":
    a , b , c = GeradorDaMatriz(n)
    d = GeradorLadoDireito(n)
    li , ui , uii = LUTridiagonal(a , b , c , n-1)
    x = SistemaTridiagonalCiclico(a , b , c , d)
    print("Vetor a:")
    print(a)
    print("Vetor b:")
    print(b)
    print("Vetor c:")
    print(c)
    print("Vetor d:")
    print(d)
    print("O vetor solução é: ", x)
    print("Os vetores da decomposição LU são, respectivamente:", li , ui , uii)   
elif P == "2" and T == "2":
    n = int(input("Qual o tamanho da matriz?"))
    a = []
    b = []
    c = []
    for i in range(n):
        a.append(int(input("Digite o valor da diagonal inferior")))
    for i in range(n):
        b.append(int(input("Digite o valor da diagonal principal")))
    for i in range(n):
        c.append(int(input("Digite o valor da diagonal superior")))
    d = []
    for i in range(n):
        d.append(int(input("Digite o valor do lado direito do sistema")))
    li , ui , uii = LUTridiagonal(a , b , c , n-1)
    x = SistemaTridiagonalCiclico(a , b , c , d)
    print("Solução do sistema:") 
    print(x)   
    print("Os vetores que definem a matriz A são:", a , b , c)
    print("Os vetores da decomposição LU são, respectivamente:", li , ui , uii)