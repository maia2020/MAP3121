#Métodos Numéricos e Aplicações - MAP3121
#Pedro Maia da Silva - 11805996
#Tiago Cavalcanti - 11804891

#----------------------------------------------------------------#

#Primeiro, importamos as bibliotecas que poderão ser usadas no programa
import math
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------#

#--------Definição dos coeficientes do Ploninômio de Legendre-------#

def pol_legendre(n):
    coeficientes = np.zeros(n+1,dtype=np.double)
    if n % 2 == 0:
        coeficientes[0] = 1.0
        m = int(n/2)
        for i in range(1,m + 1):
            coeficientes[2*i] = (2*i-n-2)*(2*i+n-1)/((2*i-1)*(2*i))*coeficientes[2*i-2]

    else:
        coeficientes[1]=1.0
        m=int((n-1)/2)
        for i in range(1,m+1):
            coeficientes[2*i+1] = (2*i-n-1)*(2*i+n)/((2*i)*(2*i+1))*coeficientes[2*i-1]
    coeficientes = coeficientes*(1/sum(coeficientes))

    return coeficientes

#--------Cálculo das Raizes do Ploninômio de Legendre-------#

def raizes_polinomio(n):

    coeficientes = pol_legendre(n)
    t = np.sort(np.roots(coeficientes[::-1])) #Raízes do polinômio de Legendre

    return t

#--------Derivação do Ploninômio de Legendre-------#

def derivada_polinomio(n):
    coeficientes = pol_legendre(n)
    coeficientes_deriv = coeficientes[1:]
    for i in range(len(coeficientes_deriv)):
        coeficientes_deriv[i] = coeficientes_deriv[i]*(i + 1)
    return coeficientes_deriv

#--------Integração do Ploninômio de Legendre-------#
#Primeiro é necessário que calculemos os pesos que serão utilizados no método de integração com a quadratura de gauss
#--------Definição dos Pesos da Quadratura de Gauss-------#

def pesos_Gauss(n):
    coeficientes = pol_legendre(n)
    t = np.sort(np.roots(coeficientes[::-1])) #Raízes do polinômio de Legendre
    coeficientes_deriv = derivada_polinomio(n)
    dp = np.zeros(len(t),dtype=np.double)
    for i in range(len(t)):
        for j in range(len(coeficientes_deriv)):
            dp[i] = dp[i] + coeficientes_deriv[j]*t[i]**j
    pesos = np.zeros(len(t),dtype=np.double)
    for i in range(len(t)):
        pesos[i] = 2/((1 - t[i]**2)*dp[i]**2)
    return pesos

#Então, assim como foi feito no EP2, fazemos uma mudança de variável para adequar as variáveis ao intervalo do problema
#--------Mudança de Variável para adequar ao intervalo [a,b]-------#

def mudanca_Variavel(a,b,t):
    i = (b - a)/2
    j = (b + a)/2
    x = i*t + j
    return x

#Como a integral neste EP é bem mais simples que aquelas do EP2, podemos utilizar uma função curta para fazer essa integração
#--------Aplicação da Quadratura de Gauss para integral simples-------#

def quadratura(pesos,y,a,b):
    return np.dot(pesos,y)*(b-a)/2 #Retorna o valor da integral

#Então precisamos definir as funções que utilizaremos para a realização das tarefas
#--------Definição das Funções a serem utilizadas-------#
#As funções serão definidas dessa forma visando uma maior facilidade de alteração para resolver as tarefas

#--------Definição de f(x)-------#
def f(x):
    return 12.0*x*(1.0-x)-2.0

#--------Definição de q(x)-------#
def q(x):
    #return 0.0
    #return 3.0
    #return 10.0
    return -3.0

#--------Definição de k(x)-------#  
def k(x):
    #return 1.0
    return 3.6 #Usado para o teste dos diferentes valores de q(x) por ser k do silício
    #return 60.0
    #return 401.0
    

#--------Definição de phi_i(x)-------#
def phi_i(x,h,i): #i varia de 1 até n
    if (x >= (i - 1)*h) and (x <= i*h):
        return (x - (i - 1)*h)/h
    elif (x > i*h) and (x <= (i + 1)*h):
        return ((i + 1)*h - x)/h
    else:
        return 0.0

#--------Definição de fbarra(x) em condições de não homogenidade-------#

def f_barra(x,a,b,L):
    return f(x) + (b - a)/L*dk(x) - q(x)*(a + (b - a)/L*x)

#--------Derivação de k(x)-------#

def dk(x):
    epsilon = 10**-6
    return (k(x + epsilon) - k(x))/epsilon
    
#--------Derivação de  phi_i(x)-------#

def dphi_i(x,h,i): #i varia de 1 até n
    if (x >= (i - 1)*h) and (x <= i*h):
        return 1.0/h
    elif (x > i*h) and (x <= (i + 1)*h):
        return -1.0/h
    else:
        return 0.0


#--------Resolução da Tarefa-------#

#--------Definição das Condições-------#
#As definições foram feitas dessa maneira com o intuito de facilitar a mudança destas
#--------Condições de Contorno------#

a = 0 #Valor da função no extremo esquerdo do intervalo [u(0) = a]
b = 0 #Valor da função no extremo direito do intervalo [u(L) = b]

#--------Definição do intervalo-------#
#Qtde de Intervalos de Discretização a serem usados
#n = 7
#n = 15
#n = 31
n = 63
L = 1 #Comprimento
h = L/(n + 1) #Tamanho dos intervalos de discretização
x = [i*h for i in range(n + 2)] #Pontos de discretização


#--------Definição para a Quadratura de Gauss-------#

m = 2
t = raizes_polinomio(m)
pesos = pesos_Gauss(m)

#--------Listas que contém os coeficientesicientes da matriz do sistema a ser resolvido-------#
#Primeiro iniciamos as variaveis para depois atualizá-las
A = np.zeros((n,n),dtype=np.double)
d = np.zeros((n,1),dtype=np.double)

for z in range(1,n + 2): #Iteração em cada intervalo [x_(i-1),x_i]
    t_adequado = mudanca_Variavel(x[z-1] , x[z] , t) #Mudança de Variaável das raízes

    #Cálculo das funções fbarra ; phi_i e dphi_i nas raízes calculadas acima
    f_barra_t_adequado = np.array([f_barra(t_adequado[i] , a  ,b , L) for i in range(len(t_adequado))])
    phii = np.zeros((n,m),dtype=np.double)
    for i in range(n):
        phii[i] = np.array([phi_i(t_adequado[j] , h , i+1) for j in range(len(t_adequado))])
    dphii = np.zeros((n,m),dtype=np.double)
    for i in range(n):
        dphii[i] = np.array([dphi_i(t_adequado[j], h, i+1) for j in range(len(t_adequado))])
    
    #Aplicação da Quadratura de Gauss no intervalo
    for i in range(n):
        y = phii[i]*f_barra_t_adequado
        d[i] = d[i] + quadratura(pesos,y,x[z - 1],x[z])
        for j in range(n):
            y = k(t_adequado)*dphii[i]*dphii[j] + q(t_adequado)*phii[i]*phii[j]
            A[i,j] = A[i,j] + quadratura(pesos,y,x[z - 1],x[z]) #Atualização da lista criada anteriormente
        
#--------Vetores que formam a matriz tridiagonal-------#

#Variáveis que ireamos alterar logo depois
p = [[0.0] for i in range(n)]
s = [[0.0] for i in range(n)]
r = [[0.0] for i in range(n)]

#Definição propriamente dita dos vetores
p[0] = 0.0
p[n-1]=A[n-2 , n-1]
for i in range(1 , n-1):
    p[i] = A[i ,i-1]

s[0] = A[0,0]
s[n-1] = A[n-1 , n-1]
for i in range(1 , n-1):
    s[i] = A[i,i]

r[0] = A[0,1]
r[n-1] = 0.0
for i in range(1 , n-1):
    r[i] = A[i , i+1]

#utilizamos o aprendizado do EP1 para definir a função responsável pela decomposição LU que será necessária

def MatrizIdentidade(n): #Função responsável pela geração de uma matriz identidade
    M = [[0.0 for i in range(n)] for i in range(n)]
    for i in range(n):
        M[i][i] = 1
    return M
def DecomposicaoLU(A): #Função responsável pela decomposição propriamente dita
    n = len(A) 

    L = MatrizIdentidade(n) #Inicializa "L" como sendo a matriz identidade

    for k in range(0 , n-1):
        for i in range(k+1 , n):
            m = -A[i][k]/A[k][k] #Cálculo do coeficientesiciente "m"
            L[i][k] = -m #Atualização da linha "i" da matriz "L"
            for j in range(k+1 , n):
                A[i][j] = m*A[k][j] + A[i][j] #Atualização da linha "i" da matriz "A"
            A[i][k] = 0 #Atualização da coluna "k" da matriz "A"
    return (L, A) #Retorno da matriz "L" e da matriz "A" modificada, no caso, a matriz "U"

#-------RESOLUÇÃO SISTEMAS LINEARES Ly=b E Ux=y-------#

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

def SistemaTridiagonal(a,b,c,d): #Esta função foi levemente adaptada em relação a que fizemos no EP1 para facilitar a manipulação dos vetores
    """Função que resolve um sistema tridiagonal não cíclico"""

    #-------Criação de listas nulas apenas para inicializar as variaveis que serão utilizadas posteriormente-------#
    n = len(d)
    u = [0.0 for i in range(n)]
    l = [0.0 for i in range(n)]
    y = [0.0 for i in range(n)]
    x = [0.0 for i in range(n)]
    
    #-------Armazenando os vetores que definem as matrizes L e U------#
    u[0] = b[0]
    for i in range(1,n):
        l[i] = a[i]/u[i-1]
        u[i] = b[i] - l[i]*c[i-1]
    
    #-------Resolvendo primeiro o sistema Ly = d-------#
    y[0] = d[0]
    for i in range(1,n):
        y[i] = d[i] - (l[i]*y[i-1])
    
    #-------Resolvendo então o sistema Ux = y-------#
    x[n-1] = y[n-1]/u[n-1]
    for i in range(n-2,-1,-1):
        x[i] = (y[i] - (c[i]*x[i+1]))/u[i]

    return x #Retornamos o vetor x de solução do sistema


#--------Resolução do Sistema-------#

alfa_tridiag = SistemaTridiagonal(p,s,r,d)

#--------Cálculo da Função v(x)-------#

v = [sum([phi_i(x[i],h,j + 1)*alfa_tridiag[j] for j in range(n)]) for i in range(n + 2)]

#--------Cálculo da Função u(x)-------#

u = [v[i] + a + (b - a)/L*x[i] for i in range(len(v))]

#--------Plotagem do Gráfico-------#

u_real = [x[i]**2*(1 - x[i])**2 for i in range(len(x))]
plt.plot(x,u_real,'b')
plt.plot(x,u,'*r')
plt.legend(['u(x) real','u(x) calculado'])
plt.show()

#--------Cálculo do Erro Absoluto Máximo-------#
P= []
o = []
P = [u[i] - u_real[i] for i in range(len(x))]
for i in range(len(P)):
    o.append(abs(P[i]))
er_abs_max = max(o) 
print("O Valor do Erro Absoluto Máximo para n=", n , "é:" , er_abs_max[0])





