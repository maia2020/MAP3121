#Métodos Numéricos e Aplicações - MAP3121
#Pedro Maia da Silva - 11805996
#Tiago Cavalcanti - 11804891

"""Tarefa: Cálculo de integrais duplas através de fórmulas iteradas
Usar fórmulas de Gauss com 'n' nós para as integrações numéricas
Os nós e os pesos são fornecidos para o intervalo [-1.1] e o program deverá fazer os ajustes
necessários par outros intervalos 
Precisão dupla (padrão do numpy)
Teste nos exemplos com n = 6 , 8 , 10
Imprima para cada n o valor de n e os valores calculados das integrais 
quando for o caso imprima tambem os valores exatos""" 

#----------------------------------------------------------------#

#Primeiro, importamos as bibliotecas que poderão ser usadas no programas
import numpy as np
import math

#Então criaremos as listas que contém os possíveis valores de 'n', no caso 6 8 e 10, e os valores dos pesos e dos pontos que nos foram dados
nos = [6,8,10]
x6 = [0.2386191860831969086305017, 0.6612093864662645136613996, 0.9324695142031520278123016]
x8 = [0.1834346424956498049394761, 0.5255324099163289858177390, 0.7966664774136267395915539, 0.9602898564975362316835609]
x10 = [0.1488743389816312108848260, 0.4333953941292471907992659, 0.6794095682990244062343274, 0.8650633666889845107320967, 0.9739065285171717200779640]
w6 = [0.4679139345726910473898703, 0.3607615730481386075698335, 0.1713244923791703450402961]
w8 = [0.3626837833783619829651504, 0.3137066458778872873379622, 0.2223810344533744705443560, 0.1012285362903762591525314]
w10 = [0.2955242247147528701738930, 0.2692667193099963550912269, 0.2190863625159820439955349, 0.1494513491505805931457763, 0.0666713443086881375935688]

#Para facilitar a utilização destas listas durante a resolução dos exemplos, criaremos o seguinte dicionário que associa as listas ao número de nós:
xdict = {6 : x6 , 8 : x8 , 10 : x10} 
wdict = {6 : w6 , 8 : w8 , 10 : w10}  

#Definiremos agora a função responsável por realizar a quadratura gaussiana:
def quadratura(a , b , c , d , n , f):
    """Parâmetros: 
    a e b: Limites de integração em x
    c e d: Limites de integração em y, que podem tambem ser funções de x [c(x),d(x)] como será explorado no código
    n é o número de nós desejado
    f é a função a ser integrada"""

    #Agora, a fim de gerar os pontos para todo o intervalo [-1,1], para qual a quadratura de gauss será aplicada
    #Criaremos duas listas que receberam os pontos negativos, os quais não foram incluidos nos dados do EP por conta da simetria
    xt = []
    wt = []

    for x in xdict[n]: #Para cada nó, adicionaremos os pontos negativos e positivos
        xt.append(x)
        xt.append(-x)
    for w in wdict[n]:
        wt.append(w)
        wt.append(w)
    
    #Então criaremos das novas listas que serão responsáveis pela primeira mudança de variáveis a fim de buscar um intervalo para o qual podemos aplicar a quadratura
    def mudancaDeVariavelponto(x):
        return(((b - a) * x + (b + a))/2)
    def mudancaDeVariavelpeso(w):
        return((b - a) * w/2)
    xf = list(map(mudancaDeVariavelponto ,  xt)) #Mudança de variável nos pontos
    wf = list(map(mudancaDeVariavelpeso , wt)) #Mudança de Variável nos pesos

    #usamos a função map para aplicarmos as funções de mudança de variável às listas xt e wt de forma mais prática
    soma2 = 0
    #Agora, aplicaremos mais uma vez a mudança de variável, desta vez para o intervalo da primeira integral a ser feita, no caso, em y
    for i in range(n): #Para cada nó faremos a mudança de variável a fim de adequar o intervalo da integração em y, calculando essa mudança utilizando a prórpia mudança já feita em x  

        xd = [(d(xf[i]) - c(xf[i])) * x + (d(xf[i]) + c(xf[i]))/2 for x in xt]
        wd = [((d(xf[i]) - c(xf[i]))/2) * w for w in wt] 

        #Agora, aplicaremos o método da quadratura de gauss propriamente dito, calculando primeiro a soma da primeira integral através do seguinte for loop:
        soma1 = 0 
        for u in range(n):
            soma1 += wd[u] * f(xf[i] , xd[u]) #aqui o xf faz o papel de x e o xd faz o papel de y
            
        #Então, calculamos a soma que aproxima a segunda integral dentro do primero for loop:
        soma2 += wf[i] * soma1
    return(soma2) #Por fim retornamos apenas a soma final

#--------------Exemplos--------------#

def exemplo1():
    #Volume do cubo e tetraedro de aresta 1:
    for n in nos:
        print("O volume do cubo calculado para n = " , n , "é: " , quadratura(0 , 1 , lambda x: 0 , lambda x: 1 , n , lambda x,y: 1))
    for n in nos:
        print("O volume do tetraedro calculado para n = " , n , "é: " , quadratura(0 , 1 , lambda x: 0 , lambda x: 1-x , n , lambda x,y: 1-x-y))

exemplo1()

def exemplo2():
    print("O valor esperado para as integras é: 0.666666")
    for n in nos:
        print("O valor calculado da primeira integral para n= " , n , "é: " , quadratura(0 , 1 , lambda x: 0 , lambda x: 1-x**2 , n , lambda x,y: 1))
    for n in nos:
        print("O valor calculado da segunda integral para n= " , n , "é: " , quadratura(0 , 1 , lambda x: 0 , lambda x: math.sqrt(1-x) , n , lambda x,y: 1))
    
# exemplo2()

def exemplo3():
    for n in nos:
        print("O valor calculado da área para n= " , n , "é: " , quadratura(0.1 , 0.5 , lambda x: x**3 , lambda x: x**2 , n , lambda x,y: 1))
    for n in nos:
        print("O valor calculado do volume para n= " , n , "é: " , quadratura(0.1 , 0.5 , lambda x: x**3 , lambda x: x**2 , n , lambda x,y: np.exp(y/x)))

# exemplo3()

def exemplo4():
    for n in nos:
        print("O volume da calota esférica calculado para n= " , n , "é: ", quadratura(0.75 , 1 , lambda x: 0 , lambda x: math.sqrt(1-x**2) , n , lambda x,y: 6.28*y))
    for n in nos:
        print("O volume do sólidos calculado para n= " , n , "é: " , quadratura(-1 , 1 , lambda x: 0 , lambda x: np.exp(-x**2) , n , lambda x,y: (2*math.pi)*y))

#exemplo4()



"""Bibligrafia: 
https://www.ufjf.br/flavia_bastos/files/2009/06/aula_integral.pdf
https://www.youtube.com/watch?v=bu8trr9Qm1Y
https://edisciplinas.usp.br/pluginfile.php/7041254/mod_resource/content/2/tarefa2_2022.pdf
https://www.digitalocean.com/community/tutorials/how-to-use-the-python-map-function-pt
"""
