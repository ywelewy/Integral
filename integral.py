import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy import integrate
from scipy.special import gammaln
from math import comb 
import math
warnings.filterwarnings("ignore", category=DeprecationWarning) 
a = 2.5 
b = 4.3 
alpha = 2 / 7 
beta = 0
epsilon = 10**(-6)
Mn = 3.3855 #  Максимальное значение производной на отрезке 
TARGET = 9.274900  # Точное значение интеграла 
znach_kvadr_form = np.empty((2,0))
def f(x):
       return np.cos(1.5*x) * np.exp(2 * x / 3) + 3 * np.sin(5.5 * x) * np.exp(-2 * x) + 2
def p(x):
    return (np.power(x - a, -alpha) * np.power(b - x, -beta)) 
def omega(x):
    return np.float_((x - a) * (x - (b - a) / 2) * (x - b))
def F(x):
    return f(x) / (np.power(x - a, -alpha) * np.power(b - x, -beta))
def momenti(a_: float = a, b_: float = b, s: int = 0, alpha_: float = alpha):
    global a
    if s == 0:
        return [(pow((b_ - a), 1 - alpha_) - pow((a_ - a), 1 - alpha_)) / (1 - alpha_)]
    else:
        res = (pow((b_ - a), s + 1 - alpha_) - pow((a_ - a), s + 1 - alpha_)) / (s + 1 - alpha_)
        mUs = momenti(a_, b_, s=s - 1)
        l_ = len(mUs)
        for num, value in enumerate(mUs):
            res += comb(s, num + 1) * pow(-1, num) * pow(a, num + 1) * mUs[num]
        return [res] + mUs

def newton_cotes(N_: int = 3, h_: int = -1, a_: float = a, b_: float = b):
    momemti1 = []
    # Задаём узлы квадратурной формулы
    if h_ != -1:
        uzli = np.arange(a_, b_ + h_, h_)
    else:
        uzli = np.linspace(a_, b_, N_)
    # Вычисляем моменты весовой функции p(x) на [a,b]
    momenti1 = momenti(a_, b_, s=len(uzli) - 1, alpha_=alpha)[::-1]
    # Решаем СЛАУ
    momemti1 = np.array(momenti1)
    A = [np.power(uzli, i) for i in range(0, len(uzli))]

    An = np.linalg.solve(A, momemti1)
    quad = np.sum(An * f(uzli))

    return quad


def Gauss(N_: int = 3, a_: float = a, b_: float = b):
    mU = []
    # 1 Вычисляем моменты весовой функции p(x) на [a,b]
    mU = momenti(a_, b_, s=2 * N_ - 1, alpha_=alpha)[::-1]
    mU_n_plus_s = np.array(list(map(lambda x: -x, mU[N_:2 * N_])))

    # 2 Решаем СЛАУ
    mU_j_plus_s = np.zeros((N_, N_))
    for s_ in range(0, N_):
        for j in range(0, N_):
            mU_j_plus_s[s_, j] = mU[j + s_]
    a_j = np.linalg.solve(mU_j_plus_s, mU_n_plus_s)[::-1]
    tmp = np.ones((len(a_j) + 1, 1))
    tmp[1:] = a_j.reshape(len(a_j), 1)
    a_j = tmp
    # 3 Находим узлы, как корни узлового многочлена
    x_j = np.roots(a_j.reshape(len(a_j), ))
    # 4 Решаем СЛАУ
    A = np.array([np.power(x_j, i) for i in range(0, N_)])
    An = np.linalg.solve(A, mU[0:N_])
    quad = np.sum(An * f(x_j))
    return quad

N = 3


value_of_integral_for_methodic_error, *_ = integrate.quad(func=lambda x_: abs(p(x_) * omega(x_)), a=a, b=b)
methodic_error = (Mn / 6) * value_of_integral_for_methodic_error

quad = newton_cotes(N_=N)
error = abs(quad - TARGET)

print('Ньютон-Котс: N = {:3d}  значение интеграла = {:10.10f}  разность с точной погрежностью = {:.10e}, '
      'методическая погрешность = {:.10e}'.format(N, quad, error, methodic_error))

N = 3
quad = Gauss()
error = abs(quad - TARGET)
print('Гаусс: N = {:3d}  значение интеграла = {:10.10f}  разность с точной погрежностью = {:.10e}, '
      'методическая погрешность = {:.10e}'.format(N, quad, error, methodic_error))



def sostavnaya_quadrature_form(method: str = 'newton_cotes', a_: float = a, b_: float = b, h_: float = abs(b - a) / 2, N_: int = 3):
    methods = {'newton_cotes': newton_cotes, 'gauss': Gauss}
    # Задаём отрезки, на которых будут строиться квадратурные формулы
    if h_ != -1:
        nodes_x = np.arange(a_, b_ + h_, h_)
    else:
        nodes_x = np.linspace(a_, b_, N_)
    Res_S = 0
    # Вычисляем результирующую сумму,суммируя значения интегралов на каждом подотрезке
    for i in range(len(nodes_x) - 1):
        Res_S += methods[method](a_=nodes_x[i], b_=nodes_x[i + 1])
    return Res_S



def Richardson(h_: float = abs(b - a) / 3, method: str = 'newton_cotes', r: int = 3, L: float = 2, m: int = 3):
    # Выбираем метод
    methods = {'newton_cotes': newton_cotes, 'gauss': Gauss}
    # Выбираем набор шагов для разложения
    hs = np.array([h_ / pow(L, k) for k in range(r + 1)])
    # Формируем матрицу из шагов
    m_drob = m % 1
    m_whole = int(m // 1)
    H_l = np.array([[pow(value, i+m_drob) for i in np.arange(m_whole, m_whole + r)] for value in hs[:-1]])
    H_r = np.array([[pow(value, i+m_drob) for i in np.arange(m_whole, m_whole + r)] for value in hs[1:]])
    H = H_l - H_r
    # Формируем вектор разностей значений КФ
    S = []
    for i in hs:
        S.append(sostavnaya_quadrature_form(h_=i, method=method))
    S = np.array(S).reshape(len(S), 1)

    S = S[1:] - S[:-1]

    # Решаем СЛАУ и находим коэффициенты Cn
    Cn = np.linalg.solve(H, S)
    L_end = pow(L, r)  
    h = np.array([pow(hs[r], k+m_drob) / L_end for k in np.arange(m_whole, m_whole + r)])
    R_h = np.matmul(Cn.reshape(1, r), h.reshape(r, 1))[0][0]
    return R_h



def Aitken_process(method: str = 'newton_cotes', h__: float = abs(b - a) / 3, L: float = 2, a_: float = a, b_: float = b):
    h3 = h__ / np.power(L, 2)
    if (np.size(znach_kvadr_form, ) == 0):  
        h1 = h__
        h2 = h__ / L
        S_h1 = sostavnaya_quadrature_form(h_=h1, method=method)
        S_h2 = sostavnaya_quadrature_form(h_=h2, method=method)

    else: 
        S_h1 = znach_kvadr_form[0]
        S_h2 = znach_kvadr_form[1]

    S_h3 = sostavnaya_quadrature_form(h_=h3, method=method)
    znach_kvadr_form[0] = S_h2
    znach_kvadr_form[1] = S_h3
    m = -(np.log((S_h3 - S_h2) / (S_h2 - S_h1)) / np.log(L))

    return m

def integral_cqd(method: str = 'newton_cotes', a_: float = a, b_: float = b, h_: float = abs(b - a) / 2, req_m: int = 3, L: int = 2):
    global znach_kvadr_form
    znach_kvadr_form = np.empty((2, 0))
    r = 1
    if not (b-a)%h_ < 1e-6:
        h_ = (b-a) / (((b-a) // h_) + 1)
    h = h_ / L
    R = Richardson(m=req_m, method=method, h_=h_, r=r) 
    print("Значение m на шагах [", h_, ",", h_ / L, ",", h_ / pow(L, 2), "]:",
          Aitken_process(method=method, h__=h_, L=L, a_=a_, b_=b_))

    while abs(R) > 1e-6:
        h = h / L
        m = Aitken_process(method=method, h__=h, L=L, a_=a_, b_=b_)
        print("Значение m на шагах [", h, ",", h / L, ",", h / pow(L, 2), "]:", m)
        r += 1
        if not math.isnan(m):
            R = Richardson(m=m, method=method, h_=h, r=1)
        

    print('Правило Ричардсона: R_h = ', R, ', где h=', h)
    ans = sostavnaya_quadrature_form(method=method, a_=a, b_=b, h_=h)
    return ans

print("Составная квадратурная формула на Ньютоне-Котсе:", integral_cqd(method='newton_cotes'))
print("-------------------------------------------------------------------------------------")
print("Составная квадратурная формула на Гауссе:", integral_cqd(method='gauss', req_m=6))


def h_opt(method: str = 'newton_cotes', h_: float = abs(b - a) / 2, m: int = 3, epsilon: float = 1e-6):
    R = Richardson(method=method, h_=h_, r=1, m=m) # Вернуть оптимальный шаг
    h_opt = h_/2 * pow(epsilon / abs(R), 1 / m)
    print(h_opt)
    quad = integral_cqd(method=method, h_=h_opt, req_m=m)
    return [quad, h_opt]

h1 = (b - a) / 2
h2 = (b - a) / 3
h3 = (b - a) / 4
print("-----------------------------------Задание 3---------------------------------------------------")
print("Ньютон-Котс:")
print("На шаге:", h1)
ans = h_opt(method='newton_cotes', h_=h1, m=3)
print("Значение квадратурной формы:", ans[0], " ", "Оптимальный шаг:", ans[1])
print("На шаге:", h2)
ans = h_opt(method='newton_cotes', h_=h2, m=3)
print("Значение квадратурной формы:", ans[0], " ", "Оптимальный шаг:", ans[1])
print("На шаге:", h3)
ans = h_opt(method='newton_cotes', h_=h3, m=3)
print("Значение квадратурной формы:", ans[0], " ", "Оптимальный шаг:", ans[1])
print("----------------------------------------------------------------------------------------------")
print("Гаусс:")
print("На шаге:", h1)
ans = h_opt(method='gauss', h_=h1, m=6)
print("Значение квадратурной формы:", ans[0], " ", "Оптимальный шаг:", ans[1])
print("На шаге:", h2)
ans = h_opt(method='gauss', h_=h2, m=6)
print("Значение квадратурной формы:", ans[0], " ", "Оптимальный шаг:", ans[1])
print("На шаге:", h3)
ans = h_opt(method='gauss', h_=h3, m=6)
print("Значение квадратурной формы:", ans[0], " ", "Оптимальный шаг:", ans[1])
print("-------------------------------------------------------------------------------------")

              
              
