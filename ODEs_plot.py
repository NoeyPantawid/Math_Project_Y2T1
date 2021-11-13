#==================== Library ====================

import numpy as np
import matplotlib.pyplot as plt


#?==================== abbreviate ====================

exp = np.exp
sin = np.sin
cos = np.cos
tan = np.tan
csc = lambda x: 1/sin(x)
sec = lambda x: 1/cos(x)
cot = lambda x: 1/tan(x)
arcsin = np.arcsin
arccos = np.arccos
arctan = np.arctan
ln = np.log
log = np.log10
log2 = np.log2


#?==================== Doc ====================

# Parameters
# ----------
# f : y' = (exp(-2*(x**2)))-(4*x*y)
# yp : position of x 
#      for example y[0]
#      yp = 0
# yx : x value at position(yp) 
#      for example y[1] = 0
#      yx = 0
# x : range of x
#      for example 1 <= x <= 2
#      x = np.arange(1, 2, h)
#     
# h : step size 
#      for example h = 0.2
#      h = 0.2

# Return
# -------
# y : list of value (y[n])
# -------------

example = """============================================
!Example!
Start = 0
Stop = 2
n = 0
Y[n] = -4.3
h = 0.2
y' = exp(-2*(x**2))-(4*x*y)
exact = x * exp(-2*x**2) - 4.3 * exp(-2*x**2)
============================================"""


#!==================== 4 Method Functions ====================

#Euler Method
def odeEuler(f, yp, yx, x, h):
    y = np.zeros(len(x))
    y[yp] = yx
    for n in range(0,len(x)-1):
        y[n+1] = y[n] + (h * f(x[n], y[n]))
    return y

#Improved Euler Method
def odeHuan(f, yp, yx, x, h):
    y = np.zeros(len(x))
    y[yp] = yx
    for n in range(0,len(x)-1):
        k1 = h * f(x[n], y[n])
        k2 = h * f(x[n+1], y[n] + k1)
        y[n+1] = y[n] + (1/2) * (k1 + k2)
    return y

#Runge-Kutta Method
def odeRunge_kutta(f, yp, yx, x, h):
    y = np.zeros(len(x))
    y[yp] = yx
    for n in range(0,len(x)-1):
        k1 = h * f(x[n], y[n])
        k2 = h * f(x[n] + (0.5 * h), y[n] + (0.5 * k1))
        k3 = h * f(x[n] + (0.5 * h), y[n] + (0.5 * k2))
        k4 = h * f(x[n] + h, y[n] + k3)
        y[n+1] = y[n] + ((1/6) * (k1 + (2 * k2) + (2 * k3) + k4))

    return y

#Runge-Kutta-Fehlberg Method (4th order)
def odeRunge_kutta_fehlberg(f, yp, yx, x, h):
    y = np.zeros(len(x))
    y[yp] = yx
    for n in range(0,len(x)-1):
        k1 = h * f(x[n], y[n])
        k2 = h * f(x[n] + ((1/4) * h), y[n] + (1/4) * k1)
        k3 = h * f(x[n] + ((3/8) * h), y[n] + ((3/32) * k1) + ((9/32) * k2))
        k4 = h * f(x[n] + ((12/13) * h), y[n] + ((1932/2197) * k1) - ((7200/2197) * k2) + ((7296/2197) * k3))
        k5 = h * f(x[n] + h, y[n] + ((439/216) * k1) - (8 * k2) + ((3680/513) * k3) - ((845/4104) * k4))
        k6 = h * f(x[n] + ((1/2) * h), y[n] - ((8/27) * k1) + (2 * k2) - ((3544/2565) * k3) + ((1859/4104) * k4) - ((11/40) * k5))
        y[n+1] = y[n] + ((16/135) * k1) + ((6656/12825) * k3) + ((28561/56430) * k4) + ((-9/50) * k5) + ((2/55) * k6)

    return y


#?==================== Error functions and create table ====================

def errorEuler():
    diff_euler = zip(y_euler, y_exact)
    n1 = 0
    s1 = yp
    Frequency1 = f'{(stop-start)/h:.2f}'
    print('______________________________________________________________________________________________')
    print(f'|Euler Method | Frequency = {Frequency1:<65}|')
    print('|____________________________________________________________________________________________|')
    print(f'|    n    |      x[n]      |       y[n]        |      exact       |          error           |')
    print('|_________|________________|___________________|__________________|__________________________|')
    for i, j in diff_euler:
        s1_f = f'{s1:.2f}'
        i1_f = f'{i:.4f}'
        j1_f = f'{j:.4f}'
        er1_f = f'{abs(i-j):.10f}'
        print(f'| n = {n1:^3} | x[{n1:^3}] = {s1_f:^5} | y[{n1:^3}] = {i1_f:>8} | exact = {j1_f:>8} | error = {er1_f:<16} |')
        n1 += 1
        s1 += h
    print('|_________|________________|___________________|__________________|__________________________|')

def errorHuan():
    diff_huan = zip(y_huan, y_exact)
    n2 = 0
    s2 = yp
    Frequency2 = f'{(stop-start)/h:.2f}'
    print('_____________________________________________________________________________________________')
    print(f'|Improved Euler Method | Frequency = {Frequency2:<56}|')
    print('|____________________________________________________________________________________________|')
    print(f'|    n    |      x[n]      |       y[n]        |      exact       |          error           |')
    print('|_________|________________|___________________|__________________|__________________________|')
    for i, j in diff_huan:
        s2_f = f'{s2:.2f}'
        i2_f = f'{i:.4f}'
        j2_f = f'{j:.4f}'
        er2_f = f'{abs(i-j):.10f}'
        print(f'| n = {n2:^3} | x[{n2:^3}] = {s2_f:^5} | y[{n2:^3}] = {i2_f:>8} | exact = {j2_f:>8} | error = {er2_f:<16} |')
        n2 += 1
        s2 += h
    print('|_________|________________|___________________|__________________|__________________________|')

def errorRunge():
    diff_Runge = zip(y_runge, y_exact)
    n3 = 0
    s3 = yp
    Frequency3 = f'{(stop-start)/h:.2f}'
    print('______________________________________________________________________________________________')
    print(f'|Runge-Kutta Method | Frequency = {Frequency3:<59}|')
    print('|____________________________________________________________________________________________|')
    print(f'|    n    |      x[n]      |       y[n]        |      exact       |          error           |')
    print('|_________|________________|___________________|__________________|__________________________|')
    for i, j in diff_Runge:
        s3_f = f'{s3:.2f}'
        i3_f = f'{i:.4f}'
        j3_f = f'{j:.4f}'
        er3_f = f'{abs(i-j):.10f}'
        print(f'| n = {n3:^3} | x[{n3:^3}] = {s3_f:^5} | y[{n3:^3}] = {i3_f:>8} | exact = {j3_f:>8} | error = {er3_f:<16} |')
        n3 += 1
        s3 += h
    print('|_________|________________|___________________|__________________|__________________________|')

def errorRungeF():
    diff_RungeF = zip(y_rungef, y_exact)
    n4 = 0
    s4 = yp
    Frequency4 = f'{(stop-start)/h:.2f}'
    print('______________________________________________________________________________________________')
    print(f'|Runge-Kutta-Fehlberg Method | Frequency = {Frequency4:<50}|')
    print('|____________________________________________________________________________________________|')
    print(f'|    n    |      x[n]      |       y[n]        |      exact       |          error           |')
    print('|_________|________________|___________________|__________________|__________________________|')
    for i, j in diff_RungeF:
        s4_f = f'{s4:.2f}'
        i4_f = f'{i:.4f}'
        j4_f = f'{j:.4f}'
        er4_f = f'{abs(i-j):.10f}'
        print(f'| n = {n4:^3} | x[{n4:^3}] = {s4_f:^5} | y[{n4:^3}] = {i4_f:>8} | exact = {j4_f:>8} | error = {er4_f:<16} |')
        n4 += 1
        s4 += h
    print('|_________|________________|___________________|__________________|__________________________|')


#?==================== Input functions =================

def start():

    global start, stop, f, yp, yx, x, h, prob_exact, zero_check, one_check, two_check, three_check, four_check

    print(example)

    check = 0

    while check != 5:

        if check == 0:
            start = input('Start = ')
            try:
                start = float(start)
                check += 1
            except:
                print('Start point must be a number!')
                continue

        elif check == 1:
            stop = input('Stop = ')
            try:
                stop = float(stop)
                check += 1
            except:
                print('Stop point must be a number!')
                continue

        elif check == 2:
            yp = input("Enter Y[n]\nn = ")
            try:
                yp = int(yp)
                check += 1
            except:
                print("n must be a integer!")
                continue

        elif check == 3:
            yx = input("Enter Y[n]\nY[n] = ")
            try:
                yx = float(yx)
                check += 1
            except:
                print('Y[n] must be a number!')
                continue

        elif check == 4:
            h = input('h (step size) = ')
            try:
                h = float(h)
                if h <= 0:
                    print('h must not be zero or negative number!')
                    continue
                else:
                    check += 1
            except:
                print('h must be a number!')
                continue

    stop += h
    x = np.arange(start, stop, h)

    danger_word = ['exit', 'system', 'import', 'clear', 'lamda', 'eval', 'for', 'if', 'and', 'or', 'not']
    while True:
        prob = input("y' = ")
        if any(word in prob for word in danger_word):
            print('You can not use that!')
            continue
        else:
            break

    while True:
        prob_exact = input("exact = ")
        if any(word in prob_exact for word in danger_word):
            print('You can not use that!')
            continue
        else:
            break
    
    print('--------------------------------')
    while True:
        zero_check = input("Do you want all method solution?\n(Y/N) : ")
        if zero_check == 'Y':
            one_check = 'Y'
            two_check = 'Y'
            three_check = 'Y'
            four_check = 'Y'
            break
        elif zero_check == 'N':
            sure_list = []
            print('--------------------------------')
            one_check = input("Do you want Euler method solution?\n(Y/N) : ")
            if one_check == 'Y':
                sure_list.append('- Euler method')
            two_check = input("Do you want Improved-Euler method solution?\n(Y/N) : ")
            if two_check == 'Y':
                sure_list.append('- Improved-Euler method')
            three_check = input("Do you want Runge-Kutta method solution?\n(Y/N) : ")
            if three_check == 'Y':
                sure_list.append('- Runge-Kutta method')
            four_check = input("Do you want Runge-Kutta-Fehlberg method (4th order) solution?\n(Y/N) : ")
            if four_check == 'Y':
                sure_list.append('- Runge-Kutta-Fehlberg 4th order method')
            print('--------------------------------')

            print("You prefer these method solution?\n")
            print(*sure_list, sep = '\n')
            r_u_sure = input("\n(Y/N) : ")
            if r_u_sure == 'Y':
                break
            else:
                print('--------------------------------')
                continue
        else:
            print('!Attention!')
            print('Invalid input please enter again')
            continue


    f = lambda x, y : eval(prob)


#?==================== Call functions ====================

def call_function():

    global y_euler, y_huan, y_runge, y_rungef, y_exact

    #Euler Method
    if one_check == 'Y':
        y_euler = odeEuler(f, yp, yx, x, h)

    #Improved Euler Method
    if two_check == 'Y':
        y_huan = odeHuan(f, yp, yx, x, h)

    #Runge-Kutta Method
    if three_check == 'Y':
        y_runge = odeRunge_kutta(f, yp, yx, x, h)

    #Runge-Kutta-Fehlberg Method
    if four_check == 'Y':
        y_rungef = odeRunge_kutta_fehlberg(f, yp, yx, x, h)

    #EXACT
    y_exact = eval(prob_exact)


#?==================== Plot graph ====================

def plot():
    call_function()
    
    plt.plot(x,y_exact,'r.-')
    legend_holder = ['Exact Solution']

    if one_check == 'Y':
        plt.plot(x,y_euler,'b-')
        legend_holder.append('Euler')
    if two_check == 'Y':
        plt.plot(x,y_huan,'g-')
        legend_holder.append('Improved-Euler')
    if three_check == 'Y':
        plt.plot(x,y_runge,'c-')
        legend_holder.append('Runge-Kutta')
    if four_check == 'Y':
        plt.plot(x,y_rungef,'m-')
        legend_holder.append('Runge-Kutta-Fehlberg 4th Order')

    plt.legend(legend_holder)
    plt.grid(True)
    plt.title("Solution")
    plt.show()


#?==================== Create table =================

def table():
    call_function()
    if one_check == 'Y':
        errorEuler()
    if two_check == 'Y':
        errorHuan()
    if three_check == 'Y':
        errorRunge()
    if four_check == 'Y':
        errorRungeF()


#!==================== Main ====================

if __name__ == '__main__':
    start()
    table()
    plot()