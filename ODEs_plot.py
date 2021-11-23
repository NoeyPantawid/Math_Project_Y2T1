#==================== Library ====================

import numpy as np
import matplotlib.pyplot as plt
import builtins


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
def odeRunge_kutta_fehlberg_fourth(f, yp, yx, x, h):
    y = np.zeros(len(x))
    y[yp] = yx
    for n in range(0,len(x)-1):
        k1 = h * f(x[n], y[n])
        k2 = h * f(x[n] + ((1/4) * h), y[n] + (1/4) * k1)
        k3 = h * f(x[n] + ((3/8) * h), y[n] + ((3/32) * k1) + ((9/32) * k2))
        k4 = h * f(x[n] + ((12/13) * h), y[n] + ((1932/2197) * k1) - ((7200/2197) * k2) + ((7296/2197) * k3))
        k5 = h * f(x[n] + h, y[n] + ((439/216) * k1) - (8 * k2) + ((3680/513) * k3) - ((845/4104) * k4))
        y[n+1] = y[n] + ((25/216) * k1) + ((1408/2565) * k3) + ((2197/4104) * k4) + ((-1/5) * k5)

    return y

#Runge-Kutta-Fehlberg Method (5th order)
def odeRunge_kutta_fehlberg_fifth(f, yp, yx, x, h):
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

def errortable(method, y, y_exact, yp):
    diff = zip(y, y_exact)
    n = 0
    s = yp
    Frequency = f'{(stop-start)/h:.2f}'

    if method == 'y_euler':
        method_name = 'Euler Method '
        Frequency = f'{Frequency:<65}'
    elif method == 'y_huan':
        method_name = 'Improved Euler Method '
        Frequency = f'{Frequency:<56}'
    elif method == 'y_runge':
        method_name = 'Runge-Kutta Method '
        Frequency = f'{Frequency:<59}'
    elif method == 'y_rungef_fourth':
        Frequency = f'{Frequency:<39}'
        method_name = 'Runge-Kutta-Fehlberg Method (4th order)'
    elif method == 'y_rungef_fifth':
        Frequency = f'{Frequency:<39}'
        method_name = 'Runge-Kutta-Fehlberg Method (5th order)'

    print('______________________________________________________________________________________________')
    print(f'|{method_name}| Frequency = {Frequency}|')
    print('|____________________________________________________________________________________________|')
    print(f'|    n    |      x[n]      |       y[n]        |      exact       |          error           |')
    print('|_________|________________|___________________|__________________|__________________________|')
    for i, j in diff:
        s_f = f'{s:.2f}'
        i_f = f'{i:.4f}'
        j_f = f'{j:.4f}'
        er_f = f'{abs(i-j):.10f}'
        print(f'| n = {n:^3} | x[{n:^3}] = {s_f:^5} | y[{n:^3}] = {i_f:>8} | exact = {j_f:>8} | error = {er_f:<16} |')
        n += 1
        s += h
    print('|_________|________________|___________________|__________________|__________________________|')


#?==================== Input functions =================

def start():

    global start, stop, f, yp, yx, x, h, prob_exact, zero_check, one_check, two_check, three_check, four_check, five_check

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

    while True:
        prob = input("y' = ")
        if any(word in prob for word in dir(builtins)):
            print('You can not use that!')
            continue
        else:
            break

    while True:
        prob_exact = input("exact = ")
        if any(word in prob_exact for word in dir(builtins)):
            print('You can not use that!')
            continue
        else:
            break
    
    print('--------------------------------')
    while True:
        zero_check = input("Do you want all method solution?\n(Y/N) : ")
        if zero_check == 'Y' or zero_check == 'y':
            one_check = 'Y'
            two_check = 'Y'
            three_check = 'Y'
            four_check = 'Y'
            five_check = 'Y'
            break
        elif zero_check == 'N' or zero_check == 'n':
            sure_list = ['- Exact']
            print('--------------------------------')
            one_check = input("Do you want Euler method solution?\n(Y/N) : ")
            if one_check == 'Y' or one_check == 'y':
                sure_list.append('- Euler method')
            two_check = input("Do you want Improved-Euler method solution?\n(Y/N) : ")
            if two_check == 'Y' or two_check == 'y':
                sure_list.append('- Improved-Euler method')
            three_check = input("Do you want Runge-Kutta method solution?\n(Y/N) : ")
            if three_check == 'Y' or three_check == 'y':
                sure_list.append('- Runge-Kutta method')
            four_check = input("Do you want Runge-Kutta-Fehlberg method (4th order) solution?\n(Y/N) : ")
            if four_check == 'Y' or four_check == 'y':
                sure_list.append('- Runge-Kutta-Fehlberg 4th order method')
            five_check = input("Do you want Runge-Kutta-Fehlberg method (5th order) solution?\n(Y/N) :")
            if five_check == 'Y' or five_check == 'y':
                sure_list.append('- Runge-Kutta-Fehlberg 5th order method')
            print('--------------------------------')

            print("You prefer these method solution?\n")
            print(*sure_list, sep = '\n')
            r_u_sure = input("\n(Y/N) : ")
            if r_u_sure == 'Y' or r_u_sure == 'y':
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

    global y_euler, y_huan, y_runge, y_rungef_fourth, y_rungef_fifth, y_exact

    #Euler Method
    if one_check == 'Y' or one_check == 'y':
        y_euler = odeEuler(f, yp, yx, x, h)

    #Improved Euler Method
    if two_check == 'Y' or two_check == 'y':
        y_huan = odeHuan(f, yp, yx, x, h)

    #Runge-Kutta Method
    if three_check == 'Y' or three_check == 'y':
        y_runge = odeRunge_kutta(f, yp, yx, x, h)

    #Runge-Kutta-Fehlberg Method 4th order
    if four_check == 'Y' or four_check == 'y':
        y_rungef_fourth = odeRunge_kutta_fehlberg_fourth(f, yp, yx, x, h)

    #Runge-Kutta-Fehlberg Method 5th order    
    if five_check == 'Y' or five_check == 'y':
        y_rungef_fifth = odeRunge_kutta_fehlberg_fifth(f, yp, yx, x, h)

    #EXACT
    y_exact = eval(prob_exact)


#?==================== Plot graph ====================

def plot():
    call_function()
    
    plt.plot(x,y_exact,'r.-')
    legend_holder = ['Exact Solution']

    if one_check == 'Y' or one_check == 'y':
        plt.plot(x,y_euler,'b-')
        legend_holder.append('Euler')
    if two_check == 'Y' or two_check == 'y':
        plt.plot(x,y_huan,'g-')
        legend_holder.append('Improved-Euler')
    if three_check == 'Y' or three_check == 'y':
        plt.plot(x,y_runge,'c-')
        legend_holder.append('Runge-Kutta')
    if four_check == 'Y' or four_check == 'y':
        plt.plot(x,y_rungef_fourth,'m-')
        legend_holder.append('Runge-Kutta-Fehlberg 4th Order')
    if five_check == 'Y' or five_check == 'y':
        plt.plot(x,y_rungef_fifth,'y-')
        legend_holder.append('Runge-Kutta-Fehlberg 5th Order')

    plt.legend(legend_holder)
    plt.grid(True)
    plt.title("Solution")
    plt.show()


#?==================== Create table =================

def table():
    call_function()
    if one_check == 'Y' or one_check == 'y':
        errortable('y_euler', y_euler, y_exact, yp)
    if two_check == 'Y' or two_check == 'y':
        errortable('y_huan', y_huan, y_exact, yp)
    if three_check == 'Y' or three_check == 'y':
        errortable('y_runge', y_runge, y_exact, yp)
    if four_check == 'Y' or four_check == 'y':
        errortable('y_rungef_fourth', y_rungef_fourth, y_exact, yp)
    if five_check == 'Y' or five_check == 'y':
        errortable('y_rungef_fifth', y_rungef_fifth, y_exact, yp)


#!==================== Main ====================

if __name__ == '__main__':
    start()
    table()
    plot()