import numpy as np


# ***********************************************************************************
#
# FUNCTION: read_geometry
#
# ***********************************************************************************
def read_geometry(filename):
    with open(filename, 'r') as f:
        dados = f.read().split('\n')

    nnos = int(dados.pop(0).split(': ')[1])
    nelem = int(dados.pop(0).split(': ')[1])

    dados.pop(0)
    dados.pop(0)
    x = np.zeros((nnos, 2))
    for i in range(nnos):
        aux = dados.pop(0).split(',')
        x[int(aux[0]), 0] = float(aux[1])
        x[int(aux[0]), 1] = float(aux[2])

    dados.pop(0)
    dados.pop(0)
    conec = np.zeros((nelem, 3))
    for i in range(nelem):
        aux = dados.pop(0).split(',')
        conec[int(aux[0]), 0] = int(aux[1])
        conec[int(aux[0]), 1] = int(aux[2])
        conec[int(aux[0]), 2] = int(aux[3])

    return nnos, nelem, x, conec


# ***********************************************************************************
#
# FUNCTION: myfunc
#
# ***********************************************************************************
def myfunc(t, p0, t0, tf):
    if t0 <= t <= tf:
        return p0 * np.sin(np.pi * t / (tf - t0))
    else:
        return 0.0


# ***********************************************************************************
#
# FUNCTION: newmark
#
# ***********************************************************************************
def newmark(F_res, delta_y_old, ydt_old, yddt_old, gamma=1 / 2, beta=1 / 6):
    lhs = M / (beta * delta_t ** 2) + C * (gamma / (beta * delta_t)) + K
    mat1 = M / (beta * delta_t ** 2) + C * (gamma / (beta * delta_t))
    mat2 = M / (beta * delta_t) + C * (gamma / beta - 1)
    mat3 = M * (1 / (2 * beta) - 1) + C * delta_t * (gamma / (2 * beta) - 1)
    rhs = F_res + np.matmul(mat1, delta_y_old) + np.matmul(mat2, ydt_old) + np.matmul(mat3, yddt_old)
    delta_y = np.linalg.solve(lhs, rhs)
    ydt = (delta_y - delta_y_old) * (gamma / (beta * delta_t)) \
          + ydt_old * (1 - gamma / beta) \
          + yddt_old * delta_t * (1 - gamma / (2 * beta))
    yddt = (delta_y - delta_y_old) / (beta * delta_t ** 2) \
           - ydt_old / (beta * delta_t) \
           - yddt_old * (1 / (2 * beta) - 1)
    return np.array([delta_y, ydt, yddt])


# ***********************************************************************************
#
# FUNCTION: contour_displacements
#
# ***********************************************************************************
def contour_displacements(delta_yc):
    yc1, yc2, theta = delta_yc
    arr1 = yc1 + rho * np.cos(alpha + theta) - x[:, 0]
    arr2 = yc2 + rho * np.sin(alpha + theta) - x[:, 1]
    dy = np.column_stack((arr1, arr2))
    return dy


# ***********************************************************************************
#
# FUNCTION: write_output
#
# ***********************************************************************************
def write_output(filename):
    posproc = []
    posproc.append('ARQUIVO DE PÓS-PROCESSAMENTO\n')
    posproc.append('\n')
    posproc.append(
        'NNOS'.rjust(8) + ' ' +
        'NELEM'.rjust(8) + ' ' +
        'NLIST'.rjust(8) + '\n'
    )
    posproc.append('#\n')
    posproc.append('{:8d} {:8d} {:8d}\n'.format(nnos, nelem, 1))
    posproc.append('\n')

    posproc.append(
        'COORD.X'.rjust(10) + ' ' +
        'COORD.Y'.rjust(10) + ' ' +
        'COORD.Z'.rjust(10) + ' ' +
        'DESL.X'.rjust(10) + ' ' +
        'DESL.Y'.rjust(10) + ' ' +
        'DESL.Z'.rjust(10) + '\n'
    )
    posproc.append('#\n')
    for i in range(nnos):
        posproc.append(
            '{:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f}\n'.format(x[i, 0], x[i, 1], 0.0, 0.0, 0.0, 0.0))
    posproc.append('\n')

    posproc.append(
        'TIPO'.rjust(8) + ' ' +
        'GRAU'.rjust(8) + ' ' +
        'NO.1'.rjust(6) + ' ' +
        'NO.2'.rjust(6) + ' ' +
        'NO.3'.rjust(6) + ' ' +
        'GRUPO'.rjust(8) + '\n'
    )
    posproc.append('#\n')
    for i in range(nelem):
        incid = ''
        for j in range(3):
            incid += '{:6} '.format(int(conec[i, j] + 1))
        posproc.append('{:8} {:8} {} {:8}\n'.format(1, 2, incid.rstrip(), 0))
    posproc.append('\n')

    posproc.append('LISTAS\n')
    posproc.append(
        'DESL.X'.rjust(10) + ' ' +
        'DESL.Y'.rjust(10) + ' ' +
        'DESL.Z'.rjust(10) + ' ' +
        'VALOR'.rjust(10) + '\n'
    )
    posproc.append('\n')

    for ir, row in enumerate(delta_y):
        dy = contour_displacements(row)

        posproc.append('#\n')
        posproc.append('DESL.x {:d}\n'.format(ir))
        for i in range(nnos):
            posproc.append('{:10.5f} {:10.5f} {:10.5f} {:10.5f}\n'.format(dy[i, 0], dy[i, 1], 0.0, dy[i, 0]))
        posproc.append('\n')

        posproc.append('#\n')
        posproc.append('DESL.y {:d}\n'.format(ir))
        for i in range(nnos):
            posproc.append('{:10.5f} {:10.5f} {:10.5f} {:10.5f}\n'.format(dy[i, 0], dy[i, 1], 0.0, dy[i, 1]))
        posproc.append('\n')

    with open(filename, 'w') as f:
        f.writelines(posproc)

    return


# -----------------------------------------------------------------------------------
# SYSTEM PARAMETERS
# -----------------------------------------------------------------------------------
m = np.array([0.771, 0.771, 0.00135])  # MASS/INERTIA
zeta = np.array([0.05, 0.05, 0.05])  # DAMPING RATIO
k = np.array([700., 700., 0.3])  # STIFFNESS

# -----------------------------------------------------------------------------------
# NATURAL FREQUENCY AND FUNDAMENTAL PERIOD
# -----------------------------------------------------------------------------------
omega0 = np.sqrt(k / m)  # rad/s
f0 = omega0 / (2 * np.pi)  # Hz
T0 = 1 / f0  # s

# -----------------------------------------------------------------------------------
# CRITICAL DAMPING
# -----------------------------------------------------------------------------------
cc = 2 * np.sqrt(k * m)
c = zeta * cc

# -----------------------------------------------------------------------------------
# MASS, DAMPING AND STIFFNESS MATRICES
# -----------------------------------------------------------------------------------
M = np.diag(m)
C = np.diag(c)
K = np.diag(k)

# -----------------------------------------------------------------------------------
# TIME PARAMETERS
# -----------------------------------------------------------------------------------
delta_t = 0.01  # TIME STEP
t_max = 3.0  # TOTAL TIME
npt = int(np.ceil(t_max / delta_t)) + 1
t = np.linspace(0., t_max, npt)

# -----------------------------------------------------------------------------------
# EXCITING FORCES
# -----------------------------------------------------------------------------------
func = np.vectorize(myfunc)
F_res = np.full((npt, 3), 0.)
F_res[:, 0] = func(t, 40., 0.0, 0.8)
F_res[:, 1] = func(t, 30., 0., 0.3)
F_res[:, 2] = func(t, 0.2, 0.0, 1.0)

# -----------------------------------------------------------------------------------
# KNOWN INITIAL CONDITIONS
# -----------------------------------------------------------------------------------
delta_y = np.full((npt, 3), 0.)
ydt = np.full((npt, 3), 0.)
yddt = np.full((npt, 3), 0.)
delta_y[0] = np.full(3, 0.)  # INITIAL POSITION
ydt[0] = np.full(3, 0.)  # INITIAL VELOCITY
yddt[0] = np.linalg.solve(M, F_res[0] - np.matmul(C, ydt[0]) - np.matmul(K, delta_y[0]))

# -----------------------------------------------------------------------------------
# TEMPORAL INTEGRATION
# -----------------------------------------------------------------------------------
for it in range(1, npt):
    delta_y[it], ydt[it], yddt[it] = newmark(F_res[it], delta_y[it - 1], ydt[it - 1], yddt[it - 1])

# -----------------------------------------------------------------------------------
# INITIALIZE GEOMETRY
# -----------------------------------------------------------------------------------
cl = 0.156  # CHORD LENGTH
x_c = np.array([0.188 * cl, 0.])  # CENTER CORDINATES
nnos, nelem, x, conec = read_geometry('entrada.txt')
x = x * cl - x_c

# -----------------------------------------------------------------------------------
# POLAR COORDINATES
# -----------------------------------------------------------------------------------
rho = np.linalg.norm(x, axis=1)
alpha = np.arctan2(x[:, 1], x[:, 0])

# -----------------------------------------------------------------------------------
# ACADVIEW OUTUPUT
# -----------------------------------------------------------------------------------
write_output('saida.ogl')
