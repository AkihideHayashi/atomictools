from math import pi

Eh = 1.0                               # energy (hartee)
me = 1.0                               # mass of electron
a0 = 1.0                               # length (bohr radius)
e  = 1.0                               # charge of electron
hbar = 1.0                             # action (h / 2pi)

alpha = 7.2973525664E-3                # fine-structure constant
alpha_inv = 137.035999139              # 1 / alpha
c = alpha_inv                          # speed of light

J = 2.2937123163853187e+17 * Eh        # joule
kg = 1.0977691228098864e+30 * me       # kilogram
m = 1.8897261255E+10 * a0              # metre
C = 6.241510734005358e+18 * e          # coulomb
Js = 9.48252172113838e+33 * hbar       # joule second

s = Js / J
m_s = m / s
kgm_s = kg * m / s
V = J / C
V_m = V / m
eV = e * V
Hz = 1 / s
THz = 1E12 * Hz
kb = 1.380649E-23 * J # / K            # Boltzmann constant have to be defined with Kelvin. Kelvin can be converted to energy by multiply k
Pa = kg / (m * s * s)
atm = 101325 * Pa

Da = 1822.8884853323707
angstrom = m * 1E-10

mol = 6.022140857E23

cal = 4.184 * J

kayser = 100 / m * hbar * 2 * pi * c

