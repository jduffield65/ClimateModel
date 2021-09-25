from sympl import get_constant

g = get_constant('gravitational_acceleration', 'm/s^2')
c_p_dry = get_constant('heat_capacity_of_dry_air_at_constant_pressure', 'J/kg/K')
sigma = get_constant('stefan_boltzmann_constant', 'W/m^2/K^4')
p_surface_earth = get_constant('reference_air_pressure', 'Pa')
p_one_atmosphere = 101325  # one atmosphere in Pa
p_toa_earth = 20  # top of atmosphere is 20 Pa
F_sun = get_constant('solar_constant', 'W/m^2')
Omega = get_constant('planetary_rotation_rate', 's^-1')
R_earth = get_constant('planetary_radius', 'm')
R_specific = get_constant('gas_constant_of_dry_air', 'J/kg/K')
Avogadro = get_constant('avogadro_constant', 'mole^-1')
speed_of_light = get_constant('speed_of_light', 'm/s')
h_planck = get_constant('planck_constant', 'J s')
k_boltzmann = get_constant('boltzmann_constant', 'J/K')
AU = 1.495978707e11  # average distance between earth and sun (m)
R_sun = 6.96340e8  # radius of sun (m)
T_sun = 5778  # temperature of sun (K).
