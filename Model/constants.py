from sympl import get_constant

g = get_constant('gravitational_acceleration', 'm/s^2')
c_p_dry = get_constant('heat_capacity_of_dry_air_at_constant_pressure', 'J/kg/K')
sigma = get_constant('stefan_boltzmann_constant', 'W/m^2/K^4')
p_surface = get_constant('reference_air_pressure', 'Pa')
p_toa = 20  # top of atmosphere is 20 Pa
F_sun = get_constant('solar_constant', 'W/m^2')

