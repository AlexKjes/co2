import numpy as np

# Average human co2 emission per sec in grams
AHCES = 0.012071759
# Moles per gram co2
MOL_G = 46


class Room:

    def __init__(self, size, init_air_quality=500, optimal_air_quality=500, occupancy=0, air_changes_per_hour=3):

        self.t = 0
        self.size = size
        self.cubic_liters = np.prod(size) * 1000
        self.optimal_air_quality = optimal_air_quality  # ventilation co2ppm
        self.co2ppm = init_air_quality
        self.occupancy = occupancy
        self.air_changes_per_second = air_changes_per_hour/(60*60)

    def add_people(self, n):
        self.occupancy += n

    def remove_people(self, n):
        if self.occupancy - n < 0:
            raise ValueError("Occupancy can not be less than 0")
        self.occupancy -= n

    def time_step(self):
        self.ventilation_co2_reduction()
        self.occupants_co2_production()

    def occupants_co2_production(self):
        emission_g = AHCES * self.occupancy
        emission_mol = emission_g/46
        emission_liter = emission_mol*22.4
        self.co2ppm += emission_liter / self.cubic_liters * 10**6

    def ventilation_co2_reduction(self):
        self.co2ppm *= 1-self.air_changes_per_second
        self.co2ppm += self.optimal_air_quality * self.air_changes_per_second
