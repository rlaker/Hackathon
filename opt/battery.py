class battery:
    def __init__(self, eta=0.85, power=1.0, capac=1.0, init_energy=0.0):
        """
        eta - efficiency
        power - power in MW
        capac - capacity in MWh
        init_energy - initial energy stored in the battery

        CAUTION: IS THE EFFICIENCY IMPLEMENTATION CORRECT

        """
        self.eta = eta
        self.power = power
        self.capac = capac
        self.energy = init_energy  # energy currently stored in the battery (in MWh)

    def charge(self, nmin):
        """
        Charges battery for a number of minutes
        """
        self.energy += self.eta * self.power * (nmin / 60.0)
        self.energy = min(self.capac, self.energy)

    def discharge(self, nmin):
        """
        Discharges battery for a number of minutes
        """
        self.energy -= self.power * (nmin / 60.0)
        self.energy = max(0.0, self.energy)
