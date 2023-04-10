import math
import random


class Sector:
    def __init__(self, sector_params):
        self.sector_id = sector_params["sector_id"] # identifier of this sector
        self.base_time = sector_params["base_time"] # best possible time (in sec) for car with optimal driver+fuel+tire parameters
        
        self.timeLoss_tireDeg = sector_params["timeLoss_tireDeg"]
        self.timeLoss_fuelMass = sector_params["timeLoss_fuelMass"]

        self.variance_min = sector_params["variance_min"]
        self.variance_max = sector_params["variance_max"]

        self.wear_fuelCon = sector_params["wear_fuelCon"] # amount of fuel burned in this sector
        self.wear_tireDeg = sector_params["wear_tireDeg"] # amount tire degregation in this sector

        self.factor_overtaking = sector_params["factor_overtaking"] # minimum distance between cars to overtake in this sector
        self.prob_code60 = sector_params["prob_code60"]

        self.code60_phases = {} # ! not implemented yet! Should have entry for each phase with start+end (comes from failed overtakes)
        
        # ! not implemented yet!
        #self.code60_timeLoss = sector_params["code60_timeLoss"] 

