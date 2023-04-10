from .sector import Sector


class Track:
    def __init__(self,track_id):
        self.track_id = track_id
        self.sectors = [
            Sector({
                "sector_id": "S1",# as of official NLS document
                "sector_length": 2745.6,# as of official NLS document
                "base_time": 66.019,
                "timeLoss_fuelMass": 0.03,
                "timeLoss_tireDeg": 1.1,
                "wear_fuelCon": 1.7,
                "wear_tireDeg": 0.5,
                "factor_overtaking": 3,
                "variance_min": 0.6,
                "variance_max": 1.8,
                "prob_code60": 0.0
            }),

            Sector({
                "sector_id": "S2",# as of official NLS document
                "sector_length": 3003.9,# as of official NLS document
                "base_time": 63.468,
                "timeLoss_fuelMass": 0.035,
                "timeLoss_tireDeg": 1.3,
                "wear_fuelCon": 1.2,
                "wear_tireDeg": 1.0,
                "factor_overtaking": 3,
                "variance_min": 1.4,
                "variance_max": 2.9,
                "prob_code60": 0.07235
            }),

            Sector({
                "sector_id": "S3",# as of official NLS document
                "sector_length": 6002.5,# as of official NLS document
                "base_time": 117.133,
                "timeLoss_fuelMass": 0.08,
                "timeLoss_tireDeg": 1.4,
                "wear_fuelCon": 1.1,
                "wear_tireDeg": 1.0,
                "factor_overtaking": 3,
                "variance_min": 3.7,
                "variance_max": 4.8,
                "prob_code60": 0.17122
            }),

            Sector({
                "sector_id": "S4",# as of official NLS document
                "sector_length": 9409.4,# as of official NLS document
                "base_time": 183.532,
                "timeLoss_fuelMass": 0.08,
                "timeLoss_tireDeg": 1.4,
                "wear_fuelCon": 1.3,
                "wear_tireDeg": 1.4,
                "factor_overtaking": 3,
                "variance_min": 8.2,
                "variance_max": 9.9,
                "prob_code60": 0.19333
            }),

            Sector({
                "sector_id": "S5",# as of official NLS document
                "sector_length": 3196.6,# as of official NLS document
                "base_time": 48.829 ,
                "timeLoss_fuelMass": 0.7,
                "timeLoss_tireDeg": 0.07,
                "wear_fuelCon": 2,
                "wear_tireDeg": 0.2,
                "factor_overtaking": 3,
                "variance_min": 0.5,
                "variance_max": 1.7,
                "prob_code60": 0.09285
            }),
        ]

        self.code60_phases = []
