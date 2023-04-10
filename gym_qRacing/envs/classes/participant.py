from rich.console import Console
console = Console()

class Participant:
    def __init__(self, participant_id, car_fuelMass, race_position):
        self.participant_id = participant_id
        
        # * car parameters
        self.car_fuelMass = car_fuelMass
        self.car_tireDeg = 0
        self.car_pitStops = []
        self.car_stintLength = 1


        # * race parameters
        self.race_time = 0
        self.race_position = race_position

        self.lastLap_time = 0
        self.lastLap_position = race_position

        self.delta_front = 0
        self.delta_back = 0


        # * status flags
        self.next_pitStop = None
        self.is_retired = False


        # * logging
        self.log = {
            "laps": {}
        }

        # structure of log dict:
        """ 
        self.log = {
            "laps": {
                   1: {                                                                                                                                       
                       'lap_time': 0.0,                                                                                                                       
                       'sectors': {                                                                                                                           
                           'S1': {'sector_time': 66.019, 'wear_tireDeg': 0, 'wear_fuelCon': 0},                                                               
                           'S2': {'sector_time': 63.468, 'wear_tireDeg': 0, 'wear_fuelCon': 0},                                                               
                           'S3': {'sector_time': 117.133, 'wear_tireDeg': 0, 'wear_fuelCon': 0},                                                              
                           'S4': {'sector_time': 183.532, 'wear_tireDeg': 0, 'wear_fuelCon': 0},                                                              
                           'S5': {'sector_time': 48.829, 'wear_tireDeg': 0, 'wear_fuelCon': 0}                                                                
                       },
                       'wear_tireDeg': 0, 
                       'wear_fuelCon': 0                                                                                                                               
                   }  
            }
        }
        """


    #
    # * this functions updates the lap_time and race_time for the participant
    #
    # TODO: implement handling of last sector and start of next lap
    def update_lapTime(self, sector_time):

        # updating last lap time
        self.lastLap_time += sector_time

        # updating race_time
        self.race_time += sector_time

        # ! this should not be here... Updating stint length
        self.car_stintLength += 1

        # TODO: also update lap_time for log dict!



    #
    # * this functions updates the wear for the participant
    #
    def update_wear(self, tire_deg, fuel_con):
        self.car_tireDeg += tire_deg
        self.car_fuelMass -= fuel_con

        # handling retirement due to high wear
        if self.car_tireDeg >= 100:
            self.is_retired = True
            self.race_time = 9999999 # this puts the participant in last position
            print(self.participant_id, " retired due to tireDeg!")
            self.car_tireDeg = 100 # resetting to avoid q_table bound exception

        # handling retirement due to high wear
        if self.car_fuelMass < 1:
            self.is_retired = True
            self.race_time = 9999999 # this puts the participant in last position
            print(self.participant_id, " retired due to fuelMass! ", fuel_con)
            self.car_fuelMass = 1 # resetting to avoid q_table bound exception

    #
    # * this functions decides if the participant is pitting
    #
    def decide_pitStop(self, current_raceLap):
        if self.participant_id == "Agent" and len(self.car_pitStops) > 0:
            if current_raceLap == self.car_pitStops[len(self.car_pitStops)-1][0]:
                return True
        elif self.participant_id == "Agent" and len(self.car_pitStops) > 0:
            if current_raceLap-1 == self.car_pitStops[len(self.car_pitStops)-1][0]:
                #print("outlap in lap #", current_raceLap)
                return True

        if self.participant_id != "Agent" and self.car_fuelMass < 15 or self.car_tireDeg > 80:
            #print("pitstop in lap #", current_raceLap)
            if not [current_raceLap, 8] in self.car_pitStops:
                self.car_pitStops.append([current_raceLap, 8])

            return True
            
        
        if self.participant_id != "Agent" and len(self.car_pitStops) > 0:
            if current_raceLap-1 == self.car_pitStops[len(self.car_pitStops)-1][0]:
                #print("outlap in lap #", current_raceLap)
                return True
        else:
            return False


    def update_log_sector(self, log_dict):

        # * check if lap key exists
        if log_dict['race_lap'] not in self.log['laps']:
            # * create lap dict!
            self.log['laps'][log_dict['race_lap']] = {
                "lap_time": 0.0,
                "sectors": {},
                'wear_tireDeg': 0.0, 
                'wear_fuelCon': 0.0,
                "current_tireDeg": self.car_tireDeg,
                "current_fuelMass": self.car_fuelMass,
                "current_racePosition": self.race_position,
                "agent_action": None
            }


        # * check if sector key exists
        if log_dict['sector_id'] not in self.log['laps'][log_dict['race_lap']]['sectors']:
            # * create sector dict!
            self.log['laps'][log_dict['race_lap']]['sectors'][log_dict['sector_id']] = log_dict['sector_data']

            # update lap time
            self.log['laps'][log_dict['race_lap']]['lap_time'] = self.log['laps'][log_dict['race_lap']]['lap_time'] + log_dict['sector_data']['sector_time']

            # set pitting flag
            self.log['laps'][log_dict['race_lap']]['agent_action'] = log_dict['agent_action']

            # update total wear for this lap
            self.log['laps'][log_dict['race_lap']]['wear_tireDeg'] = self.log['laps'][log_dict['race_lap']]['wear_tireDeg'] + log_dict['sector_data']['wear_tireDeg']
            self.log['laps'][log_dict['race_lap']]['wear_fuelCon'] = self.log['laps'][log_dict['race_lap']]['wear_fuelCon'] + log_dict['sector_data']['wear_fuelCon']


        #print(self.participant_id)
        #console.log(self.log)



