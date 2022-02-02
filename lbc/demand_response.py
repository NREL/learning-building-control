import os

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class DemandResponseProgram:

    REQUIRED_CONFIG_KEYS = {
        'TOU': ['tou_off_peak', 'tou_peak'],
        'PC': ['p_limit_dr', 'p_limit_nondr', 'base_price', 'pc_penalty'],
        'RTP': []
    }

    def __init__(self,
                 program_type: str,
                 **dr_config):

        self.default_config = {

            'TOU': {
                'dr_start_time': '12:00:00',
                'dr_end_time': '18:00:00',
                'program_type': 'TOU',
                'program_specific': {
                    'tou_off_peak': 1.,
                    'tou_peak': 10.,
                }
            },

            'PC': {
                'dr_start_time': '13:00:00',
                'dr_end_time': '17:00:00',
                'program_type': 'PC',
                'program_specific': {
                    'p_limit_dr': 10.,
                    'p_limit_nondr': 25.,
                    'base_price': 1.,
                    'pc_penalty': 10.,
                }
            },

            'RTP': {
                'program_type': 'RTP',
                'program_specific': {}
            }
        }

        self.program_type = program_type

        for (prop, default) in self.default_config[program_type].items():
            setattr(self, prop, dr_config.get(prop, default))

        self.config_sanity_check()

        self.energy_price = None
        self.power_limit = None
        self.pc_penalty = None

    def config_sanity_check(self):

        assert self.program_type in ['TOU', 'RTP', 'PC'], \
            "Demand response program type can only be 'TOU', 'RTP' or 'PC'."

        requirement_met = True
        for required_key in self.REQUIRED_CONFIG_KEYS[self.program_type]:
            if required_key not in self.program_specific.keys():
                print("%s configuration requires key: %s" % (self.program_type,
                                                             required_key))
                requirement_met = False

        if not requirement_met:
            raise ValueError("Please check DR program_specific configuration.")

    def setup_demand_response_details(self,
                                      episode_start,
                                      episode_end):

        if self.program_type == 'TOU':

            tou_off_peak = self.program_specific['tou_off_peak']
            tou_peak = self.program_specific['tou_peak']

            self.energy_price = self.generate_requirement_profile(
                'price', episode_start, episode_end, tou_off_peak, tou_peak
            )

        elif self.program_type == 'PC':

            dr_power_limit = self.program_specific['p_limit_dr']
            non_dr_power_limit = self.program_specific['p_limit_nondr']
            base_price = self.program_specific['base_price']

            self.power_limit = self.generate_requirement_profile(
                'power_limit', episode_start, episode_end,
                non_dr_power_limit, dr_power_limit
            )
            self.energy_price = self.generate_requirement_profile(
                'price', episode_start, episode_end, base_price
            )

            self.pc_penalty = self.program_specific['pc_penalty']

        elif self.program_type == 'RTP':
            pass

    def generate_requirement_profile(self,
                                     profile_name: str,
                                     episode_start_time: str,
                                     episode_end_time: str,
                                     non_dr_val: float,
                                     dr_val: float = None):
        """ Generate the DR requirement profile, this can either be a price
        profile or a power constraint profile over the episode.

        Args:
          profile_name: Name of the profile generated, used as pandas column
            name.
          episode_start_time: Timestamp string as in 'HH:MM:SS' indicating
            the start of the episode.
          episode_end_time: Timestamp string as in 'HH:MM:SS' indicating the
            end of the episode.
          non_dr_val: Value of the profile during non-DR period, e.g., off-peak
            TOU.
          dr_val: Value of the profile during the DR event, e.g., peak TOU.

        Return:
          profile: DR requirement profile in pandas data frame format.

        TODO: I don't remember why I used pandas dataframe here, might be able
        to simplify.
        """

        step_timedelta = pd.Timedelta('300s')  # Default 5-min interval.
        episode_start_timestamp = pd.Timestamp(f"{episode_start_time}")
        episode_end_timestamp = pd.Timestamp(f"{episode_end_time}")
        episode_duration = episode_end_timestamp - episode_start_timestamp
        steps_per_episode = int(episode_duration / step_timedelta) + 1

        # Create tou price signal using timestamps the convert to array.
        df_index = pd.date_range(
            start=episode_start_timestamp, end=episode_end_timestamp,
            freq=step_timedelta)

        dr_start_timestamp = pd.Timestamp(f"{self.dr_start_time}")
        dr_end_timestamp = pd.Timestamp(f"{self.dr_end_time}")

        profile = pd.DataFrame(data=non_dr_val * np.ones(steps_per_episode),
                               index=df_index, columns=[profile_name])

        if dr_val is not None:
            event_index = pd.date_range(start=dr_start_timestamp,
                                        end=dr_end_timestamp,
                                        freq=step_timedelta)
            event_index = [x for x in event_index if x in profile.index]
            profile.at[event_index] = dr_val

        return profile


if __name__ == '__main__':

    drp = DemandResponseProgram('TOU')

    print(drp.energy_price.values.squeeze())

    print('done')
