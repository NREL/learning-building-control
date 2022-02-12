import os
import random
import logging
from numbers import Number
import pickle
from typing import Union

import pandas as pd
import numpy as np

from lbc.batch import Batch
from lbc.utils import to_torch
from lbc.demand_response import DemandResponseProgram as DRP


logger = logging.getLogger(__name__)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _comfort_band_to_array(x, length, dtype=np.float32):
    """Cast the input object as array of given length and type."""
    if isinstance(x, Number):
        data = np.array((length, 2), dtype=dtype)
        data[:, 0] = x[0]
        data[:, 1] = x[1]
    else:
        assert x.shape[1] == 2, "expected an array with two columns"
        data = np.array(x)[:length, :]
    return data


class Scenario:

    action_min = np.array([.22, .22, .22, .22, .32, 10.0])
    action_max = np.array([2.2, 2.2, 2.2, 2.2, 3.2, 16.0])
    action_init = np.array([.22, .22, .22, .22, .32, 16.0])

    def __init__(
        self,
        train_start_date: str = "2020-07-01",
        train_end_date: str = "2020-07-31",
        test_start_date: str = "2020-08-01",
        test_end_date: str = "2020-08-31",
        start_time: str = "00:05:00",
        end_time: str = "23:55:00",
        action_penalty: float = 1.,
        comfort_penalty: float = 10.,
        comfort_band: Union[tuple, np.ndarray] = None,
        zone_temp_init_mean: float = None,
        zone_temp_init_std: float = None,
        data_file: str = None,
        zone_model_file: str = None,
        dr_program: DRP = DRP('TOU')
    ):
        """Class for defining training and test scenarios and generating
        samples from them.

        Args:
          scenario_config (dict): Configuration for the scenarios. The dict
          includes the following info:

            train_start_date (str): Training start date.
            train_end_date (str): Training end date.
            test_start_date (str): Test start date.
            test_end_date (str): Test end date.
            start_time (str): Scenario start time.
            end_time (str): Scenario end time.
            action_penalty (float): Soft constraint penalty for actions.
            comfort_penalty (float): Soft constraint penalty for
                thermal comfort.
            comfort_min (float): Lower comfort temperature.
            comfort_upper (float): Upper comfort temperature.
            zone_temp_mean (float, optional): Mean value of zone temperatures
              if not using exogenous. Defaults to None.
            zone_temp_std (float, optional): Std deviation around mean value of
              zone temperatures.  Required if mean is specified. Defaults to
              None.
            data_file (str, optional): CSV file. Defaults to None, in which
              case `./data/exogenous_data.csv` is used.
            zone_model_file (str, optional): Pickle file. Defaults to None, in
              which case `./data/models.p` is used.
            dr_program (DemandResponseProgram class).
        """

        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.start_time = start_time
        self.end_time = end_time
        self.action_penalty = action_penalty
        self.comfort_penalty = comfort_penalty
        self.zone_temp_init_mean = zone_temp_init_mean
        self.zone_temp_init_std = zone_temp_init_std
        self.dr_program = dr_program

        # Use the pickle file in the data directory by default.
        if zone_model_file is None:
            zone_model_file = os.path.join(THIS_DIR, "data/models.p")
        with open(zone_model_file, "rb") as f:
            self.zone_model = pickle.load(f)

        if pd.Timestamp(self.test_start_date) < pd.Timestamp(
                self.train_end_date):
            logger.warn(
                "test start date preceeds train end date, are you sure?")

        # Use the csv file in the data directory by default.
        if data_file is None:
            data_file = os.path.join(THIS_DIR, "data/exogenous_data.csv")

        # Read the csv data, slice the desired dates, and create list of
        # scenario dates.
        self.df = pd.read_csv(data_file, index_col=0)
        self.df.index = pd.DatetimeIndex(self.df.index)

        _df = self.df[self.train_start_date: self.train_end_date]
        self.train_dates = sorted(list(set([pd.Timestamp(x).date()
                                            for x in _df.index])))
        _df = self.df[self.test_start_date: self.test_end_date]
        self.test_dates = sorted(list(set([pd.Timestamp(x).date()
                                           for x in _df.index])))

        assert len(self.train_dates) > 0, \
            "configuration resulted in no valid dates"

        # Calculate the control timedelta and number of timesteps per episode
        step_timedelta = self.df.index[1] - self.df.index[0]
        day1_start_timestamp = pd.Timestamp(f"{self.train_start_date}"
                                            f" {self.start_time}")
        day1_end_timestamp = pd.Timestamp(f"{self.train_start_date}"
                                          f" {self.end_time}")
        episode_timedelta = day1_end_timestamp - day1_start_timestamp
        self.num_episode_steps = int(episode_timedelta / step_timedelta) + 1
        self.hours_per_timestep = step_timedelta / pd.Timedelta("1h")

        # Keep timestamps for plot labels later.
        self.time_index = self.df.index[:self.num_episode_steps]

        # Read in comfort band data if not provided.
        if comfort_band is None:
            comfort_band = pd.read_csv(os.path.join(
                THIS_DIR, "data/comfort-band.csv"), index_col=0)
            comfort_band.index = pd.DatetimeIndex(comfort_band.index)
            _offset = int((self.df.index[0] - day1_start_timestamp)
                          / step_timedelta)
            comfort_band = comfort_band.values[_offset:_offset +
                                               self.num_episode_steps]

        # Cast comfort band data to array and extract separate min/max for
        # readability.
        self.comfort_band = _comfort_band_to_array(
            comfort_band, self.num_episode_steps)
        self.comfort_min = self.comfort_band[:, 0]
        self.comfort_max = self.comfort_band[:, 1]

        # Setup DR program
        self.dr_program.setup_demand_response_details(
            self.start_time, self.end_time)

    def __repr__(self) -> str:
        return (f"training dates: {self.train_dates[0]} to "
                f"{self.train_dates[-1]}\n" +
                f"test dates: {self.test_dates[0]} to "
                f"{self.test_dates[-1]}\n" +
                f"time of day: {self.start_time} to {self.end_time}\n" +
                f"action penalty = {self.action_penalty}\n" +
                f"comfort penalty = {self.comfort_penalty}\n" +
                f"zone model = {self.zone_model}" +
                "dataframe:\n" +
                str(self.df.head(3)))

    def make_batch(
        self,
        batch_size: int,
        shuffle: bool = False,
        as_tensor: bool = True,
        training: bool = True
    ) -> Batch:
        """Return a batch of numpy arrays or tensors from the scenario data.
        The shape of the output is [batch_size, num_times, 5] for
        exogenous data, and [batch_size, 5] for initial conditions.

        Args:
            batch_size (int): batch size, cannot exceed number of dates in
              scenario.
            shuffle (bool, optional): whether dates be shuffled.
              Defaults to True.
            as_tensor (bool, optional): return a torch tensor. Defaults to
              True.
            training (bool, optional): Use the training set.  Defaults to
              True. Set to False to generate batches from the test set.

        Returns:
            Batch instance.
        """

        if training:
            _df = self.df[self.train_start_date:self.train_end_date].copy()
            _dates = self.train_dates.copy()
        else:
            _df = self.df[self.test_start_date:self.test_end_date].copy()
            _dates = self.test_dates.copy()

        # Limit the batch size to the number of scenario dates.
        batch_size = min(batch_size, len(_dates))
        assert batch_size > 0, "batch size must be > 0"

        # Create a list of batch dates, optionally shuffled.
        if shuffle:
            random.shuffle(_dates)
        _dates = _dates[:batch_size].copy()

        # Create a list of tuples of start and end timestamps.
        start_times = [f"{d} {self.start_time}" for d in _dates]
        end_times = [f"{d} {self.end_time}" for d in _dates]
        times = sorted(list(zip(start_times, end_times)))

        # Exogenous data
        # Outdoor air temperature.
        temp_oa = np.stack([_df["T_oa"].loc[s:e].values for s, e in times])
        temp_oa = temp_oa if not as_tensor else to_torch(temp_oa)

        # Solar irradiance.
        cols = [f"Q_solar_{i+1}" for i in range(5)]
        q_solar = np.array([_df[cols].loc[s:e].values for s, e in times])
        q_solar = q_solar if not as_tensor else to_torch(q_solar)

        # Initial conditions
        # Initial cooling energy (this depends on control actions
        # for later timesteps).
        cols = [f"Q_cool_{i+1}" for i in range(5)]
        q_cool = np.array([_df[cols].loc[s].values for s, _ in times])
        q_cool = q_cool if not as_tensor else to_torch(q_cool)

        # Comfort band
        comfort_min = to_torch(self.comfort_min, batch_size=batch_size)
        comfort_max = to_torch(self.comfort_max, batch_size=batch_size)

        # Energy price
        if self.dr_program.program_type == 'RTP':
            energy_price = np.stack([_df["RTP"].loc[s:e].values
                                     for s, e in times])

            predicted_energy_price = np.stack([_df["DAP"].loc[s:e].values
                                               for s, e in times])
        else:  # same for both 'TOU' and 'PC'
            energy_price = np.stack([
                self.dr_program.energy_price.values.squeeze()
                for s, e in times])
            predicted_energy_price = energy_price

        energy_price = energy_price if not as_tensor else to_torch(
            energy_price)
        predicted_energy_price = predicted_energy_price \
            if not as_tensor else to_torch(predicted_energy_price)

        # Initial zone temperature. Evolves according to control dynamics for
        # later timesteps.
        if self.zone_temp_init_mean is not None:
            std = (self.zone_temp_init_std
                   if self.zone_temp_init_std is not None
                   else 0.)
            zone_temp = ((self.zone_temp_init_mean + std * np.random.randn(5))
                         * np.ones((batch_size, 5)))
        else:
            cols = [f"T_{i+1}" for i in range(5)]
            zone_temp = np.array([_df[cols].loc[s].values for s, _ in times])
        zone_temp = zone_temp if not as_tensor else to_torch(zone_temp)

        return Batch(temp_oa, q_solar, q_cool, zone_temp, comfort_min,
                     comfort_max, energy_price, predicted_energy_price)
