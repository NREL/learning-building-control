
class Batch:
    """Simple data structure to store batches of scenario data.

    Expected shapes:
    [Initial conditions]
        q_cool: (batch_size,)
        zone_temp: (batch_size, num_zones)
    [Exogenous variables]
        temp_oa: (batch_size, time)
        q_solar: (batch_size, time)
        comfort_min: (batch_size, time)
        comfort_max: (batch_size, time)
        energy_price: (batch_size, time)
        predicted_energy_price: (batch_size, time)
    """

    def __init__(
        self,
        temp_oa: any = None,
        q_solar: any = None,
        q_cool: any = None,
        zone_temp: any = None,
        comfort_min: any = None,
        comfort_max: any = None,
        energy_price: any = None,
        predicted_energy_price: any = None,
        actions_to_imitate: any = None
    ):

        # These are used only for initial conditions (if at all)
        self.q_cool = q_cool
        self.zone_temp = zone_temp

        # These variables drive the simulation and are needed at all timesteps.
        self.temp_oa = temp_oa
        self.q_solar = q_solar
        self.comfort_min = comfort_min
        self.comfort_max = comfort_max
        self.energy_price = energy_price
        self.predicted_energy_price = predicted_energy_price
        self.actions_to_imitate = actions_to_imitate

    def get_time(
        self,
        t: int,
        device: str = "cpu"
    ):
        """ Returns the batch tensors for non-initial condition variables at
        time index t.
        """

        # Get current values of exogenous data (excluding initial condition
        # data).
        temp_oa = self.temp_oa[:, t].unsqueeze(-1).to(device)
        q_solar = self.q_solar[:, t, :].to(device)
        comfort_min = self.comfort_min[:, t].unsqueeze(-1).to(device)
        comfort_max = self.comfort_max[:, t].unsqueeze(-1).to(device)
        energy_price = self.energy_price[:, t].unsqueeze(-1).to(device)
        predicted_price = self.predicted_energy_price[:, t]
        predicted_energy_price = predicted_price.unsqueeze(-1).to(device)
        
        actions = None
        if self.actions_to_imitate is not None:
            actions = self.actions_to_imitate[:, t, :].to(device)

        return (temp_oa, q_solar, comfort_min, comfort_max,
                energy_price, predicted_energy_price, actions)
