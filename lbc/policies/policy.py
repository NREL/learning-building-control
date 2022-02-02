from abc import abstractmethod
from typing import Tuple

from lbc.scenario import Scenario, Batch


class Policy:

    @abstractmethod
    def __call__(
        self,
        scenario: Scenario = None,
        batch: Batch = None,
        t: int = None,
        x: any = None,
        u: any = None,
        zone_temp: any = None,
        action_init: any = None,
        training: bool = True,
        **kwargs
    ) -> Tuple[any, dict]:
        """Returns the policy actions and metadata.  The keyword arguments
        will always be provided in simulate.py, it is up to the user what
        to do with them.

        Args:
            scenario:  scenario instance
            batch:  batch instance for exogenous data
            t: time index of current control step
            x: state tensor at current control step
            zone_temp:  zone temperature tensor at current control step
            action_init:  initial (or previous) action for linearization
            training:  whether we are actively training a policy or not
            **kwargs:  for customization and forward compatibility

        Returns:
            actions:  tensor of computed actions
            meta:  dictionary of metadata
        """

        raise NotImplementedError
