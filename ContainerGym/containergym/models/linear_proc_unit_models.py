from typing import Optional, Union

import numpy as np

"""
Processing unit models are linear of the form:
	processing_time ~ slopes_proc_unit_(i, container_id) * n_products
where i is the index of the requested processing unit.
"""


class ProcUnitModel:
    """
    A class that models processing unit behavior based on container and processing parameters.

    Parameters
    ----------
    enabled_containers : list or None
        List of containers that are connected to a PU. If None, no containers are connected.

    slopes : dict or list or np.ndarray, optional
        Dictionary, list, or numpy array with the slopes for each PU. If a dictionary is provided,
        it must have a key for each container that is connected to a PU. If a list or array is provided,
        it must have length n_containers, even if a PU is not connected to a container. In that case,
        set that corresponding container's value to None.

    offsets : dict or list or np.ndarray, optional
        Dictionary, list, or numpy array with the offsets for each PU. If a dictionary is provided,
        it must have a key for each container that is connected to a PU. If a list or array is provided,
        it must have length n_containers, even if a PU is not connected to a container. In that case,
        set that corresponding container's value to None.

    Attributes
    ----------
    slopes : np.ndarray
        Array with the slopes for each processing unit.

    offsets : np.ndarray
        Array with the offsets for each processing unit.

    Methods
    -------
    get_processing_time(current_time, time_prev_processing_done, container_idx, n_products)
        Calculate how long it takes until processing unit is free.
    """

    def __init__(
        self,
        enabled_containers: Optional[list],
        slopes: Union[dict, list, np.ndarray] = {"C1-20": 264.9},
        offsets: Union[dict, list, np.ndarray] = {"C1-20": 106.798502},
    ):
        # "Overloading" of the constructor to allow dict or list/array input.
        # We assume all parameters are of the same type (either dict, list, or ndarray)
        # List and array parameters must be of length n_containers, even if a PU is not connected to a container.
        # In that case, set that corresponding container's value to None
        if type(slopes) == dict and enabled_containers:
            self.slopes = np.array(
                [slopes.get(container, None) for container in enabled_containers]
            )
            self.offsets = np.array(
                [offsets.get(container, None) for container in enabled_containers]
            )
        elif type(slopes) == list:
            self.slopes = np.array(slopes)
            self.offsets = np.array(offsets)
        elif type(slopes) == np.ndarray:
            self.slopes = slopes
            self.offsets = offsets
        else:
            raise ValueError(
                "Parameters must be of types dict, list or ndarray. They must all be of the same type."
            )

    def get_processing_time(
        self, current_time, time_prev_processing_done, container_idx, n_products
    ):
        """
        Calculate how long it takes until a processing unit is free.

        Parameters
        ----------
        current_time : float
            The current time since the start of the simulation.

        time_prev_processing_done : float
            The time at which the last process finished.

        container_idx : int
            The index of the container that will be emptied.

        n_products : int
            How many products will be processed.

        Returns
        -------
        float or None
            Time in seconds until processing will be finished. None if PU is not free.
        """
        if current_time > time_prev_processing_done or not self.slopes[container_idx]:
            # Proc_unit is not free or container not compatible with this proc_unit
            return None
        return (
            current_time
            + self.slopes[container_idx] * n_products
            + self.offsets[container_idx]
        )
