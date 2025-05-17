"""
Store system model parameters and generate datasets for non-linear cases.

This module defines the SystemModel class, which is responsible for:
- Storing system model parameters such as state transition function (f),
  observation function (h), process noise covariance (Q), observation noise
  covariance (R), sequence lengths (T, T_test), state dimension (m), and
  observation dimension (n).
- Generating training, validation, and test datasets.
"""
from tqdm import tqdm
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import Callable, Optional, List, Tuple

class SystemModel:
    """
    Represents the system model for state estimation.

    Attributes:
        f (Callable): State transition function.
        m (int): State dimension.
        Q (torch.Tensor): Process noise covariance matrix.
        h (Callable): Observation function.
        n (int): Observation dimension.
        R (torch.Tensor): Observation noise covariance matrix.
        T (int): Sequence length for training and validation.
        T_test (int): Sequence length for testing.
        prior_Q (torch.Tensor): Prior for process noise covariance.
        prior_Sigma (torch.Tensor): Prior for state covariance.
        prior_S (torch.Tensor): Prior for observation noise covariance.
        m1x_0 (torch.Tensor): Initial mean of the state.
        m2x_0 (torch.Tensor): Initial covariance of the state.
        m1x_0_batch (torch.Tensor): Batched initial mean of the state.
        x_prev (torch.Tensor): Previous state in batched sequence generation.
        m2x_0_batch (torch.Tensor): Batched initial covariance of the state.
        Input (List[torch.Tensor]): List of input sequences for the model.
        Target (List[torch.Tensor]): List of target sequences for the model.
    """

    def __init__(self, f: Callable, Q: torch.Tensor, h: Callable, R: torch.Tensor,
                 T: int, T_test: int, m: int, n: int,
                 prior_Q: Optional[torch.Tensor] = None,
                 prior_Sigma: Optional[torch.Tensor] = None,
                 prior_S: Optional[torch.Tensor] = None):
        """
        Initializes the SystemModel.

        Args:
            f: State transition function.
            Q: Process noise covariance matrix.
            h: Observation function.
            R: Observation noise covariance matrix.
            T: Sequence length for training and validation.
            T_test: Sequence length for testing.
            m: State dimension.
            n: Observation dimension.
            prior_Q: Prior for process noise covariance. Defaults to Q.
            prior_Sigma: Prior for state covariance. Defaults to a zero matrix.
            prior_S: Prior for observation noise covariance. Defaults to R.
        """

        ####################
        ### Motion Model ###
        ####################
        self.f = f
        self.m = m
        self.Q = Q

        #########################
        ### Observation Model ###
        #########################
        self.h = h
        self.n = n
        self.R = R

        ################
        ### Sequence ###
        ################
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        self.prior_Q = Q if prior_Q is None else prior_Q
        self.prior_Sigma = torch.zeros((self.m, self.m)) if prior_Sigma is None else prior_Sigma
        self.prior_S = R if prior_S is None else prior_S

        # Initialize attributes that will be set later
        self.m1x_0: Optional[torch.Tensor] = None
        self.m2x_0: Optional[torch.Tensor] = None
        self.m1x_0_batch: Optional[torch.Tensor] = None
        self.x_prev: Optional[torch.Tensor] = None
        self.m2x_0_batch: Optional[torch.Tensor] = None
        self.Input: List[torch.Tensor] = []
        self.Target: List[torch.Tensor] = []


    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0: torch.Tensor, m2x_0: torch.Tensor) -> None:
        """Initializes the mean and covariance of the initial state."""
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    def Init_batched_sequence(self, m1x_0_batch: torch.Tensor, m2x_0_batch: torch.Tensor) -> None:
        """Initializes batched mean and covariance of the initial state."""
        self.m1x_0_batch = m1x_0_batch
        self.x_prev = m1x_0_batch # Assuming x_prev should be initialized with the mean
        self.m2x_0_batch = m2x_0_batch

    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Matrix(self, Q: torch.Tensor, R: torch.Tensor) -> None:
        """Updates the process and observation noise covariance matrices."""
        self.Q = Q
        self.R = R

    def GenerateBatch(self, args: object, dataloader: torch.utils.data.DataLoader,
                      batch_size: int, randomInit: bool = False) -> None:
        """
        Generates a batch of data.

        Iterates through the provided dataloader to populate self.Input and self.Target.
        The initial conditions for the batch are set using self.m1x_0 and self.m2x_0.

        Args:
            args: Configuration object (expected to have attributes like randomInit_train/cv).
            dataloader: DataLoader providing the raw data.
            batch_size: The number of sequences to generate in this batch.
                       Note: The current implementation iterates through the entire
                       dataloader, and `batch_size` is primarily used for initializing
                       batched sequences. If `len(dataloader)` is different from
                       `batch_size`, this might lead to unexpected behavior or
                       inefficient use of `initConditions`.
            randomInit: Flag to indicate if random initialization should be used
                        (currently not directly used in this method but passed via args).
        """
        if self.m1x_0 is None or self.m2x_0 is None:
            raise ValueError("Initial sequence (m1x_0, m2x_0) must be set using InitSequence before generating a batch.")

        # Expand initial conditions to batch size
        initConditions_m1x_0 = self.m1x_0.view(1, self.m, 1).expand(batch_size, -1, -1)
        # Assuming m2x_0 might also need similar batch expansion if it's used per batch item
        # For now, m2x_0 is passed as is to Init_batched_sequence.
        self.Init_batched_sequence(initConditions_m1x_0, self.m2x_0)

        self.Input = []
        self.Target = []
        # self.filenames = [] # Removed as it was unused

        # The loop iterates through the dataloader.
        # If batch_size is intended to limit the number of items processed from dataloader,
        # the loop should be adjusted (e.g., using itertools.islice or breaking after batch_size items).
        # Current assumption: dataloader itself yields items that constitute the batch,
        # or we process the entire dataloader content up to `batch_size` if dataloader is larger.
        # However, the code processes the *entire* dataloader.
        for idx, data_loaded in enumerate(tqdm(dataloader, desc="Generating Batch")):
            # if idx >= batch_size: # Uncomment this to strictly limit processing to batch_size
            #     break
            # Data format from dataloader: data[0] is target, data[1] is input
            # This is based on the original assignment:
            # pose_input = data[1]
            # pose_target = data[0]
            if not (isinstance(data_loaded, (list, tuple)) and len(data_loaded) >= 2):
                raise ValueError(f"DataLoader expected to yield tuples/lists of at least 2 elements. Got: {type(data_loaded)}")

            pose_target, pose_input = data_loaded[0], data_loaded[1]

            if not isinstance(pose_input, torch.Tensor) or not isinstance(pose_target, torch.Tensor):
                raise TypeError(f"Expected tensors for pose_input and pose_target. Got: {type(pose_input)}, {type(pose_target)}")

            self.Input.append(pose_input)
            self.Target.append(pose_target)





