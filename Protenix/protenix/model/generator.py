# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional

import torch

from protenix.model.utils import centre_random_augmentation


class TrainingNoiseSampler:
    """
    Sample the noise-level of of training samples
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Sampler for training noise-level

        Args:
            p_mean (float, optional): gaussian mean. Defaults to -1.2.
            p_std (float, optional): gaussian std. Defaults to 1.5.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        print(f"train scheduler {self.sigma_data}")

    def __call__(
        self, size: torch.Size, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sampling

        Args:
            size (torch.Size): the target size
            device (torch.device, optional): target device. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: sampled noise-level
        """
        rnd_normal = torch.randn(size=size, device=device)
        noise_level = (rnd_normal * self.p_std + self.p_mean).exp() * self.sigma_data
        return noise_level


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps)
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Scheduler parameters

        Args:
            s_max (float, optional): maximal noise level. Defaults to 160.0.
            s_min (float, optional): minimal noise level. Defaults to 4e-4.
            rho (float, optional): the exponent numerical part. Defaults to 7.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho
        print(f"inference scheduler {self.sigma_data}")

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    diffusion_steps: int = 0,
    input_atom_array_path: str = "",
    motif_fixed_positions: Optional[torch.Tensor] = None,
    motif_fixed_mask: Optional[torch.Tensor] = None,
    motif_projection_weight: Optional[float] = None,
    motif_noisy_projection_weight: Optional[float] = None,
    motif_denoising_projection_weight: Optional[float] = None,
    motif_x0_projection_weight: Optional[float] = None,
    motif_smoothing_weights: Optional[torch.Tensor] = None,
    motif_smoothing_source_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype
    use_motif_projection = motif_fixed_positions is not None and motif_fixed_mask is not None
    if motif_noisy_projection_weight is None:
        motif_noisy_projection_weight = (
            1.0 if motif_projection_weight is None else motif_projection_weight
        )
    if motif_denoising_projection_weight is None:
        motif_denoising_projection_weight = (
            1.0
            if motif_projection_weight is None
            else motif_projection_weight
        )
    # x0 guidance is independent from the legacy/post-step projection fields.
    # A missing value keeps the original sampler behavior unchanged.
    if motif_x0_projection_weight is None:
        motif_x0_projection_weight = 0.0
    for projection_name, projection_weight in (
        ("motif_noisy_projection_weight", motif_noisy_projection_weight),
        ("motif_denoising_projection_weight", motif_denoising_projection_weight),
        ("motif_x0_projection_weight", motif_x0_projection_weight),
    ):
        if not 0.0 <= float(projection_weight) <= 1.0:
            raise ValueError(f"{projection_name} must be between 0 and 1.")
    if use_motif_projection:
        motif_fixed_positions = motif_fixed_positions.to(device=device, dtype=dtype)
        motif_fixed_mask = motif_fixed_mask.to(device=device, dtype=torch.bool)
        if motif_fixed_mask.shape != input_feature_dict["atom_to_token_idx"].shape:
            raise ValueError(
                "Motif projection mask does not match Protenix atom layout: "
                f"mask={tuple(motif_fixed_mask.shape)}, "
                f"expected={tuple(input_feature_dict['atom_to_token_idx'].shape)}"
            )
        if motif_fixed_positions.shape != motif_fixed_mask.shape + (3,):
            raise ValueError(
                "Motif projection positions do not match Protenix atom layout: "
                f"positions={tuple(motif_fixed_positions.shape)}, "
                f"expected={tuple(motif_fixed_mask.shape) + (3,)}"
            )
        if (motif_smoothing_weights is None) != (
            motif_smoothing_source_indices is None
        ):
            raise ValueError(
                "Motif smoothing weights and source indices must be provided together."
            )
        use_motif_smoothing = motif_smoothing_weights is not None
        if use_motif_smoothing:
            if (motif_smoothing_weights.shape != motif_fixed_mask.shape or
                    motif_smoothing_source_indices.shape != motif_fixed_mask.shape):
                raise ValueError(
                    "Motif smoothing arrays do not match Protenix atom layout: "
                    f"weights={tuple(motif_smoothing_weights.shape)}, "
                    f"source_indices={tuple(motif_smoothing_source_indices.shape)}, "
                    f"expected={tuple(motif_fixed_mask.shape)}"
                )
            motif_smoothing_weights = motif_smoothing_weights.to(
                device=device, dtype=dtype
            )
            motif_smoothing_source_indices = motif_smoothing_source_indices.to(
                device=device, dtype=torch.long
            )

    def apply_motif_projection(
        positions: torch.Tensor,
        reference_positions: Optional[torch.Tensor],
        projection_weight: float,
    ) -> torch.Tensor:
        if not use_motif_projection:
            return positions
        if reference_positions is None:
            reference_positions = motif_fixed_positions
        delta = reference_positions - positions
        if use_motif_smoothing:
            source_delta = torch.index_select(
                delta, dim=-2, index=motif_smoothing_source_indices
            )
            neighbor_weights = torch.where(
                motif_fixed_mask, torch.zeros_like(motif_smoothing_weights),
                motif_smoothing_weights,
            )
            soft_positions = positions + float(projection_weight) * (
                delta * motif_fixed_mask[..., None]
                + source_delta * neighbor_weights[..., None]
            )
            hard_positions = torch.where(
                motif_fixed_mask[..., None], reference_positions, positions
            ) + source_delta * neighbor_weights[..., None]
            if float(projection_weight) >= 1.0:
                return hard_positions
            return soft_positions
        soft_positions = positions + float(projection_weight) * delta * motif_fixed_mask[..., None]
        if float(projection_weight) >= 1.0:
            return torch.where(motif_fixed_mask[..., None], reference_positions, positions)
        return soft_positions

    def rigid_align_reference(
        reference_positions: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Fit reference motif coordinates to their current diffusion pose."""
        weights = motif_fixed_mask.to(dtype=target_positions.dtype)
        weights = torch.broadcast_to(weights, target_positions.shape[:-1])
        reference_positions = torch.broadcast_to(
            reference_positions, target_positions.shape
        )
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        reference_center = (
            reference_positions * weights[..., None]
        ).sum(dim=-2, keepdim=True) / weight_sum[..., None]
        target_center = (
            target_positions * weights[..., None]
        ).sum(dim=-2, keepdim=True) / weight_sum[..., None]
        reference_centered = reference_positions - reference_center
        target_centered = target_positions - target_center
        covariance = torch.einsum(
            "...ni,...nj->...ij",
            reference_centered * weights[..., None],
            target_centered,
        )
        left, _, right_transposed = torch.linalg.svd(
            covariance.float(), full_matrices=False
        )
        left = left.to(dtype=target_positions.dtype)
        right_transposed = right_transposed.to(dtype=target_positions.dtype)
        handedness = torch.where(
            torch.linalg.det(left @ right_transposed) < 0,
            target_positions.new_tensor(-1.0),
            target_positions.new_tensor(1.0),
        )
        correction = torch.diag_embed(
            torch.stack(
                [torch.ones_like(handedness), torch.ones_like(handedness), handedness],
                dim=-1,
            )
        )
        rotation = left @ correction @ right_transposed
        rigid_aligned = reference_centered @ rotation + target_center
        translated = reference_positions - reference_center + target_center
        return torch.where(
            (weight_sum >= 3.0)[..., None], rigid_aligned, translated
        )

    total_schedule_steps = len(noise_schedule) - 1
    num_denoise_iterations = diffusion_steps if diffusion_steps > 0 else len(noise_schedule) - 1
    if diffusion_steps < 0:
        raise ValueError(f"diffusion_steps must be non-negative, got {diffusion_steps}.")
    if num_denoise_iterations > total_schedule_steps:
        raise ValueError(
            "diffusion_steps cannot exceed the noise schedule length: "
            f"got {num_denoise_iterations}, max {total_schedule_steps}."
        )
    print(f"Actual number of denoising iterations: {num_denoise_iterations}")

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        # init noise
        # [..., N_sample, N_atom, 3]
        start_idx = total_schedule_steps - num_denoise_iterations
        x_l = noise_schedule[start_idx] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )  # NOTE: set seed in distributed training
        #print(f"Initialized x_l with noise. Shape: {x_l.shape}")
        end_idx = total_schedule_steps # For c_tau_last
        if input_atom_array_path:
            origin_positions = torch.load(input_atom_array_path, map_location=device)
            origin_positions = torch.as_tensor(
                origin_positions, device=device, dtype=dtype
            )
            if origin_positions.ndim != 2 or origin_positions.shape[-1] != 3:
                raise ValueError(
                    "Reference coordinates must have shape [N_atom, 3], got "
                    f"{tuple(origin_positions.shape)}."
                )
            if origin_positions.shape[0] != N_atom:
                raise ValueError(
                    "Reference coordinate atom count does not match the Protenix "
                    f"input: reference={origin_positions.shape[0]}, input={N_atom}."
                )
            # Keep the full current structure in the same centered frame as
            # the aligned motif reference. Never center around the motif: that
            # would move the scaffold and can amplify chain breaks.
            origin_positions = origin_positions - origin_positions.mean(
                dim=-2, keepdim=True
            )
            origin_positions = origin_positions.unsqueeze(0).repeat(chunk_n_sample, 1, 1)
            
            x_l = origin_positions
            noise = torch.randn_like(x_l)
            x_l = x_l + 1 * noise_schedule[start_idx].item() * noise
            print(f"Initialized x_l from {input_atom_array_path}. Shape: {x_l.shape}")
        else:
            print(f"Initialized x_l with noise. Shape: {x_l.shape}")
        #print(f"Starting {num_denoise_iterations} denoising steps.")
        
        
        
        #print(f"Starting denoising from noise_schedule index {start_idx} to {end_idx - 1} (inclusive).")
        
        # Iterate through the selected noise schedule steps
        for k, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[start_idx : end_idx], noise_schedule[start_idx + 1 : end_idx + 1])
        ):
            #print(f"Denoising step {k+1}/{num_denoise_iterations} (c_tau_last={c_tau_last:.4f}, c_tau={c_tau:.4f})")
            
            # [..., N_sample, N_atom, 3]
            x_l = (
                centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )
            motif_reference = (
                rigid_align_reference(motif_fixed_positions, x_l)
                if use_motif_projection else None
            )
            # Denoise with a predictor-corrector sampler
            # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            noise = noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )
            noisy_reference = (
                motif_reference + noise if motif_reference is not None else None
            )
            x_noisy = apply_motif_projection(
                x_l + noise,
                noisy_reference,
                motif_noisy_projection_weight,
            )

            # 2. Denoise from x_{t_hat} to x_{c_tau}
            # Euler step only
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )

            x_denoised = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
            )

            # The denoiser predicts a complete x0.  Fit the native motif to
            # that prediction's motif pose and apply only a local residual
            # correction before computing the diffusion update direction.
            x0_aligned_reference = motif_reference
            if use_motif_projection:
                x0_aligned_reference = rigid_align_reference(
                    motif_fixed_positions, x_denoised
                )
            x0_conditioned = apply_motif_projection(
                x_denoised,
                x0_aligned_reference,
                motif_x0_projection_weight,
            )

            delta = (x_noisy - x0_conditioned) / t_hat[
                ..., None, None
            ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
            dt = c_tau - t_hat
            x_l = apply_motif_projection(
                x_noisy + step_scale_eta * dt[..., None, None] * delta,
                motif_reference,
                motif_denoising_projection_weight,
            )

        return x_l

    if diffusion_chunk_size is None:
        x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        x_l = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            chunk_x_l = _chunk_sample_diffusion(
                chunk_n_sample, inplace_safe=inplace_safe
            )
            x_l.append(chunk_x_l)
        x_l = torch.cat(x_l, -3)  # [..., N_sample, N_atom, 3]
    #torch.save(x_l, "/storage/caolongxingLab/fangminchao/software/Protenix/coord.pkl")
    
    #if input_atom_array_path:
    #    origin_positions = torch.load(input_atom_array_path, map_location=device)
    #    mean_position = origin_positions.mean(dim=0)
    #    recenter_positions = origin_positions - mean_position
    #    recenter_positions = recenter_positions.unsqueeze(0).repeat(chunk_n_sample, 1, 1)
    #    
    #    if recenter_positions.shape != x_l.shape:
    #        plus = recenter_positions.shape[1] - x_l.shape[1] 
    #        last_step = recenter_positions[:, plus:, :]  # shape: (1, 1, 3)
    #        x_l = torch.cat([recenter_positions, last_step], dim=1)  # shape: (1, L+1, 3)
    #        mean_position = x_l.mean(dim=0)
    #        recenter_positions = x_l - mean_position
    #    x_l = recenter_positions
    return x_l


def sample_diffusion_training(
    noise_sampler: TrainingNoiseSampler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
) -> tuple[torch.Tensor, ...]:
    """Implements diffusion training as described in AF3 Appendix at page 23.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        label_dict (dict, optional) : a dictionary containing the followings.
            "coordinate": the ground-truth coordinates
                [..., N_atom, 3]
            "coordinate_mask": whether true coordinates exist.
                [..., N_atom]
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        N_sample (int): number of training samples
    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    batch_size_shape = label_dict["coordinate"].shape[:-2]
    device = label_dict["coordinate"].device
    dtype = label_dict["coordinate"].dtype
    # Areate N_sample versions of the input structure by randomly rotating and translating
    x_gt_augment = centre_random_augmentation(
        x_input_coords=label_dict["coordinate"],
        N_sample=N_sample,
        mask=label_dict["coordinate_mask"],
    ).to(
        dtype
    )  # [..., N_sample, N_atom, 3]

    # Add independent noise to each structure
    # sigma: independent noise-level [..., N_sample]
    sigma = noise_sampler(size=(*batch_size_shape, N_sample), device=device).to(dtype)
    # noise: [..., N_sample, N_atom, 3]
    noise = torch.randn_like(x_gt_augment, dtype=dtype) * sigma[..., None, None]

    # Get denoising outputs [..., N_sample, N_atom, 3]
    if diffusion_chunk_size is None:
        x_denoised = denoise_net(
            x_noisy=x_gt_augment + noise,
            t_hat_noise_level=sigma,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
        )
    else:
        x_denoised = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            x_noisy_i = (x_gt_augment + noise)[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            t_hat_noise_level_i = sigma[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
            ]
            x_denoised_i = denoise_net(
                x_noisy=x_noisy_i,
                t_hat_noise_level=t_hat_noise_level_i,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
            )
            x_denoised.append(x_denoised_i)
        x_denoised = torch.cat(x_denoised, dim=-3)

    return x_gt_augment, x_denoised, sigma
