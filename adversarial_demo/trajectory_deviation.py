

from plonk.pipe import PlonkPipeline
from plonk.pipe import _gps_degrees_to_cartesian
from PIL import Image
import torch
import torch.nn.functional as F
import tqdm as tqdm
import numpy as np
from itertools import product

############################################################################################
# Attempt 1 training pipeline: universal perturbation on a single source image
# Objective: min E_{x0, eps, t} <psi(x_t | c+delta), eps>^2
# where x_t = sqrt(gamma_t) * x0 + sqrt(1-gamma_t) * eps

def build_x0_bank_from_clean_model(
    pipeline,
    source_image,
    n_samples=256,
    num_steps=200,
    cfg=0.0,
    device="cuda",
):
    """
    Build a bank of plausible x0 states by sampling the clean model on the source image.
    This approximates expectation over x0 in the objective.
    """
    with torch.no_grad():
        gps_samples = pipeline(
            source_image,
            batch_size=n_samples,
            num_steps=num_steps,
            cfg=cfg,
        )
    if isinstance(gps_samples, tuple):
        gps_samples = gps_samples[0]
    return _gps_degrees_to_cartesian(gps_samples, device=device)


def train_diffusion_perturbation(
    source_image,
    pipeline,
    n_steps=400,
    train_batch_size=64,
    lr=2e-2,
    eps_max=1.0,
    anchor_samples=256,
    clean_num_steps=200,
    log_every=20,
    target_pure_noise = False,
    reconstruction_loss_weight=0,
    device="cuda",
):
    # Freeze PLONK denoiser and embedding model parameters (we only optimize delta)
    pipeline.network.eval().requires_grad_(False)
    pipeline.cond_preprocessing.emb_model.eval().requires_grad_(False)

    # Preprocess once; perturbation is optimized in normalized-image space
    source_tensor = (
        pipeline.cond_preprocessing.augmentation(source_image)
        .unsqueeze(0)
        .to(device)
    )  # [1, 3, H, W]

    # Universal perturbation parameter
    delta = torch.zeros_like(source_tensor, requires_grad=True)
    # optimizer = torch.optim.Adam([delta], lr=lr)
    #we use sign sgd
    optimizer = torch.optim.SGD([delta], lr=lr)

    # Approximate x0 distribution for this source image
    x0_bank = build_x0_bank_from_clean_model(
        pipeline,
        source_image,
        n_samples=anchor_samples,
        num_steps=clean_num_steps,
        cfg=0.0,
    )  # [N, 3]

    history = []
    pbar = tqdm.trange(n_steps, desc="PGD attack training")

    for step in pbar:
        optimizer.zero_grad(set_to_none=True)

        # Sample x0 from bank, noise eps, and continuous t ~ U(0,1)
        idx = torch.randint(0, x0_bank.shape[0], (train_batch_size,), device=device)
        x0 = x0_bank[idx]
        eps = torch.randn_like(x0)

        t = torch.rand(train_batch_size, device=device)
        gamma = pipeline.scheduler(t)  # [B]

        x_t = (
            torch.sqrt(gamma).unsqueeze(-1) * x0
            + torch.sqrt(1.0 - gamma).unsqueeze(-1) * eps
        )

        # Compute conditional embedding of perturbed source image
        perturbed_source = source_tensor + delta
        emb_single = pipeline.cond_preprocessing.emb_model(perturbed_source).squeeze(0)
        emb = emb_single.unsqueeze(0).repeat(train_batch_size, 1)
  
        # Denoiser epsilon prediction, using PLONK expected batch keys
        model_batch_perturbed = {
            "y": x_t,
            "emb": emb,
            "gamma": gamma,
        }
        eps_pred_perturbed = pipeline.model(model_batch_perturbed)

        if not target_pure_noise:
            #compute conditional embedding of unperturbed source image
            emb_source = pipeline.cond_preprocessing.emb_model(source_tensor).squeeze(0)
            emb_source = emb_source.unsqueeze(0).repeat(train_batch_size, 1)

            model_batch = {
                "y": x_t,
                "emb": emb_source,
                "gamma": gamma,
            }
            eps_pred = pipeline.model(model_batch)
            eps= eps_pred


        # Orthogonality objective: minimize squared cosine similarity
        cos = F.cosine_similarity(eps, eps_pred_perturbed, dim=-1)
        loss = (cos**2).mean()
        
        if reconstruction_loss_weight > 0:
            # Add image reconstruction loss to ensure perturbation does not degrade image quality too much
            loss_x = torch.nn.functional.l1_loss(perturbed_source, source_tensor)
            loss= loss + reconstruction_loss_weight*loss_x


        loss.backward()
        
        #perform sign sgd in the l inf ball of radius eps_max:
        with torch.no_grad():
            delta.grad = torch.sign(delta.grad)
            optimizer.step()
            delta.data = torch.clamp(delta.data, -eps_max, eps_max)
            delta.grad.zero_()
        

        history.append(loss.item())
        if (step + 1) % log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    return delta.detach(), history, source_tensor

###########################################################################################################

# Then, we can evaluate the trained perturbation by adding it to the source image, and running the pipeline forward with the perturbed image as conditioning input.
# We especially want to visualize the diffusion trajectory. This is possible via the "return_trajectories" argument of the ddim_sampler. 
#This requires changing the _call method of the PlonkPipeline to return trajectories when this argument is set to True.

class PlonkPipelineTrajectory(PlonkPipeline):
    def __call__(
        self,
        images,
        batch_size=None,
        x_N=None,
        num_steps=None,
        scheduler=None,
        cfg=0,
        generator=None,
        return_trajectories=False
    ):
        """
        Extends the __call__ method of the PlonkPipeline by allowing to track trajectories.
        The rest of the code is identical
        """
        # Set up batch size and initial noise
        shape = [3]
        if not isinstance(images, list):
            images = [images]
        if x_N is None:
            if batch_size is None:
                if isinstance(images, list):
                    batch_size = len(images)
                else:
                    batch_size = 1
            x_N = torch.randn(
                batch_size, *shape, device=self.device, generator=generator
            )
        else:
            x_N = x_N.to(self.device)
            if x_N.ndim == 3:
                x_N = x_N.unsqueeze(0)
            batch_size = x_N.shape[0]

        # Set up batch with conditioning
        batch = {"y": x_N}
        batch["img"] = images
        batch = self.cond_preprocessing(batch)
        if len(images) > 1:
            assert len(images) == batch_size
        else:
            batch["emb"] = batch["emb"].repeat(batch_size, 1)

        # Use default sampler/scheduler if not provided
        sampler = self.sampler
        if scheduler is None:
            scheduler = self.scheduler
        # Sample from model
        traj = None
        if num_steps is None:
            if return_trajectories:
                output, traj = sampler(
                    self.model,
                    batch,
                    conditioning_keys="emb",
                    scheduler=scheduler,
                    cfg_rate=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )
            else:
                output = sampler(
                    self.model,
                    batch,
                    conditioning_keys="emb",
                    scheduler=scheduler,
                    cfg_rate=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )
        else:
            if return_trajectories:
                output, traj = sampler(
                    self.model,
                    batch,
                    conditioning_keys="emb",
                    scheduler=scheduler,
                    num_steps=num_steps,
                    cfg_rate=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )
            else:
                output = sampler(
                    self.model,
                    batch,
                    conditioning_keys="emb",
                    scheduler=scheduler,
                    num_steps=num_steps,
                    cfg_rate=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )

        # Apply postprocessing to final samples
        output = self.postprocessing(output)
        output = np.degrees(output.detach().cpu().numpy())
        
        # Apply postprocessing to each trajectory step if requested.
        # Keep the same conversion path as `output`: cartesian -> radians -> degrees.
        if traj is not None:
            if isinstance(traj, torch.Tensor):
                traj_steps = [traj[i] for i in range(traj.shape[0])]
            else:
                traj_steps = list(traj)

            traj_gps = []
            for batch_traj in traj_steps:
                if not isinstance(batch_traj, torch.Tensor):
                    batch_traj = torch.as_tensor(batch_traj, device=self.device)
                batch_traj_gps = self.postprocessing(batch_traj)
                batch_traj_gps = np.degrees(batch_traj_gps.detach().cpu().numpy())
                traj_gps.append(batch_traj_gps)
            traj = np.stack(traj_gps, axis=0)
        
        if return_trajectories and traj is not None:
            return output, traj
        else:
            return output


def build_diffusion_hparam_grid(
    lrs,
    train_batch_sizes,
    anchor_samples_list,
    clean_num_steps_list,
    eps_max_list=None,
):
    """
    Build cartesian-product hyperparameter configurations for diffusion attack training.

    Returns a list of dicts with keys:
      - lr
      - train_batch_size
      - anchor_samples
      - clean_num_steps
      - eps_max (optional; included when eps_max_list is provided)
    """
    if eps_max_list is None:
        eps_max_list = [None]

    grid = []
    for lr, train_batch_size, anchor_samples, clean_num_steps, eps_max_value in product(
        lrs,
        train_batch_sizes,
        anchor_samples_list,
        clean_num_steps_list,
        eps_max_list,
    ):
        if lr is None or train_batch_size is None or anchor_samples is None or clean_num_steps is None:
            raise ValueError("lrs, train_batch_sizes, anchor_samples_list and clean_num_steps_list cannot contain None")
        config = {
            "lr": float(lr),
            "train_batch_size": int(train_batch_size),
            "anchor_samples": int(anchor_samples),
            "clean_num_steps": int(clean_num_steps),
        }
        if eps_max_value is not None:
            config["eps_max"] = float(eps_max_value)
        grid.append(config)
    return grid


def _score_history(history, tail_k=25):
    """Lower is better: mean of the last k losses (or all if shorter)."""
    if len(history) == 0:
        return float("inf")
    k = min(int(tail_k), len(history))
    return float(np.mean(history[-k:]))


def run_diffusion_hparam_search(
    source_image,
    pipeline,
    hparam_grid,
    n_steps=400,
    eps_max=1,
    target_pure_noise=False,
    reconstruction_loss_weight=0,
    device="cuda",
    log_every=20,
    score_tail_k=25,
    keep_deltas=False,
    show_progress=True,
):
    """
    Run a hyperparameter search over `train_diffusion_perturbation`.

    Args:
        source_image: PIL image used for training.
        pipeline: PLONK pipeline.
        hparam_grid: iterable of dicts with keys
            {"lr", "train_batch_size", "anchor_samples", "clean_num_steps"}
            and optionally "eps_max".
        n_steps, eps_max, target_pure_noise, reconstruction_loss_weight, device, log_every:
            forwarded to train_diffusion_perturbation.
        score_tail_k: score = mean of the last k training losses.
        keep_deltas: if True, each trial stores trained perturbation tensor.
        show_progress: tqdm over trials.

    Returns:
        dict with keys:
          - best_index
          - best_config
          - best_score
          - trials (list of per-trial dicts)
            each trial has: config, score, final_loss, min_loss, history, (optional) delta
    """
    trials = []
    best_index = None
    best_score = float("inf")

    iterator = enumerate(hparam_grid)
    if show_progress:
        iterator = tqdm.tqdm(iterator, total=len(hparam_grid), desc="Hyperparameter search")

    for trial_index, config in iterator:
        required = {"lr", "train_batch_size", "anchor_samples", "clean_num_steps"}
        missing = required.difference(config.keys())
        if missing:
            raise ValueError(f"Missing keys in config {trial_index}: {sorted(missing)}")

        trial_eps_max = float(config["eps_max"]) if "eps_max" in config else float(eps_max)

        delta, history, _ = train_diffusion_perturbation(
            source_image=source_image,
            pipeline=pipeline,
            n_steps=n_steps,
            train_batch_size=int(config["train_batch_size"]),
            lr=float(config["lr"]),
            eps_max=trial_eps_max,
            anchor_samples=int(config["anchor_samples"]),
            clean_num_steps=int(config["clean_num_steps"]),
            log_every=log_every,
            target_pure_noise=target_pure_noise,
            reconstruction_loss_weight=reconstruction_loss_weight,
            device=device,
        )

        score = _score_history(history, tail_k=score_tail_k)
        trial = {
            "config": {
                "lr": float(config["lr"]),
                "train_batch_size": int(config["train_batch_size"]),
                "anchor_samples": int(config["anchor_samples"]),
                "clean_num_steps": int(config["clean_num_steps"]),
            },
            "score": score,
            "final_loss": float(history[-1]) if len(history) > 0 else float("inf"),
            "min_loss": float(np.min(history)) if len(history) > 0 else float("inf"),
            "history": history,
        }
        if "eps_max" in config:
            trial["config"]["eps_max"] = trial_eps_max
        if keep_deltas:
            trial["delta"] = delta

        trials.append(trial)

        if score < best_score:
            best_score = score
            best_index = trial_index

    best_config = trials[best_index]["config"] if best_index is not None else None
    return {
        "best_index": best_index,
        "best_config": best_config,
        "best_score": best_score,
        "trials": trials,
    }