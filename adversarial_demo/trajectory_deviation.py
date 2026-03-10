

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
    reconstruction_loss_weight=0.0,
    num_restarts=1,
    restart_selection_metric="mean_displacement",
    restart_eval_batch_size=256,
    restart_eval_cfg=10.0,
    restart_eval_num_steps=None,
    restart_eval_seed=1234,
    print_restart_results=True,
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

    # Approximate x0 distribution for this source image
    x0_bank = build_x0_bank_from_clean_model(
        pipeline,
        source_image,
        n_samples=anchor_samples,
        num_steps=clean_num_steps,
        cfg=0.0,
    )  # [N, 3]

    metric_aliases = {
        "mean": "mean_displacement",
        "final": "final_displacement",
    }
    if restart_selection_metric in metric_aliases:
        restart_selection_metric = metric_aliases[restart_selection_metric]
    valid_metrics = {"mean_displacement", "final_displacement"}
    if restart_selection_metric not in valid_metrics:
        raise ValueError(
            f"Unknown restart_selection_metric: {restart_selection_metric}. "
            f"Expected one of {sorted(valid_metrics)}"
        )
    if int(num_restarts) < 1:
        raise ValueError("num_restarts must be >= 1")

    # Utility metric for restart selection: mean displacement at the final step.
    def _compute_final_step_displacement(gps_traj_source, gps_traj_perturbed):
        if isinstance(gps_traj_source, torch.Tensor):
            gps_traj_source = gps_traj_source.detach().cpu().numpy()
        if isinstance(gps_traj_perturbed, torch.Tensor):
            gps_traj_perturbed = gps_traj_perturbed.detach().cpu().numpy()

        src = np.asarray(gps_traj_source, dtype=np.float64)
        per = np.asarray(gps_traj_perturbed, dtype=np.float64)
        if src.ndim != 3 or src.shape[-1] != 2:
            raise ValueError("gps_traj_source must have shape [num_steps, batch_size, 2]")
        if per.ndim != 3 or per.shape[-1] != 2:
            raise ValueError("gps_traj_perturbed must have shape [num_steps, batch_size, 2]")
        if src.shape != per.shape:
            raise ValueError("gps_traj_source and gps_traj_perturbed must have the same shape")

        src_last = src[-1]
        per_last = per[-1]
        valid_last = np.isfinite(src_last).all(axis=1) & np.isfinite(per_last).all(axis=1)
        if not np.any(valid_last):
            return 0.0

        dlat = per_last[valid_last, 0] - src_last[valid_last, 0]
        dlon = per_last[valid_last, 1] - src_last[valid_last, 1]
        dist = np.sqrt(dlat ** 2 + dlon ** 2)
        return float(np.mean(dist))

    restart_summaries = []
    best_score = -float("inf")
    best_delta = None
    best_history = None
    best_restart = None

    from adversarial_utils import add_perturbation_to_image

    for restart_idx in range(int(num_restarts)):
        # Universal perturbation parameter (fresh start at each restart)
        delta = torch.zeros_like(source_tensor, requires_grad=True)
        #we use sign sgd
        optimizer = torch.optim.SGD([delta], lr=lr)

        history = []
        pbar = tqdm.trange(n_steps, desc=f"PGD attack training (restart {restart_idx + 1}/{int(num_restarts)})")

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

        # Evaluate restart quality on trajectory displacement with a shared initial noise.
        perturbed_image = add_perturbation_to_image(source_image, delta.detach(), pipeline)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(restart_eval_seed) + restart_idx)
        x_N = torch.randn(int(restart_eval_batch_size), 3, device=device, generator=generator)

        eval_kwargs = {
            "batch_size": int(restart_eval_batch_size),
            "cfg": float(restart_eval_cfg),
            "x_N": x_N,
            "return_trajectories": True,
        }
        if restart_eval_num_steps is not None:
            eval_kwargs["num_steps"] = int(restart_eval_num_steps)

        _, traj_source = pipeline(source_image, **eval_kwargs)
        _, traj_perturbed = pipeline(perturbed_image, **eval_kwargs)

        mean_displacement = compute_mean_perturbation_effect_over_steps(traj_source, traj_perturbed)
        final_displacement = _compute_final_step_displacement(traj_source, traj_perturbed)
        score = mean_displacement if restart_selection_metric == "mean_displacement" else final_displacement

        summary = {
            "restart": restart_idx,
            "mean_displacement": float(mean_displacement),
            "final_displacement": float(final_displacement),
            "score": float(score),
            "final_loss": float(history[-1]) if len(history) > 0 else float("inf"),
            "min_loss": float(np.min(history)) if len(history) > 0 else float("inf"),
        }
        restart_summaries.append(summary)

        if print_restart_results:
            print(
                f"[restart {restart_idx + 1}/{int(num_restarts)}] "
                f"final_loss={summary['final_loss']:.6f}, "
                f"min_loss={summary['min_loss']:.6f}, "
                f"mean_disp={summary['mean_displacement']:.6f}, "
                f"final_disp={summary['final_displacement']:.6f}, "
                f"selection_score={summary['score']:.6f}"
            )

        if score > best_score:
            best_score = float(score)
            best_delta = delta.detach().clone()
            best_history = list(history)
            best_restart = restart_idx

    if best_delta is None or best_history is None or best_restart is None:
        raise RuntimeError("No restart produced a valid perturbation")

    if print_restart_results and int(num_restarts) > 1:
        print(
            f"Selected restart {best_restart + 1}/{int(num_restarts)} using "
            f"{restart_selection_metric} (score={best_score:.6f})"
        )

    return best_delta, best_history, source_tensor

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
    reconstruction_loss_weight_list=None,
):
    """
    Build cartesian-product hyperparameter configurations for diffusion attack training.

    Returns a list of dicts with keys:
      - lr
      - train_batch_size
      - anchor_samples
      - clean_num_steps
      - eps_max (optional; included when eps_max_list is provided)
      - reconstruction_loss_weight (optional; included when reconstruction_loss_weight_list is provided)
    """
    if eps_max_list is None:
        eps_max_list = [None]
    if reconstruction_loss_weight_list is None:
        reconstruction_loss_weight_list = [None]

    grid = []
    for (
        lr,
        train_batch_size,
        anchor_samples,
        clean_num_steps,
        eps_max_value,
        reconstruction_loss_weight_value,
    ) in product(
        lrs,
        train_batch_sizes,
        anchor_samples_list,
        clean_num_steps_list,
        eps_max_list,
        reconstruction_loss_weight_list,
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
        if reconstruction_loss_weight_value is not None:
            config["reconstruction_loss_weight"] = float(reconstruction_loss_weight_value)
        grid.append(config)
    return grid


def _score_history(history, tail_k=25):
    """Lower is better: mean of the last k losses (or all if shorter)."""
    if len(history) == 0:
        return float("inf")
    k = min(int(tail_k), len(history))
    return float(np.mean(history[-k:]))


def compute_mean_perturbation_effect_over_steps(gps_traj_source, gps_traj_perturbed):
    """
    Compute the perturbation-effect metric used in plotting utilities.

    This reproduces the displacement computation in `plot_gps_trajectories_on_map`:
      displacement(step, sample) = sqrt((dlat)^2 + (dlon)^2)
    and returns the mean perturbation effect over all steps.
    """
    if isinstance(gps_traj_source, torch.Tensor):
        gps_traj_source = gps_traj_source.detach().cpu().numpy()
    if isinstance(gps_traj_perturbed, torch.Tensor):
        gps_traj_perturbed = gps_traj_perturbed.detach().cpu().numpy()

    src = np.asarray(gps_traj_source, dtype=np.float64)
    per = np.asarray(gps_traj_perturbed, dtype=np.float64)

    if src.ndim != 3 or src.shape[-1] != 2:
        raise ValueError("gps_traj_source must have shape [num_steps, batch_size, 2]")
    if per.ndim != 3 or per.shape[-1] != 2:
        raise ValueError("gps_traj_perturbed must have shape [num_steps, batch_size, 2]")
    if src.shape != per.shape:
        raise ValueError("gps_traj_source and gps_traj_perturbed must have the same shape")

    valid = np.isfinite(src).all(axis=2) & np.isfinite(per).all(axis=2)
    dlat = per[:, :, 0] - src[:, :, 0]
    dlon = per[:, :, 1] - src[:, :, 1]
    displacement = np.sqrt(dlat ** 2 + dlon ** 2)
    displacement[~valid] = np.nan

    valid_counts = valid.sum(axis=1)
    sum_disp = np.nansum(displacement, axis=1)
    mean_disp = np.divide(
        sum_disp,
        valid_counts,
        out=np.zeros_like(sum_disp, dtype=np.float64),
        where=valid_counts > 0,
    )

    step_has_valid = valid_counts > 0
    if not np.any(step_has_valid):
        return 0.0
    return float(np.mean(mean_disp[step_has_valid]))


def compute_mean_final_prediction_distance(gps_coords_source, gps_coords_perturbed):
    """
    Compute mean distance between final source and perturbed GPS predictions.

    Inputs are arrays shaped [batch_size, 2] with [lat, lon].
    Distance is Euclidean in degree space, consistent with plotting displacement.
    """
    src = np.asarray(gps_coords_source, dtype=np.float64)
    per = np.asarray(gps_coords_perturbed, dtype=np.float64)

    if src.ndim != 2 or src.shape[-1] != 2:
        raise ValueError("gps_coords_source must have shape [batch_size, 2]")
    if per.ndim != 2 or per.shape[-1] != 2:
        raise ValueError("gps_coords_perturbed must have shape [batch_size, 2]")
    if src.shape != per.shape:
        raise ValueError("gps_coords_source and gps_coords_perturbed must have the same shape")

    valid = np.isfinite(src).all(axis=1) & np.isfinite(per).all(axis=1)
    if not np.any(valid):
        return 0.0

    dlat = per[valid, 0] - src[valid, 0]
    dlon = per[valid, 1] - src[valid, 1]
    dist = np.sqrt(dlat ** 2 + dlon ** 2)
    return float(np.mean(dist))


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
    num_restarts=1,
    eval_batch_size=256,
    eval_cfg=10.0,
    eval_num_steps=None,
    eval_seed=1234,
    eval_metric="mean_perturbation_effect_over_steps",
):
    """
    Run a hyperparameter search over `train_diffusion_perturbation`.

    Args:
        source_image: PIL image used for training.
        pipeline: PLONK pipeline.
        hparam_grid: iterable of dicts with keys
            {"lr", "train_batch_size", "anchor_samples", "clean_num_steps"}
            and optionally "eps_max" and "reconstruction_loss_weight".
        n_steps, eps_max, target_pure_noise, reconstruction_loss_weight, device, log_every:
            forwarded to train_diffusion_perturbation.
        score_tail_k: score = mean of the last k training losses (kept for logging).
        keep_deltas: if True, each trial stores trained perturbation tensor.
        show_progress: tqdm over trials.
        num_restarts: number of independent training runs per config.
        eval_batch_size, eval_cfg, eval_num_steps, eval_seed:
            parameters used to evaluate trajectory perturbation effect.
        eval_metric: one of
            - "mean_perturbation_effect_over_steps"
            - "mean_final_prediction_distance"

    Returns:
        dict with keys:
          - best_index
          - best_config
          - best_score
          - trials (list of per-trial dicts)
                each trial has:
                  config, score, metric_name, final_loss, min_loss, history,
                  best_restart, restart_summaries, (optional) delta
    """
    trials = []
    best_index = None
    best_score = -float("inf")

    iterator = enumerate(hparam_grid)
    if show_progress:
        iterator = tqdm.tqdm(iterator, total=len(hparam_grid), desc="Hyperparameter search")

    valid_metrics = {
        "mean_perturbation_effect_over_steps",
        "mean_final_prediction_distance",
    }
    if eval_metric not in valid_metrics:
        raise ValueError(
            f"Unknown eval_metric: {eval_metric}. Expected one of {sorted(valid_metrics)}"
        )

    for trial_index, config in iterator:
        required = {"lr", "train_batch_size", "anchor_samples", "clean_num_steps"}
        missing = required.difference(config.keys())
        if missing:
            raise ValueError(f"Missing keys in config {trial_index}: {sorted(missing)}")

        trial_eps_max = float(config["eps_max"]) if "eps_max" in config else float(eps_max)
        trial_reconstruction_loss_weight = (
            float(config["reconstruction_loss_weight"])
            if "reconstruction_loss_weight" in config
            else float(reconstruction_loss_weight)
        )

        restart_summaries = []
        best_restart_effect = -float("inf")
        best_restart_idx = None
        best_restart_payload = None

        for restart_idx in range(int(num_restarts)):
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
                reconstruction_loss_weight=trial_reconstruction_loss_weight,
                device=device,
            )

            # Build a perturbed image and evaluate perturbation effect on trajectories.
            from adversarial_utils import add_perturbation_to_image

            perturbed_image = add_perturbation_to_image(source_image, delta, pipeline)
            gen = torch.Generator(device=device)
            gen.manual_seed(int(eval_seed) + trial_index * 1000 + restart_idx)
            x_N = torch.randn(eval_batch_size, 3, device=device, generator=gen)

            if eval_num_steps is None:
                gps_source_eval, traj_source = pipeline(
                    source_image,
                    batch_size=eval_batch_size,
                    cfg=eval_cfg,
                    x_N=x_N.clone(),
                    return_trajectories=True,
                )
                gps_perturbed_eval, traj_perturbed = pipeline(
                    perturbed_image,
                    batch_size=eval_batch_size,
                    cfg=eval_cfg,
                    x_N=x_N.clone(),
                    return_trajectories=True,
                )
            else:
                gps_source_eval, traj_source = pipeline(
                    source_image,
                    batch_size=eval_batch_size,
                    cfg=eval_cfg,
                    x_N=x_N.clone(),
                    num_steps=eval_num_steps,
                    return_trajectories=True,
                )
                gps_perturbed_eval, traj_perturbed = pipeline(
                    perturbed_image,
                    batch_size=eval_batch_size,
                    cfg=eval_cfg,
                    x_N=x_N.clone(),
                    num_steps=eval_num_steps,
                    return_trajectories=True,
                )

            if eval_metric == "mean_perturbation_effect_over_steps":
                effect_score = compute_mean_perturbation_effect_over_steps(traj_source, traj_perturbed)
            else:
                effect_score = compute_mean_final_prediction_distance(gps_source_eval, gps_perturbed_eval)
            history_score = _score_history(history, tail_k=score_tail_k)
            restart_summary = {
                "restart": restart_idx,
                "effect_score": effect_score,
                "history_score": history_score,
                "final_loss": float(history[-1]) if len(history) > 0 else float("inf"),
                "min_loss": float(np.min(history)) if len(history) > 0 else float("inf"),
            }
            restart_summaries.append(restart_summary)

            if effect_score > best_restart_effect:
                best_restart_effect = effect_score
                best_restart_idx = restart_idx
                best_restart_payload = {
                    "delta": delta,
                    "history": history,
                    "history_score": history_score,
                }

        if best_restart_payload is None:
            raise RuntimeError("No restart completed successfully for this hyperparameter config")
        if best_restart_idx is None:
            raise RuntimeError("Unable to select best restart for this hyperparameter config")

        score = float(best_restart_effect)
        history = best_restart_payload["history"]
        trial = {
            "config": {
                "lr": float(config["lr"]),
                "train_batch_size": int(config["train_batch_size"]),
                "anchor_samples": int(config["anchor_samples"]),
                "clean_num_steps": int(config["clean_num_steps"]),
            },
            "score": score,
            "metric_name": eval_metric,
            "best_restart": best_restart_idx,
            "restart_summaries": restart_summaries,
            "history_score": float(best_restart_payload["history_score"]),
            "final_loss": float(history[-1]) if len(history) > 0 else float("inf"),
            "min_loss": float(np.min(history)) if len(history) > 0 else float("inf"),
            "history": history,
        }
        if "eps_max" in config:
            trial["config"]["eps_max"] = trial_eps_max
        if "reconstruction_loss_weight" in config:
            trial["config"]["reconstruction_loss_weight"] = trial_reconstruction_loss_weight
        if keep_deltas:
            trial["delta"] = best_restart_payload["delta"]

        trials.append(trial)

        if score > best_score:
            best_score = score
            best_index = trial_index

    best_config = trials[best_index]["config"] if best_index is not None else None
    return {
        "best_index": best_index,
        "best_config": best_config,
        "best_score": best_score,
        "trials": trials,
    }