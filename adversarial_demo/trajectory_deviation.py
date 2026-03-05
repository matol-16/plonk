

from plonk.pipe import PlonkPipeline
from plonk.pipe import _gps_degrees_to_cartesian
from PIL import Image
import torch
import torch.nn.functional as F
import tqdm as tqdm
import numpy as np

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
    eps_max=1,
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
    optimizer = torch.optim.Adam([delta], lr=lr)

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
        optimizer.step()

        # Project onto l_inf ball (PGD)
        with torch.no_grad():
            delta.clamp_(-eps_max, eps_max)

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

        # Apply postprocessing and return
        output = self.postprocessing(output)
        # To degrees
        output = np.degrees(output.detach().cpu().numpy())
        
        #apply postprocessing to trajectories if they are returned. This must be done for each step
        if traj is not None:
            traj_gps = np.zeros((len(traj), batch_size, 2))
            for i_batch_traj, batch_traj in enumerate(traj):
                batch_traj_gps = self.postprocessing(batch_traj)
                batch_traj_gps = np.degrees(batch_traj_gps.detach().cpu().numpy())
                traj_gps[i_batch_traj] = batch_traj_gps
            traj = traj_gps
        
        if return_trajectories and traj is not None:
            if isinstance(traj, (list, tuple)):
                traj = torch.stack([
                    t.detach().to("cpu") if isinstance(t, torch.Tensor) else torch.as_tensor(t)
                    for t in traj
                ])
            elif isinstance(traj, torch.Tensor):
                traj = traj.detach().to("cpu")
            else:
                traj = torch.as_tensor(traj)
            traj = np.degrees(traj.numpy())
            return output, traj
        else:
            return output