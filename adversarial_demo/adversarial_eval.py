from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from plonk.metrics.utils import haversine

from attacks import run_attack


#################Auxiliary functions for computing displacement metrics between trajectories.

def trajectory_displacement(gps_traj_source, gps_traj_perturbed, metric = "haversine") -> torch.Tensor:
    """
    Computes distances per step and batch element.
    """
    # Pipeline outputs degrees; haversine expects radians
    gps_traj_source = torch.deg2rad(gps_traj_source.float())
    gps_traj_perturbed = torch.deg2rad(gps_traj_perturbed.float())
    distance = torch.zeros(gps_traj_source.shape[0], gps_traj_source.shape[1])
    for step in range(gps_traj_source.shape[0]):
        distance[step] = haversine(gps_traj_source[step], gps_traj_perturbed[step])
        if metric == "geoscore":
            distance[step] = 5000 * torch.exp(-distance[step] / 1492.7)
    return distance

def mean_trajectory_displacement(gps_traj_source, gps_traj_perturbed, metric = "haversine") -> float:
	"""Mean displacement over all diffusion steps and batch elements."""
	distance = trajectory_displacement(gps_traj_source, gps_traj_perturbed, metric=metric)	
	return float(distance.mean().mean().item())


def mean_final_prediction_distance(gps_coords_source, gps_coords_perturbed, metric="haversine") -> float:
	"""Mean Euclidean distance in degree space between final predictions."""
	distance = trajectory_displacement(gps_coords_source, gps_coords_perturbed, metric=metric)
	return float(distance[-1].mean().item())

def select_displacement_score(
	mean_step_disp: float,
	final_step_disp: float,
	metric_name: str,
) -> float:
	"""Select scalar score from explicit displacement metric choice."""
	if metric_name == "mean_step_displacement":
		return float(mean_step_disp)
	if metric_name == "final_step_displacement":
		return float(final_step_disp)
	raise ValueError(
		f"Unknown displacement metric: {metric_name}. "
		"Expected one of ['mean_step_displacement', 'final_step_displacement']"
	)


def evaluate_displacement_metrics(gps_traj_source, gps_traj_perturbed) -> Dict[str, float]:
	"""Return both canonical displacement metrics for a pair of trajectories"""
	#convert to tensor if numpy (metrics requires tensors)
	if isinstance(gps_traj_source, np.ndarray):
		gps_traj_source = torch.from_numpy(gps_traj_source)
	if isinstance(gps_traj_perturbed, np.ndarray):
		gps_traj_perturbed = torch.from_numpy(gps_traj_perturbed)
	
	mean_step = mean_trajectory_displacement(gps_traj_source, gps_traj_perturbed)
	final_step = mean_final_prediction_distance(gps_traj_source, gps_traj_perturbed)
	return {
		"mean_step_displacement": float(mean_step),
		"final_step_displacement": float(final_step),
	}

def evaluate_source_perturbation(
	pipeline,
	source_image,
	perturbed_image,
	batch_size: int = 256,
	cfg: float = 10.0,
	num_steps: Optional[int] = None,
	seed: Optional[int] = 1234,
	device: str = "cuda",
) -> Dict[str, Any]:
	"""Run source and perturbed images from the same initial noise and compute metrics."""
	generator = torch.Generator(device=device)
	if seed is not None:
		generator.manual_seed(int(seed))
	x_n = torch.randn(int(batch_size), 3, device=device, generator=generator)

	kwargs = {
		"batch_size": int(batch_size),
		"cfg": float(cfg),
		"x_N": x_n,
		"return_trajectories": True,
	}
	if num_steps is not None:
		kwargs["num_steps"] = int(num_steps)

	gps_source, traj_source = pipeline(source_image, **kwargs)
	gps_perturbed, traj_perturbed = pipeline(perturbed_image, **kwargs)

	return {
		"gps_source": gps_source,
		"gps_perturbed": gps_perturbed,
		"traj_source": traj_source,
		"traj_perturbed": traj_perturbed,
		"mean_trajectory_displacement": mean_trajectory_displacement(traj_source, traj_perturbed),
		"mean_final_prediction_distance": mean_final_prediction_distance(gps_source, gps_perturbed),
	}


####################### Methods to evaluate a an attack across a dataset of source and perturbed images.

#We evaluate first on OSV-5M's test set. We may also evaluate on YFCC4k

from huggingface_hub import hf_hub_download
import os
import csv
import random
import zipfile
from PIL import Image


def load_osv5m_test(local_dir="datasets/osv5m"):
    #only download if the data is not already present
    if os.path.exists(os.path.join(local_dir, "images", "test")) and \
       any(os.path.isdir(os.path.join(local_dir, "images", "test", d)) for d in os.listdir(os.path.join(local_dir, "images", "test"))):
        return
    for i in range(5):
        hf_hub_download(repo_id="osv5m/osv5m", filename=str(i).zfill(2)+'.zip', subfolder="images/test", repo_type='dataset', local_dir=local_dir)
    hf_hub_download(repo_id="osv5m/osv5m", filename="README.md", repo_type='dataset', local_dir=local_dir)
    hf_hub_download(repo_id="osv5m/osv5m", filename="test.csv", repo_type='dataset', local_dir=local_dir)
    # extract zip files
    img_dir = os.path.join(local_dir, "images", "test")
    for f in os.listdir(img_dir):
        if f.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(img_dir, f), 'r') as z:
                z.extractall(img_dir)
    return

def retrieve_osv_images(n_images_to_eval: int = 100, use_real_gps: bool = False):
    local_dir = "datasets/osv5m"
	download_osv5m_test(local_dir=local_dir)  # download & extract if needed

	# Load test metadata from CSV
	csv_path = os.path.join(local_dir, "test.csv")
	with open(csv_path, "r") as f:
		reader = csv.DictReader(f)
		rows = list(reader)

	img_dir = os.path.join(local_dir, "images", "test")
	subdirs = sorted(d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d)))
	# Build a lookup: image_id -> file path
	id_to_path = {}
	for sd in subdirs:
		sd_path = os.path.join(img_dir, sd)
		for fname in os.listdir(sd_path):
			img_id = os.path.splitext(fname)[0]
			id_to_path[img_id] = os.path.join(sd_path, fname)

	# Keep only rows whose image exists on disk
	rows = [r for r in rows if r["id"] in id_to_path]

	# Sample n_images_to_eval random images
	rng = random.Random(42)
	samples = rng.sample(rows, min(n_images_to_eval, len(rows)))

	source_images = [Image.open(id_to_path[s["id"]]) for s in samples]
	if use_real_gps:
		source_gps = [(float(s["latitude"]), float(s["longitude"])) for s in samples]
	print(f"Loaded {len(source_images)} images from OSV-5M test set.")
	return source_images, source_gps if use_real_gps else None
  
def evaluate_attack_on_dataset(
    attack_type: str,
    pipeline,
    dataset_name,
    use_real_gps: bool = False,
    n_images_to_eval: int = 100,
    **kwargs):
	"""
		Evaluate an attack on images from a test dataset.

		Args:
			attack_type: "encoder" or "diffusion".
			pipeline: Plonk pipeline.
			dataset_name: Name of the dataset to evaluate on. "osvm" or "YFCC". THis corresponds to corresponding
				test datasets (YFCC4K for YFCC). 
            use_real_gps: Whether use real gps coords from dataset as source trajectory, instead of the clean predicted one for evaluation.
			**kwargs: forwarded to the corresponding attack function.
	"""

	if dataset_name == "osv":
		source_images, source_gps = retrieve_osv_images(n_images_to_eval=n_images_to_eval, use_real_gps=use_real_gps)
		
	else:
		raise ValueError(f"Unknown dataset_name={dataset_name}. Expected one of ['osv']")
    
    #run the attack on each image and collect results
	results = []
	for i, source_image in enumerate(source_images):
    	print(f"Evaluating image {i+1}/{len(source_images)}")
		attack_result = run_attack(
			attack_type=attack_type,
			source_image=source_image,
			pipeline=pipeline,
			**kwargs,
		)
		if use_real_gps:
			attack_result["source_gps"] = source_gps[i]
		results.append(attack_result)
	
	#compute summary metrics across the dataset: final loss, final displacement.
 
	return
 
  


if __name__ == "__main__":
	# download_osv5m_test()
	evaluate_attack_on_dataset(
		attack_type="diffusion",
  		dataset_name="osv",
		pipeline=None,  
	)