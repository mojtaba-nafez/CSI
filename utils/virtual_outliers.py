from cv2 import normalize
import numpy as np
import torch
import faiss
import faiss.contrib.torch_utils

def normalize_embeddings(embeddings):
    norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    return embeddings / norms

def create_faiss_index(embeddings, use_gpu=True):
    res = faiss.StandardGpuResources() if use_gpu else None
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    if use_gpu:
        # Convert to Numpy array if embeddings is a PyTorch tensor
        embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    index.add(embeddings_np)
    return index


def find_boundary_points(normalized_embeddings, K, number_of_boundary_points):
    index = create_faiss_index(normalized_embeddings.cpu().numpy())  # Ensure embeddings are on CPU for FAISS
    D, I = index.search(normalized_embeddings.cpu().numpy(), K + 1)  # D for distances, I for indices

    # Sort by the Kth nearest distance in descending order and get the corresponding indices
    sorted_indices = np.argsort(-D[:, K])[:number_of_boundary_points]  # Use -D[:, K] for descending sort

    # Extract the indices of the points with the highest Kth neighbor distance
    boundary_indices = I[sorted_indices, K]

    return boundary_indices

def generate_candidate_outliers(boundary_embeddings_unnorm, number_of_candidates, std_dev):
    d = boundary_embeddings_unnorm.shape[1]
    noise = torch.randn((boundary_embeddings_unnorm.shape[0] * number_of_candidates, d), device=boundary_embeddings_unnorm.device) * std_dev
    candidates = boundary_embeddings_unnorm.unsqueeze(1).repeat(1, number_of_candidates, 1).view(-1, d) + noise
    return candidates


def select_best_outliers(all_candidates, index, num_boundary_points, num_candidates_per_boundary, num_outliers_needed, K):
    # Search for the Kth nearest neighbor in the index for each candidate
    D, _ = index.search(normalize_embeddings(all_candidates).cpu().numpy(), K + 1)
    distances_to_kth_neighbor = torch.from_numpy(np.sqrt(D[:, K])).squeeze()

    # Reshape distances to group by boundary point
    distances_to_kth_neighbor = distances_to_kth_neighbor.view(num_boundary_points, num_candidates_per_boundary)

    # Select the best (furthest) candidates from each group based on distance to the Kth nearest neighbor
    _, selected_indices = torch.topk(distances_to_kth_neighbor, num_outliers_needed // num_boundary_points, largest=True)

    # Flatten the indices to get the global index in the all_candidates tensor
    global_indices = (torch.arange(num_boundary_points).unsqueeze(1) * num_candidates_per_boundary + selected_indices).view(-1)
    return all_candidates[global_indices]

def synthesize_outliers(inlier_embeddings, num_outliers_needed, K, num_boundary_points, num_candidate_outliers, std_dev):
    normalized_embeddings = normalize_embeddings(inlier_embeddings)
    boundary_indices = find_boundary_points(normalized_embeddings, K, num_boundary_points)
    boundary_embeddings_unnorm = inlier_embeddings[boundary_indices]
    
    all_candidates = generate_candidate_outliers(boundary_embeddings_unnorm, num_candidate_outliers, std_dev)
    index = create_faiss_index(normalized_embeddings)
    
    selected_outliers = select_best_outliers(all_candidates, index, num_boundary_points, num_candidate_outliers, num_outliers_needed, K)

    return selected_outliers


#####################################
## Generate outliers with GMM Method
#####################################

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

def calculate_epsilon_from_samples(mean, covariance, t, num_samples=1000):
    mvn = MultivariateNormal(mean, covariance)
    samples = mvn.sample((num_samples,))
    distances = (samples - mean).unsqueeze(1).matmul(covariance.inverse().unsqueeze(0)).matmul((samples - mean).unsqueeze(-1)).squeeze()
    sorted_distances, _ = torch.sort(distances)
    epsilon = sorted_distances[t - 1]  # t-th smallest value, assuming t starts from 1
    return epsilon.item()  # Return as a Python float for later comparison

def synthesize_outliers_with_gaussian(inlier_embeddings, num_outliers_needed, t, batch_size=10000):
    # Move to GPU for acceleration, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inlier_embeddings = torch.tensor(inlier_embeddings, dtype=torch.float).to(device)
    
    # Estimate Gaussian parameters: mean and covariance
    mean = torch.mean(inlier_embeddings, dim=0)
    covariance = torch.cov(inlier_embeddings.t()) + torch.eye(inlier_embeddings.shape[1], device=device) * 1e-6
    
    # Calculate epsilon based on the t-th smallest likelihood from the sample
    epsilon = calculate_epsilon_from_samples(mean, covariance, t)
    print(epsilon)

    mvn = MultivariateNormal(mean, covariance)
    selected_outliers = []
    
    # Continue sampling until we have enough outliers
    while len(selected_outliers) < num_outliers_needed:
        print(len(selected_outliers))
        # Sample batch from Gaussian
        samples = mvn.sample((batch_size,))
        
        # Compute the Mahalanobis distance for each sample
        mahalanobis_distances = (samples - mean).unsqueeze(1).matmul(covariance.inverse().unsqueeze(0)).matmul((samples - mean).unsqueeze(-1)).squeeze()
        
        # Filter samples based on the calculated epsilon
        for i, distance in enumerate(mahalanobis_distances):
            if distance < epsilon:
                selected_outliers.append(samples[i])
                
            # Stop if we've collected enough
            if len(selected_outliers) >= num_outliers_needed:
                break
    
    return torch.stack(selected_outliers)


#############################################
## Generate outliers with Window-Based Method
#############################################

def generate_outliers_with_exact_magnitude(embedding, K, magnitude):
    d = embedding.shape[1]
    noise = torch.randn((K, d), device=embedding.device)
    noise_norms = torch.norm(noise, p=2, dim=1, keepdim=True)
    normalized_noise = noise / noise_norms
    scaled_noise = normalized_noise * magnitude
    return scaled_noise

def create_faiss_index_window(embeddings, use_gpu=True):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    if use_gpu:
        faiss_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(faiss_res, 0, index)
    index.add(embeddings.cpu().numpy())
    return index

def validate_and_find_magnitude(embedding, embeddings, K, initial_magnitude, threshold=0.5, tau=2, use_gpu=True):
    magnitude = initial_magnitude
    if len(embedding.shape) == 1:
        embedding = embedding.unsqueeze(0)
        
    normalized_embedding = normalize_embeddings(embedding)

    # Normalize all embeddings to ensure a fair comparison
    all_embeddings_normalized = normalize_embeddings(embeddings)
    candidates_init = generate_outliers_with_exact_magnitude(embedding, K, 1)

    # Create a FAISS index for all embeddings including the original dataset
    index = create_faiss_index_window(all_embeddings_normalized, use_gpu=use_gpu)

    while magnitude > 0.01:
        # Generate outliers for the specific embedding
        candidates = embedding + candidates_init * magnitude
        # Normalize the candidates to ensure a fair comparison
        candidates_normalized = normalize_embeddings(candidates)
        # Add these candidates to the index
        temp_index = faiss.IndexFlatL2(all_embeddings_normalized.shape[1])
        if use_gpu:
            faiss_res = faiss.StandardGpuResources()  # Use GPU resources if available
            temp_index = faiss.index_cpu_to_gpu(faiss_res, 0, temp_index)
        temp_index.add(np.concatenate([all_embeddings_normalized.cpu().numpy(), candidates_normalized.cpu().numpy()]))

        # Search for the original embedding's nearest neighbors among all points including the generated outliers
        D, I = temp_index.search(normalized_embedding.cpu().numpy(), K + 1)

        # Calculate the ratio of neighbors that are generated outliers
        # Count as outliers only those neighbors that are not part of the original dataset
        num_outliers = sum([1 for i in I[0] if i >= len(embeddings)])
        ratio = num_outliers / K

        if ratio >= threshold:
            return magnitude
        magnitude /= tau

    return magnitude

def find_magnitudes_for_all_embeddings(embeddings, K=10, initial_magnitude=1.0, threshold=0.5, use_gpu=True):
    magnitudes = []
    for i, embedding in enumerate(embeddings):
        magnitude = validate_and_find_magnitude(embedding, embeddings, K, initial_magnitude, threshold, use_gpu=use_gpu)
        magnitudes.append(magnitude)
        print(f"Embedding {i+1}/{embeddings.shape[0]}: Magnitude = {magnitude}")
    return magnitudes


def synthesize_outliers_window(inlier_embeddings, num_outliers_needed, K, num_boundary_points, num_candidate_outliers, std_dev, initial_magnitude, threshold):
    normalized_embeddings = normalize_embeddings(inlier_embeddings)
    magnitudes = find_magnitudes_for_all_embeddings(inlier_embeddings, K, initial_magnitude, threshold)
    sorted_indices = np.argsort(magnitudes)[::-1]  # Sort magnitudes in descending order and get indices
    boundary_indices = sorted_indices[:num_boundary_points] 
    boundary_indices = torch.tensor(boundary_indices)
    boundary_embeddings_unnorm = inlier_embeddings[boundary_indices]
    
    all_candidates = generate_candidate_outliers(boundary_embeddings_unnorm, num_candidate_outliers, std_dev)
    index = create_faiss_index(normalized_embeddings)
    
    selected_outliers = select_best_outliers(all_candidates, index, num_boundary_points, num_candidate_outliers, num_outliers_needed, K)

    return selected_outliers