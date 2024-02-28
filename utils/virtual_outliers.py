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
    _, I = index.search(normalized_embeddings.cpu().numpy(), K + 1)  # Move tensor to CPU before conversion
    boundary_indices = np.unique(I[:, -1])[:number_of_boundary_points]  # Last column, unique indices
    return boundary_indices

def generate_candidate_outliers(boundary_embeddings_unnorm, number_of_candidates, std_dev):
    d = boundary_embeddings_unnorm.shape[1]
    noise = torch.randn((boundary_embeddings_unnorm.shape[0] * number_of_candidates, d), device=boundary_embeddings_unnorm.device) * std_dev
    candidates = boundary_embeddings_unnorm.unsqueeze(1).repeat(1, number_of_candidates, 1).view(-1, d) + noise
    return candidates

def select_best_outliers(all_candidates, index, num_boundary_points, num_candidates_per_boundary, num_outliers_needed):
    D, _ = index.search(all_candidates.cpu().numpy(), 1)
    distances = torch.from_numpy(np.sqrt(D)).squeeze()

    # Reshape distances to group by boundary point
    distances = distances.view(num_boundary_points, num_candidates_per_boundary)

    # Select the best (furthest) candidates from each group
    _, selected_indices = torch.topk(distances, num_outliers_needed // num_boundary_points, largest=True)

    # Flatten the indices to get the global index in the all_candidates tensor
    global_indices = (torch.arange(num_boundary_points).unsqueeze(1) * num_candidates_per_boundary + selected_indices).view(-1)
    return all_candidates[global_indices]

def synthesize_outliers(inlier_embeddings, num_outliers_needed, K, num_boundary_points, num_candidate_outliers, std_dev):
    normalized_embeddings = normalize_embeddings(inlier_embeddings)
    boundary_indices = find_boundary_points(normalized_embeddings, K, num_boundary_points)
    boundary_embeddings_unnorm = inlier_embeddings[boundary_indices]
    
    all_candidates = generate_candidate_outliers(boundary_embeddings_unnorm, num_candidate_outliers, std_dev)
    index = create_faiss_index(normalized_embeddings)
    
    selected_outliers = select_best_outliers(all_candidates, index, num_boundary_points, num_candidate_outliers, num_outliers_needed)

    return selected_outliers