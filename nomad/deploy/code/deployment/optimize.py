import os
import shutil
import yaml
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import networkx as nx

# Import NoMaD components
from deploy.code.models.nomad import NoMaD, DenseNetwork
from deploy.code.models.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False", category=UserWarning)

def load_config(config_path):
    # Load configuration from YAML file provided as an argument.
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(config, device):
    # Build the vision encoder and replace BatchNorm with GroupNorm.
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    # Build the distance prediction network.
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    # Create the NoMaD model with only the distance prediction branch active.
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=None,
        dist_pred_net=dist_pred_network,
    )
    return model.to(device)

def preprocess_image(image_path, image_size, transform):
    # Open the image, convert to RGB, resize, and apply normalization.
    image = Image.open(image_path).convert("RGB")
    image = image.resize(tuple(image_size))
    return transform(image)

def load_images_from_directory(directory):
    # Assumes images are named "0.png", "1.png", ... and sorts them numerically.
    files = sorted(
        [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    return [os.path.join(directory, f) for f in files]

def compute_distance(model, obs_context, goal_img, goal_mask, device):
    # Pass the observation context and goal image through the model to predict the distance.
    with torch.no_grad():
        obsgoal_cond = model("vision_encoder", obs_img=obs_context, goal_img=goal_img, input_goal_mask=goal_mask)
        dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
    return dist_pred.squeeze().item()

def compute_sequence(image_paths, i_skip, step):
    
    # Create a list of indices from 0 to len(image_paths)-1.
    indices = list(range(len(image_paths)))
    # Sub-sample the indices using the step value.
    subsampled = indices[::step]
    # For the first jump, skip the next (i_skip - 1) entries in the subsampled list.
    # That is, keep index 0, then start from subsampled[i_skip].
    sequence = [subsampled[0]] + subsampled[i_skip:]
    
    # Print the results
    # print("Full indices:", indices)
    print("Subsampled (step={}):".format(step), subsampled)
    print("Final sequence (i_skip={}):".format(i_skip), sequence)
    
    
    return sequence

def graph_shortest_path(candidate_list, image_paths, model, config, device, threshold):

    # Prepare transform and parameters.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image_size = config["image_size"]
    num_context = config["context_size"] + 1
    goal_mask = torch.zeros(1, dtype=torch.long, device=device)
    
    # Preprocess candidate images.
    preprocessed = {}
    for idx in candidate_list:
        path_img = image_paths[idx]
        img_tensor = preprocess_image(path_img, image_size, transform)
        obs_context = torch.cat([img_tensor] * num_context, dim=0).unsqueeze(0)
        goal_img = img_tensor.unsqueeze(0)
        preprocessed[idx] = (obs_context.to(device), goal_img.to(device))
    
    # Build directed graph.
    G = nx.DiGraph()
    for idx in candidate_list:
        G.add_node(idx)
    
    # Add default edges between adjacent candidates.
    for i in range(len(candidate_list) - 1):
        u = candidate_list[i]
        v = candidate_list[i+1]
        G.add_edge(u, v, weight=1)
    
    # Add extra edges for non-adjacent candidates if they satisfy the threshold.
    for i in tqdm(range(len(candidate_list)), desc="Building extra graph edges"):
        for j in range(i + 2, len(candidate_list)):
            u = candidate_list[i]
            v = candidate_list[j]
            obs_context, _ = preprocessed[u]
            _, goal_img = preprocessed[v]
            cost = compute_distance(model, obs_context, goal_img, goal_mask, device)
            if cost < threshold:
                G.add_edge(u, v, weight=1)
    
    # Find the shortest path (fewest hops) from the first candidate to the last.
    try:
        shortest_path = nx.shortest_path(G, source=candidate_list[0],
                                         target=candidate_list[-1],
                                         weight="weight")
        print("Graph-based shortest path:", " -> ".join(map(str, shortest_path)))
    except nx.NetworkXNoPath:
        print("No path found in the candidate graph!")
        return None, None
    
    # Identify non-adjacent jumps.
    jump_index = []
    # For each consecutive pair in the shortest path, check if they're adjacent in candidate_list.
    for i in range(len(shortest_path) - 1):
        u, v = shortest_path[i], shortest_path[i+1]
        pos_u = candidate_list.index(u)
        pos_v = candidate_list.index(v)
        # If v is not exactly the next candidate after u, record the edge.
        if pos_v != pos_u + 1:
            jump_index.append((u, v))
    
    return shortest_path, jump_index

def print_path(sparse_path, jump_index):
    
    full_seq = []
    # Iterate over each consecutive edge in the sparse path.
    for i in range(len(sparse_path) - 1):
        u = sparse_path[i]
        v = sparse_path[i+1]
        # Check if this edge is marked as a jump.
        if (u, v) in jump_index:
            # If it's a jump edge, simply add the endpoints (avoid duplicating u if already present).
            if not full_seq or full_seq[-1] != u:
                full_seq.append(u)
            full_seq.append(v)
        else:
            # Otherwise, they are adjacent in the candidate list.
            # Fill in all indices between u and v.
            # For the first edge, include u through v.
            # For subsequent edges, avoid duplicating the starting element.
            if i == 0:
                full_seq.extend(range(u, v + 1))
            else:
                full_seq.extend(range(u + 1, v + 1))
    print("Shortest full path (all original indices):", " -> ".join(map(str, full_seq)))
    
def copy_path_images(image_paths, jump_index, destination_folder):
    
    os.makedirs(destination_folder, exist_ok=True)
    
    # Function to check if an index is within any of the jump intervals.
    def in_jump_interval(idx, intervals):
        for u, v in intervals:
            if u < idx < v:
                return True
        return False

    new_idx = 0
    # Iterate over all image indices.
    for idx in tqdm(range(len(image_paths)), desc="Copying images"):
        if in_jump_interval(idx, jump_index):
            continue  # Skip indices within any jump interval.
        src = image_paths[idx]
        dst = os.path.join(destination_folder, f"{new_idx}.png")
        shutil.copy(src, dst)
        new_idx += 1

    print(f"Copied {new_idx} images to folder '{destination_folder}' with sequential naming (0.png to {new_idx-1}.png).")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Find shortest path from 0.png to N.png using NoMaD distance predictor, and copy path images."
    )
    parser.add_argument("-d", "--directory", type=str, required=True, help="Path to the image directory")
    parser.add_argument("-n", "--checkpoint", type=str, required=True, help="Path to the NoMaD model checkpoint (.pth file)")
    parser.add_argument("-y", "--config", type=str, required=True, help="Path to the model YAML configuration file")
    parser.add_argument("-o", "--destination", type=str, required=True, help="Destination folder to copy images")
    parser.add_argument("-i", "--i_skip", type=int, default=5, help="Minimum index skip between image pairs")
    parser.add_argument("-s", "--step", type=int, default=20, help="Step size for candidate images")
    parser.add_argument("-t", "--threshold", type=float, default=2.0, help="Distance threshold for similarity")

    args = parser.parse_args()

    # Load config from user-provided YAML path
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded config from '{args.config}'")
    else:
        print(f"No config found at '{args.config}'. Exiting.")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)

    # Load provided checkpoint path for the NoMaD distance predictor.
    if os.path.exists(args.checkpoint):
        print(f"Loading NoMaD model checkpoint from '{args.checkpoint}'...")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"No checkpoint found at '{args.checkpoint}'. Exiting.")
        exit(1)
    model.eval()

    image_paths = load_images_from_directory(args.directory)
    if not image_paths:
        print("No images found in the directory.")
        exit(1)
    print(f"Found {len(image_paths)} images in the directory.")

    candidate = compute_sequence(image_paths, args.i_skip, args.step)

    path, jump = graph_shortest_path(candidate, image_paths, model, config, device, args.threshold)
    
    if path is not None:
        print_path(path, jump)
        print(f"Skipped jumps: {jump}")

        destination_folder = os.path.abspath(args.destination)

        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)

        copy_path_images(image_paths, jump, destination_folder)
    else:
        print("No valid path found, skipping image copying.")

if __name__ == "__main__":
    main()
