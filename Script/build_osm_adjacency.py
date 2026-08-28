#!/usr/bin/env python3
"""Build model-agnostic OSM road-network adjacency matrices.

The graph nodes are countline-direction pairs from a stable node list. Each node
is map-matched to the nearest OpenStreetMap drive-network node. Pairwise
shortest-path road-network distances are calculated in metres, symmetrised for
STGCN-style adjacency use, and converted to weights with a thresholded Gaussian
kernel.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer


REQUIRED_COLUMNS = ["node_order", "node_id", "Countline", "Direction", "mean_lat", "mean_lon"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node-list",
        type=Path,
        default=Path("data_processed/speed/node_selection_v2/stable_node_list_v2.csv"),
        help="Stable node list. The node_order column defines matrix row/column order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_processed/adjacency/osm"),
        help="Output folder for OSM adjacency files.",
    )
    parser.add_argument(
        "--model-input-dir",
        type=Path,
        default=Path("data_processed/model_inputs/adjacency/osm"),
        help="Output folder for model-ready OSM adjacency matrices.",
    )
    parser.add_argument(
        "--model-input-epsilon",
        type=float,
        default=0.3,
        help="Single epsilon value exported to model_inputs as the default model-ready graph.",
    )
    parser.add_argument(
        "--common-input-dir",
        type=Path,
        default=Path("data_processed/model_inputs/common"),
        help="Output folder for shared model inputs, including node_list.csv.",
    )
    parser.add_argument(
        "--graphml-cache",
        type=Path,
        default=Path("data_external/osm_roads/nottingham_drive_network.graphml"),
        help="Cached OSM drive-network GraphML. Downloaded if missing unless --no-download is set.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="Threshold values used to remove weak Gaussian-kernel edges.",
    )
    parser.add_argument(
        "--bbox-buffer-m",
        type=float,
        default=2000.0,
        help="Buffer around stable nodes used to download the OSM drive network.",
    )
    parser.add_argument(
        "--network-type",
        default="drive",
        help="OSMnx network_type used when downloading the road network.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require --graphml-cache to exist and do not download OSM data.",
    )
    return parser.parse_args()


def epsilon_label(epsilon: float) -> str:
    return f"eps{epsilon:g}"


def load_nodes(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Node list is missing required columns: {missing}")

    nodes = data.copy()
    nodes["node_order"] = pd.to_numeric(nodes["node_order"], errors="raise").astype(int)
    nodes = nodes.sort_values("node_order").reset_index(drop=True)
    expected = list(range(len(nodes)))
    actual = nodes["node_order"].tolist()
    if actual != expected:
        raise ValueError("node_order must be consecutive integers starting from 0.")
    if nodes[["mean_lat", "mean_lon"]].isna().any().any():
        raise ValueError("mean_lat/mean_lon contain missing values.")
    return nodes


def project_coordinates(nodes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    x, y = transformer.transform(nodes["mean_lon"].to_numpy(), nodes["mean_lat"].to_numpy())
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def buffered_bbox_wgs84(nodes: pd.DataFrame, buffer_m: float) -> tuple[float, float, float, float]:
    to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    to_wgs = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    x, y = to_bng.transform(nodes["mean_lon"].to_numpy(), nodes["mean_lat"].to_numpy())
    left_x, right_x = float(np.min(x) - buffer_m), float(np.max(x) + buffer_m)
    bottom_y, top_y = float(np.min(y) - buffer_m), float(np.max(y) + buffer_m)
    west, south = to_wgs.transform(left_x, bottom_y)
    east, north = to_wgs.transform(right_x, top_y)
    return float(west), float(south), float(east), float(north)


def load_or_download_graph(args: argparse.Namespace, nodes: pd.DataFrame) -> nx.MultiDiGraph:
    if args.graphml_cache.exists():
        graph = ox.load_graphml(args.graphml_cache)
    else:
        if args.no_download:
            raise FileNotFoundError(f"GraphML cache does not exist: {args.graphml_cache}")
        bbox = buffered_bbox_wgs84(nodes, args.bbox_buffer_m)
        ox.settings.use_cache = True
        ox.settings.cache_folder = str(args.graphml_cache.parent / "osmnx_cache")
        ox.settings.log_console = True
        graph = ox.graph_from_bbox(
            bbox,
            network_type=args.network_type,
            simplify=True,
            retain_all=False,
            truncate_by_edge=True,
        )
        args.graphml_cache.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(graph, args.graphml_cache)

    return ox.project_graph(graph, to_crs="EPSG:27700")


def map_match_nodes(
    graph: nx.MultiDiGraph, nodes: pd.DataFrame, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    nearest, distances = ox.distance.nearest_nodes(graph, X=x, Y=y, return_dist=True)
    nearest = np.asarray(nearest)
    distances = np.asarray(distances, dtype=float)
    rows = []
    for idx, row in nodes.iterrows():
        rows.append(
            {
                "node_order": int(row["node_order"]),
                "node_id": row["node_id"],
                "Countline": row["Countline"],
                "Direction": row["Direction"],
                "mean_lat": row["mean_lat"],
                "mean_lon": row["mean_lon"],
                "x_27700": x[idx],
                "y_27700": y[idx],
                "matched_osm_node": nearest[idx],
                "match_distance_m": distances[idx],
            }
        )
    return nearest, distances, pd.DataFrame(rows)


def shortest_path_distance_matrix(graph: nx.MultiDiGraph, matched_nodes: np.ndarray) -> np.ndarray:
    n = len(matched_nodes)
    distance = np.full((n, n), np.inf, dtype=float)
    unique_sources = sorted(set(matched_nodes.tolist()))
    lengths_by_source = {
        source: nx.single_source_dijkstra_path_length(graph, source, weight="length")
        for source in unique_sources
    }
    for i, source in enumerate(matched_nodes.tolist()):
        lengths = lengths_by_source[source]
        for j, target in enumerate(matched_nodes.tolist()):
            if source == target:
                distance[i, j] = 0.0
            elif target in lengths:
                distance[i, j] = float(lengths[target])
    return distance


def symmetrise_distances_min(directed_distance: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        symmetric = np.minimum(directed_distance, directed_distance.T)
    both_missing = ~np.isfinite(directed_distance) & ~np.isfinite(directed_distance.T)
    symmetric[both_missing] = np.inf
    np.fill_diagonal(symmetric, 0.0)
    return symmetric


def gaussian_weights(distances: np.ndarray, sigma: float) -> np.ndarray:
    weights = np.zeros_like(distances, dtype=float)
    finite = np.isfinite(distances)
    weights[finite] = np.exp(-np.square(distances[finite]) / (sigma**2))
    np.fill_diagonal(weights, 0.0)
    return weights


def threshold_weights(weights: np.ndarray, epsilon: float) -> np.ndarray:
    adjacency = np.where(weights >= epsilon, weights, 0.0)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def same_countline_opposite_direction_mask(nodes: pd.DataFrame) -> np.ndarray:
    countlines = nodes["Countline"].astype(str).to_numpy()
    directions = nodes["Direction"].astype(str).str.lower().to_numpy()
    mask = (countlines[:, None] == countlines[None, :]) & (
        directions[:, None] != directions[None, :]
    )
    np.fill_diagonal(mask, False)
    return mask


def mask_same_countline_opposite_directions(
    adjacency: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    masked = adjacency.copy()
    masked[mask] = 0.0
    np.fill_diagonal(masked, 0.0)
    return masked


def save_matrix_csv(matrix: np.ndarray, nodes: pd.DataFrame, path: Path) -> None:
    labels = nodes["node_id"].tolist()
    out = pd.DataFrame(matrix, index=labels, columns=labels)
    out = out.replace([np.inf, -np.inf], np.nan)
    out.to_csv(path, index_label="node_id")


def build_edge_list(
    adjacency: np.ndarray, distances: np.ndarray, nodes: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    node_ids = nodes["node_id"].tolist()
    countlines = nodes["Countline"].tolist()
    directions = nodes["Direction"].tolist()
    source_idx, target_idx = np.nonzero(adjacency)
    for i, j in zip(source_idx.tolist(), target_idx.tolist(), strict=False):
        rows.append(
            {
                "source_order": i,
                "source_node_id": node_ids[i],
                "source_countline": countlines[i],
                "source_direction": directions[i],
                "target_order": j,
                "target_node_id": node_ids[j],
                "target_countline": countlines[j],
                "target_direction": directions[j],
                "network_distance_m": distances[i, j],
                "weight": adjacency[i, j],
            }
        )
    return pd.DataFrame(rows)


def diagnostics_row(
    adjacency: np.ndarray,
    distances: np.ndarray,
    epsilon: float,
    sigma: float,
    direction_mask: np.ndarray,
) -> dict[str, float]:
    n = adjacency.shape[0]
    nonzero = adjacency > 0
    neighbours = nonzero.sum(axis=1)
    possible = n * (n - 1)
    nonzero_weights = adjacency[nonzero]
    nonzero_distances = distances[nonzero]
    finite_offdiag = np.isfinite(distances) & ~np.eye(n, dtype=bool)
    return {
        "epsilon": epsilon,
        "sigma_m": sigma,
        "nodes": n,
        "finite_directed_pairs_excluding_diagonal": int(finite_offdiag.sum()),
        "nonzero_directed_edges": int(nonzero.sum()),
        "edge_density_excluding_diagonal": float(nonzero.sum() / possible) if possible else 0.0,
        "mean_neighbours_per_node": float(neighbours.mean()) if n else 0.0,
        "median_neighbours_per_node": float(np.median(neighbours)) if n else 0.0,
        "min_neighbours_per_node": int(neighbours.min()) if n else 0,
        "max_neighbours_per_node": int(neighbours.max()) if n else 0,
        "isolated_nodes": int((neighbours == 0).sum()),
        "masked_same_countline_opposite_direction_pairs": int(direction_mask.sum()),
        "remaining_same_countline_opposite_direction_edges": int((nonzero & direction_mask).sum()),
        "mean_edge_distance_m": float(nonzero_distances.mean()) if nonzero_distances.size else 0.0,
        "median_edge_distance_m": float(np.median(nonzero_distances)) if nonzero_distances.size else 0.0,
        "mean_edge_weight": float(nonzero_weights.mean()) if nonzero_weights.size else 0.0,
        "median_edge_weight": float(np.median(nonzero_weights)) if nonzero_weights.size else 0.0,
    }


def write_summary(
    path: Path,
    nodes: pd.DataFrame,
    graph: nx.MultiDiGraph,
    directed_distances: np.ndarray,
    symmetric_distances: np.ndarray,
    match_summary: pd.DataFrame,
    sigma: float,
    epsilons: list[float],
    diagnostics: pd.DataFrame,
    graphml_cache: Path,
    direction_mask: np.ndarray,
) -> None:
    finite_distances = symmetric_distances[np.isfinite(symmetric_distances) & (symmetric_distances > 0)]
    lines = [
        "# OSM Road-Network Adjacency Build Summary",
        "",
        "## Inputs",
        "",
        f"- Nodes: {len(nodes)} countline-direction nodes",
        "- Node order: `node_order` from the stable node list",
        "- Coordinate source: `mean_lat` / `mean_lon`",
        "- Source CRS: EPSG:4326 (WGS84 latitude/longitude)",
        "- Network CRS after projection: EPSG:27700 (British National Grid)",
        "- Distance unit: metres along the OSM drive network",
        f"- GraphML cache: `{graphml_cache}`",
        "",
        "## OSM Network",
        "",
        f"- OSM nodes: {graph.number_of_nodes()}",
        f"- OSM edges: {graph.number_of_edges()}",
        "",
        "## Method",
        "",
        "1. A drivable OSM road network was loaded/downloaded using OSMnx.",
        "2. The OSM graph and countline coordinates were projected to EPSG:27700.",
        "3. Each countline-direction node was matched to its nearest OSM network node.",
        "4. Directed shortest-path distances were calculated using edge length in metres.",
        "5. A symmetric distance matrix was produced using the minimum of both directed distances.",
        "6. Symmetric distances were converted to edge weights using a thresholded Gaussian kernel:",
        "7. Edges between opposite directions of the same physical countline were masked to zero.",
        "",
        "`w_ij = exp(-d_ij^2 / sigma^2)` if `i != j` and `w_ij >= epsilon`; otherwise `w_ij = 0`.",
        "",
        f"- sigma method: standard deviation of all finite non-zero symmetric OSM distances",
        f"- sigma: {sigma:.6f} m",
        f"- epsilon values tested: {', '.join(str(e) for e in epsilons)}",
        "- diagonal entries: 0",
        f"- same-countline opposite-direction pairs masked: {int(direction_mask.sum())} directed pairs",
        "",
        "## Map-Matching Diagnostics",
        "",
        f"- Mean match distance: {match_summary['match_distance_m'].mean():.3f} m",
        f"- Median match distance: {match_summary['match_distance_m'].median():.3f} m",
        f"- Maximum match distance: {match_summary['match_distance_m'].max():.3f} m",
        "",
        "## Distance Diagnostics",
        "",
        f"- Finite non-zero symmetric distances: {finite_distances.size}",
        f"- Minimum finite non-zero distance: {finite_distances.min():.3f} m",
        f"- Median finite non-zero distance: {np.median(finite_distances):.3f} m",
        f"- Mean finite non-zero distance: {finite_distances.mean():.3f} m",
        f"- Maximum finite non-zero distance: {finite_distances.max():.3f} m",
        f"- Unreachable directed pairs: {int((~np.isfinite(directed_distances)).sum())}",
        "",
        "## Sparsity Diagnostics",
        "",
        diagnostics.to_markdown(index=False),
        "",
        "## Outputs",
        "",
        "- `osm_network_distance_matrix_directed_m.csv`: directed shortest-path distance matrix.",
        "- `osm_network_distance_matrix_m.csv`: symmetric shortest-path distance matrix used for adjacency.",
        "- `osm_weight_matrix_<epsilon>.csv`: thresholded Gaussian adjacency matrix.",
        "- `osm_edge_list_<epsilon>.csv`: non-zero directed matrix entries with distances and weights.",
        "- `adj_osm_<epsilon>.npy`: numpy adjacency matrix.",
        "- `adj_osm_<epsilon>.npz`: adjacency matrix saved under keys `adj`, `adj_mx`, and `data`.",
        "- `osm_node_map_matching.csv`: nearest OSM node and match distance for every modelling node.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    model_input_dir = args.model_input_dir
    common_input_dir = args.common_input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_input_dir.mkdir(parents=True, exist_ok=True)
    common_input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes(args.node_list)
    x, y = project_coordinates(nodes)
    nodes_out = nodes.copy()
    nodes_out["x_27700"] = x
    nodes_out["y_27700"] = y
    # Keep one common modelling node order shared by all model inputs.
    nodes_out.to_csv(common_input_dir / "node_list.csv", index=False)

    for stale in model_input_dir.glob("*osm*"):
        stale.unlink()

    graph = load_or_download_graph(args, nodes)
    matched_nodes, _, match_summary = map_match_nodes(graph, nodes, x, y)
    match_summary.to_csv(output_dir / "osm_node_map_matching.csv", index=False)

    directed_distances = shortest_path_distance_matrix(graph, matched_nodes)
    symmetric_distances = symmetrise_distances_min(directed_distances)

    finite_nonzero = symmetric_distances[np.isfinite(symmetric_distances) & (symmetric_distances > 0)]
    sigma = float(np.std(finite_nonzero))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Unable to calculate a positive sigma from finite non-zero OSM distances.")

    weights = gaussian_weights(symmetric_distances, sigma)
    direction_mask = same_countline_opposite_direction_mask(nodes)
    save_matrix_csv(directed_distances, nodes, output_dir / "osm_network_distance_matrix_directed_m.csv")
    save_matrix_csv(symmetric_distances, nodes, output_dir / "osm_network_distance_matrix_m.csv")

    diagnostics = []
    for epsilon in args.epsilons:
        label = epsilon_label(epsilon)
        adjacency = threshold_weights(weights, epsilon)
        adjacency = mask_same_countline_opposite_directions(adjacency, direction_mask)
        save_matrix_csv(adjacency, nodes, output_dir / f"osm_weight_matrix_{label}.csv")
        build_edge_list(adjacency, symmetric_distances, nodes).to_csv(
            output_dir / f"osm_edge_list_{label}.csv", index=False
        )
        np.save(output_dir / f"adj_osm_{label}.npy", adjacency)
        np.savez(output_dir / f"adj_osm_{label}.npz", adj=adjacency, adj_mx=adjacency, data=adjacency)
        if np.isclose(epsilon, args.model_input_epsilon):
            save_matrix_csv(adjacency, nodes, model_input_dir / f"osm_weight_matrix_{label}.csv")
            np.save(model_input_dir / f"adj_osm_{label}.npy", adjacency)
            np.savez(
                model_input_dir / f"adj_osm_{label}.npz",
                adj=adjacency,
                adj_mx=adjacency,
                data=adjacency,
            )
        diagnostics.append(diagnostics_row(adjacency, symmetric_distances, epsilon, sigma, direction_mask))

    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df.to_csv(output_dir / "osm_adjacency_diagnostics.csv", index=False)
    write_summary(
        output_dir / "graph_build_summary.md",
        nodes_out,
        graph,
        directed_distances,
        symmetric_distances,
        match_summary,
        sigma,
        args.epsilons,
        diagnostics_df,
        args.graphml_cache,
        direction_mask,
    )

    print(f"Built OSM adjacency for {len(nodes)} nodes.")
    print(f"Outputs written to: {output_dir}")
    print(f"Default model-ready adjacency written to: {model_input_dir}")
    print(diagnostics_df.to_string(index=False))


if __name__ == "__main__":
    main()
