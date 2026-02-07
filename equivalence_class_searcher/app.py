from flask import Flask, render_template, request, jsonify, send_file
import json, random, os, uuid, shutil, time
from collections import defaultdict
from core_logic import ling_equiv_checker
from core_logic.my_utils import (
    digraph_adjmat_to_config, digraph_config_to_adjmat, \
    digraph_adjmat_to_edges_frozenset, digraph_edges_iterable_to_adjmat, \
    plot_digraph, is_DAG
)

app = Flask(__name__)
PRELOADED_DIR = "static/preloaded"
TEMP_SESSIONS_DIR = "static/tmp_uploads"
SESSION_LIFETIME_HOURS = 6
def cleanup_old_sessions():
    now = time.time()
    cutoff = now - SESSION_LIFETIME_HOURS * 3600
    for folder in os.listdir(TEMP_SESSIONS_DIR):
        folder_path = os.path.join(TEMP_SESSIONS_DIR, folder)
        if os.path.isdir(folder_path) and folder.startswith("session_") and "PROTECTED" not in folder:
            last_modified = os.path.getmtime(folder_path)
            if last_modified < cutoff:
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    pass

@app.route("/")
def index():
    cleanup_old_sessions()
    return render_template("index.html")

@app.route("/random_one", methods=["GET"])
def random_one():
    nx_lx = random.choice([
        d for d in os.listdir(PRELOADED_DIR)
        if os.path.isdir(os.path.join(PRELOADED_DIR, d)) and "n" in d and "_l" in d
    ])
    nx_lx_path = os.path.join(PRELOADED_DIR, nx_lx)
    dist_dir = random.choice([
        d for d in os.listdir(nx_lx_path)
        if d.startswith("dist_") and os.path.isdir(os.path.join(nx_lx_path, d))
    ])
    dist_path = os.path.join(nx_lx_path, dist_dir)

    res_path = os.path.join(dist_path, "res.json")
    with open(res_path, "r") as f: res = json.load(f)

    edge_sections = []
    for edgenumstr in sorted(res["n_edge_to_noniso_graphs"].keys(), key=lambda x: int(x)):
        info = res["n_edge_to_noniso_graphs"][edgenumstr]
        edgenum = int(edgenumstr)
        n_graphs = int(info["n_graphs"])
        n_first_dags = int(info["n_first_dags"])
        edge_dir = os.path.join(dist_path, f"{edgenumstr}_edges")
        svg_paths = [os.path.join(edge_dir, f'{gid}.svg') for gid in range(n_graphs)]
        edge_sections.append({
            "edge_count": edgenum,
            "n_graphs": n_graphs,
            "n_first_dags": n_first_dags,
            "svg_paths": svg_paths,
        })

    return render_template(
        "random_one.html",
        n_L_noniso_graphs=int(res["n_L_noniso_graphs"]),
        n_dags=int(res["n_dags"]),
        edge_sections=edge_sections,
        representation_svg_path=os.path.join(dist_path, "representation.svg"),
    )

@app.route("/specify_graph", methods=["GET", "POST"])
def specify_graph():
    if request.method == "GET": return render_template("specify_graph.html")
    try:
        n_nodes = int(request.form["numnodes"])
        n_latents = int(request.form["n_latents"])
        assert n_nodes >= 3 and n_nodes <= 10
        assert n_latents >= 0 and n_latents <= n_nodes - 2
    except:
        return render_template("specify_graph.html", specify_message="Error in the number of nodes or latents. Please input again.")
    diedges_str = request.form["diedges"]
    L_names = [f'L{i+1}' for i in range(n_latents)]
    X_names = [f'X{i+1}' for i in range(n_nodes - n_latents)]
    V_names = L_names + X_names

    diedges_in_Vnames = set()
    for oneline in diedges_str.strip().split('\n'):
        oneline = oneline.strip()
        if not oneline: continue
        if "-->" not in oneline:
            return render_template("specify_graph.html", specify_message=f"Error in line: '{oneline}'. Invalid edge format. Please input again.")
        src_str, tgt_str = [s.strip() for s in oneline.split("-->")]
        if (src_str not in V_names) or (tgt_str not in V_names):
            return render_template("specify_graph.html", specify_message=f"Error in line: '{oneline}'. Node names not recognized. Please input again.")
        if src_str == tgt_str:
            return render_template("specify_graph.html", specify_message=f"Error in line: '{oneline}'. Self-loops are not allowed. Please input again.")
        diedges_in_Vnames.add((src_str, tgt_str))

    is_irreducible, L_nodeset_reduced, edges_reduced = ling_equiv_checker.check_and_get_an_equivalent_irreducible_model(set(L_names), set(X_names), set(diedges_in_Vnames))
    L_names_new = sorted(L_nodeset_reduced)
    V_names_new = L_names_new + X_names
    V_names_new_name_to_idx = {name: i for i, name in enumerate(V_names_new)}
    edges_new_in_idx = [(V_names_new_name_to_idx[src], V_names_new_name_to_idx[tgt]) for src, tgt in edges_reduced]
    n_nodes_new, n_latents_new = len(V_names_new), len(L_names_new)
    graph_config = digraph_adjmat_to_config(digraph_edges_iterable_to_adjmat(edges_new_in_idx, n_nodes_new, fill_diag_val=1))
    resdict = ling_equiv_checker.traverse_dist_equiv_class_from_an_irreducible_graph_config(
        n_nodes_new,
        n_latents_new,
        graph_config,
        max_class_size_for_column_augmentation=3000,
        max_time_seconds_for_column_augmentation=0.5,
        max_class_size_for_L_bipartite_traverse=1000,
        max_time_seconds_for_L_bipartite_traverse=0.5,
        max_class_size_for_LX_combining=5000,
        max_time_seconds_for_LX_combining=0.5,
        max_class_size_for_L_rename_deduplicate=10000,
        max_time_seconds_for_L_rename_deduplicate=1,
    )

    session_id = str(uuid.uuid4().hex)
    session_dir = os.path.join(TEMP_SESSIONS_DIR, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    to_plots = [('input.svg', L_names, X_names, diedges_in_Vnames)]
    if not is_irreducible: to_plots.append(('reduced.svg', L_names_new, X_names, edges_reduced))

    cpdag_representation_mat = resdict["cpdag_like_representation"]
    cpdag_representation_edges_to_style = dict()
    for vb in range(n_nodes_new):
        for va in range(n_nodes_new):
            if vb == va or cpdag_representation_mat[vb, va] == 0: continue
            cpdag_representation_edges_to_style[(V_names_new[va], V_names_new[vb])] = "solid" if cpdag_representation_mat[vb, va] == 2 else "dashed"
    to_plots.append(('representation.svg', L_names_new, X_names, cpdag_representation_edges_to_style))

    for fname, Lns, Xns, edges in to_plots:
        nodes = list(Lns) + list(Xns)
        vlabels = ({l: "" for l in Lns} | {x: x for x in Xns}) if isinstance(edges, dict) else {i: i for i in nodes}
        vcolors = {i: "lightgrey" if "L" in i else "white" for i in nodes}
        vshapes = {i: "square" if "L" in i else "circle" for i in nodes}
        vsizes = {i: 0.4 if "L" in i else 0.5 for i in nodes}
        if isinstance(edges, dict):
            plot_digraph(nodes, list(edges.keys()), vlabels, vcolors, vshapes, vsizes, os.path.join(session_dir, fname), edge_styles=edges)
        else: plot_digraph(nodes, edges, vlabels, vcolors, vshapes, vsizes, os.path.join(session_dir, fname))

    n_graphs_total = len(resdict['deduplicated_graph_configs_after_L_rename'])
    n_edge_to_configs = defaultdict(list)
    for config in resdict['deduplicated_graph_configs_after_L_rename']: n_edge_to_configs[config.bit_count()].append(config)

    resdict = {
        "n_nodes": len(V_names_new),
        "n_latents": len(L_names_new),
        "is_irreducible": is_irreducible,
        'is_search_finished': resdict['is_search_finished'],
        'is_L_rename_deduplicate_finished': resdict['is_L_rename_deduplicate_finished'],
        "n_L_noniso_graphs": n_graphs_total,
        "n_dags": 0,
        "n_edge_to_noniso_graphs": {}
    }

    for n_edge, configs in n_edge_to_configs.items():
        config_to_is_dag = {cfg: is_DAG(digraph_config_to_adjmat(cfg, n_nodes, fill_diag_val=1)) for cfg in configs}
        n_dags = sum(config_to_is_dag.values())
        configs = sorted(configs, key=lambda c: not config_to_is_dag[c])
        resdict["n_edge_to_noniso_graphs"][n_edge] = {
            "n_graphs": len(configs),
            "n_first_dags": n_dags,
            "configs": configs,
        }
        resdict["n_dags"] += n_dags
        # for now, do not plot all these graphs, that will be too slow...

    with open(os.path.join(session_dir, 'res.json'), 'w') as f:
        f.write(json.dumps(resdict, indent=4))

    return render_template("redirect_post_to_show_class.html", session_id=session_id)

@app.route("/show_specified_class", methods=["POST"])
def show_specified_class():
    session_id = request.form.get("session_id")
    if not session_id: return "No session ID provided.", 400
    session_dir = os.path.join(TEMP_SESSIONS_DIR, f"session_{session_id}")
    if not os.path.exists(session_dir): return "Session not found or expired.", 404
    with open(os.path.join(session_dir, 'res.json')) as f: resdict = json.load(f)

    is_irreducible = bool(resdict["is_irreducible"])
    n_L_noniso_graphs = int(resdict["n_L_noniso_graphs"])
    n_dags = int(resdict["n_dags"])

    edge_sections = []
    for edgenumstr in sorted(resdict["n_edge_to_noniso_graphs"].keys(), key=lambda x: int(x)):
        info = resdict["n_edge_to_noniso_graphs"][edgenumstr]
        edgenum = int(edgenumstr)
        n_graphs = int(info["n_graphs"])
        n_first_dags = int(info["n_first_dags"])
        edge_sections.append({
            "edge_count": edgenum,
            "n_graphs": n_graphs,
            "n_first_dags": n_first_dags,
            "config_indices": list(range(n_graphs))
        })
    return render_template(
        "show_specified_class.html",
        session_id=session_id,
        is_irreducible=is_irreducible,
        n_L_noniso_graphs=n_L_noniso_graphs,
        n_dags=n_dags,
        edge_sections=edge_sections,
        is_search_finished=resdict["is_search_finished"],
        is_L_rename_deduplicate_finished=resdict["is_L_rename_deduplicate_finished"],
    )

@app.route("/get_svg/<session_id>/<int:edge_count>/<int:graph_index>")
def get_svg(session_id, edge_count, graph_index):
    session_dir = os.path.join(TEMP_SESSIONS_DIR, f"session_{session_id}")
    svg_path = os.path.join(session_dir, f"{edge_count}_edges", f"{graph_index}.svg")
    if not os.path.exists(svg_path):
        with open(os.path.join(session_dir, 'res.json')) as f:
            resdict = json.load(f)
        n_nodes, n_latents = resdict["n_nodes"], resdict["n_latents"]
        cfg = resdict["n_edge_to_noniso_graphs"][str(edge_count)]["configs"][graph_index]
        adjmat = digraph_config_to_adjmat(cfg, n_nodes, fill_diag_val=1)
        edges = digraph_adjmat_to_edges_frozenset(adjmat)
        nodes = list(range(n_nodes))
        vlabels = {i: "" if i < n_latents else f"X{i - n_latents + 1}" for i in nodes}
        vcolors = {i: "lightgrey" if i < n_latents else "white" for i in nodes}
        vshapes = {i: "square" if i < n_latents else "circle" for i in nodes}
        vsizes = {i: 0.4 if i < n_latents else 0.5 for i in nodes}
        os.makedirs(os.path.dirname(svg_path), exist_ok=True)
        plot_digraph(nodes, edges, vlabels, vcolors, vshapes, vsizes, svg_path)
    return send_file(svg_path, mimetype='image/svg+xml')


@app.route("/example1_Y_latent", methods=["GET"])
@app.route("/example1_C_latent", methods=["GET"])
@app.route("/example1_YC_latent", methods=["GET"])
def example1_iclr():
    path = request.path
    session_id = path.split("/")[-1] + "_PROTECTED"
    session_dir = os.path.join(TEMP_SESSIONS_DIR, f"session_{session_id}")
    with open(os.path.join(session_dir, 'res.json')) as f: resdict = json.load(f)
    is_irreducible = bool(resdict["is_irreducible"])
    n_L_noniso_graphs = int(resdict["n_L_noniso_graphs"])
    n_dags = int(resdict["n_dags"])
    edge_sections = []
    for edgenumstr in sorted(resdict["n_edge_to_noniso_graphs"].keys(), key=lambda x: int(x)):
        info = resdict["n_edge_to_noniso_graphs"][edgenumstr]
        edgenum = int(edgenumstr)
        n_graphs = int(info["n_graphs"])
        n_first_dags = int(info["n_first_dags"])
        edge_sections.append({
            "edge_count": edgenum,
            "n_graphs": n_graphs,
            "n_first_dags": n_first_dags,
            "config_indices": list(range(n_graphs))
        })
    return render_template(
        "show_specified_class.html",
        session_id=session_id,
        is_irreducible=is_irreducible,
        n_L_noniso_graphs=n_L_noniso_graphs,
        n_dags=n_dags,
        edge_sections=edge_sections,
        is_search_finished=resdict["is_search_finished"],
        is_L_rename_deduplicate_finished=resdict["is_L_rename_deduplicate_finished"],
    )



if __name__ == "__main__":
    app.run(debug=True)
