import pickle
import base64
import os

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor

rdDepictor.SetPreferCoordGen(True)

import numpy as np

import networkx as nx

import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot

from pathlib import Path
from b01_utility import *

# Adapted from code by Vincent F. Scalfani (BSD 3-Clause License)
# Original Copyright (c) 2022, Vincent F. Scalfani
# Modifications made by Elliot Chan, 2025
# Modifications of note: modularized the visualization steps into a class, use of plotly instead of matplotlib, with interactive maps and dynamic sizing + node spacing
# continued: base64 molecule image encoding (plotly-specific), graph cosmetic customizability (node sizing, options of displaying 2D images or nodes, coloring), error handling
# added weight method, allowing for graphs to be based on Tanimoto, Tanimoto MCS, or a hybrid weight, with biases towards overall or MCSubstructure determined in the config file


class ChemNet:
    def __init__(self, model_name, network_type, weight_method):
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")

        # get validated config
        self.cfg = validate_config()

        # handle improper weighing method
        if weight_method not in ['tan_sim', 'tan_mcs', 'hybrid']:
            raise ChemNetError(f"{weight_method} is not a valid weight method. Choose between: 'tan_sim', 'tan_mcs', 'hybrid'")
        self.weight_method = weight_method

        # second validation of model_name
        # -> make sure it's got a file in the network_folder database - meaning it's been run on CSNodes
        self.model_name = model_name

        self.network_path = Path(self.cfg['database']) / self.cfg['network_folder']
        target_folder_name = f"{self.model_name}_network_database"
        folders = os.listdir(self.network_path)
        if target_folder_name not in folders:
            raise ChemNetError(f"{self.model_name} does not exist in the network data storage: Check if this is a valid stored model or if it's saved data has been processed by a05")

        # tanimoto similarity calculation dataframe filepath
        self.tandata_path  = self.network_path / target_folder_name  # graph folder)

        # handle improper network type
        self.network_type = network_type
        if self.network_type not in ['optima', 'optimized']:
            raise ValueError(f"Invalid network_type: {self.network_type}. Must be either 'optima' or 'optimized'")


        # initialize settings
        self.scaling_constant = 0.4
        self.top_percent = 0
        self.k_dist = 0
        self.a = self.cfg['tanimoto_bias']

        if (self.a > 1) or (self.a < 0):
            raise ValueError(f"Config's 'tanimoto_bias' needs to be between 0 and 1, right now it's at {self.a}")

        # toggle settings from cfg
        self.node_opacity = 0 if self.cfg['transparent_nodes'] is True else 1
        self.node_size = 10 if self.cfg['node_toggle'] is True else 0
        self.label_size = 10 if self.cfg['label_toggle'] is True else 0
        if not isinstance(self.cfg['colorscale'], str):
            raise ChemNetError("Invalid colorscale configuration")

        # we'll need this later
        self.node_count = None

        # make folder within prediction / result file to upload html files of CSN graph
        self.upload_folder = Path(self.cfg['predictions']) / f"{self.model_name}_CSN_graphs"
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        self.savepath = self.upload_folder / f"{self.model_name}_{self.network_type}_CSN_graph.html"

    def graph_data(self):
        """Determine greatest % of subset data to display"""
        subsets_filepath = Path(self.tandata_path) / f"{self.model_name}_{self.network_type}_subsets.pkl"
        nodes_filepath = Path(self.tandata_path) / f"{self.model_name}_{self.network_type}_node_data.pkl"

        # safely load pickle files - handles file.exist and st_size
        subsets = self.pkl_safeload(subsets_filepath)
        node_data = self.pkl_safeload(nodes_filepath)


        # handle empty / insufficient data
        if not subsets:
            raise ChemNetError("No tanimoto similarity data available for CSN visualization")
        if len(node_data) < 2:
            raise ChemNetError("Insufficient nodes for network visualization: min 2 nodes needed")

        # if data is too high - don't process it: may need to adjust this threshold after enough testing
        if len(subsets) > 1000000:
            raise ChemNetError(f"Dataset too large for visualization: {len(subsets)} pairs. Consider trimming the optima / optimized data files down first.")

        scored_subsets = {}
        # ['tan_sim', 'tan_mcs', 'hybrid']
        for key, score in subsets.items():
            try:
                if self.weight_method == 'tan_sim':
                    plot_value = score['tan_similarity']
                elif self.weight_method == 'tan_mcs':
                    plot_value = score['tan_mcs']
                elif self.weight_method == 'hybrid':
                    tan_sim = score['tan_similarity']
                    tan_mcs = score['tan_mcs']
                    plot_value = (self.a * tan_sim) + ((1 - self.a) * tan_mcs)
                else:
                    raise ChemNetError(f"Invalid weight method: {self.weight_method}")

                scored_subsets[key] = plot_value
            except Exception as e:
                raise ChemNetError(f"Unexpected error generating scored subsets: {e}")

        sorted_subset = sorted(scored_subsets.items(), key=lambda x: x[1], reverse=True)
        # scored subsets is a dictionary, with key as molecule pair index and value as similarity score
        # sorted subset is a list of tuples, with value 1 as idx and value 2 as the similarity score

        # [[DATA SCALING HERE]]
        # adaptive number of edges to amount of data for readability
        n_total = len(sorted_subset)
        if n_total > 10000:
            self.top_percent = 0.03
            self.k_dist = 0.5
        elif n_total > 5000:
            self.top_percent = 0.05
            self.k_dist = 0.7
        elif n_total > 2000:
            self.top_percent = 0.10
            self.k_dist = 0.9
        elif n_total > 1000:
            self.top_percent = 0.15
            self.k_dist = 1.0
        elif n_total > 500:
            self.top_percent = 0.3
            self.k_dist = 1.2
        else:
            self.top_percent = 1
            self.k_dist = 1.2

        # get the greatest % subsets that we will display in the CSN
        top_n = int(len(sorted_subset) * self.top_percent)
        top_keys = [key for (key, score) in sorted_subset[:top_n]]
        csn_subsets = {key: subsets[key] for key in top_keys}

        # adapt sizing according to how dense the network actually is
        density = n_total / len(node_data.keys())
        if density > 0.1:
            self.k_dist *= 1.5
        elif density > 0.05:
            self.k_dist *= 1.2

        self.interactive_network(csn_subsets, node_data, scored_subsets)
        return True


    def interactive_network(self, subsets, nodes, scored_subsets):
        """Uses data generated by get_graph to visualize the chemical space network"""
        try:
            G1 = nx.Graph()

            for key, value in subsets.items():
                G1.add_edge(value['smi1'], value['smi2'], weight=scored_subsets[key])

            custom_label = {}
            for smile, pIC50 in nodes.items():
                G1.add_node(smile, ID=smile, pIC50=pIC50)  # add node data
                custom_label[smile] = str(smile)

            # [[DISTANCING BETWEEN NODES HERE]]
            # network plot -> determine how close or far each node is going to be from each other
            pos = nx.spring_layout(G1, self.k_dist, seed=40)
        except nx.NetworkXPointlessConcept:
            raise ChemNetError(f"Could not create layout for {self.network_type} compounds")
        except nx.NetworkXError as e:
            raise ChemNetError(f"Network layout calculation failed: {e}")
        except Exception as e:
            raise ChemNetError(f"Unexpected error during network setup: {e}")

        # [[COLORMAPPING]] -> need to fix this
        # colormap the potency values using percentile-based strategy
        pic_values = np.array([data['pIC50'] for node, data in G1.nodes(data=True)])
        color_values = np.zeros_like(pic_values)

        for i, value in enumerate(pic_values):
            percentile_rank = sum(pic_values <= value) / len(pic_values) * 100
            color_values[i] = percentile_rank

        cmin, cmax = 0, 100
        cmap = px.colors.sample_colorscale(self.cfg['colorscale'], np.linspace(0, 1, 100))


        # [[MAKE EDGE TRACES]]
        edge_x = []
        edge_y = []
        for edge in G1.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        # plot edges
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='grey'),
            hoverinfo='none',
            mode='lines'
        )


        # [[MAKE NODES AND HOVER TEXT]]
        node_x = []
        node_y = []
        node_images = []
        node_colours = []
        node_hover_text = []
        hover_image = {}

        failed_images = 0

        # make nodes - god this is hard
        self.node_count = len(G1.nodes())
        smile_2_color = {}
        for i, (smile, data) in enumerate(G1.nodes(data=True)):
            smile_2_color[smile] = color_values[i]
            color_idx = min(int(smile_2_color[smile]), len(cmap) - 1)
            node_colours.append(color_values[i])
            x, y = pos[smile]
            node_x.append(x)
            node_y.append(y)

            rgb_string = cmap[color_idx]
            custom_rgb=rgb_string.replace('rgb(', '').replace(')', '')
            r, g, b = map(int, custom_rgb.split(', '))
            normalized_custom_rgb = (r/255.0, g/255.0, b/255.0)

            # generate molecule image for hover-text
            mol_image = self.molecular_image(smile, normalized_custom_rgb, size=(250, 250))

            # handle missing molecular images
            if mol_image is None:
                failed_images += 1
            if failed_images > len(G1.nodes()) * 0.1: # if more than 10% of molecules fail to generate images...
                print(f"Warning: {failed_images} molecular images failed to generate")


            # [[BUILD HOVER TEXT TO BE DISPLAYED AND 2D MOL IMAGES]]
            if mol_image:
                hover_image[smile] = mol_image
                hover_text = f"""
                                <b>SMILES</b> {smile}<br>
                                <b>pIC50:</b> {data["pIC50"]:.4f}<br>
                                <b>Percentile:</b> {color_values[i]:.4f}%<br>"""
                node_hover_text.append(hover_text)
            else:   # if there is no molecular image -> hover text still but with a note saing image generation failed
                hover_text = f"""
                                <b>SMILES</b> {smile}<br>
                                <b>pIC50:</b> {data["pIC50"]:.4f}<br>
                                <b>Percentile:</b> {color_values[i]:.4f}%<br>
                                <b>Note:</b> Image generation failed<br>"""
                node_hover_text.append(hover_text)
                raise ValueError("No structure detected, analyze dataset and see if a molecule is missing")

            molecule_size = self.adaptive_sizing(pos)

            if self.cfg['2D_molecules'] is True:
                node_images.append(dict(
                    source=f"data:image/png;base64,{mol_image}",
                    xref="x",
                    yref="y",
                    x=node_x[i],
                    y=node_y[i],
                    sizex=molecule_size,
                    sizey=molecule_size,
                    xanchor="center",
                    yanchor="middle",
                    layer="above"
                ))

        # NODES TO BE USED IN GOFIGURE
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            opacity=self.node_opacity,
            mode='markers+text' if self.label_size else 'markers',
            hoverinfo='text',
            text=node_hover_text,
            textposition='middle center',
            textfont=dict(size=8),
            marker=dict(
                size=self.node_size,
                color=node_colours,
                colorscale=self.cfg['colorscale'],
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(title="pIC50 Potency % Rank"),
                line=dict(width=1, color='black')
            )
        )

        if self.weight_method == 'hybrid':
            title_text = f"Chemical Space Network Graph: {self.model_name} {self.network_type} compounds - Top {self.top_percent}% similarity | {self.weight_method} | {self.a} bias"
        else:
            title_text = f"Chemical Space Network Graph: {self.model_name} {self.network_type} compounds - Top {self.top_percent}% similarity | {self.weight_method}"

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=dict(
                                text=title_text,
                                font=dict(size=25)),
                            images=node_images,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))

        # handle errors in saving file
        try:
            plot(fig, filename=str(self.savepath) , auto_open=True)

        except Exception as e:  # handle output errors -> backup file location
            try:
                fallback_savepath = Path.cwd() / f"{self.model_name}_{self.network_type}_fallbackCSNgraph.html"
                plot(fig, filename = str(fallback_savepath), auto_open=False)
                print(f"Something went wrong with the original save's filepath: {e} \nFallback save location: {fallback_savepath}")
            except Exception:
                raise ChemNetError(f"Failed to save network visualization")


    @staticmethod
    def molecular_image(smiles, rgb=None, size=(300, 300)):
        """
        convert the molecule's smiles into base64 image that plotly can use
        :return: molecules b64-encoded 2D image base64
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # draw molecule
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            drawer.drawOptions().clearBackground = False
            drawer.drawOptions().addStereoAnnotation = False

            if rgb:
                # get indices for atoms and bonds so we can highlight them
                atoms = [atom.GetIdx() for atom in mol.GetAtoms()]
                bonds = [bond.GetIdx() for bond in mol.GetBonds()]

                # highlighting stuff
                drawer.drawOptions().fillHighlights = True
                drawer.drawOptions().setHighlightColour(rgb[:3] + (0.3,))
                drawer.drawOptions().highlightBondWidthMultiplier = 5
                drawer.drawOptions().highlightRadius = 0.3

                rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=atoms, highlightBonds=bonds)
            else:
                rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()

            # convert from png to base64 so plotly can use it
            mol_png = drawer.GetDrawingText()
            encoded_mol = base64.b64encode(mol_png).decode()

            return encoded_mol
        except Exception as e:
            # keep track of the error but don't fail the entire visualization process
            print(f"Warning: Could not generate image for {smiles}: {e}")
            return None


    def adaptive_sizing(self, pos):
        """use n_total and layout space to calculate molecule size"""
        # get bounding box of layout
        x_bound = [coord[0] for coord in pos.values()]
        y_bound = [coord[1] for coord in pos.values()]

        x_range = max(x_bound) - min(x_bound)
        y_range = max(y_bound) - min(y_bound)

        layout_area = x_range * y_range

        # base size
        area_per_node = layout_area / self.node_count

        # calculate base size
        base_size = np.sqrt(area_per_node)

        molecule_size = base_size * self.scaling_constant

        return molecule_size


    def pkl_safeload(self, filepath):
        """Safely load pickle file with validation"""
        file_desc = str(f"{self.model_name} {self.network_type} Tanimoto similarity data file")

        try:
            if not filepath.exists():
                raise ChemNetError(f"{file_desc} not found: {filepath}")

            if filepath.stat().st_size == 0:
                raise ChemNetError(f"{file_desc} is empty: {filepath} ")

            with open(filepath, 'rb') as handle:
                data = pickle.load(handle)

            if not data:
                raise ChemNetError(f"{file_desc} is empty: {filepath}")

            return data

        except (pickle.PickleError, EOFError) as e:
            raise ChemNetError(f"Corrupted {file_desc}: {filepath} | {e}") from e
        except PermissionError:
            raise ChemNetError(f"Permission denied accessing {file_desc}: {filepath}")
        except Exception as e:
            raise ChemNetError(f"Failed to load {file_desc}: {filepath} | {e}")