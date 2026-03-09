from benchopt import BasePlot
import re


class Plot(BasePlot):
    name = "Communication Ratio"
    type = "scatter"
    options = {
        "dataset": ['mlp'],
        "batch_size": ["local", "global"],
    }

    def plot(self, df, dataset, batch_size):
        df = df[df["dataset_name"].str.contains(dataset, na=False)]

        plots = []
        for solver, df_solver in df.groupby("solver_name"):
            for dataset_name, df_dataset in df_solver.groupby("dataset_name"):
                if "objective_comm_ratio" not in df_dataset.columns:
                    continue
                y = df_dataset["objective_comm_ratio"].values.tolist()
                solver_name = solver.split("[")[0]
                global_batch_size = int(re.search(r"batch_size=(\d+)", solver).group(1))
                n_nodes = int(re.search(r"slurm_nodes=(\d+)", solver).group(1))
                local_batch_size = global_batch_size // n_nodes
                if batch_size == "local":
                    batch_size_val = local_batch_size
                else:
                    batch_size_val = global_batch_size
                d = dataset_name.split("d=")[1].split(",")[0]
                solver_label = f"{solver_name}[batch_size={batch_size_val},d={d},nodes={n_nodes}]"
                curve_data = {
                    "x": [batch_size_val / int(d)] * len(y),
                    "y": y,
                    "label": solver_label,
                    **self.get_style(solver_label)
                }

                plots.append(curve_data)

        return plots

    def get_metadata(self, df, dataset, batch_size):
        title = f"Communication Ratio\nData: {dataset} "
        return {
            "title": title,
            "xlabel": "Batch Ratio",
            "ylabel": "Communication Ratio",
        }
