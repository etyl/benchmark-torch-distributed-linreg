from benchopt import BasePlot


class Plot(BasePlot):
    name = "Communication Ratio"
    type = "scatter"
    options = {
        "dataset": ['mlp'],
        "batch_size": ["local", "global"],
    }

    def plot(self, df, dataset):
        df = df[df["dataset_name"].str.contains(dataset, na=False)]

        plots = []
        for solver, df_solver in df.groupby("solver_name"):
            for dataset_name, df_dataset in df_solver.groupby("dataset_name"):
                if "objective_comm_ratio" not in df_dataset.columns:
                    continue
                y = df_dataset["objective_comm_ratio"].values.tolist()
                solver_name = solver.split("[")[0]
                global_batch_size = int(solver.split("batch_size=")[1].split(",")[0])
                n_nodes = int(solver.split("n_nodes=")[1].split(",")[0])
                local_batch_size = global_batch_size // n_nodes
                if self.batch_size == "local":
                    batch_size = local_batch_size
                else:
                    batch_size = global_batch_size
                d = dataset_name.split("d=")[1].split(",")[0]
                solver_label = f"{solver_name}[batch_size={batch_size},d={d},nodes={n_nodes}]"
                curve_data = {
                    "x": [int(batch_size) / int(d)] * len(y),
                    "y": y,
                    "label": solver_label,
                    **self.get_style(solver_label)
                }

                plots.append(curve_data)

        return plots

    def get_metadata(self, df, dataset):
        title = f"Communication Ratio\nData: {dataset} "
        return {
            "title": title,
            "xlabel": "Batch Ratio",
            "ylabel": "Communication Ratio",
        }
