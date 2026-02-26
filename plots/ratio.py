from benchopt import BasePlot


class Plot(BasePlot):
    name = "Communication Ratio"
    type = "scatter"
    options = {
        "dataset": ['simulated']
    }

    def plot(self, df, dataset):
        df = df[df["dataset_name"].str.contains(dataset, na=False)]

        plots = []
        for k in range(len(df)):
            solver= df["solver_name"].iloc[k]
            y = df["objective_comm_ratio"].iloc[k]
            solver_name = solver.split("[")[0]
            batch_size = solver.split("batch_size=")[1].split(",")[0]
            d1 = df["dataset_name"].iloc[k]
            d1 = d1.split("d1=")[1].split(",")[0]
            solver_label = f"{solver_name}[batch_size={batch_size},d1={d1}]"
            curve_data = {
                "x": [int(batch_size) / int(d1)],
                "y": [y],
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
