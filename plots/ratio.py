from benchopt import BasePlot


class ObjectiveCurvePlot(BasePlot):
    name = "Communication Ratio"
    type = "scatter"
    options = {
        "dataset": ['simulated']
    }

    def plot(self, df, dataset):
        df = df.filter(df["dataset_name"].str.contains(dataset))

        plots = []
        for solver, df_filtered in df.groupby('solver_name'):
            medians = df_filtered.groupby('stop_val').median(numeric_only=True)
            y = medians["comm_ratio"].values.tolist()
            x = medians["batch_ratio"].values.tolist()
            solver_name = solver.split("[")[0]
            batch_size = solver.split("batch_size=")[1].split(",")[0]
            d1 = df_filtered["dataset_name"].iloc[0]
            d1 = d1.split("d1=")[1].split(",")[0]
            solver_label = f"{solver_name}[batch_size={batch_size},d1={d1}]"
            curve_data = {
                "x": x,
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
