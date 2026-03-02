from benchopt import BasePlot


class Plot(BasePlot):
    name = "Communication Time"
    type = "boxplot"
    options = {
        "objective": ...,
        "dataset": ...,
        "record_device": ["cpu", "gpu"],
    }

    def plot(self, df, objective, dataset, record_device):
        df = df.query(f"objective_name == '{objective}' and dataset_name == '{dataset}'")
        if record_device == "gpu":
            objective_column = "objective_comm_time"
        else:
            objective_column = "objective_comm_time_cpu"
        plot_data = []
        for solver, df_filtered in df.groupby('solver_name'):
            y = [df_filtered[objective_column].values.tolist()]
            x = [solver]

            plot_data.append({
                "x": x,
                "y": y,
                "label": solver,
                "color": self.get_style(solver)["color"],
            })

        return plot_data

    def get_metadata(self, df, objective, dataset, record_device):
        return {
            "title": f"Communication Time\n{objective}\nData: {dataset}",
        }