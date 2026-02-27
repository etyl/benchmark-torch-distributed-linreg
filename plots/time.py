from benchopt import BasePlot


class Plot(BasePlot):
    name = "Communication Time"
    type = "boxplot"
    options = {
        "objective": ...,
        "dataset": ...,
    }

    def plot(self, df, objective, dataset):
        df = df.query(f"objective_name == '{objective}' and dataset_name == '{dataset}'")

        plot_data = []
        for solver, df_filtered in df.groupby('solver_name'):
            y = [df_filtered['objective_comm_time'].values.tolist()]
            x = [solver]

            plot_data.append({
                "x": x,
                "y": y,
                "label": solver,
                "color": self.get_style(solver)["color"],
            })

        return plot_data

    def get_metadata(self, df, objective, dataset):
        return {
            "title": f"Communication Time\n{objective}\nData: {dataset}",
        }