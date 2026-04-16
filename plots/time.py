from benchopt import BasePlot


class Plot(BasePlot):
    name = "Metrics"
    type = "bar_chart"
    options = {
        "objective": ...,
        "dataset": ...,
        "metric": ["comm_time", "comm_time_cpu", "run_time"],
    }

    def plot(self, df, objective, dataset, metric):
        df = df.query(f"objective_name == '{objective}' and dataset_name == '{dataset}'")
        objective_column = f"objective_{metric}"
        plot_data = []

        solvers = df['solver_name'].unique()
        solver_params = [solver for solver in solvers if 'ddp' in solver]
        total_solver_params = [solver for solver in solvers if 'ddp' in solver]
        total_solver_params.sort(key=lambda s: int(s.split('slurm_nodes=')[1].split(']')[0]))

        for solver in total_solver_params:
            solver_params = solver.split('[')[1]
            total_solvers = [s for s in solvers if solver_params in s]

            for s in total_solvers:
                solver_type = s.split('[')[0]
                df_filtered = df[df['solver_name'] == s]
                if objective_column not in df_filtered.columns:
                    continue

                y = df_filtered[objective_column].dropna().values.tolist()
                if len(y) == 0:
                    continue

                plot_data.append({
                    "y": y,
                    "label": s,
                    "color": self.get_style(solver_type)["color"],
                })

        return plot_data

    def get_metadata(self, df, objective, dataset, metric):
        return {
            "title": f"Communication Time\n{objective}\nData: {dataset}",
        }