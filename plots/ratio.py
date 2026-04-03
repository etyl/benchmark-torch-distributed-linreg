from benchopt import BasePlot
import re


class Plot(BasePlot):
    name = "Communication Ratio"
    type = "boxplot"
    options = {
    }

    def plot(self, df):
        plot_data = []

        solvers = df['solver_name'].unique()
        datasets = df['dataset_name'].unique()
        all_reduce_solvers = [solver for solver in solvers if 'all-reduce' in solver]
        all_reduce_solvers.sort(key=lambda s: int(s.split('slurm_nodes=')[1].split(']')[0]))

        for solver in all_reduce_solvers:
            n_nodes = re.search(r'slurm_nodes=(\d+)', solver).group(1)
            solver_params = solver.split('[')[1]
            df_solver = df[df['solver_name'].str.contains(solver_params)]

            for dataset in datasets:
                df_filtered = df_solver[df_solver['dataset_name'] == dataset]

                communication_times = df_filtered[df_filtered["solver_name"].str.contains("all-reduce")]["objective_comm_time"].dropna().values.tolist()
                run_times = df_filtered[df_filtered["solver_name"].str.contains("all-reduce-nolock")]["objective_run_time"].dropna().values.tolist()
                run_times_ddp = df_filtered[df_filtered["solver_name"].str.contains("ddp")]["objective_run_time"].dropna().values.tolist()
                communication_ratio = [comm_time / run_time for comm_time, run_time in zip(communication_times, run_times) if run_time > 0]
                comm_times_ddp = [communication_times[i] - (run_times[i] - run_times_ddp[i]) for i in range(len(run_times_ddp))]
                communication_ratio_ddp = [comm_time / run_time for comm_time, run_time in zip(comm_times_ddp, run_times_ddp) if run_time > 0]

                dataset_label = dataset.split('[')[0]
                plot_data.append({
                    "y": [communication_ratio],
                    "x": [f"{n_nodes} nodes"],
                    "label": f"{dataset_label}",
                    "color": self.get_style(dataset_label)["color"],
                })
                plot_data.append({
                    "y": [communication_ratio_ddp],
                    "x": [f"{n_nodes} nodes (DDP)"],
                    "label": f"{dataset_label} (DDP)",
                    "color": self.get_style(f"{dataset_label} (DDP)")["color"],
                })

        return plot_data

    def get_metadata(self, df):
        return {
            "title": "Communication Ratio",
            "ylabel": "Communication Time (% of Total Time)",
        }
