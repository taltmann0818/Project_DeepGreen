import wandb
import wandb_workspaces.reports.v2 as wr
import logging
from typing import Dict, Any, Optional, List
import plotly.graph_objects as go
from pathlib import Path


class WandbReportGenerator:
    """
    Custom class for creating and publishing enterprise-grade W&B reports
    with comprehensive model performance analysis, metrics, and visualizations.
    """

    def __init__(self, project_name: str, run_name: str):
        """
        Initialize the W&B Report Generator

        Args:
            project_name: W&B project name
            run_name: Name of the W&B run to create report for
        """
        self.project_name = project_name
        self.run_name = run_name
        self.api = wandb.Api()
        self.target_run = None
        self.report = None

        # Initialize logging
        self.logger = logging.getLogger(__name__)

    def get_run_metrics(self, run_name: str):
        """
        Search for a run by name and extract specified metrics from run.summary
        Returns the run and its metrics, plus the previous run's metrics
        """
        runs = self.api.runs(self.project_name)
        target_run = None
        target_run_index = None

        # Find the target run
        for i, run in enumerate(runs):
            if run_name in run.name or run.name == run_name:
                target_run = run
                target_run_index = i
                break

        if target_run is None:
            return None, None, None, None

        # Get previous run (runs are sorted with earliest first, so previous is at index-1)
        previous_run = None
        if target_run_index > 0:
            previous_run = runs[target_run_index - 1]

        # Extract metrics from target run
        target_metrics = {}
        metrics_to_extract = ['train_loss_epoch', 'val_loss', 'val_MAE', 'val_MAPE', 'val_RMSE', 'val_SMAPE']

        for metric in metrics_to_extract:
            target_metrics[metric] = target_run.summary.get(metric, 'N/A')

        # Extract metrics from previous run
        previous_metrics = {}
        if previous_run:
            for metric in metrics_to_extract:
                previous_metrics[metric] = previous_run.summary.get(metric, 'N/A')

        return target_run, target_metrics, previous_run, previous_metrics

    def create_metrics_markdown_table(self, target_run, target_metrics, previous_run, previous_metrics):
        """
        Create a markdown table comparing current and previous run metrics with percent change
        """
        markdown = "## Model Training Metrics\n\n"

        # Table header
        markdown += "| Metric | Current Run | Previous Run | % Change |\n"
        markdown += "|--------|-------------|-------------|----------|\n"

        # Add run names as first row
        current_run_name = target_run.name if target_run else 'N/A'
        previous_run_name = previous_run.name if previous_run else 'N/A'
        markdown += f"| **Run Name** | {current_run_name} | {previous_run_name} | - |\n"

        # Add metrics rows
        metrics_to_show = ['train_loss_epoch', 'val_loss', 'val_MAE', 'val_MAPE', 'val_RMSE', 'val_SMAPE']

        for metric in metrics_to_show:
            current_value = target_metrics.get(metric, 'N/A')
            previous_value = previous_metrics.get(metric, 'N/A') if previous_metrics else 'N/A'

            # Calculate percent change
            percent_change = 'N/A'
            if (isinstance(current_value, (int, float)) and 
                isinstance(previous_value, (int, float)) and 
                previous_value != 0):
                change = ((current_value - previous_value) / previous_value) * 100
                percent_change = f"{change:+.2f}%"

            # Format numeric values to 4 decimal places if they're numbers
            current_formatted = f"{current_value:.4f}" if isinstance(current_value, (int, float)) else current_value
            previous_formatted = f"{previous_value:.4f}" if isinstance(previous_value, (int, float)) else previous_value

            markdown += f"| {metric} | {current_formatted} | {previous_formatted} | {percent_change} |\n"

        return markdown

    def create_hyperparameters_markdown_table(self, target_run):
        """
        Create a markdown table showing key model training hyperparameters
        """
        markdown = "\n## Model Training Hyperparameters\n\n"

        # Key hyperparameters to display
        key_params = [
            'epochs', 'batch_size', 'learning_rate', 'dropout', 'weight_decay',
            'hidden_size', 'lstm_layers', 'attention_head_size', 'gradient_clip',
            'years', 'prediction_window', 'max_encoder_length', 'optimizer',
            'precision', 'accelerator', 'early_stopping_patience'
        ]

        # Table header
        markdown += "| Parameter | Value |\n"
        markdown += "|-----------|-------|\n"

        # Add hyperparameters
        config = target_run.config
        for param in key_params:
            if param in config:
                value = config[param]
                # Format boolean values
                if isinstance(value, bool):
                    value = str(value)
                # Format lists (truncate if too long)
                elif isinstance(value, list) and len(str(value)) > 100:
                    value = f"[{len(value)} items]"
                # Format dictionaries (truncate if too long)
                elif isinstance(value, dict) and len(str(value)) > 100:
                    value = f"{{{len(value)} items}}"

                markdown += f"| {param} | {value} |\n"

        # Add notes if available
        if 'notes' in config and config['notes']:
            markdown += f"\n**Notes:** {config['notes']}\n"

        return markdown

    def add_training_plots_to_report(self, target_run):
        """
        Add model training plot artifacts from the run to the report
        """
        files = target_run.files()
        plot_files = [f for f in files if f.name.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf'))]

        if plot_files:
            # Add a markdown section for training plots
            plots_markdown = "\n## Model Training Plots and Visualizations\n\n"
            self.report.blocks.append(wr.MarkdownBlock(text=plots_markdown))

            # Group plots by type
            prediction_plots = [f for f in plot_files if 'prediction_plot' in f.name]
            importance_plots = [f for f in plot_files if 'variable_importance' in f.name]

            # Add prediction plots
            if prediction_plots:
                self.report.blocks.append(wr.MarkdownBlock(text="### Prediction Plots\n"))
                for plot_file in sorted(prediction_plots, key=lambda x: x.name):
                    try:
                        image = wr.Image(plot_file.url)
                        self.report.blocks.append(image)
                        self.logger.info(f"Added prediction plot: {plot_file.name}")
                    except Exception as e:
                        self.logger.warning(f"Could not add plot {plot_file.name}: {e}")

            # Add variable importance plots using markdown table for compact layout
            if importance_plots:
                self.report.blocks.append(wr.MarkdownBlock(text="### Variable Importance Plots\n"))
                try:
                    # Create a 2-column markdown table for importance plots
                    markdown_content = "| Plot 1 | Plot 2 |\n|--------|--------|\n"

                    sorted_importance = sorted(importance_plots, key=lambda x: x.name)
                    for i in range(0, len(sorted_importance), 2):
                        plot1 = sorted_importance[i]
                        plot2 = sorted_importance[i+1] if i+1 < len(sorted_importance) else None

                        # Clean plot names
                        plot1_name = plot1.name.replace('media/images/', '').replace('.png', '').replace('_', ' ').title()
                        plot1_md = f"**{plot1_name}**<br>![{plot1_name}]({plot1.url})"

                        if plot2:
                            plot2_name = plot2.name.replace('media/images/', '').replace('.png', '').replace('_', ' ').title()
                            plot2_md = f"**{plot2_name}**<br>![{plot2_name}]({plot2.url})"
                        else:
                            plot2_md = ""

                        markdown_content += f"| {plot1_md} | {plot2_md} |\n"

                    table_block = wr.MarkdownBlock(text=markdown_content)
                    self.report.blocks.append(table_block)
                    self.logger.info(f"Created markdown table with {len(importance_plots)} importance plots")

                except Exception as e:
                    self.logger.warning(f"Could not create table for importance plots: {e}")
                    # Fallback to individual images
                    for plot_file in sorted(importance_plots, key=lambda x: x.name):
                        try:
                            image = wr.Image(plot_file.url)
                            self.report.blocks.append(image)
                            self.logger.info(f"Added importance plot (fallback): {plot_file.name}")
                        except Exception as e2:
                            self.logger.warning(f"Could not add plot {plot_file.name}: {e2}")

    def create_wandb_report(self, llm_summary: str, benchmark_plots: Dict[str, Any], 
                               wandb_run):
        """
        Create a comprehensive enterprise-grade report with all components

        Args:
            llm_summary: LLM-generated analysis summary
            benchmark_plots: Dictionary of benchmark plots to include
            wandb_run: Active W&B run for logging artifacts
            export_path: Optional path to export HTML report
        """
        try:
            # Get run metrics and create the report
            target_run, target_metrics, previous_run, previous_metrics = self.get_run_metrics(self.run_name)

            if not target_run:
                self.logger.error(f"Could not find run: {self.run_name}")
                return None

            self.target_run = target_run

            # Create the main report
            self.report = wr.Report(
                project=self.project_name,
                title=f"Model Performance Report - {self.run_name}",
                description=f"Comprehensive analysis of model performance, benchmarks, and training metrics for run {self.run_name}",
                width='fluid'
            )

            # 1. Executive Summary (LLM Analysis)
            executive_summary = f"# Executive Summary\n\n{llm_summary['exec_summary']}\n\n"
            self.report.blocks.append(wr.MarkdownBlock(text=executive_summary))

            performance_analysis = f"# 1.1 Performance Analysis\n\n{llm_summary['performance_analysis']}\n\n"
            self.report.blocks.append(wr.MarkdownBlock(text=performance_analysis))

            risk_assessment = f"# 1.2 Risk Assessment\n\n{llm_summary['risk_assessment']}\n\n"
            self.report.blocks.append(wr.MarkdownBlock(text=risk_assessment))

            recommendations = f"# 1.3 Recommendation(s)\n\n{llm_summary['recommendations']}\n\n"
            self.report.blocks.append(wr.MarkdownBlock(text=recommendations))

            # 2. Backtesting Results
            backtesting_header = "# 2 Backtesting Result(s)\n"
            self.report.blocks.append(wr.MarkdownBlock(text=backtesting_header))
            self.report.blocks.append(
                wr.PanelGrid(
                    panels=[
                        wr.CustomChart(
                            chart_name="performance_comparison"
                        )
                    ]
                )
            )

            self.report.blocks.append(
                wr.PanelGrid(
                    panels=[
                        wr.CustomChart(
                            chart_name="metrics_comparison"
                        )
                    ]
                )
            )

            # 3. Model Training Metrics
            metrics_table = self.create_metrics_markdown_table(target_run, target_metrics, previous_run, previous_metrics)
            metrics_markdown = f"# 3.1 Training Performance\n\n{metrics_table}\n\n"
            self.report.blocks.append(wr.MarkdownBlock(text=metrics_markdown))
                        # 5. Training Plots and Visualizations
            self.add_training_plots_to_report(target_run)

            # 3. Model Training Hyperparameters
            hyperparams_table = self.create_hyperparameters_markdown_table(target_run)
            hyperparams_markdown = f"# 3.2 Training Parameters\n\n{hyperparams_table}\n\n"
            self.report.blocks.append(wr.MarkdownBlock(text=hyperparams_markdown))


            # Save the report
            self.report.save()

            return self.report

        except Exception as e:
            self.logger.error(f"Error creating W&B report: {e}")
            return None
