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

    def add_benchmark_plots_to_report(self, plots: Dict[str, go.Figure], wandb_run):
        """
        Add benchmark plots to the report by first logging them as PNG artifacts,
        then adding them to the report as wr.Image URIs
        """
        if not plots:
            self.logger.warning("No benchmark plots provided")
            return

        plot_uris = {}

        try:
            # Log plots as PNG artifacts first
            for plot_name, fig in plots.items():
                # Save plot as PNG temporarily
                temp_path = f"/tmp/{plot_name}.png"
                fig.write_image(temp_path, format="png", width=1200, height=800)

                # Log as artifact
                artifact = wandb.Artifact(f"{plot_name}_plot", type="plot")
                artifact.add_file(temp_path)
                wandb_run.log_artifact(artifact)

                # Get the artifact URI for the report
                plot_uris[plot_name] = artifact.get_path(f"{plot_name}.png").download()

                self.logger.info(f"Logged benchmark plot as PNG artifact: {plot_name}")

        except Exception as e:
            self.logger.error(f"Error logging benchmark plots as artifacts: {e}")

        # Add benchmark results section to report
        benchmark_markdown = "\n## Benchmark Results\n\n"
        benchmark_markdown += "### Performance Comparison\n"
        benchmark_markdown += "The following charts show the comparative performance of all tested models:\n\n"

        self.report.blocks.append(wr.MarkdownBlock(text=benchmark_markdown))

        # Add plots to report using PNG URIs with wr.Image()
        for plot_name, fig in plots.items():
            try:
                if plot_name in plot_uris:
                    # Use PNG artifact URI with wr.Image()
                    image = wr.Image(plot_uris[plot_name])
                    self.report.blocks.append(image)
                    self.logger.info(f"Added benchmark plot to report as PNG image: {plot_name}")
            except Exception as e:
                self.logger.warning(f"Could not add benchmark plot {plot_name} to report: {e}")

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

    def create_enterprise_report(self, llm_summary: str, benchmark_plots: Dict[str, go.Figure], 
                               wandb_run, export_path: Optional[str] = None):
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
                title=f"Enterprise Model Performance Report - {self.run_name}",
                description=f"Comprehensive analysis of model performance, benchmarks, and training metrics for run {self.run_name}",
                width='fluid'
            )

            # 1. Executive Summary (LLM Analysis)
            executive_summary = f"# Executive Summary\n\n{llm_summary}\n\n"
            self.report.blocks.append(wr.MarkdownBlock(text=executive_summary))

            # 2. Model Training Metrics
            metrics_markdown = self.create_metrics_markdown_table(target_run, target_metrics, previous_run, previous_metrics)
            self.report.blocks.append(wr.MarkdownBlock(text=metrics_markdown))

            # 3. Benchmark Results (including plots)
            self.add_benchmark_plots_to_report(benchmark_plots, wandb_run)

            # 4. Model Training Hyperparameters
            hyperparams_markdown = self.create_hyperparameters_markdown_table(target_run)
            self.report.blocks.append(wr.MarkdownBlock(text=hyperparams_markdown))

            # 5. Training Plots and Visualizations
            self.add_training_plots_to_report(target_run)

            # Save the report
            self.report.save()
            self.logger.info("✅ Enterprise report created and saved to W&B")

            # Export as HTML if path provided
            if export_path:
                self.export_html_report(export_path)

            return self.report

        except Exception as e:
            self.logger.error(f"Error creating enterprise report: {e}")
            return None

    def export_html_report(self, export_path: str):
        """
        Export the report as HTML to the specified path

        Args:
            export_path: Path where to save the HTML report
        """
        try:
            if not self.report:
                self.logger.error("No report available to export")
                return

            # Create export directory if it doesn't exist
            export_dir = Path(export_path).parent
            export_dir.mkdir(parents=True, exist_ok=True)

            # Get report URL and save as HTML
            report_url = self.report.url

            # Create a simple HTML wrapper with the report content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Enterprise Model Performance Report - {self.run_name}</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .report-header {{ text-align: center; margin-bottom: 30px; }}
                    .report-link {{ margin: 20px 0; padding: 10px; background-color: #f0f0f0; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="report-header">
                    <h1>Enterprise Model Performance Report</h1>
                    <h2>Run: {self.run_name}</h2>
                    <p>Generated on: {wandb.util.generate_id()}</p>
                </div>

                <div class="report-link">
                    <p><strong>Interactive W&B Report:</strong> <a href="{report_url}" target="_blank">{report_url}</a></p>
                    <p>This report contains comprehensive model analysis including metrics, benchmarks, and visualizations.</p>
                </div>

                <iframe src="{report_url}" width="100%" height="800px" frameborder="0"></iframe>
            </body>
            </html>
            """

            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info(f"✅ HTML report exported to: {export_path}")

        except Exception as e:
            self.logger.error(f"Error exporting HTML report: {e}")
