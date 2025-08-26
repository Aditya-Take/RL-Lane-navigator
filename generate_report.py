#!/usr/bin/env python3
"""
HTML report generator for autonomous driving agent visualizations and results.
"""

import os
import json
from pathlib import Path
from datetime import datetime
import base64


def encode_image_to_base64(image_path):
    """Encode image to base64 for HTML embedding."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except:
        return None


def generate_html_report(plots_dir="./plots", results_dir="./evaluation_results", 
                        models_dir="./models", output_file="./training_report.html"):
    """Generate comprehensive HTML report."""
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous Driving Agent - Training Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }
        h3 {
            color: #2c3e50;
            margin-top: 25px;
        }
        .plot-section {
            margin: 20px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        .plot-container {
            text-align: center;
            margin: 15px 0;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .summary-section {
            background-color: #e8f4fd;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }
        .timestamp {
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
            margin-top: 30px;
        }
        .file-list {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .file-list ul {
            margin: 0;
            padding-left: 20px;
        }
        .file-list li {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 Autonomous Driving Agent - Training Report</h1>
        
        <div class="summary-section">
            <h3>📋 Report Summary</h3>
            <p>This report contains the complete training results, visualizations, and evaluation metrics for the vision-based autonomous driving agent using imitation learning and reinforcement learning.</p>
            <p><strong>Generated:</strong> {timestamp}</p>
        </div>
    """
    
    # Add training plots section
    plots_path = Path(plots_dir)
    if plots_path.exists():
        plot_files = list(plots_path.glob("*.png")) + list(plots_path.glob("*.jpg"))
        
        if plot_files:
            html_content += """
        <h2>📊 Training Visualizations</h2>
        <div class="file-list">
            <strong>Generated Plot Files:</strong>
            <ul>
            """
            
            for plot_file in plot_files:
                html_content += f"<li>{plot_file.name}</li>"
            
            html_content += """
            </ul>
        </div>
            """
            
            # Add individual plots
            for plot_file in plot_files:
                img_base64 = encode_image_to_base64(plot_file)
                if img_base64:
                    html_content += f"""
        <div class="plot-section">
            <h3>{plot_file.stem.replace('_', ' ').title()}</h3>
            <div class="plot-container">
                <img src="data:image/png;base64,{img_base64}" alt="{plot_file.name}">
            </div>
        </div>
                    """
    
    # Add evaluation results section
    results_path = Path(results_dir)
    if results_path.exists():
        results_file = results_path / "evaluation_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            html_content += """
        <h2>📈 Evaluation Results</h2>
        <div class="metrics-grid">
            """
            
            # Add performance metrics
            if 'mean_reward' in results:
                html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{results['mean_reward']:.2f}</div>
                <div class="metric-label">Mean Reward</div>
            </div>
                """
            
            if 'success_rate' in results:
                html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{results['success_rate']:.1%}</div>
                <div class="metric-label">Success Rate</div>
            </div>
                """
            
            if 'collision_rate' in results:
                html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{results['collision_rate']:.1%}</div>
                <div class="metric-label">Collision Rate</div>
            </div>
                """
            
            if 'off_road_rate' in results:
                html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{results['off_road_rate']:.1%}</div>
                <div class="metric-label">Off-road Rate</div>
            </div>
                """
            
            html_content += """
        </div>
            """
    
    # Add model information section
    models_path = Path(models_dir)
    if models_path.exists():
        html_content += """
        <h2>🤖 Model Information</h2>
        """
        
        model_types = ['cnn_imitation', 'ppo_driving', 'hybrid_agent']
        
        for model_type in model_types:
            model_path = models_path / model_type
            if model_path.exists():
                html_content += f"""
        <div class="plot-section">
            <h3>{model_type.replace('_', ' ').title()} Model</h3>
                """
                
                # Check for config file
                config_file = model_path / "config.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    html_content += "<div class='file-list'><strong>Configuration:</strong><ul>"
                    
                    if 'imitation_learning' in config:
                        il_config = config['imitation_learning']
                        html_content += f"<li>CNN Layers: {il_config['cnn']['conv_layers']}</li>"
                        html_content += f"<li>FC Layers: {il_config['cnn']['fc_layers']}</li>"
                        html_content += f"<li>Learning Rate: {il_config['training']['learning_rate']}</li>"
                    
                    if 'reinforcement_learning' in config:
                        rl_config = config['reinforcement_learning']
                        html_content += f"<li>PPO Learning Rate: {rl_config['ppo']['learning_rate']}</li>"
                        html_content += f"<li>Total Timesteps: {rl_config['training']['total_timesteps']:,}</li>"
                    
                    html_content += "</ul></div>"
                
                # Check for training history
                history_file = model_path / "training_history.json"
                if history_file.exists():
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                    
                    html_content += "<div class='file-list'><strong>Training History:</strong><ul>"
                    
                    if 'train_loss' in history:
                        html_content += f"<li>Training Epochs: {len(history['train_loss'])}</li>"
                        if history['train_loss']:
                            html_content += f"<li>Final Train Loss: {history['train_loss'][-1]:.4f}</li>"
                            html_content += f"<li>Final Val Loss: {history['val_loss'][-1]:.4f}</li>"
                    
                    html_content += "</ul></div>"
                
                html_content += "</div>"
    
    # Add conclusion
    html_content += """
        <h2>🎯 Conclusion</h2>
        <div class="summary-section">
            <p>This training report demonstrates the complete pipeline of the vision-based autonomous driving agent, from data collection through imitation learning and reinforcement learning to final evaluation.</p>
            <p>The system successfully combines multiple learning paradigms to create a robust and capable autonomous driving agent that can handle various traffic scenarios with domain randomization and multi-agent interactions.</p>
        </div>
        
        <div class="timestamp">
            Report generated on {timestamp}
        </div>
    </div>
</body>
</html>
    """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Write HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"✅ HTML report generated: {output_file}")
    print(f"📖 Open {output_file} in your web browser to view the complete report")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate HTML report for autonomous driving agent")
    parser.add_argument("--plots_dir", type=str, default="./plots",
                       help="Directory containing plot files")
    parser.add_argument("--results_dir", type=str, default="./evaluation_results",
                       help="Directory containing evaluation results")
    parser.add_argument("--models_dir", type=str, default="./models",
                       help="Directory containing trained models")
    parser.add_argument("--output", type=str, default="./training_report.html",
                       help="Output HTML file path")
    
    args = parser.parse_args()
    
    generate_html_report(
        plots_dir=args.plots_dir,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        output_file=args.output
    )


if __name__ == "__main__":
    main()