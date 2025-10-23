# src/tools/viz_exp_store.py
"""
Visualization Experiment Store - Track dashboard creation history
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter


class VizExperimentStore:
    """Storage for visualization dashboard experiments."""
    
    def __init__(self, runs_file="runs/viz_experiments.jsonl"):
        self.runs_file = Path(runs_file)
        
        # Create the directory if it doesn't exist
        self.runs_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the file if it doesn't exist
        if not self.runs_file.exists():
            self.runs_file.touch()
    
    def clear_all(self):
        """Delete all stored experiments."""
        self.runs_file.write_text("", encoding="utf-8")
    
    def save_experiment(self, experiment_data: Dict):
        """
        Save a dashboard experiment.
        
        Expected fields:
        - dashboard_id: str
        - title: str
        - num_charts: int
        - num_widgets: int
        - chart_types: List[str]
        - widget_types: List[str]
        - layout: str
        - theme: str
        - filepath: str
        - success: bool
        - error: str (optional)
        """
        # Add timestamp
        experiment_data["timestamp"] = time.time()
        experiment_data["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Append to file
        with self.runs_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(experiment_data) + "\n")
    
    def load_all_experiments(self) -> List[Dict]:
        """Load all saved experiments."""
        experiments = []
        
        if not self.runs_file.exists():
            return experiments
        
        with self.runs_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        experiment = json.loads(line)
                        experiments.append(experiment)
                    except json.JSONDecodeError:
                        # Skip corrupted lines
                        continue
        
        return experiments
    
    def get_experiment_summary(self) -> Dict:
        """Get summary statistics about all experiments."""
        experiments = self.load_all_experiments()
        
        if not experiments:
            return {
                'total_experiments': 0,
                'successful_experiments': 0,
                'failed_experiments': 0,
                'success_rate': 'N/A',
                'chart_usage': {},
                'widget_usage': {},
                'layout_usage': {},
                'theme_usage': {}
            }
        
        # Count successes/failures
        successful = [e for e in experiments if e.get('success', False)]
        failed = [e for e in experiments if not e.get('success', True)]
        
        # Calculate success rate
        success_rate = f"{len(successful) / len(experiments) * 100:.1f}%"
        
        # Count chart types
        chart_types = []
        for exp in successful:
            chart_types.extend(exp.get('chart_types', []))
        chart_usage = dict(Counter(chart_types).most_common())
        
        # Count widget types
        widget_types = []
        for exp in successful:
            widget_types.extend(exp.get('widget_types', []))
        widget_usage = dict(Counter(widget_types).most_common())
        
        # Count layouts
        layouts = [exp.get('layout', 'unknown') for exp in successful]
        layout_usage = dict(Counter(layouts).most_common())
        
        # Count themes
        themes = [exp.get('theme', 'unknown') for exp in successful]
        theme_usage = dict(Counter(themes).most_common())
        
        # Average charts/widgets per dashboard
        avg_charts = sum(e.get('num_charts', 0) for e in successful) / len(successful) if successful else 0
        avg_widgets = sum(e.get('num_widgets', 0) for e in successful) / len(successful) if successful else 0
        
        return {
            'total_experiments': len(experiments),
            'successful_experiments': len(successful),
            'failed_experiments': len(failed),
            'success_rate': success_rate,
            'chart_usage': chart_usage,
            'widget_usage': widget_usage,
            'layout_usage': layout_usage,
            'theme_usage': theme_usage,
            'avg_charts_per_dashboard': f"{avg_charts:.1f}",
            'avg_widgets_per_dashboard': f"{avg_widgets:.1f}"
        }
    
    def get_recent_experiments(self, limit: int = 10) -> List[Dict]:
        """Get the most recent experiments."""
        experiments = self.load_all_experiments()
        
        # Sort by timestamp (most recent first)
        experiments.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        return experiments[:limit]
    
    def count_experiments(self) -> int:
        """Count total number of experiments."""
        return len(self.load_all_experiments())
    
    def get_successful_experiments(self) -> List[Dict]:
        """Get all successful experiments."""
        experiments = self.load_all_experiments()
        return [e for e in experiments if e.get('success', False)]
    
    def get_failed_experiments(self) -> List[Dict]:
        """Get all failed experiments."""
        experiments = self.load_all_experiments()
        return [e for e in experiments if not e.get('success', True)]
    
    def find_experiment_by_id(self, dashboard_id: str) -> Optional[Dict]:
        """Find experiment by dashboard ID."""
        experiments = self.load_all_experiments()
        for exp in experiments:
            if exp.get('dashboard_id') == dashboard_id:
                return exp
        return None
    
    def get_most_used_chart(self) -> Optional[str]:
        """Get the most commonly used chart type."""
        summary = self.get_experiment_summary()
        chart_usage = summary.get('chart_usage', {})
        if not chart_usage:
            return None
        return max(chart_usage, key=chart_usage.get)
    
    def get_most_used_layout(self) -> Optional[str]:
        """Get the most commonly used layout."""
        summary = self.get_experiment_summary()
        layout_usage = summary.get('layout_usage', {})
        if not layout_usage:
            return None
        return max(layout_usage, key=layout_usage.get)