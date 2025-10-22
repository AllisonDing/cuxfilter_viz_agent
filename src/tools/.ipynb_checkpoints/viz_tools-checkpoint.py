# src/tools/viz_tools.py
"""
Visualization Tools - 5 CuxFilter Modules

Implements all 5 CuxFilter modules:
1. cuxfilter.DataFrame  - Data loading
2. cuxfilter.DashBoard  - Dashboard creation/export
3. cuxfilter.charts     - All chart types
4. cuxfilter.layouts    - Dashboard layouts
5. cuxfilter.themes     - Visual themes
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import traceback


import cudf
import pandas as pd
import cuxfilter as cxf


class VizTools:
    """Visualization tools implementing 5 CuxFilter modules."""
    
    def __init__(self, output_dir="viz_outputs"):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # MODULE 1: DataFrame
        self.df = None
        self.cxf_df = None
        
        # MODULE 2: DashBoard
        self.active_dashboard = None
        self.dashboard_counter = 0
        
        self.use_gpu = self._check_gpu()
    
    def _check_gpu(self) -> bool:
        try:
            test = cudf.DataFrame({'a': [1]})
            return True
        except:
            return False
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODULE 1: cuxfilter.DataFrame
    # ═══════════════════════════════════════════════════════════════════════
    
    def load_data(self, filepath: str) -> Dict[str, Any]:
        """Load data and create cuxfilter.DataFrame."""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return {"success": False, "error": f"File not found"}
            
            if filepath.suffix == '.csv':
                self.df = cudf.read_csv(str(filepath)) if self.use_gpu else pd.read_csv(str(filepath))
            elif filepath.suffix == '.parquet':
                self.df = cudf.read_parquet(str(filepath)) if self.use_gpu else pd.read_parquet(str(filepath))
            elif filepath.suffix == '.json':
                self.df = cudf.read_json(str(filepath)) if self.use_gpu else pd.read_json(str(filepath))
            else:
                return {"success": False, "error": f"Unsupported type"}
            
            self.cxf_df = cxf.DataFrame.from_dataframe(self.df)
            
            return {
                "success": True,
                "filepath": str(filepath),
                "shape": self.df.shape,
                "columns": list(self.df.columns),
                "using_gpu": self.use_gpu
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get info about loaded data."""
        if self.df is None:
            return {"success": False, "error": "No data"}
        return {
            "success": True,
            "shape": self.df.shape,
            "columns": list(self.df.columns)
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODULE 2: cuxfilter.DashBoard
    # ═══════════════════════════════════════════════════════════════════════
    
    def create_dashboard(self, charts: List[Any], layout_type: str = "auto",
                        theme_name: str = "rapids_dark", title: str = "Dashboard") -> Dict[str, Any]:
        """Create cuxfilter.DashBoard."""
        try:
            if not self.cxf_df or not charts:
                return {"success": False, "error": "No data or charts"}
            
            layout = self.get_layout(layout_type, len(charts))
            theme = self.get_theme(theme_name)
            
            self.active_dashboard = self.cxf_df.dashboard(
                charts=charts, layout=layout, theme=theme, title=title
            )
            
            self.dashboard_counter += 1
            return {
                "success": True,
                "dashboard_id": f"dashboard_{self.dashboard_counter}",
                "num_charts": len(charts)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def export_dashboard(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Export cuxfilter.DashBoard to HTML."""
        try:
            if not self.active_dashboard:
                return {"success": False, "error": "No dashboard"}
            
            if not filename:
                filename = f"dashboard_{self.dashboard_counter}.html"
            
            filepath = self.output_dir / filename
            self.active_dashboard.export(str(filepath))
            
            return {"success": True, "filepath": str(filepath)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODULE 3: cuxfilter.charts
    # ═══════════════════════════════════════════════════════════════════════
    
    def create_bar_chart(self, x: str, y: Optional[str] = None, 
                        aggregate_fn: str = "count", title: str = "") -> Any:
        """cuxfilter.charts.bar()"""
        return cxf.charts.bar(x=x, y=y, aggregate_fn=aggregate_fn, title=title) if y else cxf.charts.bar(x=x, title=title)
    
    def create_line_chart(self, x: str, y: str, color: Optional[str] = None, title: str = "") -> Any:
        """cuxfilter.charts.line()"""
        return cxf.charts.line(x=x, y=y, color=color, title=title) if color else cxf.charts.line(x=x, y=y, title=title)
    
    def create_scatter_chart(self, x: str, y: str, color: Optional[str] = None, 
                            size: Optional[str] = None, title: str = "") -> Any:
        """cuxfilter.charts.scatter()"""
        kwargs = {"x": x, "y": y, "title": title}
        if color: kwargs["color"] = color
        if size: kwargs["size"] = size
        return cxf.charts.scatter(**kwargs)
    
    def create_histogram(self, x: str, bins: int = 30, title: str = "") -> Any:
        """cuxfilter.charts.histogram()"""
        return cxf.charts.histogram(x=x, bins=bins, title=title)
    
    def create_heatmap(self, x: str, y: str, aggregate_fn: str = "count", title: str = "") -> Any:
        """cuxfilter.charts.heatmap()"""
        return cxf.charts.heatmap(x=x, y=y, aggregate_fn=aggregate_fn, title=title)
    
    def create_point_map(self, x: str, y: str, color: Optional[str] = None, title: str = "") -> Any:
        """cuxfilter.charts.datashader.scatter_geo()"""
        kwargs = {"x": x, "y": y, "title": title}
        if color: kwargs["color_column"] = color
        return cxf.charts.datashader.scatter_geo(**kwargs)
    
    def create_hex_map(self, x: str, y: str, aggregate_fn: str = "count", title: str = "") -> Any:
        """cuxfilter.charts.datashader.hexbin()"""
        return cxf.charts.datashader.hexbin(x=x, y=y, aggregate_fn=aggregate_fn, title=title)
    
    def create_range_slider(self, x: str, title: str = "") -> Any:
        """cuxfilter.charts.range_slider()"""
        return cxf.charts.range_slider(x=x, title=title)
    
    def create_dropdown(self, x: str, title: str = "") -> Any:
        """cuxfilter.charts.dropdown()"""
        return cxf.charts.dropdown(x=x, title=title)
    
    def create_multi_select(self, x: str, title: str = "") -> Any:
        """cuxfilter.charts.multi_select()"""
        return cxf.charts.multi_select(x=x, title=title)
    
    def create_number_display(self, x: str, aggregate_fn: str = "mean", title: str = "") -> Any:
        """cuxfilter.charts.number()"""
        return cxf.charts.number(x=x, aggregate_fn=aggregate_fn, title=title)
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODULE 4: cuxfilter.layouts
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_layout(self, layout_type: str, num_charts: int = None) -> Any:
        """Get cuxfilter.layouts object."""
        if layout_type == "auto" and num_charts:
            if num_charts == 1: return cxf.layouts.single_feature
            elif num_charts == 2: return cxf.layouts.double_feature
            elif num_charts == 3: return cxf.layouts.triple_feature
            else: return cxf.layouts.quad_feature
        
        layouts = {
            "single_feature": cxf.layouts.single_feature,
            "double_feature": cxf.layouts.double_feature,
            "triple_feature": cxf.layouts.triple_feature,
            "quad_feature": cxf.layouts.quad_feature
        }
        return layouts.get(layout_type, cxf.layouts.double_feature)
    
    def list_layouts(self) -> Dict[str, str]:
        """List available layouts."""
        return {
            "single_feature": "1 chart",
            "double_feature": "2 charts side-by-side",
            "triple_feature": "3 charts",
            "quad_feature": "4 charts in grid"
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODULE 5: cuxfilter.themes
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_theme(self, theme_name: str) -> Any:
        """Get cuxfilter.themes object."""
        themes = {
            "rapids_dark": cxf.themes.rapids_dark,
            "rapids": cxf.themes.rapids,
            "dark": cxf.themes.dark,
            "light": cxf.themes.light
        }
        return themes.get(theme_name, cxf.themes.rapids_dark)
    
    def list_themes(self) -> Dict[str, str]:
        """List available themes."""
        return {
            "rapids_dark": "RAPIDS dark (default)",
            "rapids": "RAPIDS light",
            "dark": "Standard dark",
            "light": "Standard light"
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # HELPER
    # ═══════════════════════════════════════════════════════════════════════
    
    def analyze_columns(self) -> Dict[str, Any]:
        """Analyze data and suggest charts."""
        if self.df is None:
            return {"success": False, "error": "No data"}
        
        analysis = {"success": True, "columns": {}, "suggestions": []}
        
        for col in self.df.columns:
            info = {"name": col, "dtype": str(self.df[col].dtype)}
            
            if self.df[col].dtype in ['int64', 'float64']:
                info["type"] = "numeric"
                info["suggested_charts"] = ["histogram", "scatter", "line"]
            elif self.df[col].dtype == 'object':
                info["type"] = "categorical"
                info["suggested_charts"] = ["bar", "dropdown"]
            else:
                info["type"] = "other"
                info["suggested_charts"] = []
            
            analysis["columns"][col] = info
        
        # Suggestions
        numeric = [c for c, i in analysis["columns"].items() if i["type"] == "numeric"]
        categorical = [c for c, i in analysis["columns"].items() if i["type"] == "categorical"]
        
        if len(numeric) >= 2:
            analysis["suggestions"].append({"chart": "scatter", "x": numeric[0], "y": numeric[1]})
        if numeric:
            analysis["suggestions"].append({"chart": "histogram", "x": numeric[0]})
        if categorical:
            analysis["suggestions"].append({"chart": "bar", "x": categorical[0]})
        
        return analysis