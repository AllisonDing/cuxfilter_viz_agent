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
import math

import cudf
import pandas as pd
import cuxfilter as cxf
from cuxfilter import DataFrame, charts, layouts, themes
from bokeh import palettes
import panel as pn

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
                return {"success": False, "error": f"File not found", "cxf_df": None}
            
            if filepath.suffix == '.csv':
                self.df = cudf.read_csv(str(filepath))
                self.cxf_df = cxf.DataFrame.from_dataframe(self.df)
            elif filepath.suffix == '.parquet':
                self.df = cudf.read_parquet(str(filepath))
                self.cxf_df = cxf.DataFrame.from_dataframe(self.df)
            elif filepath.suffix == '.json':
                self.df = cudf.read_json(str(filepath))
                self.cxf_df = cxf.DataFrame.from_dataframe(self.df)
            elif filepath.suffix == '.arrow':
                self.cxf_df = DataFrame.from_arrow(str(filepath))
            else:
                return {"success": False, "error": f"Unsupported type", "cxf_df": None}
            
            return self.cxf_df
        
        except Exception as e:
            return {"success": False, "error": str(e), "df": None, "cxf_df": None}
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get info about loaded data."""
        if self.cxf_df is None:
            return {"success": False, "error": "No data"}
        return {
            "success": True,
            "shape": self.cxf_df.data.shape,
            "columns": list(self.cxf_df.data.columns)
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODULE 2: cuxfilter.DashBoard
    # ═══════════════════════════════════════════════════════════════════════
    
    def create_dashboard(self, charts: List[Any], sidebar: List[Any] = None, layout_type: str = "auto", layout_array: list = None,
                        theme_name: str = "rapids_dark", title: str = "Dashboard") -> Any:
        """Create cuxfilter.DashBoard."""
        try:
            if not self.cxf_df:
                raise ValueError("No cuxfilter DataFrame loaded (cxf_df is None)")
            if not charts:
                raise ValueError("No charts provided")
            
            layout = self.get_layout(layout_type, len(charts))
            theme = self.get_theme(theme_name)

            # For preset layouts, use 'layout' parameter
            if isinstance(layout, list):
                self.active_dashboard = self.cxf_df.dashboard(
                    charts=charts, 
                    sidebar=sidebar,
                    layout_array=layout,  # Use layout_array for custom layouts
                    theme=theme, 
                    title=title
                )
            else:
                self.active_dashboard = self.cxf_df.dashboard(
                    charts=charts, 
                    sidebar=sidebar,
                    layout=layout,  # Use layout for presets
                    theme=theme, 
                    title=title
                )
            
            self.dashboard_counter += 1

            return self.active_dashboard           
            
        except Exception as e:
            print(f"Error creating dashboard: {str(e)}")
            raise
    
    def export_dashboard(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Export cuxfilter.DashBoard to HTML."""
        try:
            if not self.active_dashboard:
                return {"success": False, "error": "No dashboard"}
            
            if not filename:
                filename = f"dashboard_{self.dashboard_counter}.html"
            
            filepath = os.path.join(self.output_dir, filename)
            self.active_dashboard.app().save(str(filepath))
            
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
        return cxf.charts.datashader.line(x=x, y=y, color=color, title=title) if color else cxf.charts.datashader.line(x=x, y=y, title=title)
    
    def create_scatter_chart(self, x: str, y: str, color: Optional[str] = None, 
                            size: Optional[str] = None, title: str = "") -> Any:
        """cuxfilter.charts.scatter()"""
        kwargs = {"x": x, "y": y, "title": title}
        if color: kwargs["color"] = color
        if size: kwargs["size"] = size
        return cxf.charts.scatter(**kwargs)
    
    def create_stacked_lines_chart(self, x: str, y: List[str], title: str = "") -> Any:
        """cuxfilter.charts.stacked_lines()"""
        return cxf.charts.stacked_lines(x=x, y=y, title=title)
    
    def create_heatmap(self, x: str, y: str, aggregate_fn: str = "count", title: str = "") -> Any:
        """cuxfilter.charts.heatmap()"""
        return cxf.charts.heatmap(x=x, y=y, aggregate_fn=aggregate_fn, title=title)
    
    def create_range_slider(self, x: str) -> Any:
        """cuxfilter.charts.range_slider()"""
        return cxf.charts.range_slider(x=x)

    def create_date_range_slider(self, x: str) -> Any:
        """cuxfilter.charts.date_range_slider()"""
        return cxf.charts.date_range_slider(x=x)

    def create_float_slider(self, x: str) -> Any:
        """cuxfilter.charts.float_slider()"""
        return cxf.charts.float_slider(x=x, step_size=0.5)

    def create_int_slider(self, x: str) -> Any:
        """cuxfilter.charts.int_slider()"""
        return cxf.charts.int_slider(x=x)
    
    def create_drop_down(self, x: str) -> Any:
        """cuxfilter.charts.dropdown()"""
        return cxf.charts.drop_down(x=x)
    
    def create_multi_select(self, x: str) -> Any:
        """cuxfilter.charts.multi_select()"""
        return cxf.charts.multi_select(x=x)
    
    def create_number_chart(self, x: str, aggregate_fn: str = "mean", title: str = "") -> Any:
        """cuxfilter.charts.number()"""
        return cxf.charts.number(expression=x, aggregate_fn=aggregate_fn, title=title)

    def create_view_dataframe(self, x: List[str]) -> Any:
        """cuxfilter.charts.view_dataframe([column names])"""
        return cxf.charts.view_dataframe(x)
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODULE 4: cuxfilter.layouts
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_layout(self, layout_type: str, num_charts: int = None) -> Any:
        """
        Get cuxfilter layout based on type and number of charts.
        
        Args:
            layout_type: Layout type - 'auto', 'single_feature', 'feature_and_base', 
                        'double_feature', 'left_feature_right_double', 'triple_feature',
                        'feature_and_double_base', 'two_by_two', 'feature_and_triple_base',
                        'feature_and_quad_base', 'feature_and_five_edge', 'two_by_three',
                        'double_feature_quad_base', 'three_by_three', 'grid', 'horizontal', 'vertical'
            num_charts: Number of charts (required for 'auto' and custom layouts)
        
        Returns:
            Layout preset object or custom layout array
        """

        import math
    
        # Preset layouts mapping
        presets = {
            'single_feature': cxf.layouts.single_feature,
            'feature_and_base': cxf.layouts.feature_and_base,
            'double_feature': cxf.layouts.double_feature,
            'left_feature_right_double': cxf.layouts.left_feature_right_double,
            'triple_feature': cxf.layouts.triple_feature,
            'feature_and_double_base': cxf.layouts.feature_and_double_base,
            'two_by_two': cxf.layouts.two_by_two,
            'feature_and_triple_base': cxf.layouts.feature_and_triple_base,
            'feature_and_quad_base': cxf.layouts.feature_and_quad_base,
            'feature_and_five_edge': cxf.layouts.feature_and_five_edge,
            'two_by_three': cxf.layouts.two_by_three,
            'double_feature_quad_base': cxf.layouts.double_feature_quad_base,
            'three_by_three': cxf.layouts.three_by_three
        }
        
        # Return preset if available
        if layout_type in presets:
            return presets[layout_type]
        
        # Auto-select based on number of charts
        if layout_type == 'auto':
            if num_charts is None:
                raise ValueError("num_charts required for 'auto' layout")
            
            if num_charts == 1:
                return presets['single_feature']
            elif num_charts == 2:
                return presets['double_feature']
            elif num_charts == 3:
                return presets['triple_feature']
            elif num_charts == 4:
                return presets['two_by_two']
            elif num_charts <= 6:
                return presets['two_by_three']
            elif num_charts <= 9:
                return presets['three_by_three']
            else:
                # Fall back to grid for many charts
                layout_type = 'grid'      

        # Custom layout arrays
        if num_charts is None:
            raise ValueError(f"num_charts required for '{layout_type}' layout")
        
        if layout_type == 'grid':
            # Balanced grid layout with consistent row widths
            cols = math.ceil(math.sqrt(num_charts))
            rows = math.ceil(num_charts / cols)
            
            layout = []
            chart_idx = 1
            
            for row in range(rows):
                row_layout = []
                # Calculate how many charts go in this row
                remaining_charts = num_charts - chart_idx + 1
                charts_in_row = min(cols, remaining_charts)
                
                # Each chart gets equal width units
                for col in range(charts_in_row):
                    if chart_idx <= num_charts:
                        row_layout.extend([chart_idx] * 2)  # Each chart gets 2 units
                        chart_idx += 1
                
                # Pad the last row to match width of other rows if needed
                if row_layout:
                    # Calculate expected row width (cols * 2 units per chart)
                    expected_width = cols * 2
                    current_width = len(row_layout)
                    
                    # If last row is shorter, stretch the last chart to fill
                    if current_width < expected_width and row == rows - 1:
                        # Extend the last chart to fill remaining space
                        padding_needed = expected_width - current_width
                        row_layout.extend([row_layout[-1]] * padding_needed)
                    
                    layout.append(row_layout)
            
            return layout

        elif layout_type == 'horizontal':
            # Single row
            return [list(range(1, num_charts + 1))]
        
        elif layout_type == 'vertical':
            # Single column
            return [[i] for i in range(1, num_charts + 1)]
        
        else:
            raise ValueError(f"Unknown layout_type: {layout_type}")
    
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
        themes_map = {
            "rapids_dark": cxf.themes.rapids_dark,
            "rapids": cxf.themes.rapids,
            "dark": cxf.themes.dark,
            "default": cxf.themes.default
        }
        return themes_map.get(theme_name, cxf.themes.rapids_dark)
    
    def list_themes(self) -> Dict[str, str]:
        """List available themes."""
        return {
            "rapids_dark": "RAPIDS Dark",
            "rapids": "RAPIDS",
            "dark": "Dark",
            "default": "Default"
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # HELPER
    # ═══════════════════════════════════════════════════════════════════════
        
    def analyze_columns(self) -> Dict[str, Any]:
        """Analyze data and suggest charts."""
        if self.cxf_df is None:
            return {"success": False, "error": "No data"}
        
        analysis = {"success": True, "columns": {}, "suggestions": []}
        
        for col in self.cxf_df.data.columns:
            info = {"name": col, "dtype": str(self.cxf_df.data[col].dtype)}
            
            # Check for numeric types (including int32, float32, uint8, etc.)
            if self.cxf_df.data[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64']:
                info["type"] = "numeric"
                info["suggested_charts"] = ["bar", "line", "scatter", "heatmap", "range_slider", "float_slider", "int_slider", "number_chart"]
            elif 'datetime' in str(self.cxf_df.data[col].dtype):
                info["type"] = "datetime"
                info["suggested_charts"] = ["line", "date_range_slider"]
            elif self.cxf_df.data[col].dtype == 'object':
                info["type"] = "categorical"
                info["suggested_charts"] = ["bar", "drop_down", "multi_select"]
            else:
                info["type"] = "other"
                info["suggested_charts"] = ["bar"]
            
            analysis["columns"][col] = info
        
        # Suggestions
        numeric = [c for c, i in analysis["columns"].items() if i["type"] == "numeric"]
        categorical = [c for c, i in analysis["columns"].items() if i["type"] == "categorical"]
        datetime_cols = [c for c, i in analysis["columns"].items() if i["type"] == "datetime"]
        
        # Scatter plot suggestions
        if len(numeric) >= 2:
            analysis["suggestions"].append({
                "chart": "scatter",
                "params": {"x": numeric[0], "y": numeric[1]},
                "description": f"Scatter plot of {numeric[0]} vs {numeric[1]}"
            })
        
        # Line chart suggestions
        if datetime_cols and numeric:
            analysis["suggestions"].append({
                "chart": "line",
                "params": {"x": datetime_cols[0], "y": numeric[0]},
                "description": f"Line chart of {numeric[0]} over {datetime_cols[0]}"
            })
        elif len(numeric) >= 2:
            analysis["suggestions"].append({
                "chart": "line",
                "params": {"x": numeric[0], "y": numeric[1]},
                "description": f"Line chart of {numeric[1]} by {numeric[0]}"
            })
        
        # Bar chart suggestions
        if categorical:
            analysis["suggestions"].append({
                "chart": "bar",
                "params": {"x": categorical[0]},
                "description": f"Bar chart of {categorical[0]}"
            })
        if numeric:
            analysis["suggestions"].append({
                "chart": "bar",
                "params": {"x": numeric[0]},
                "description": f"Bar chart of {numeric[0]} distribution"
            })
        
        # Heatmap suggestion
        if len(numeric) >= 2:
            analysis["suggestions"].append({
                "chart": "heatmap",
                "params": {"x": numeric[0], "y": numeric[1], "aggregate_fn": "count"},
                "description": f"Heatmap of {numeric[0]} vs {numeric[1]}"
            })
        
        # Stacked lines suggestion
        if len(numeric) >= 3 and datetime_cols:
            analysis["suggestions"].append({
                "chart": "stacked_lines",
                "params": {"x": datetime_cols[0], "y": numeric[:3]},
                "description": f"Stacked lines of multiple metrics over {datetime_cols[0]}"
            })
        elif len(numeric) >= 3:
            analysis["suggestions"].append({
                "chart": "stacked_lines",
                "params": {"x": numeric[0], "y": numeric[1:4]},
                "description": f"Stacked lines of multiple metrics"
            })
        
        # Widget suggestions
        if numeric:
            analysis["suggestions"].append({
                "chart": "range_slider",
                "params": {"x": numeric[0]},
                "description": f"Range slider for {numeric[0]}"
            })
            analysis["suggestions"].append({
                "chart": "number_chart",
                "params": {"x": numeric[0], "aggregate_fn": "mean"},
                "description": f"Average {numeric[0]}"
            })
        
        if datetime_cols:
            analysis["suggestions"].append({
                "chart": "date_range_slider",
                "params": {"x": datetime_cols[0]},
                "description": f"Date range slider for {datetime_cols[0]}"
            })
        
        if categorical:
            analysis["suggestions"].append({
                "chart": "multi_select",
                "params": {"x": categorical[0]},
                "description": f"Multi-select filter for {categorical[0]}"
            })
        
        # View dataframe suggestion
        all_cols = list(self.cxf_df.data.columns)[:10]  # Limit to first 10 columns
        analysis["suggestions"].append({
            "chart": "view_dataframe",
            "params": {"x": all_cols},
            "description": f"Data table view"
        })
        
        return analysis