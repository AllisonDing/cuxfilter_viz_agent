# src/chat_agent.py
"""
Chat Agent - Orchestrator for Visualization Tasks

Coordinates between:
- llm.py (NVIDIA NIM API)
- tools/viz_tools.py (Visualization)
- tools/exp_store.py (Experiment tracking)
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from llm import create_client
from tools.viz_tools import VizTools
from tools.exp_store import VizExperimentStore


class ChatAgent:
    """
    Agent orchestrator coordinating LLM with visualization tools.
    """
    
    def __init__(self, exp_store: Optional[VizExperimentStore] = None):
        """
        Initialize chat agent.
        
        Args:
            exp_store: VizExperimentStore instance (optional)
        """
        # Initialize LLM client
        self.llm = create_client()
        
        # Initialize tools
        self.viz_tools = VizTools()
        self.exp_store = exp_store if exp_store else VizExperimentStore()
        
        # Conversation history
        self.messages = []
        
        # System prompt
        self.system_prompt = self._default_system_prompt()
        self.messages.append({"role": "system", "content": self.system_prompt})
        
        # Last visualization file
        self.last_viz_file = None
    
    def _default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a helpful AI assistant with GPU-accelerated visualization capabilities using CuxFilter.

You can:
1. Load and analyze data (CSV, Parquet, JSON, Arrow)
2. Create TWO types of visualizations:
   - **Single Chart**: Use create_chart() -> chart.view() (opens in browser, no HTML file)
   - **Dashboard**: Use create_dashboard() -> exports HTML file with multiple charts
3. Track experiments

Available chart types:
- **Basic**: bar, line, scatter, stacked_lines, heatmap
- **Widgets**: range_slider, date_range_slider, float_slider, int_slider, drop_down, multi_select, number_chart
- **Data**: view_dataframe, card, graph
- **Geospatial**: choropleth_2d, choropleth_3d (requires geoJSON source and latitude/longitude columns)

Available layouts (for dashboards only):
- **Preset**: single_feature, double_feature, triple_feature, quad_feature, feature_and_base
- **Grid**: grid(cols=2), grid(cols=3), grid(cols=4)
- **Custom**: horizontal, vertical

Available themes:
- rapids_dark (default), rapids, dark, default

When to use which:
- User wants "a chart" or "single chart" → use create_chart (calls .view())
- User wants "dashboard" or "multiple charts" → use create_dashboard (exports HTML)

For choropleth maps:
- choropleth_2d: Requires x (lat/lon column), geoJSONSource (URL or path), and optional color_column
- choropleth_3d: Additionally requires elevation_column for 3D visualization

Be helpful and explain what you're creating."""
    
    def get_tool_definitions(self) -> List[Dict]:
        """
        Get tool definitions for LLM.
        
        Returns:
            List of tool definitions in OpenAI/NIM format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "load_data",
                    "description": "Load data from CSV, Parquet, JSON, or Arrow file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path to the data file"
                            }
                        },
                        "required": ["filepath"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_data",
                    "description": "Analyze loaded data to understand columns, data types, and get chart suggestions",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_chart",
                    "description": "Create a single standalone chart and export as HTML file (chart_{i}.html). Use when user asks for 'a chart', 'single chart', 'standalone chart', or 'just one chart'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chart_type": {
                                "type": "string",
                                "enum": ["bar", "line", "scatter", "stacked_lines", "heatmap", 
                                        "number_chart", "choropleth_2d", "choropleth_3d"],
                                "description": "Type of chart to create"
                            },
                            "x": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "string"}}
                                ],
                                "description": "Column(s) for x-axis"
                            },
                            "y": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "string"}},
                                    {"type": "null"}
                                ],
                                "description": "Column(s) for y-axis (optional)"
                            },
                            "aggregate_fn": {
                                "type": "string",
                                "enum": ["count", "mean", "sum", "min", "max"],
                                "description": "Aggregation function (optional)"
                            },
                            "color": {
                                "type": "string",
                                "description": "Color column (optional, for scatter/line)"
                            },
                            "size": {
                                "type": "string",
                                "description": "Size column (optional, for scatter)"
                            },
                            "geoJSONSource": {
                                "type": "string",
                                "description": "GeoJSON source (required for choropleth)"
                            },
                            "color_column": {
                                "type": "string",
                                "description": "Color column for choropleth (optional, default: 'color')"
                            },
                            "color_aggregate_fn": {
                                "type": "string",
                                "enum": ["mean", "sum", "min", "max", "count"],
                                "description": "Color aggregation for choropleth (optional)"
                            },
                            "elevation_column": {
                                "type": "string",
                                "description": "Elevation column for 3D choropleth (optional)"
                            },
                            "elevation_factor": {
                                "type": "integer",
                                "description": "Elevation scale factor for 3D (optional, default: 1000)"
                            },
                            "title": {
                                "type": "string",
                                "description": "Chart title"
                            }
                        },
                        "required": ["chart_type", "x"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_dashboard",
                    "description": "Create a dashboard with multiple charts and widgets, then export as HTML. Use when user asks for 'dashboard', 'multiple charts', or 'visualization with filters'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "charts": {
                                "type": "array",
                                "description": "List of charts for main dashboard area",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": ["bar", "line", "scatter", "stacked_lines", "heatmap", 
                                                    "view_dataframe", "card", "graph", "number_chart", 
                                                    "choropleth_2d", "choropleth_3d"],
                                            "description": "Type of chart"
                                        },
                                        "x": {
                                            "anyOf": [
                                                {"type": "string"},
                                                {"type": "array", "items": {"type": "string"}}
                                            ],
                                            "description": "Column(s) for x-axis"
                                        },
                                        "y": {
                                            "anyOf": [
                                                {"type": "string"},
                                                {"type": "array", "items": {"type": "string"}},
                                                {"type": "null"}
                                            ],
                                            "description": "Column(s) for y-axis (optional)"
                                        },
                                        "aggregate_fn": {
                                            "type": "string",
                                            "enum": ["count", "mean", "sum", "min", "max"],
                                            "description": "Aggregation function (optional)"
                                        },
                                        "geoJSONSource": {
                                            "type": "string",
                                            "description": "GeoJSON source URL or path (required for choropleth)"
                                        },
                                        "color_column": {
                                            "type": "string",
                                            "description": "Column for color mapping (optional, default: 'color')"
                                        },
                                        "color_aggregate_fn": {
                                            "type": "string",
                                            "enum": ["mean", "sum", "min", "max", "count"],
                                            "description": "Color aggregation function (optional, default: 'mean')"
                                        },
                                        "elevation_column": {
                                            "type": "string",
                                            "description": "Column for elevation in 3D choropleth (optional, default: 'elevation')"
                                        },
                                        "elevation_factor": {
                                            "type": "integer",
                                            "description": "Elevation scale factor for 3D (optional, default: 1000)"
                                        },
                                        "elevation_aggregate_fn": {
                                            "type": "string",
                                            "enum": ["mean", "sum", "min", "max", "count"],
                                            "description": "Elevation aggregation function (optional, default: 'mean')"
                                        },
                                        "title": {"type": "string", "description": "Chart title"}
                                    },
                                    "required": ["type", "x"]
                                }
                            },
                            "sidebar": {
                                "type": "array",
                                "description": "List of widgets for sidebar (filters/controls)",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": ["range_slider", "date_range_slider", "float_slider", 
                                                    "int_slider", "drop_down", "multi_select"],
                                            "description": "Type of widget"
                                        },
                                        "x": {"type": "string", "description": "Column name"}
                                    },
                                    "required": ["type", "x"]
                                }
                            },
                            "layout": {
                                "type": "string",
                                "enum": ["auto", "single_feature", "double_feature", "triple_feature", 
                                        "quad_feature", "grid", "horizontal", "vertical"],
                                "description": "Dashboard layout (default: auto)"
                            },
                            "layout_cols": {
                                "type": "integer",
                                "description": "Number of columns for grid layout (2, 3, or 4)"
                            },
                            "theme": {
                                "type": "string",
                                "enum": ["rapids_dark", "rapids", "dark", "default"],
                                "description": "Visual theme (default: rapids_dark)"
                            },
                            "title": {
                                "type": "string",
                                "description": "Dashboard title"
                            }
                        },
                        "required": ["charts"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_data_info",
                    "description": "Get information about currently loaded data",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Process user message and execute tasks.
        
        Args:
            user_message: User's message
            
        Returns:
            Dict with response and visualization info
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_message})
        
        # Get LLM response with tools
        response = self.llm.chat(
            messages=self.messages,
            tools=self.get_tool_definitions()
        )
        
        # Extract message
        message = response['choices'][0]['message']
        
        # Check for tool calls
        if message.get('tool_calls'):
            tool_results = []
            
            for tool_call in message['tool_calls']:
                tool_name = tool_call['function']['name']
                tool_args = json.loads(tool_call['function']['arguments'])
                
                # Execute tool
                result = self._execute_tool(tool_name, tool_args)
                tool_results.append(result)
                
                # Add tool result to conversation
                self.messages.append({
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": tool_call['id']
                })
            
            # Get final response after tool execution
            final_response = self.llm.chat(
                messages=self.messages,
                tools=self.get_tool_definitions()
            )
            final_message = final_response['choices'][0]['message']
            
            self.messages.append({"role": "assistant", "content": final_message['content']})
            
            return {
                "response": final_message['content'],
                "tool_results": tool_results,
                "viz_file": self.last_viz_file
            }
        
        else:
            # No tool calls
            self.messages.append({"role": "assistant", "content": message['content']})
            
            return {
                "response": message['content'],
                "tool_results": [],
                "viz_file": None
            }
    
    def _execute_tool(self, tool_name: str, tool_args: Dict) -> Dict[str, Any]:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            if tool_name == "load_data":
                # Returns cxf_df directly or error dict
                result = self.viz_tools.load_data(tool_args['filepath'])
                
                # Check if it's an error dict or cxf_df object
                if isinstance(result, dict) and not result.get('success', True):
                    return result
                else:
                    # Success - cxf_df loaded
                    info = self.viz_tools.get_data_info()
                    return {
                        "success": True,
                        "message": f"Data loaded successfully: {info['shape'][0]:,} rows × {info['shape'][1]} columns",
                        "shape": info['shape'],
                        "columns": info['columns']
                    }
            
            elif tool_name == "analyze_data":
                result = self.viz_tools.analyze_columns()
                return result
            
            elif tool_name == "get_data_info":
                result = self.viz_tools.get_data_info()
                return result
            
            elif tool_name == "create_chart":
                # Create single standalone chart and export with .view().save()
                chart_type = tool_args.get('chart_type')
                x = tool_args.get('x')
                y = tool_args.get('y')
                title = tool_args.get('title', '')
                aggregate_fn = tool_args.get('aggregate_fn', 'count')
                
                # Build chart config
                chart_config = {
                    'type': chart_type,
                    'x': x,
                    'y': y,
                    'title': title,
                    'aggregate_fn': aggregate_fn
                }
                
                # Add optional parameters
                if 'color' in tool_args:
                    chart_config['color'] = tool_args['color']
                if 'size' in tool_args:
                    chart_config['size'] = tool_args['size']
                if 'geoJSONSource' in tool_args:
                    chart_config['geoJSONSource'] = tool_args['geoJSONSource']
                if 'color_column' in tool_args:
                    chart_config['color_column'] = tool_args['color_column']
                if 'color_aggregate_fn' in tool_args:
                    chart_config['color_aggregate_fn'] = tool_args['color_aggregate_fn']
                if 'elevation_column' in tool_args:
                    chart_config['elevation_column'] = tool_args['elevation_column']
                if 'elevation_factor' in tool_args:
                    chart_config['elevation_factor'] = tool_args['elevation_factor']
                
                # Create the chart
                chart = self._create_chart(chart_config)
                
                if chart is None:
                    return {"success": False, "error": f"Failed to create {chart_type} chart"}
                
                # Export using viz_tools.export_chart()
                try:
                    self.viz_tools.chart_counter += 1
                    chart_filename = f"chart_{self.viz_tools.chart_counter}.html"
                    
                    print(f"Attempting to export chart to: {chart_filename}")
                    
                    # Use viz_tools.export_chart()
                    export_result = self.viz_tools.export_chart(chart, chart_filename)
                    
                    if not export_result['success']:
                        return {
                            "success": False,
                            "error": f"Chart export failed: {export_result.get('error')}"
                        }
                    
                    print(f"Chart exported successfully to: {export_result['filepath']}")
                    
                    self.last_viz_file = export_result['filepath']
                    
                    # Track experiment
                    self.exp_store.save_experiment({
                        'chart_id': f"chart_{self.viz_tools.chart_counter}",
                        'title': title if title else f"{chart_type.title()} Chart",
                        'chart_type': chart_type,
                        'filepath': export_result['filepath'],
                        'success': True
                    })
                    
                    return {
                        "success": True,
                        "message": f"Created {chart_type} chart: '{title}'",
                        "chart_type": chart_type,
                        "filepath": export_result['filepath'],
                        "note": f"Chart exported as {chart_filename}"
                    }
                    
                except Exception as e:
                    print(f"ERROR exporting chart: {e}")
                    import traceback
                    traceback.print_exc()
                    return {
                        "success": False,
                        "error": f"Chart created but export failed: {str(e)}"
                    }
            
            elif tool_name == "create_dashboard":
                # Create main charts
                charts = []
                for chart_config in tool_args.get('charts', []):
                    chart = self._create_chart(chart_config)
                    if chart is not None:  # Explicit None check
                        charts.append(chart)
                    else:
                        print(f"Warning: Chart creation returned None for config: {chart_config}")
                
                # Create sidebar widgets
                sidebar = []
                for widget_config in tool_args.get('sidebar', []):
                    widget = self._create_widget(widget_config)
                    if widget is not None:  # Explicit None check
                        sidebar.append(widget)
                
                if not charts:
                    return {"success": False, "error": "No valid charts created. Check that data is loaded and chart parameters are correct."}
                
                # Determine layout
                layout_type = tool_args.get('layout', 'auto')
                layout_cols = tool_args.get('layout_cols', 2)
                
                # Create dashboard
                try:
                    dashboard = self.viz_tools.create_dashboard(
                        charts=charts,
                        sidebar=sidebar if sidebar else [],  # Pass empty list instead of None
                        layout_type=layout_type,
                        layout_array=None,  # Will be calculated in create_dashboard
                        theme_name=tool_args.get('theme', 'rapids_dark'),
                        title=tool_args.get('title', 'Dashboard')
                    )
                    
                    # Export to HTML
                    export_result = self.viz_tools.export_dashboard()
                    
                    if export_result['success']:
                        self.last_viz_file = export_result['filepath']
                        
                        # Track experiment
                        self.exp_store.save_experiment({
                            'dashboard_id': f"dashboard_{self.viz_tools.dashboard_counter}",
                            'title': tool_args.get('title', 'Dashboard'),
                            'num_charts': len(charts),
                            'num_widgets': len(sidebar),
                            'chart_types': [c['type'] for c in tool_args.get('charts', [])],
                            'widget_types': [w['type'] for w in tool_args.get('sidebar', [])],
                            'layout': tool_args.get('layout', 'auto'),
                            'theme': tool_args.get('theme', 'rapids_dark'),
                            'filepath': export_result['filepath'],
                            'success': True
                        })
                        
                        return {
                            "success": True,
                            "message": f"Dashboard created with {len(charts)} charts" + (f" and {len(sidebar)} widgets" if sidebar else ""),
                            "filepath": export_result['filepath'],
                            "num_charts": len(charts),
                            "num_widgets": len(sidebar)
                        }
                    else:
                        # Track failed export
                        self.exp_store.save_experiment({
                            'dashboard_id': f"dashboard_{self.viz_tools.dashboard_counter}",
                            'title': tool_args.get('title', 'Dashboard'),
                            'num_charts': len(charts),
                            'num_widgets': len(sidebar),
                            'success': False,
                            'error': export_result.get('error', 'Unknown error')
                        })
                        return export_result
                
                except Exception as e:
                    # Track failed creation
                    self.exp_store.save_experiment({
                        'title': tool_args.get('title', 'Dashboard'),
                        'success': False,
                        'error': str(e)
                    })
                    return {"success": False, "error": f"Dashboard creation failed: {str(e)}"}
            
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _create_chart(self, chart_config: Dict) -> Any:
        """
        Create a chart from configuration.
        
        Args:
            chart_config: Chart configuration dict
            
        Returns:
            Chart object or None
        """
        chart_type = chart_config['type']
        x = chart_config['x']
        y = chart_config.get('y')
        title = chart_config.get('title', '')
        aggregate_fn = chart_config.get('aggregate_fn', 'count')
        
        try:
            if chart_type == 'bar':
                print(f"Creating bar chart: x={x}, y={y}, aggregate_fn={aggregate_fn}, title={title}")
                chart = self.viz_tools.create_bar_chart(x=x, y=y, aggregate_fn=aggregate_fn, title=title)
                print(f"Bar chart created: {chart}")
                return chart
            
            elif chart_type == 'line':
                if not y:
                    return None
                return self.viz_tools.create_line_chart(x=x, y=y, title=title)
            
            elif chart_type == 'scatter':
                if not y:
                    return None
                return self.viz_tools.create_scatter_chart(x=x, y=y, title=title)
            
            elif chart_type == 'stacked_lines':
                if not y or not isinstance(y, list):
                    return None
                return self.viz_tools.create_stacked_lines_chart(x=x, y=y, title=title)
            
            elif chart_type == 'heatmap':
                if not y:
                    return None
                return self.viz_tools.create_heatmap(x=x, y=y, aggregate_fn=aggregate_fn, title=title)
            
            elif chart_type == 'view_dataframe':
                # x should be a list of columns
                if isinstance(x, str):
                    x = [x]
                return self.viz_tools.create_view_dataframe(x=x)
            
            elif chart_type == 'card':
                return self.viz_tools.create_card_chart(x=x, title=title)
            
            elif chart_type == 'number_chart':
                return self.viz_tools.create_number_chart(x=x, aggregate_fn=aggregate_fn, title=title)
            
            elif chart_type == 'graph':
                if not y:
                    return None
                return self.viz_tools.create_graph_chart(node_x=x, node_y=y, title=title)
            
            elif chart_type == 'choropleth_2d':
                # 2D choropleth map
                geoJSONSource = chart_config.get('geoJSONSource')
                if not geoJSONSource:
                    print("Error: geoJSONSource required for choropleth_2d")
                    return None
                
                color_column = chart_config.get('color_column', 'color')
                color_aggregate_fn = chart_config.get('color_aggregate_fn', 'mean')
                
                return self.viz_tools.create_2d_choropleth(
                    x=x,
                    geoJSONSource=geoJSONSource,
                    color_column=color_column,
                    color_aggregate_fn=color_aggregate_fn,
                    add_interaction=True
                )
            
            elif chart_type == 'choropleth_3d':
                # 3D choropleth map
                geoJSONSource = chart_config.get('geoJSONSource')
                if not geoJSONSource:
                    print("Error: geoJSONSource required for choropleth_3d")
                    return None
                
                color_column = chart_config.get('color_column', 'color')
                color_aggregate_fn = chart_config.get('color_aggregate_fn', 'mean')
                elevation_column = chart_config.get('elevation_column', 'elevation')
                elevation_factor = chart_config.get('elevation_factor', 1000)
                elevation_aggregate_fn = chart_config.get('elevation_aggregate_fn', 'mean')
                
                return self.viz_tools.create_3d_choropleth(
                    x=x,
                    geoJSONSource=geoJSONSource,
                    color_column=color_column,
                    color_aggregate_fn=color_aggregate_fn,
                    elevation_column=elevation_column,
                    elevation_factor=elevation_factor,
                    elevation_aggregate_fn=elevation_aggregate_fn,
                    add_interaction=True
                )
            
            else:
                return None
                
        except Exception as e:
            print(f"Error creating chart {chart_type}: {e}")
            return None
    
    def _create_widget(self, widget_config: Dict) -> Any:
        """
        Create a widget from configuration.
        
        Args:
            widget_config: Widget configuration dict
            
        Returns:
            Widget object or None
        """
        widget_type = widget_config['type']
        x = widget_config['x']
        
        try:
            if widget_type == 'range_slider':
                return self.viz_tools.create_range_slider(x=x)
            
            elif widget_type == 'date_range_slider':
                return self.viz_tools.create_date_range_slider(x=x)
            
            elif widget_type == 'float_slider':
                return self.viz_tools.create_float_slider(x=x)
            
            elif widget_type == 'int_slider':
                return self.viz_tools.create_int_slider(x=x)
            
            elif widget_type == 'drop_down':
                return self.viz_tools.create_drop_down(x=x)
            
            elif widget_type == 'multi_select':
                return self.viz_tools.create_multi_select(x=x)
            
            else:
                return None
                
        except Exception as e:
            print(f"Error creating widget {widget_type}: {e}")
            return None
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.messages.copy()