# src/chat_agent.py
"""
Chat Agent - Orchestrator for Visualization Tasks

Coordinates between:
- llm.py (NVIDIA NIM API)
- tools/viz_tools.py (Visualization)
- tools/exp_store.py (Experiment tracking)
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.llm import create_client
from src.tools.viz_tools import VizTools
from src.tools.exp_store import VizExperimentStore


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
2. Create interactive dashboards with multiple charts
3. Track experiments

Available chart types:
- **Basic**: bar, line, scatter, stacked_lines, heatmap
- **Widgets**: range_slider, date_range_slider, float_slider, int_slider, drop_down, multi_select, number_chart
- **Data**: view_dataframe, card, graph, choropleth

Available layouts:
- **Preset**: single_feature, double_feature, triple_feature, quad_feature, feature_and_base
- **Grid**: grid(cols=2), grid(cols=3), grid(cols=4)
- **Custom**: horizontal, vertical

Available themes:
- rapids_dark (default), rapids, dark, default

When users ask for visualizations:
1. First check if data is loaded
2. Analyze columns to understand data types
3. Create appropriate charts based on data types
4. Use sidebar for filters/widgets, main area for charts
5. Export as interactive HTML dashboard

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
                    "name": "create_visualization",
                    "description": "Create interactive dashboard with charts and widgets, then export as HTML",
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
                                                    "view_dataframe", "card", "graph", "number_chart", "choropleth"],
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
            
            elif tool_name == "create_visualization":
                # Create main charts
                charts = []
                for chart_config in tool_args.get('charts', []):
                    chart = self._create_chart(chart_config)
                    if chart:
                        charts.append(chart)
                
                # Create sidebar widgets
                sidebar = []
                for widget_config in tool_args.get('sidebar', []):
                    widget = self._create_widget(widget_config)
                    if widget:
                        sidebar.append(widget)
                
                if not charts:
                    return {"success": False, "error": "No valid charts created"}
                
                # Determine layout
                layout_type = tool_args.get('layout', 'auto')
                layout_cols = tool_args.get('layout_cols', 2)
                
                # Create dashboard
                try:
                    dashboard = self.viz_tools.create_dashboard(
                        charts=charts,
                        sidebar=sidebar if sidebar else None,
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
                return self.viz_tools.create_bar_chart(x=x, y=y, aggregate_fn=aggregate_fn, title=title)
            
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
            
            elif chart_type == 'choropleth':
                if not y:
                    return None
                return self.viz_tools.create_choropleth_chart(x=x, y=y, title=title)
            
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