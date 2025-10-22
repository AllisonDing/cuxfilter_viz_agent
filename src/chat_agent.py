# src/chat_agent.py
"""
Chat Agent - Orchestrator for ML and Visualization Tasks

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
from src.tools.exp_store import ExperimentStore


class ChatAgent:
    """
    Agent orchestrator coordinating LLM with visualization and ML tools.
    """
    
    def __init__(self, exp_store: Optional[ExperimentStore] = None):
        """
        Initialize chat agent.
        
        Args:
            exp_store: ExperimentStore instance (optional)
        """
        # Initialize LLM client
        self.llm = create_client()
        
        # Initialize tools
        self.viz_tools = VizTools()
        self.exp_store = exp_store if exp_store else ExperimentStore()
        
        # Conversation history
        self.messages = []
        
        # System prompt
        self.system_prompt = self._default_system_prompt()
        self.messages.append({"role": "system", "content": self.system_prompt})
        
        # Last visualization file
        self.last_viz_file = None
    
    def _default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a helpful AI assistant with visualization capabilities.

You can:
1. Load and analyze data (CSV, Parquet, JSON)
2. Create interactive GPU-accelerated dashboards using CuxFilter
3. Track and visualize ML experiments

Available visualizations:
- Charts: bar, line, scatter, histogram, heatmap
- Geographic: point maps, hex maps
- Widgets: sliders, dropdowns, filters
- Layouts: single, double, triple, quad feature
- Themes: rapids_dark, rapids, dark, light

When users ask for visualizations:
1. First load their data if not loaded
2. Analyze columns to understand data types
3. Create appropriate charts
4. Export as interactive HTML dashboard

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
                    "description": "Load data from CSV, Parquet, or JSON file for visualization",
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
                    "description": "Analyze loaded data to understand columns and suggest visualizations",
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
                    "description": "Create interactive dashboard with charts and export as HTML",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "charts": {
                                "type": "array",
                                "description": "List of charts to create",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": ["bar", "line", "scatter", "histogram", "heatmap", 
                                                    "point_map", "hex_map", "range_slider", "dropdown"],
                                            "description": "Type of chart"
                                        },
                                        "x": {"type": "string", "description": "Column for x-axis"},
                                        "y": {"type": "string", "description": "Column for y-axis (optional)"},
                                        "title": {"type": "string", "description": "Chart title"}
                                    },
                                    "required": ["type", "x"]
                                }
                            },
                            "layout": {
                                "type": "string",
                                "enum": ["auto", "single_feature", "double_feature", "triple_feature", "quad_feature"],
                                "description": "Dashboard layout (default: auto)"
                            },
                            "theme": {
                                "type": "string",
                                "enum": ["rapids_dark", "rapids", "dark", "light"],
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
                result = self.viz_tools.load_data(tool_args['filepath'])
                return result
            
            elif tool_name == "analyze_data":
                result = self.viz_tools.analyze_columns()
                return result
            
            elif tool_name == "get_data_info":
                result = self.viz_tools.get_data_info()
                return result
            
            elif tool_name == "create_visualization":
                # Create charts
                charts = []
                for chart_config in tool_args['charts']:
                    chart = self._create_chart(chart_config)
                    if chart:
                        charts.append(chart)
                
                if not charts:
                    return {"success": False, "error": "No valid charts created"}
                
                # Create dashboard
                dashboard_result = self.viz_tools.create_dashboard(
                    charts=charts,
                    layout_type=tool_args.get('layout', 'auto'),
                    theme_name=tool_args.get('theme', 'rapids_dark'),
                    title=tool_args.get('title', 'Dashboard')
                )
                
                if dashboard_result['success']:
                    # Export to HTML
                    export_result = self.viz_tools.export_dashboard()
                    
                    if export_result['success']:
                        self.last_viz_file = export_result['filepath']
                        return {
                            "success": True,
                            "message": f"Dashboard created with {len(charts)} charts",
                            "filepath": export_result['filepath'],
                            "dashboard_info": dashboard_result
                        }
                
                return dashboard_result
            
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
        
        try:
            if chart_type == 'bar':
                return self.viz_tools.create_bar_chart(x=x, y=y, title=title)
            elif chart_type == 'line':
                return self.viz_tools.create_line_chart(x=x, y=y, title=title)
            elif chart_type == 'scatter':
                return self.viz_tools.create_scatter_chart(x=x, y=y, title=title)
            elif chart_type == 'histogram':
                return self.viz_tools.create_histogram(x=x, title=title)
            elif chart_type == 'heatmap':
                return self.viz_tools.create_heatmap(x=x, y=y, title=title)
            elif chart_type == 'point_map':
                return self.viz_tools.create_point_map(x=x, y=y, title=title)
            elif chart_type == 'hex_map':
                return self.viz_tools.create_hex_map(x=x, y=y, title=title)
            elif chart_type == 'range_slider':
                return self.viz_tools.create_range_slider(x=x, title=title)
            elif chart_type == 'dropdown':
                return self.viz_tools.create_dropdown(x=x, title=title)
            else:
                return None
        except Exception as e:
            print(f"Error creating chart {chart_type}: {e}")
            return None
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.messages.copy()