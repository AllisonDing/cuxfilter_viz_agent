# user_interface.py
"""
Streamlit User Interface for Visualization Agent
Two-column layout with file management on left, chat on right
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import sys
import pandas as pd
import tempfile
import os
import time
from contextlib import redirect_stdout, redirect_stderr
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.chat_agent import ChatAgent
from src.tools.exp_store import VizExperimentStore

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data_preview(filepath: str):
    """
    Load data preview for UI display.
    Uses same logic as viz_tools.load_data() but returns pandas DataFrame for preview.
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        # Mirror viz_tools.load_data() logic
        if filepath.suffix == '.csv':
            df = pd.read_csv(str(filepath))
        elif filepath.suffix == '.parquet':
            df = pd.read_parquet(str(filepath))
        elif filepath.suffix == '.json':
            df = pd.read_json(str(filepath))
        elif filepath.suffix == '.arrow':
            # For Arrow files, try multiple methods
            try:
                import pyarrow as pa
                # Method 1: Try IPC file reader
                with pa.memory_map(str(filepath), 'r') as source:
                    table = pa.ipc.open_file(source).read_all()
                df = table.to_pandas()
            except Exception as e1:
                print(f"Arrow method 1 failed: {e1}")
                try:
                    # Method 2: Try direct IPC open
                    import pyarrow.ipc as ipc
                    with pa.OSFile(str(filepath), 'r') as f:
                        reader = ipc.open_file(f)
                        table = reader.read_all()
                    df = table.to_pandas()
                except Exception as e2:
                    print(f"Arrow method 2 failed: {e2}")
                    # Method 3: Try feather (alternative Arrow format)
                    try:
                        import pyarrow.feather as feather
                        table = feather.read_table(str(filepath))
                        df = table.to_pandas()
                    except Exception as e3:
                        print(f"Arrow method 3 failed: {e3}")
                        print(f"All Arrow loading methods failed. Cannot preview Arrow file.")
                        return None
        else:
            print(f"Unsupported file type: {filepath.suffix}")
            return None
        
        return df
    
    except Exception as e:
        print(f"Preview load error for {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Dashboard Agent", 
    page_icon="📊", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        margin-top: 10px;
    }
    .uploaded-file-info {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .metric-container {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .chat-message {
        margin-bottom: 15px;
    }
    .dashboard-preview {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state."""
    if 'agent' not in st.session_state:
        exp_store = VizExperimentStore()
        st.session_state.agent = ChatAgent(exp_store=exp_store)
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    
    if 'dashboard_created' not in st.session_state:
        st.session_state.dashboard_created = False
    
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    
    if 'pending_input' not in st.session_state:
        st.session_state.pending_input = None
    
    if 'last_viz_file' not in st.session_state:
        st.session_state.last_viz_file = None

# ============================================================================
# LEFT COLUMN - FILE MANAGEMENT
# ============================================================================

def render_left_column():
    """Render left column with file management."""
    st.header("📊 Dashboard Agent")
    st.markdown("*GPU-Accelerated Interactive Dashboards*")
    
    # Choose data loading method
    data_method = st.radio(
        "Data Loading Method",
        ["📁 File Path (for large files)", "⬆️ Upload Files"],
        help="Use File Path for files > 200MB"
    )
    
    uploaded_files = None
    
    # ========== FILE PATH METHOD ==========
    if data_method == "📁 File Path (for large files)":
        st.subheader("📁 Load from File Path")
        file_path = st.text_input(
            "Enter file path",
            placeholder="./data/sales_data.csv",
            help="Full or relative path to your data file"
        )
        
        if st.button("Load File") and file_path:
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                file_key = file_name
                
                # Check if already loaded
                if file_key not in st.session_state.uploaded_files:
                    try:
                        file_size = os.path.getsize(file_path)
                        
                        # For Arrow files, load directly to viz_tools (skip pandas preview)
                        if file_path.endswith('.arrow'):
                            # Load data into viz_tools
                            result = st.session_state.agent.viz_tools.load_data(file_path)
                            
                            # Check if result is dict (error) or cxf_df (success)
                            if isinstance(result, dict) and not result.get('success', True):
                                st.error(f"Error: {result['error']}")
                            else:
                                # Get info from viz_tools
                                info = st.session_state.agent.viz_tools.get_data_info()
                                
                                st.session_state.uploaded_files[file_key] = {
                                    'path': file_path,
                                    'name': file_name,
                                    'size': file_size,
                                    'shape': info['shape'],
                                    'columns': info['columns']
                                }
                                
                                st.success(f"✅ Loaded {file_name}")
                                st.rerun()
                        else:
                            # For other formats, load preview first
                            df = load_data_preview(file_path)
                            
                            if df is None:
                                st.error(f"Could not load file for preview")
                            else:
                                st.session_state.uploaded_files[file_key] = {
                                    'path': file_path,
                                    'name': file_name,
                                    'size': file_size,
                                    'shape': df.shape,
                                    'columns': list(df.columns)
                                }
                                
                                # Load data into viz_tools
                                result = st.session_state.agent.viz_tools.load_data(file_path)
                                
                                # Check if result is dict (error) or cxf_df (success)
                                if isinstance(result, dict) and not result.get('success', True):
                                    st.error(f"Error: {result['error']}")
                                else:
                                    # Success
                                    st.success(f"✅ Loaded {file_name}")
                                    st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
                else:
                    st.warning("File already loaded!")
            else:
                st.error(f"File not found: {file_path}")
    
    # ========== FILE UPLOAD METHOD ==========
    else:
        uploaded_files = st.file_uploader(
            "Upload Datasets", 
            type=['csv', 'parquet', 'arrow'], 
            accept_multiple_files=True,
            help="Upload CSV, Parquet, or Arrow files (max 200MB each)"
        )
    
    # Process uploaded files
    if uploaded_files:
        st.subheader("📁 Uploaded Files")
        
        for uploaded_file in uploaded_files:
            file_key = uploaded_file.name
            
            # Save file if not already saved
            if file_key not in st.session_state.uploaded_files:
                # Create temp file
                suffix = f'.{uploaded_file.name.split(".")[-1]}'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Store file info
                st.session_state.uploaded_files[file_key] = {
                    'path': tmp_path,
                    'name': uploaded_file.name,
                    'size': len(uploaded_file.getvalue())
                }
                
                # Load and show basic info
                try:
                    # For Arrow files, load directly to viz_tools (skip pandas preview)
                    if uploaded_file.name.endswith('.arrow'):
                        # Load into viz_tools
                        result = st.session_state.agent.viz_tools.load_data(tmp_path)
                        
                        # Check for errors
                        if isinstance(result, dict) and not result.get('success', True):
                            st.error(f"Error loading {uploaded_file.name}: {result['error']}")
                            continue
                        
                        # Get info from viz_tools
                        info = st.session_state.agent.viz_tools.get_data_info()
                        st.session_state.uploaded_files[file_key]['shape'] = info['shape']
                        st.session_state.uploaded_files[file_key]['columns'] = info['columns']
                    else:
                        # For other formats, load preview first
                        df = load_data_preview(tmp_path)
                        
                        if df is None:
                            st.error(f"Could not load {uploaded_file.name} for preview")
                            continue
                        
                        st.session_state.uploaded_files[file_key]['shape'] = df.shape
                        st.session_state.uploaded_files[file_key]['columns'] = list(df.columns)
                        
                        # Load into viz_tools
                        result = st.session_state.agent.viz_tools.load_data(tmp_path)
                        
                        # Check for errors
                        if isinstance(result, dict) and not result.get('success', True):
                            st.error(f"Error loading {uploaded_file.name}: {result['error']}")
                            continue
                    
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                    continue
    
    # Display file info
    if st.session_state.uploaded_files:
        st.divider()
        for file_key, file_info in st.session_state.uploaded_files.items():
            if 'shape' in file_info:
                shape = file_info['shape']
                size_mb = file_info['size'] / (1024 * 1024)
                
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{file_info['name']}**")
                        st.caption(f"{shape[0]:,} rows × {shape[1]} columns | {size_mb:.1f} MB")
                    with col2:
                        if st.button("🗑️", key=f"del_{file_key}"):
                            # Remove from session
                            del st.session_state.uploaded_files[file_key]
                            # Remove temp file
                            if os.path.exists(file_info['path']):
                                try:
                                    os.unlink(file_info['path'])
                                except:
                                    pass
                            st.rerun()
                
                # Show columns in expander
                with st.expander(f"Columns ({len(file_info['columns'])})"):
                    for col in file_info['columns'][:20]:  # Show first 20
                        st.text(f"• {col}")
                    if len(file_info['columns']) > 20:
                        st.caption(f"... and {len(file_info['columns']) - 20} more")
    
    # ========== QUICK EXAMPLES ==========
    st.divider()
    st.subheader("💡 Example Prompts")
    
    if st.button("📊 'Analyze my data'", use_container_width=True):
        st.session_state.pending_input = "Analyze my data"
        st.session_state.is_processing = True
        st.rerun()
    
    if st.button("📈 'Create a bar chart'", use_container_width=True):
        st.session_state.pending_input = "Create a standalone bar chart"
        st.session_state.is_processing = True
        st.rerun()
    
    if st.button("📊 'Build dashboard'", use_container_width=True):
        st.session_state.pending_input = "Build a dashboard with multiple charts"
        st.session_state.is_processing = True
        st.rerun()
    
    # ========== REFERENCE DOCS ==========
    st.divider()
    
    # Chart types reference
    with st.expander("📊 Available Charts", expanded=False):
        st.markdown("""
        **Basic Charts:**
        - `bar` - Bar chart
        - `line` - Line chart
        - `scatter` - Scatter plot
        - `heatmap` - Heatmap
        
        **Data Display:**
        - `view_dataframe` - Data table

        **Advanced:**
        - `choropleth` - 2D/3D Choropleth map
        
        **Widgets:**
        - `range_slider` - Range slider
        - `date_range_slider` - Date range slider
        - `float_slider` - Float slider
        - `int_slider` - Integer slider
        - `drop_down` - Dropdown selector
        - `multi_select` - Multi-select
        - `number_chart` - Number display
        """)
    
    # Layouts reference
    with st.expander("📐 Layouts", expanded=False):
        st.markdown("""
        **Preset Layouts:**
        - `single_feature`: 1 chart
        - `feature_base`: 1 feature chart and 1 base chart
        - `double_feature`: 2 charts
        - `left_feature_right_double`: 1 feature chart on the left and 2 charts on the right
        - `triple_feature`: 3 charts
        - `feature_and_double_base`: 1 feature chart and 2 base charts
        - `two_by_two`: 4 charts
        - `feature_and_triple_base`: 1 feature chart and 3 base charts
        - `feature_and_quad_base`: 1 feature chart and 4 base charts
        - `feature_and_five_edge`: 1 feature chart and 5 edge charts
        - `two_by_three`: 2x3 charts
        - `double_feature_quad_base`: 2 feature charts and 4 base charts
        - `three_by_three`: 3x3 charts
        
        **Dynamic Layouts:**
        - `auto`: Automatic selection
        - `grid`: Grid layout (specify cols with layout_cols parameter)
        """)
    
    # Themes reference
    with st.expander("🎨 Themes", expanded=False):
        st.markdown("""
        - `rapids_dark` - RAPIDS dark
        - `rapids` - RAPIDS
        - `dark` - Dark
        - `default` - Default
        """)

# ============================================================================
# RIGHT COLUMN - CHAT INTERFACE
# ============================================================================

def render_right_column():
    """Render right column with chat interface."""
    st.header("💬 Chat with Dashboard Agent")
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("ℹ️ Data Info"):
            info = st.session_state.agent.viz_tools.get_data_info()
            if info['success']:
                info_msg = f"""**Data Information:**
                
📊 Shape: {info['shape'][0]:,} rows × {info['shape'][1]} columns

**Columns:** {', '.join(info['columns'])}
"""
                st.session_state.messages.append({"role": "assistant", "content": info_msg})
                st.rerun()
            else:
                st.warning("No data loaded")
    
    with col3:
        if st.button("📈 Stats"):
            try:
                summary = st.session_state.agent.exp_store.get_experiment_summary()
                
                stats_msg = f"""**Dashboard Statistics:**
                
📊 Total: {summary['total_experiments']}
✅ Success: {summary['successful_experiments']} ({summary.get('success_rate', 'N/A')})
❌ Failed: {summary['failed_experiments']}

**Popular Chart Types:**
{chr(10).join([f"• {k}: {v}" for k, v in list(summary.get('chart_usage', {}).items())[:5]])}

**Popular Layouts:**
{chr(10).join([f"• {k}: {v}" for k, v in list(summary.get('layout_usage', {}).items())[:5]])}

**Averages:**
• Charts per dashboard: {summary.get('avg_charts_per_dashboard', 'N/A')}
• Widgets per dashboard: {summary.get('avg_widgets_per_dashboard', 'N/A')}
"""
                st.session_state.messages.append({"role": "assistant", "content": stats_msg})
                st.rerun()
            except Exception as e:
                st.error(f"Could not load stats: {e}")
    
    # Display messages with dashboard rendering
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display dashboard if this message has one
            if message.get("viz_file"):
                viz_path = Path(message["viz_file"])
                if viz_path.exists():
                    try:
                        with open(viz_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        st.subheader("📊 Interactive Dashboard")
                        components.html(html_content, height=800, scrolling=True)
                        
                        # Download button for each dashboard
                        st.download_button(
                            label="⬇️ Download Dashboard",
                            data=html_content,
                            file_name=viz_path.name,
                            mime="text/html",
                            key=f"download_{idx}"
                        )
                    except Exception as e:
                        st.error(f"Error displaying dashboard: {e}")
    
    # Show input only if not processing
    if not st.session_state.is_processing:
        user_input = st.chat_input("Ask about dashboards or give visualization commands...")
        
        if user_input:
            st.session_state.pending_input = user_input
            st.session_state.is_processing = True
            st.rerun()
    
    else:
        # Process pending input
        if st.session_state.pending_input:
            user_input = st.session_state.pending_input
            st.session_state.pending_input = None
            
            start_time = time.time()
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Create placeholders
            user_msg_container = st.empty()
            assistant_msg_container = st.empty()
            
            # Show user message
            with user_msg_container.container():
                with st.chat_message("user"):
                    st.markdown(user_input)
            
            # Show processing message
            with assistant_msg_container.container():
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    response_placeholder.markdown("⏳ Processing your request...")
            
            captured_output = io.StringIO()
            try:
                # Get response from agent
                with redirect_stdout(captured_output), redirect_stderr(captured_output):
                    result = st.session_state.agent.chat(user_input)
                
                response = result['response']
                
                # Check if dashboard was created
                if result.get('viz_file'):
                    st.session_state.dashboard_created = True
                    st.session_state.last_viz_file = result['viz_file']
                
                # Update processing message
                response_placeholder.markdown("💭 Generating response...")
                time.sleep(0.3)
                
                # Combine with terminal output
                terminal_output = captured_output.getvalue().strip()
                if terminal_output:
                    full_response = response + "\n\n**Output:**\n```\n" + terminal_output + "\n```"
                else:
                    full_response = response
                
                # Add timing
                end_time = time.time()
                execution_time = end_time - start_time
                
                if execution_time < 60:
                    time_str = f"{execution_time:.2f} seconds"
                else:
                    minutes = int(execution_time // 60)
                    seconds = execution_time % 60
                    time_str = f"{minutes} min {seconds:.1f} sec"
                
                full_response += f"\n\n---\n⏱️ **Processing time: {time_str}**"
                
                # Display visualization if created
                if result.get('viz_file'):
                    viz_path = Path(result['viz_file'])
                    if viz_path.exists():
                        full_response += f"\n\n📊 **Dashboard:** {viz_path.name}"
                        
                        # Read HTML
                        with open(viz_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        # Clear placeholder and show final response
                        response_placeholder.empty()
                        
                        with assistant_msg_container.container():
                            with st.chat_message("assistant"):
                                st.markdown(full_response)
                                st.subheader("📊 Interactive Dashboard")
                                components.html(html_content, height=800, scrolling=True)
                                
                                # Download button
                                st.download_button(
                                    label="⬇️ Download Dashboard",
                                    data=html_content,
                                    file_name=viz_path.name,
                                    mime="text/html"
                                )
                        
                        # Save message with viz_file reference
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response,
                            "viz_file": str(viz_path)
                        })
                    else:
                        full_response += "\n\n⚠️ Dashboard file not found"
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Stream response (if no viz)
                else:
                    words = full_response.split()
                    for i in range(1, len(words) + 1):
                        partial_response = " ".join(words[:i])
                        response_placeholder.markdown(partial_response)
                        time.sleep(0.02)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                
                end_time = time.time()
                execution_time = end_time - start_time
                if execution_time < 60:
                    time_str = f"{execution_time:.2f} seconds"
                else:
                    minutes = int(execution_time // 60)
                    seconds = execution_time % 60
                    time_str = f"{minutes} min {seconds:.1f} sec"
                
                error_msg += f"\n\n---\n⏱️ **Processing time: {time_str}**"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            finally:
                st.session_state.is_processing = False
                st.rerun()

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar."""
    if st.session_state.uploaded_files:
        with st.sidebar:
            st.header("📋 Quick Reference")
            st.subheader("Loaded Files")
            for file_key, file_info in st.session_state.uploaded_files.items():
                st.write(f"**{file_info['name']}**")
                if 'shape' in file_info:
                    st.caption(f"{file_info['shape'][0]:,} × {file_info['shape'][1]}")
            
            st.divider()
            
            # Quick actions
            st.subheader("⚡ Quick Actions")
            
            # Download latest dashboard button
            if st.session_state.last_viz_file:
                viz_path = Path(st.session_state.last_viz_file)
                if viz_path.exists():
                    try:
                        with open(viz_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        st.download_button(
                            label="📥 Download Latest Dashboard",
                            data=html_content,
                            file_name=viz_path.name,
                            mime="text/html",
                            use_container_width=True,
                            type="primary"
                        )
                        st.caption(f"📄 {viz_path.name}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            if st.button("🔄 Reload Agent", use_container_width=True):
                exp_store = VizExperimentStore()
                st.session_state.agent = ChatAgent(exp_store=exp_store)
                st.success("Agent reloaded!")
                time.sleep(1)
                st.rerun()
            
            if st.button("📁 Open Outputs", use_container_width=True):
                output_dir = Path("viz_outputs")
                if not output_dir.exists():
                    output_dir.mkdir()
                st.info(f"Outputs folder: {output_dir.absolute()}")
            
            st.divider()
            st.caption("📊 Dashboard Agent v1.0")
            st.caption("GPU-Accelerated with RAPIDS")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main application."""
    # Initialize
    init_session_state()
    
    # Layout
    left_col, right_col = st.columns([3, 7])
    
    with left_col:
        render_left_column()
    
    with right_col:
        render_right_column()
    
    # Sidebar
    render_sidebar()

if __name__ == "__main__":
    main()