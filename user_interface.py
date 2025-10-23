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
                        # Load file info
                        if file_path.endswith('.parquet'):
                            df = pd.read_parquet(file_path)
                        elif file_path.endswith('.arrow'):
                            df = pd.read_feather(file_path)
                        else:
                            df = pd.read_csv(file_path)
                        
                        file_size = os.path.getsize(file_path)
                        
                        st.session_state.uploaded_files[file_key] = {
                            'path': file_path,
                            'name': file_name,
                            'size': file_size,
                            'shape': df.shape,
                            'columns': list(df.columns)
                        }
                        
                        # Load data into viz_tools
                        result = st.session_state.agent.viz_tools.load_data(file_path)
                        
                        if result['success']:
                            st.success(f"✅ Loaded {file_name}")
                            st.rerun()
                        else:
                            st.error(f"Error: {result['error']}")
                        
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
                    if uploaded_file.name.endswith('.parquet'):
                        df = pd.read_parquet(tmp_path)
                    elif uploaded_file.name.endswith('.arrow'):
                        df = pd.read_feather(tmp_path)
                    else:
                        df = pd.read_csv(tmp_path)
                    
                    st.session_state.uploaded_files[file_key]['shape'] = df.shape
                    st.session_state.uploaded_files[file_key]['columns'] = list(df.columns)
                    
                    # Load into viz_tools
                    result = st.session_state.agent.viz_tools.load_data(tmp_path)
                    
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
    
    if st.button("📈 'Create histogram'", use_container_width=True):
        st.session_state.pending_input = "Create a histogram of the first numeric column"
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
        - `histogram` - Histogram
        - `heatmap` - Heatmap
        
        **Geographic:**
        - `point_map` - Point map
        - `hex_map` - Hexbin map
        
        **Widgets:**
        - `range_slider` - Range slider
        - `dropdown` - Dropdown selector
        - `multi_select` - Multi-select
        """)
    
    # Layouts reference
    with st.expander("📐 Layouts", expanded=False):
        st.markdown("""
        - `single_feature` - 1 chart
        - `double_feature` - 2 charts
        - `triple_feature` - 3 charts
        - `quad_feature` - 4 charts
        - `auto` - Automatic selection
        """)
    
    # Themes reference
    with st.expander("🎨 Themes", expanded=False):
        st.markdown("""
        - `rapids_dark` - RAPIDS dark (default)
        - `rapids` - RAPIDS light
        - `dark` - Standard dark
        - `light` - Standard light
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
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
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
                        
                        # Read and display HTML
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
                    else:
                        full_response += "\n\n⚠️ Dashboard file not found"
                
                # Stream response (if no viz)
                if not result.get('viz_file'):
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