# viz_user_interface.py - Streamlit UI for Cuxfilter Dashboard Agent
import streamlit as st
import pandas as pd
import tempfile
import os
import io
from contextlib import redirect_stdout, redirect_stderr
import time
from functools import wraps

from src.chat_agent import ChatAgent

# ============= TIMING DECORATOR =============
def track_execution_time(method_name):
    """Decorator to track and display execution time for dashboard methods"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            print(f"Starting {method_name}...")
            
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            if execution_time < 60:
                time_str = f"{execution_time:.2f} seconds"
            else:
                minutes = int(execution_time // 60)
                seconds = execution_time % 60
                time_str = f"{minutes} min {seconds:.1f} sec"
            
            print(f"\nTotal execution time for {method_name}: {time_str}")
            print(f"{method_name} completed successfully!")
            
            return result
        return wrapper
    return decorator

# Monkey-patch ChatAgent methods for timing
if "timing_patched" not in st.session_state:
    st.session_state.timing_patched = True
# ============= END TIMING DECORATOR =============

st.set_page_config(page_title="Dashboard Agent", page_icon="📊", layout="wide")

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

# Initialize session state
if "viz_agent" not in st.session_state:
    st.session_state.viz_agent = ChatAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "dashboard_created" not in st.session_state:
    st.session_state.dashboard_created = False

# Layout
left_col, right_col = st.columns([3, 7])

with left_col:
    st.header("📊 Dashboard Agent")
    st.markdown("*GPU-Accelerated Interactive Dashboards*")
    
    # Choose data loading method
    data_method = st.radio(
        "Data Loading Method",
        ["📁 File Path (for large files)", "⬆️ Upload Files"],
        help="Use File Path for files > 200MB"
    )
    
    # FIX: Initialize uploaded_files to avoid NameError
    uploaded_files = None
    
    if data_method == "📁 File Path (for large files)":
        # File path input
        st.subheader("📁 Load from File Path")
        file_path = st.text_input(
            "Enter file path",
            placeholder="./data/auto_accidents.arrow",
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
                        
                        # Register with agent
                        st.session_state.viz_agent.uploaded_files[file_name] = file_path
                        base_name = file_name.split('.')[0]
                        st.session_state.viz_agent.uploaded_files[base_name] = file_path
                        
                        st.success(f"✅ Loaded {file_name}")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
                else:
                    st.warning("File already loaded!")
            else:
                st.error(f"File not found: {file_path}")
    
    else:
        # Original file upload
        uploaded_files = st.file_uploader(
            "Upload Datasets", 
            type=['csv', 'parquet', 'arrow'], 
            accept_multiple_files=True,
            help="Upload CSV, Parquet, or Arrow files (max 200MB each)"
        )
    
    # Process uploaded files (only if files were uploaded)
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
                    
                    # Register file with viz agent
                    st.session_state.viz_agent.uploaded_files[uploaded_file.name] = tmp_path
                    
                    # Also register without extension
                    base_name = uploaded_file.name.split('.')[0]
                    st.session_state.viz_agent.uploaded_files[base_name] = tmp_path
                    
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                    continue
        
        # Display file info
        for file_key, file_info in st.session_state.uploaded_files.items():
            if 'shape' in file_info:
                shape = file_info['shape']
                size_mb = file_info['size'] / (1024 * 1024)
                
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{file_info['name']}**")
                        st.caption(f"{shape[0]:,} rows × {shape[1]} cols | {size_mb:.2f} MB")
                    with col2:
                        if st.button("📋", key=f"copy_{file_key}", help="Copy filename"):
                            st.code(file_info['name'])
                
                # Show columns in expander
                with st.expander(f"Columns", expanded=False):
                    cols_text = ", ".join(file_info['columns'])
                    st.text(cols_text)
        
        # Show helpful message
        st.info("💡 To load a file, type: `load data filename.arrow`")
        
        # Show example for first file
        if st.session_state.uploaded_files:
            first_file = list(st.session_state.uploaded_files.values())[0]
            st.code(f"load data {first_file['name']}")
    
    # Display file info for files loaded via path method
    elif st.session_state.uploaded_files and data_method == "📁 File Path (for large files)":
        st.subheader("📁 Loaded Files")
        for file_key, file_info in st.session_state.uploaded_files.items():
            if 'shape' in file_info:
                shape = file_info['shape']
                size_mb = file_info['size'] / (1024 * 1024)
                
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{file_info['name']}**")
                        st.caption(f"{shape[0]:,} rows × {shape[1]} cols | {size_mb:.2f} MB")
                    with col2:
                        if st.button("📋", key=f"copy_{file_key}", help="Copy filename"):
                            st.code(file_info['name'])
                
                # Show columns in expander
                with st.expander(f"Columns", expanded=False):
                    cols_text = ", ".join(file_info['columns'])
                    st.text(cols_text)
        
        # Show helpful message
        st.info("💡 To load a file, type: `load data filename.arrow`")
        
        # Show example for first file
        if st.session_state.uploaded_files:
            first_file = list(st.session_state.uploaded_files.values())[0]
            st.code(f"load data {first_file['name']}")
    
    # Dashboard status
    st.divider()
    st.subheader("📈 Dashboard Status")
    
    if st.session_state.dashboard_created:
        st.success("✅ Dashboard created")
    else:
        st.info("ℹ️ No dashboard yet")
    
    # Check for exported dashboards
    if os.path.exists("outputs"):
        dashboard_files = [f for f in os.listdir("outputs") if f.endswith('.html')]
        if dashboard_files:
            st.write(f"**Exported:** {len(dashboard_files)} dashboard(s)")
            with st.expander("View exports", expanded=False):
                for df_file in dashboard_files:
                    st.write(f"• {df_file}")
    
    # Quick examples
    st.divider()
    with st.expander("💡 Example Commands", expanded=False):
        st.markdown("""
        **Load data:**
        ```
        load data auto_accidents.arrow
        load data sales.csv
        ```
        
        **Create dashboard:**
        ```
        create dashboard with:
        - scatter map
        - bar chart for categories
        - layout: feature_and_base
        - theme: rapids_dark
        ```
        
        **Quick dashboard:**
        ```
        create dashboard with scatter, bar, and line charts
        use rapids_dark theme
        ```
        
        **Add charts:**
        ```
        add heatmap and histogram
        add filter for region column
        ```
        
        **Customize:**
        ```
        change theme to dark
        change layout to two_by_two
        ```
        
        **Show & Export:**
        ```
        show dashboard
        export dashboard to outputs/my_viz.html
        ```
        """)
    
    # Chart types reference
    with st.expander("📊 Chart Types", expanded=False):
        st.markdown("""
        **Available Charts:**
        - `scatter` - Scatter plot (with optional map tiles)
        - `bar` - Bar chart
        - `line` - Line chart
        - `histogram` - Histogram
        - `heatmap` - Heatmap
        - `choropleth` - Geographic map
        - `number` - KPI/metric widget
        
        **Widgets:**
        - `multi_select` - Multi-select filter
        - `drop_down` - Dropdown filter
        - `range_slider` - Range slider
        - `date_range` - Date range picker
        """)
    
    # Layouts reference
    with st.expander("📐 Layouts", expanded=False):
        st.markdown("""
        - `feature_and_base`
        - `two_by_two`
        - `three_by_three`
        - `single_feature`
        - `double_feature`
        - `triple_feature`
        - `feature_and_triple_base`
        - And 8 more...
        """)
    
    # Themes reference
    with st.expander("🎨 Themes", expanded=False):
        st.markdown("""
        - `default` - Light theme
        - `dark` - Dark theme
        - `rapids` - Rapids light
        - `rapids_dark` - Rapids dark
        """)

with right_col:
    st.header("💬 Chat with Dashboard Agent")
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("📊 Show History"):
            response = st.session_state.viz_agent._show_history()
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col3:
        if st.button("📈 Stats"):
            try:
                from src.tools.viz_exp_store import VizExperimentStore
                store = VizExperimentStore()
                summary = store.get_experiment_summary()
                
                stats_msg = f"""**Dashboard Statistics:**
                
📊 Total: {summary['total_experiments']}
✅ Success: {summary['successful_experiments']} ({summary.get('success_rate', 'N/A')})
❌ Failed: {summary['failed_experiments']}

**Popular Chart Types:**
{chr(10).join([f"• {k}: {v}" for k, v in list(summary.get('chart_usage', {}).items())[:5]])}

**Popular Layouts:**
{chr(10).join([f"• {k}: {v}" for k, v in list(summary.get('layout_usage', {}).items())[:5]])}
"""
                st.session_state.messages.append({"role": "assistant", "content": stats_msg})
                st.rerun()
            except Exception as e:
                st.error(f"Could not load stats: {e}")
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Initialize flags
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None

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
                    response_placeholder.markdown("⏳ Creating your dashboard...")

            captured_output = io.StringIO()
            try:
                # Get response from agent
                with redirect_stdout(captured_output), redirect_stderr(captured_output):
                    response = st.session_state.viz_agent.chat(user_input)
                
                # Check if dashboard was created
                if "create_dashboard" in user_input.lower() and "✓" in response:
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
                
                # Stream response
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

# Sidebar with file reference
if st.session_state.uploaded_files:
    with st.sidebar:
        st.header("📋 Quick Reference")
        st.subheader("Uploaded Files")
        for file_key, file_info in st.session_state.uploaded_files.items():
            st.write(f"**{file_info['name']}**")
            if 'shape' in file_info:
                st.caption(f"{file_info['shape'][0]:,} × {file_info['shape'][1]}")
        
        st.divider()
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🔄 Reload Agent", use_container_width=True):
            st.session_state.viz_agent = ChatAgent()
            st.success("Agent reloaded!")
            time.sleep(1)
            st.rerun()
        
        if st.button("📁 Open Outputs Folder", use_container_width=True):
            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            st.info("Outputs folder: ./outputs/")

# Cleanup temp files on session end
import atexit

def cleanup_temp_files():
    for file_info in st.session_state.get('uploaded_files', {}).values():
        if 'path' in file_info and os.path.exists(file_info['path']):
            try:
                os.unlink(file_info['path'])
            except:
                pass

atexit.register(cleanup_temp_files)

# Footer
st.sidebar.divider()
st.sidebar.caption("📊 Cuxfilter Dashboard Agent v1.0")
st.sidebar.caption("GPU-Accelerated Dashboards with RAPIDS")