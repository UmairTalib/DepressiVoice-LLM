import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys
from datetime import datetime

# Add project root to sys.path for importing local modules
sys.path.append(os.getcwd())

# Import PDF generator if available
try:
    from src.phase5_llm.generate_pdf_report import generate_pdf
except ImportError:
    st.error("Could not import generate_pdf from src.phase5_llm.generate_pdf_report")
    def generate_pdf(json_path, output_path):
        st.error("PDF Generation logic not found.")

# Configuration
SUMMARY_LOG_PATH = "data/output/explanations/summary_log.csv"
JSON_DIR = "data/output/explanations/"
PROB_THRESHOLD = 0.06

st.set_page_config(
    page_title="DepressiVoice | Clinical Cockpit",
    page_icon="🩺",
    layout="wide",
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
    }
    .risk-low {
        color: #27ae60;
        font-weight: bold;
    }
    .sidebar-disclaimer {
        font-size: 0.8rem;
        color: #7f8c8d;
        padding-top: 20px;
        border-top: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if not os.path.exists(SUMMARY_LOG_PATH):
        st.error(f"Data file not found: {SUMMARY_LOG_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(SUMMARY_LOG_PATH)
    # Ensure Participant_ID is string for easier matching
    df['Participant_ID'] = df['Participant_ID'].astype(str)
    return df

def load_patient_json(patient_id):
    json_path = os.path.join(JSON_DIR, f"v7_clinical_summary_{patient_id}.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        return json.load(f)

# --- Sidebar ---
st.sidebar.title("🩺 DepressiVoice-LLM")
st.sidebar.subheader("Clinical Decision Support")

data = load_data()

if data.empty:
    st.sidebar.warning("No screening data found. Please run validation to generate results.")
    st.title("🛡️ Dashboard Recovery Mode")
    st.info("The dashboard source code has been recovered. However, the patient screening data (`summary_log.csv`) is missing due to the Git rebase. Please run your validation scripts to repopulate the experiments folder.")
    st.stop()

page = st.sidebar.radio("Navigation", ["Global Screening View", "Patient Deep-Dive"])

selected_patient_id = None
if page == "Patient Deep-Dive":
    patient_options = []
    for _, row in data.iterrows():
        label = row['Participant_ID']
        if row['Prob'] > PROB_THRESHOLD:
            label = f"🚩 {label}"
        patient_options.append(label)
    
    selected_label = st.sidebar.selectbox("Select Participant ID", options=patient_options, index=0)
    selected_patient_id = selected_label.replace("🚩 ", "").strip()

st.sidebar.markdown("""
<div class="sidebar-disclaimer">
<b>Safety Disclaimer:</b> Decision Support Tool only. Not a substitute for professional psychiatric diagnosis.
</div>
""", unsafe_allow_html=True)

# --- Main Page ---

if page == "Global Screening View":
    st.title("🛡️ Global Screening View: Population Overview")
    
    kpi1, kpi2, kpi3 = st.columns(3)
    high_risk_count = (data['Prob'] > PROB_THRESHOLD).sum()
    kpi1.metric(f"Total Screened ({len(data)})", len(data))
    kpi2.metric("High Risk (Prob > 0.06)", high_risk_count, delta=f"{high_risk_count/len(data)*100:.1f}%", delta_color="inverse")
    kpi3.metric("Mean Error (±3.0 RMSE)", "±3.0", help="Model calibration anchor")

    st.write(f"Screening results for current cohort. High Risk threshold: {PROB_THRESHOLD}.")
    
    tab1, tab2 = st.tabs(["📋 Detailed Screening Log", "📊 Model Insights"])
    
    with tab1:
        search_query = st.text_input("🔍 Search Participant ID", "")
        filtered_df = data.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df['Participant_ID'].str.contains(search_query)]
        
        def color_risk(val):
            color = 'red' if val > PROB_THRESHOLD else 'black'
            return f'color: {color}'
        
        st.dataframe(filtered_df.style.applymap(color_risk, subset=['Prob']), use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Model Honesty: AI Prob/Severity Alignment")
        plot_df = data.copy()
        def get_sev_val(s):
            try: return float(str(s).split('±')[0].strip())
            except: return 0.0
        plot_df['Severity_Val'] = plot_df['Severity'].apply(get_sev_val)
        
        fig_quad = px.scatter(
            plot_df, x='Severity_Val', y='Prob', text='Participant_ID',
            labels={'Severity_Val': 'Predicted PHQ-8 Severity', 'Prob': 'AI-Detected Probability'},
            title="Model Honesty Quadrant Analysis", hover_data=['Participant_ID', 'Severity']
        )
        fig_quad.add_hline(y=PROB_THRESHOLD, line_dash="dash", line_color="red", 
                          annotation_text=f"Clinical Threshold ({PROB_THRESHOLD})", annotation_position="bottom right")
        
        fig_quad.add_annotation(x=3, y=PROB_THRESHOLD/2, text="✅ Healthy Alignment", showarrow=False, font=dict(color="green", size=12))
        fig_quad.add_annotation(x=3, y=0.8, text="🚩 Masked / High-Risk", showarrow=False, font=dict(color="red", size=12))
        fig_quad.add_annotation(x=18, y=PROB_THRESHOLD/2, text="⚖️ Situational Burnout", showarrow=False, font=dict(color="orange", size=12))
        fig_quad.add_annotation(x=18, y=0.8, text="🔥 Overt Crisis", showarrow=False, font=dict(color="darkred", size=12))

        fig_quad.update_traces(marker=dict(size=12, opacity=0.6, line=dict(width=1, color='DarkSlateGrey')), textposition='top center')
        fig_quad.update_layout(height=600, xaxis=dict(range=[0, 24]), yaxis=dict(range=[0, 1.05]))
        st.plotly_chart(fig_quad, use_container_width=True)

elif page == "Patient Deep-Dive" and selected_patient_id:
    patient_data = data[data['Participant_ID'] == selected_patient_id].iloc[0]
    json_data = load_patient_json(selected_patient_id)
    st.title(f"👤 Patient Analysis: {selected_patient_id}")
    
    if json_data:
        st.subheader("📊 Diagnostic Summary")
        m_col1, m_col2 = st.columns([1, 1])
        with m_col1:
            prob = float(patient_data['Prob'])
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = prob, title = {'text': "Depression Probability", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [0, 1]}, 'bar': {'color': "#3498db"},
                    'steps': [{'range': [0, PROB_THRESHOLD], 'color': '#ebf5f1'}, {'range': [PROB_THRESHOLD, 1], 'color': '#fdedec'}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'value': PROB_THRESHOLD}
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with m_col2:
            sev_str = str(patient_data['Severity'])
            try:
                sev_val = float(sev_str.split('±')[0].strip())
                err_val = float(sev_str.split('±')[1].strip())
            except: sev_val, err_val = 0.0, 3.0
            
            fig_sev = go.Figure()
            fig_sev.add_shape(type="rect", x0=max(0, sev_val - err_val), x1=min(24, sev_val + err_val), y0=-0.2, y1=0.2, fillcolor="rgba(230, 126, 34, 0.2)", line=dict(width=0))
            fig_sev.add_trace(go.Bar(y=['PHQ-8 Score'], x=[sev_val], orientation='h', marker=dict(color='#e67e22'), width=0.4))
            fig_sev.update_layout(title="Severity Spectrum (0-24) | Confidence: ±3.0 RMSE", xaxis=dict(range=[0, 24]), height=300, showlegend=False)
            st.plotly_chart(fig_sev, use_container_width=True)

        st.subheader("🕒 Attention Rollout Timeline")
        spikes = json_data.get('attention_spikes', [])
        if spikes:
            spike_df = pd.DataFrame(spikes)
            def get_spike_label(row):
                if row['timestamp'] == "0:00": return "📍 Baseline Anchor: Establishing Vocal Identity & Cognitive Load"
                return "Prosodic/Non-Verbal (Acoustic)" if "[Non-verbal distressed cues]" in row['text'] else "Grammatical/Semantics (Textual)"
            spike_df['Modality'] = spike_df.apply(get_spike_label, axis=1)
            spike_df['seconds'] = spike_df['timestamp'].apply(lambda ts: int(ts.split(':')[0])*60 + int(ts.split(':')[1]) if ':' in ts else 0)
            fig_timeline = px.line(spike_df.sort_values('seconds'), x='seconds', y='importance_score', title="Model Attention Spikes Throughout Interview", markers=True, color='Modality',
                                   color_discrete_map={"📍 Baseline Anchor: Establishing Vocal Identity & Cognitive Load": "#2c3e50", "Prosodic/Non-Verbal (Acoustic)": "#e74c3c", "Grammatical/Semantics (Textual)": "#3498db"})
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.subheader("📝 Senior Psychiatrist Report")
        narrative = json_data.get('clinical_narrative', "Narrative unavailable.")
        st.markdown(f'<div style="background-color: #ffffff; padding: 25px; border-radius: 12px; border-left: 6px solid #16a085; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px;"><h4 style="color: #16a085; margin-top: 0; margin-bottom: 10px;">Clinical Findings & AI Justification</h4></div>', unsafe_allow_html=True)
        st.markdown(narrative)
        
        output_pdf = f"clinical_report_{selected_patient_id}.pdf"
        st.download_button(label="📄 Download Clinical Report (PDF)", data=open(output_pdf, "rb").read() if os.path.exists(output_pdf) else generate_pdf(os.path.join(JSON_DIR, f"v7_clinical_summary_{selected_patient_id}.json"), output_pdf) if True else None, file_name=output_pdf, mime="application/pdf", use_container_width=True)
    else:
        st.error(f"Detailed data (JSON) not found for Patient {selected_patient_id}")
