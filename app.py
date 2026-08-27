# ============================================================
#  AI-Based Skin Cancer Detection Using Deep Learning
#  on Dermoscopic Images
#  Developed by: Golam Muktadir Al Sabir & Jahidul Islam Jisan
#  v5.1 — working save, auto-clear form, persistent clinician,
#          patterned clinical background, MOBILE RESPONSIVE LAYER
# ============================================================

import os
import re
import json
import uuid
import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageStat, ImageFilter
from ultralytics import YOLO

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY = True
except ImportError:
    PLOTLY = False


# ------------------------------------------------------------
# FORCE LIGHT (DAY) THEME — auto-create .streamlit/config.toml
# ------------------------------------------------------------
def ensure_light_theme():
    cfg_dir = ".streamlit"
    cfg = os.path.join(cfg_dir, "config.toml")
    body = (
        "[theme]\n"
        'base = "light"\n'
        'primaryColor = "#0b3d63"\n'
        'backgroundColor = "#eef4f9"\n'
        'secondaryBackgroundColor = "#ffffff"\n'
        'textColor = "#0f2438"\n'
        'font = "sans serif"\n\n'
        "[client]\n"
        'toolbarMode = "minimal"\n'
    )
    try:
        os.makedirs(cfg_dir, exist_ok=True)
        if not os.path.exists(cfg):
            with open(cfg, "w", encoding="utf-8") as fh:
                fh.write(body)
            return True
    except Exception:
        pass
    return False


THEME_CREATED = ensure_light_theme()

# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Skin Cancer Detection | Dermoscopic Analysis",
    page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

DATA_DIR = "records"
IMG_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "patient_records.csv")
DOC_PATH = os.path.join(DATA_DIR, "clinician.json")
os.makedirs(IMG_DIR, exist_ok=True)

# ------------------------------------------------------------
# CSS — clinical day theme with soft medical pattern background
# ------------------------------------------------------------
st.markdown("""
<style>
:root{
    --ink:#0f2438; --muted:#5a7186; --line:#dde7ef;
    --surface:#ffffff; --bg:#eef4f9;
    --navy:#0b3d63; --navy2:#125e8a; --teal:#0d9488;
    --gold:#b9821f; --red:#c62828; --green:#1f8a4c;
}

/* ---------- PATTERNED CLINICAL BACKGROUND ---------- */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"]{
    color:var(--ink) !important;
    background-color:#eef4f9 !important;
    background-image:
        radial-gradient(circle at 12% 8%, rgba(13,148,136,.10) 0%, rgba(13,148,136,0) 42%),
        radial-gradient(circle at 88% 4%, rgba(18,94,138,.12) 0%, rgba(18,94,138,0) 40%),
        radial-gradient(circle at 78% 92%, rgba(11,61,99,.09) 0%, rgba(11,61,99,0) 45%),
        url("data:image/svg+xml;utf8,\
<svg xmlns='http://www.w3.org/2000/svg' width='90' height='90' viewBox='0 0 90 90'>\
<g fill='none' stroke='%230b3d63' stroke-opacity='0.055' stroke-width='2'>\
<path d='M20 8v24M8 20h24'/>\
<path d='M70 58v24M58 70h24'/>\
<circle cx='20' cy='20' r='15'/>\
<circle cx='70' cy='70' r='15'/>\
</g></svg>") !important;
    background-attachment:fixed !important;
    background-repeat:no-repeat, no-repeat, no-repeat, repeat !important;
}
[data-testid="block-container"]{ background:transparent !important; }
[data-testid="stHeader"], [data-testid="stToolbar"]{
    background:rgba(238,244,249,.88) !important;
    backdrop-filter:blur(6px); }
[data-testid="stHeader"] *, [data-testid="stToolbar"] *{
    color:var(--ink) !important; }
[data-testid="stAppViewContainer"] .main .block-container{
    padding:1.1rem 2.1rem 2.4rem; max-width:1540px; }

/* ---------- GLOBAL TEXT VISIBILITY ---------- */
[data-testid="stMain"] p, [data-testid="stMain"] span,
[data-testid="stMain"] li, [data-testid="stMain"] label,
[data-testid="stMain"] h1, [data-testid="stMain"] h2,
[data-testid="stMain"] h3, [data-testid="stMain"] h4,
[data-testid="stMain"] h5, [data-testid="stMain"] h6,
[data-testid="stMain"] div[data-testid="stMarkdownContainer"]{
    color:var(--ink) !important; }
[data-testid="stMain"] .hero *{ color:#ffffff !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *{
    color:var(--muted) !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] div{
    color:var(--ink) !important; font-size:.83rem !important;
    font-weight:700 !important; letter-spacing:.2px; margin-bottom:3px !important; }

/* ---------- inputs ---------- */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-baseweb="input"] input, [data-baseweb="base-input"] input{
    background:var(--surface) !important; color:var(--ink) !important;
    border:1.4px solid var(--line) !important; border-radius:9px !important;
    -webkit-text-fill-color:var(--ink) !important; }
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus{
    border-color:var(--navy2) !important;
    box-shadow:0 0 0 3px rgba(18,94,138,.12) !important; }
.stTextInput input::placeholder, .stTextArea textarea::placeholder{
    color:#a3b4c2 !important; -webkit-text-fill-color:#a3b4c2 !important; }
[data-testid="stNumberInput"] button{
    background:#eef3f8 !important; color:var(--navy) !important;
    border:1px solid var(--line) !important; }
[data-testid="stNumberInput"] button svg{ fill:var(--navy) !important; }

div[data-baseweb="select"] > div{
    background:var(--surface) !important; color:var(--ink) !important;
    border:1.4px solid var(--line) !important; border-radius:9px !important; }
div[data-baseweb="select"] *, div[data-baseweb="select"] svg{
    color:var(--ink) !important; fill:var(--navy) !important; }
div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"]{
    background:var(--surface) !important; }
div[data-baseweb="popover"] li, ul[role="listbox"] li,
div[data-baseweb="popover"] li *{ color:var(--ink) !important; }
div[data-baseweb="popover"] li:hover{ background:#e9f1f7 !important; }
[data-baseweb="tag"]{ background:var(--navy2) !important; }
[data-baseweb="tag"] *{ color:#fff !important; }

/* ---------- radio ---------- */
[data-testid="stRadio"] label, [data-testid="stRadio"] label p,
[data-testid="stRadio"] div[role="radiogroup"] label,
[data-testid="stRadio"] div[role="radiogroup"] label div,
[data-testid="stRadio"] div[role="radiogroup"] label p{
    color:var(--ink) !important; font-weight:600 !important; font-size:.87rem !important; }
[data-testid="stRadio"] div[role="radiogroup"]{ gap:18px; }
[data-testid="stRadio"] div[role="radiogroup"] > label{
    background:var(--surface); border:1.4px solid var(--line);
    border-radius:9px; padding:7px 15px 7px 10px; margin:0;
    box-shadow:0 2px 6px rgba(15,60,90,.05); }

/* ---------- file uploader ---------- */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"]{
    background:var(--surface) !important; border:2px dashed #b9cddd !important;
    border-radius:12px !important; }
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzoneInstructions"] *{ color:var(--ink) !important; }
[data-testid="stFileUploader"] small,
[data-testid="stFileUploaderDropzoneInstructions"] small{ color:var(--muted) !important; }
[data-testid="stFileUploader"] svg{ fill:var(--navy2) !important; }
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button{
    background:linear-gradient(90deg,var(--navy),var(--teal)) !important;
    color:#fff !important; border:none !important; border-radius:8px !important;
    font-weight:700 !important; }
[data-testid="stFileUploader"] button *{ color:#fff !important; }
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFile"] *{
    color:var(--ink) !important; }

/* ---------- camera ---------- */
[data-testid="stCameraInput"] *{ color:var(--ink) !important; }
[data-testid="stCameraInput"] button{
    background:linear-gradient(90deg,var(--navy),var(--teal)) !important;
    color:#fff !important; border:none !important; border-radius:8px !important;
    font-weight:700 !important; }
[data-testid="stCameraInput"] button *{ color:#fff !important; }
[data-testid="stCameraInput"] > div{
    background:var(--surface) !important; border:1.4px solid var(--line) !important;
    border-radius:12px !important; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"]{
    gap:6px; background:rgba(231,239,246,.92) !important; padding:5px;
    border-radius:11px; border:1px solid #dbe6ef; }
.stTabs [data-baseweb="tab"]{ border-radius:8px; padding:8px 20px; background:transparent; }
.stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] div,
.stTabs [data-baseweb="tab"] span{
    color:var(--navy) !important; font-size:.88rem !important; font-weight:700 !important; }
.stTabs [aria-selected="true"]{
    background:var(--surface) !important; box-shadow:0 3px 9px rgba(15,60,90,.15); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{
    background:transparent !important; }

/* ---------- expander / alerts / tables ---------- */
[data-testid="stExpander"]{ background:var(--surface) !important;
    border:1px solid var(--line) !important; border-radius:11px !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary *,
[data-testid="stExpander"] p{ color:var(--ink) !important; font-weight:600; }
[data-testid="stExpander"] svg{ fill:var(--navy) !important; }
[data-testid="stAlert"]{ border-radius:10px !important; }
[data-testid="stAlert"] p, [data-testid="stAlert"] div,
[data-testid="stAlert"] li, [data-testid="stAlert"] strong{ color:#0f2438 !important; }
[data-testid="stNotification"] *{ color:#0f2438 !important; }
[data-testid="stDataFrame"], [data-testid="stDataEditor"]{
    background:var(--surface) !important; border:1px solid var(--line) !important;
    border-radius:10px !important; }
[data-testid="stDataFrame"] *, [data-testid="stDataEditor"] *{ color:var(--ink) !important; }
[data-testid="stSlider"] label, [data-testid="stSlider"] div{ color:var(--ink) !important; }

/* ---------- containers / metrics ---------- */
[data-testid="stVerticalBlockBorderWrapper"]{
    background:rgba(255,255,255,.97) !important; border-radius:14px !important;
    box-shadow:0 4px 16px rgba(15,60,90,.07); }
[data-testid="stMetricValue"]{ color:var(--navy) !important;
    font-size:1.25rem !important; font-weight:800 !important; }
[data-testid="stMetricLabel"] p{ color:var(--muted) !important;
    font-size:.71rem !important; font-weight:700 !important;
    text-transform:uppercase; letter-spacing:.5px; }
hr{ border-color:var(--line) !important; margin:.7rem 0 !important; }

/* ---------- hero ---------- */
.hero{ background:linear-gradient(120deg,#0b3d63 0%,#125e8a 48%,#0d9488 100%);
    padding:24px 30px; border-radius:16px;
    box-shadow:0 12px 32px rgba(11,61,99,.28); margin-bottom:16px;
    position:relative; overflow:hidden; }
.hero::after{ content:""; position:absolute; right:-70px; top:-70px;
    width:240px; height:240px;
    background:radial-gradient(circle,rgba(255,255,255,.15) 0%,rgba(255,255,255,0) 70%); }
.hero h1{ margin:0; font-size:1.5rem; font-weight:800; position:relative; z-index:1; }
.hero p{ margin:7px 0 0; font-size:.89rem; max-width:800px;
    position:relative; z-index:1; opacity:.96; }
.hero .tag{ display:inline-block; margin:12px 8px 0 0; padding:4px 13px;
    background:rgba(255,255,255,.17); border:1px solid rgba(255,255,255,.3);
    border-radius:20px; font-size:.73rem; font-weight:600;
    position:relative; z-index:1; }

/* ---------- custom blocks ---------- */
.sec{ font-size:1rem; font-weight:800; color:var(--navy) !important;
    border-bottom:2.5px solid var(--line); padding-bottom:6px; margin:16px 0 10px; }
.mbox{ background:var(--surface); border:1px solid var(--line);
    border-left:5px solid var(--navy2); border-radius:11px; padding:11px 15px;
    box-shadow:0 3px 10px rgba(15,60,90,.06); }
.mbox .lbl{ font-size:.65rem; text-transform:uppercase; letter-spacing:1px;
    color:var(--muted) !important; font-weight:700; }
.mbox .val{ font-size:1.2rem; font-weight:800; color:var(--navy) !important; line-height:1.4; }
.rk{ padding:14px 18px; border-radius:11px; font-weight:700; font-size:.92rem;
     margin:8px 0; border:1px solid transparent; }
.rk small{ font-weight:500; display:block; margin-top:4px; }
.rk-high{ background:#fdecea; border-left:5px solid var(--red); border-color:#f5cdc9; }
.rk-high, .rk-high *{ color:#8e1b16 !important; }
.rk-mid{ background:#fff6e5; border-left:5px solid var(--gold); border-color:#efdfb4; }
.rk-mid, .rk-mid *{ color:#77510f !important; }
.rk-low{ background:#e9f7ee; border-left:5px solid var(--green); border-color:#c3e8d2; }
.rk-low, .rk-low *{ color:#155c33 !important; }
.rk-unk{ background:#eef1f5; border-left:5px solid #607d8b; border-color:#d6dde3; }
.rk-unk, .rk-unk *{ color:#37474f !important; }
.sugg{ background:var(--surface); border:1px solid var(--line); border-radius:13px;
    padding:16px 20px; box-shadow:0 3px 12px rgba(15,60,90,.06); }
.sugg h4{ margin:0 0 10px; font-size:.95rem; color:var(--navy) !important; font-weight:800; }
.step{ display:flex; gap:11px; align-items:flex-start; padding:7px 0;
    border-bottom:1px dashed var(--line); font-size:.86rem; }
.step, .step *{ color:var(--ink) !important; }
.step:last-child{ border-bottom:none; }
.num{ min-width:23px; height:23px; border-radius:50%;
    background:linear-gradient(135deg,var(--navy),var(--navy2));
    font-size:.72rem; display:flex; align-items:center; justify-content:center;
    font-weight:800; flex-shrink:0; }
.num, .num *{ color:#fff !important; }
.flag{ display:inline-block; background:#fdecea; font-size:.72rem; padding:4px 11px;
    border-radius:14px; margin:3px 5px 3px 0; font-weight:700; border:1px solid #f5cdc9; }
.flag, .flag *{ color:#8e1b16 !important; }
.chip{ display:inline-block; background:#e7f1f8; font-size:.72rem; padding:4px 11px;
    border-radius:14px; margin:3px 5px 3px 0; font-weight:700; border:1px solid #d2e3ef; }
.chip, .chip *{ color:var(--navy) !important; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0b2540 0%,#0b3d63 100%) !important; }
section[data-testid="stSidebar"] *{ color:#e7f0f8 !important; }
section[data-testid="stSidebar"] [data-testid="stMetricValue"]{ color:#fff !important; }
section[data-testid="stSidebar"] .stTextInput input{
    background:rgba(255,255,255,.10) !important; color:#fff !important;
    -webkit-text-fill-color:#fff !important;
    border:1px solid rgba(255,255,255,.25) !important; }
section[data-testid="stSidebar"] .stTextInput input::placeholder{
    color:#a9c2d6 !important; -webkit-text-fill-color:#a9c2d6 !important; }
section[data-testid="stSidebar"] [data-testid="stAlert"]{
    background:rgba(255,214,120,.14) !important;
    border:1px solid rgba(255,214,120,.35) !important; }
section[data-testid="stSidebar"] [data-testid="stAlert"] *{ color:#ffe9b8 !important; }
.sb{ background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.10);
     border-radius:11px; padding:12px 15px; margin-bottom:11px; font-size:.83rem; }

/* ---------- buttons ---------- */
.stButton>button, .stDownloadButton>button{
    background:linear-gradient(90deg,var(--navy),var(--teal)) !important;
    color:#fff !important; border:none !important; border-radius:9px !important;
    font-weight:700 !important; font-size:.86rem !important;
    padding:.5rem 1.1rem !important;
    box-shadow:0 4px 12px rgba(11,61,99,.18) !important; }
.stButton>button *, .stDownloadButton>button *{ color:#fff !important; }
.stButton>button:hover, .stDownloadButton>button:hover{ filter:brightness(1.1); }
[data-testid="stPopover"] button{ color:#fff !important; }

#MainMenu, header, footer{ visibility:hidden !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# ADD-ON v5.1 — EDGE BACKGROUND FIX + MOBILE RESPONSIVE LAYER
# (nothing above is removed; these rules only override sizes)
# ------------------------------------------------------------
st.markdown("""
<style>
/* 1) phone-e page-er charpashe brown/dark strip bondho */
html, body{
    background-color:#eef4f9 !important;
    margin:0 !important; padding:0 !important;
    overflow-x:hidden !important; }
#root, .stApp, [data-testid="stAppViewContainer"]{
    background-color:#eef4f9 !important; min-height:100vh !important; }
[data-testid="stDecoration"]{ background:transparent !important; }

/* mobile hint pill — desktop-e lukano thakbe */
.mob-hint{ display:none; }

@media (max-width: 768px){

  /* NEW — right side-er extra strip / horizontal scroll bondho */
  html, body, #root, .stApp{
      max-width:100vw !important; overflow-x:hidden !important; }
  [data-testid="stAppViewContainer"]{ overflow-x:hidden !important; }

  /* fixed background mobile browser-e kepe, tai scroll kora holo */
  html, body, .stApp,
  [data-testid="stAppViewContainer"], [data-testid="stMain"]{
      background-attachment:scroll !important;

      background-size:auto, auto, auto, 60px 60px !important; }

  /* du-pasher nosto hoya faka jayga firiye ana */
  [data-testid="stAppViewContainer"] .main .block-container{
      padding:0.6rem 0.7rem 2rem !important; max-width:100% !important; }

  /* hero card compact */
  .hero{ padding:14px 15px !important; border-radius:13px !important;
         margin-bottom:11px !important; }
  .hero h1{ font-size:1.02rem !important; line-height:1.35 !important; }
  .hero p{ font-size:.76rem !important; margin-top:5px !important; }
  .hero .tag{ font-size:.62rem !important; padding:3px 9px !important;
              margin:7px 5px 0 0 !important; }
  .hero::after{ display:none !important; }

  /* tabs kata na pore — pashe scroll hobe */
  .stTabs [data-baseweb="tab-list"]{
      overflow-x:auto !important; flex-wrap:nowrap !important;
      -webkit-overflow-scrolling:touch;
      padding:4px !important; gap:3px !important; }
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar{ display:none; }
  .stTabs [data-baseweb="tab"]{ padding:6px 11px !important; flex:0 0 auto !important; }
  .stTabs [data-baseweb="tab"] p{ font-size:.75rem !important; }

  /* columns 2x2 grid hobe, ek-column lomba hobe na */
  [data-testid="stHorizontalBlock"]{ flex-wrap:wrap !important; gap:.5rem !important; }
  [data-testid="stHorizontalBlock"] > [data-testid="column"],
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{
      min-width:calc(50% - .5rem) !important;
      flex:1 1 calc(50% - .5rem) !important; }

  /* heading + card gulo choto */
  .sec{ font-size:.9rem !important; margin:12px 0 8px !important; }
  .mbox{ padding:9px 11px !important; }
  .mbox .val{ font-size:1rem !important; }
  .mbox .lbl{ font-size:.58rem !important; }
  .rk{ padding:11px 13px !important; font-size:.83rem !important; }
  .sugg{ padding:12px 13px !important; }
  .step{ font-size:.79rem !important; }
  .chip, .flag{ font-size:.65rem !important; padding:3px 8px !important; }

  /* metric + widget + button */
  [data-testid="stMetricValue"]{ font-size:1rem !important; }
  [data-testid="stMetricLabel"] p{ font-size:.6rem !important; }
  [data-testid="stWidgetLabel"] p{ font-size:.76rem !important; }
  .stButton>button, .stDownloadButton>button{
      width:100% !important; font-size:.8rem !important;
      padding:.55rem .6rem !important; }

  /* radio card gulo upor-niche */
  [data-testid="stRadio"] div[role="radiogroup"]{
      flex-direction:column !important; gap:8px !important; }
  [data-testid="stRadio"] div[role="radiogroup"] > label{ width:100% !important; }

  /* table nijer moto scroll korbe, page chaparbe na */
  [data-testid="stDataFrame"], [data-testid="stDataEditor"]{
      overflow-x:auto !important; font-size:.72rem !important; }

  /* sidebar drawer puro screen dhake na */
  section[data-testid="stSidebar"]{ width:82vw !important; min-width:0 !important; }

  /* menu hint sudhu phone-e dekhabe */
  .mob-hint{
      display:block !important; background:#e7f1f8; border:1px solid #d2e3ef;
      border-left:4px solid #125e8a; border-radius:9px; padding:7px 11px;
      font-size:.74rem; font-weight:700; color:#0b3d63 !important; margin:0 0 9px; }
}

@media (max-width: 430px){
  .hero h1{ font-size:.94rem !important; }
  .stTabs [data-baseweb="tab"] p{ font-size:.7rem !important; }
  [data-testid="stAppViewContainer"] .main .block-container{
      padding:0.5rem 0.5rem 1.6rem !important; }
}
</style>
""", unsafe_allow_html=True)

# medical inline SVG logo (crescent + cross, works offline)
MED_LOGO = """
<svg width="62" height="62" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <path d="M44 6a26 26 0 1 0 0 52 21 21 0 1 1 0-52z" fill="#e63946"/>
  <path d="M38 22c5 0 9 4 9 9 0 6-9 13-9 13s-9-7-9-13c0-5 4-9 9-9z" fill="#e63946"/>
  <path d="M35 24h6v5h5v6h-5v5h-6v-5h-5v-6h5z" fill="#ffffff"
        stroke="#0b2540" stroke-width="1.6" stroke-linejoin="round"/>
</svg>
"""

# ------------------------------------------------------------
# KNOWLEDGE BASE
# ------------------------------------------------------------
CLASS_INFO = {
    "melanoma": dict(risk="HIGH", w=1.0, full="Malignant Melanoma",
        desc="Aggressive malignancy of melanocytes; early wide excision is critical.",
        urgency="Urgent — refer within 1–2 weeks (2-week-wait pathway)",
        rec=["Urgent dermatology / skin-cancer MDT referral",
             "Excisional biopsy with 2 mm clinical margin (avoid shave biopsy)",
             "Breslow thickness, ulceration and mitotic rate on histology",
             "Full-body skin survey plus regional lymph node examination",
             "Discuss sentinel node biopsy if Breslow > 0.8 mm"]),
    "basal cell carcinoma": dict(risk="HIGH", w=.75, full="Basal Cell Carcinoma",
        desc="Commonest skin cancer; locally invasive, metastasis is rare.",
        urgency="Semi-urgent — refer within 4 weeks",
        rec=["Routine dermatology referral for definitive treatment",
             "Punch or shave biopsy for histological subtype",
             "Mohs micrographic surgery for H-zone / facial lesions",
             "Photoprotection counselling and annual skin surveillance"]),
    "squamous cell carcinoma": dict(risk="HIGH", w=.85, full="Squamous Cell Carcinoma",
        desc="Keratinocyte malignancy with genuine metastatic potential.",
        urgency="Urgent — refer within 2 weeks",
        rec=["Urgent dermatology referral",
             "Biopsy with assessment of depth and differentiation",
             "Palpate regional lymph nodes; image if suspicious",
             "Review immunosuppression status (transplant, haematological disease)"]),
    "akiec": dict(risk="MODERATE", w=.45,
        full="Actinic Keratosis / Intraepithelial Carcinoma",
        desc="Pre-malignant, UV-induced dysplasia that may progress to SCC.",
        urgency="Routine — review within 4–8 weeks",
        rec=["Dermatology review for field-directed therapy",
             "Cryotherapy, 5-fluorouracil or imiquimod as appropriate",
             "Strict daily broad-spectrum photoprotection",
             "Six-monthly surveillance of the sun-damaged field"]),
    "bkl": dict(risk="LOW", w=.15, full="Benign Keratosis-like Lesion",
        desc="Benign group including seborrhoeic keratosis and solar lentigo.",
        urgency="Routine — no referral usually required",
        rec=["Reassure the patient; treatment is cosmetic only",
             "Re-image if the lesion changes in size, colour or texture",
             "Baseline dermoscopic photograph for future comparison"]),
    "nv": dict(risk="LOW", w=.10, full="Melanocytic Nevus",
        desc="Common benign mole.",
        urgency="Routine — annual skin check",
        rec=["Teach ABCDE self-examination",
             "Annual review, sooner if any change is noticed",
             "Baseline photography for patients with many naevi"]),
    "df": dict(risk="LOW", w=.10, full="Dermatofibroma",
        desc="Benign fibrohistiocytic nodule, often on the limbs.",
        urgency="Routine — reassurance only",
        rec=["Reassure; excision only if symptomatic or diagnostically uncertain"]),
    "vasc": dict(risk="LOW", w=.12, full="Vascular Lesion",
        desc="Benign vascular proliferation such as angioma or haemorrhage.",
        urgency="Routine — reassurance only",
        rec=["Reassure the patient",
             "Refer only if bleeding, ulcerating or rapidly enlarging"]),
    "benign": dict(risk="LOW", w=.12, full="Benign Lesion",
        desc="No malignant morphological features detected.",
        urgency="Routine monitoring",
        rec=["Routine monitoring and photoprotection",
             "Re-screen if the lesion changes"]),
    "malignant": dict(risk="HIGH", w=.90, full="Malignant Lesion",
        desc="Morphology suspicious for malignancy.",
        urgency="Urgent — refer within 2 weeks",
        rec=["Urgent dermatology referral", "Histopathological confirmation"]),
}
for a, b in [("mel", "melanoma"), ("bcc", "basal cell carcinoma"),
             ("scc", "squamous cell carcinoma"), ("nevus", "nv"),
             ("actinic keratosis", "akiec"), ("seborrheic keratosis", "bkl"),
             ("dermatofibroma", "df"), ("vascular lesion", "vasc")]:
    CLASS_INFO[a] = CLASS_INFO[b]

DEFAULT_INFO = dict(risk="UNDETERMINED", w=.5, full="Unclassified Lesion",
    desc="Class not present in the internal knowledge base.",
    urgency="Specialist review advised",
    rec=["Clinical correlation required", "Specialist dermoscopic review"])


def lesion_info(name: str):
    k = str(name).strip().lower()
    if k in CLASS_INFO and CLASS_INFO[k]:
        return CLASS_INFO[k]
    for key, val in CLASS_INFO.items():
        if val and (key in k or k in key):
            return val
    return DEFAULT_INFO


RED_FLAGS = {
    r"bleed": "Spontaneous bleeding",
    r"ulcer": "Ulceration",
    r"itch|pruri": "Persistent pruritus",
    r"pain|tender|sore": "Pain or tenderness",
    r"grow|enlarg|increas|bigger|size": "Documented enlargement",
    r"colou?r|dark|black|multi": "Colour change / variegation",
    r"irregular|border|asymmetr": "Irregular border or asymmetry",
    r"crust|scab|scale": "Crusting or scaling",
    r"family|hereditar|father|mother|sibling": "Family history of skin cancer",
    r"immunosuppress|transplant|chemo|hiv|steroid": "Immunosuppression",
    r"sunburn|tanning|outdoor|uv|sun exposure": "Significant UV exposure history",
    r"new|recent|month|week": "Recent-onset lesion",
}
HIGH_UV_SITES = {"Face", "Scalp", "Neck", "Upper Limb", "Hand"}
SITES = ["Face", "Scalp", "Neck", "Chest", "Back", "Abdomen",
         "Upper Limb", "Lower Limb", "Hand", "Foot", "Other"]


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLOv11 weights...")
def load_model(path="best.pt"):
    return YOLO(path)


def image_quality(img: Image.Image) -> dict:
    g = img.convert("L")
    s = ImageStat.Stat(g)
    bright, contrast = s.mean[0], s.stddev[0]
    sharp = float(np.asarray(g.filter(ImageFilter.FIND_EDGES), np.float32).var())
    w, h = img.size
    notes = []
    if bright < 55: notes.append("Under-exposed image")
    if bright > 205: notes.append("Over-exposed / glare present")
    if contrast < 22: notes.append("Low contrast — lesion border unclear")
    if sharp < 120: notes.append("Possible blur or out-of-focus capture")
    if min(w, h) < 224: notes.append("Resolution below optimal (224 px minimum)")
    score = 100.0
    score -= min(abs(bright - 128) / 128 * 30, 30)
    score -= max(0, 30 - contrast) * .8
    score -= 0 if sharp > 300 else min((300 - sharp) / 300 * 25, 25)
    score -= 0 if min(w, h) >= 448 else 10
    return dict(brightness=bright, contrast=contrast, sharpness=sharp, width=w,
                height=h, score=max(0, min(100, score)),
                notes=notes or ["No acquisition issues detected"])


def entropy_of(p):
    p = np.clip(np.asarray(p, np.float64), 1e-12, 1)
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def detect_flags(text: str):
    t = (text or "").lower()
    return sorted({v for k, v in RED_FLAGS.items() if re.search(k, t)})


def auto_suggest(cls, conf, unc, info, age, sex, site, symptoms, quality, gate):
    flags = detect_flags(symptoms)
    score = info["w"] * (conf / 100) * 60
    score += min(len(flags), 5) * 4
    if age >= 65: score += 8
    elif age >= 50: score += 5
    elif age >= 35: score += 2
    if site in HIGH_UV_SITES: score += 5
    if unc > .55: score += 4
    if quality["score"] < 55: score += 3
    score = round(min(100, score), 1)

    if score >= 62 or info["risk"] == "HIGH":
        band, css, icon = "URGENT REVIEW", "rk-high", "🚨"
        window = "Refer within 1–2 weeks"
    elif score >= 38 or info["risk"] == "MODERATE":
        band, css, icon = "PRIORITY REVIEW", "rk-mid", "⚠️"
        window = "Refer within 4–6 weeks"
    elif info["risk"] == "UNDETERMINED":
        band, css, icon = "INCONCLUSIVE", "rk-unk", "❔"
        window = "Specialist dermoscopic review"
    else:
        band, css, icon = "ROUTINE MONITORING", "rk-low", "✅"
        window = "Routine annual skin check"

    steps = list(info["rec"])
    if conf < gate:
        steps.insert(0, f"Model confidence ({conf:.1f}%) is below the {gate}% "
                        "threshold — treat as inconclusive and repeat imaging.")
    if unc > .55:
        steps.append("High output entropy: probability is spread across several "
                     "classes, so a specialist opinion is strongly advised.")
    if quality["score"] < 55:
        steps.append("Poor image quality may have degraded the prediction — "
                     "re-capture with even lighting and a stable, focused frame.")
    if flags:
        steps.append("Document the reported red-flag symptoms (" +
                     ", ".join(flags) + ") in the referral letter.")
    if age >= 50 and info["risk"] != "LOW":
        steps.append("Age above 50 raises baseline malignancy probability — "
                     "lower the threshold for biopsy.")
    if site in HIGH_UV_SITES:
        steps.append(f"{site} is a chronically UV-exposed site — perform a full "
                     "field examination for additional actinic damage.")
    steps.append("Arrange interval dermoscopic photography to allow objective "
                 "comparison at follow-up.")
    return dict(score=score, band=band, css=css, icon=icon, window=window,
                flags=flags, steps=steps)


def gauge(value, title, height=185):
    if not PLOTLY:
        st.progress(int(value)); st.caption(f"{title}: {value:.1f}%"); return
    color = "#c62828" if value >= 62 else "#b9821f" if value >= 38 else "#1f8a4c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"font": {"size": 28, "color": "#0b3d63"}},
        title={"text": title, "font": {"size": 12, "color": "#5a7186"}},
        gauge={"axis": {"range": [0, 100], "tickfont": {"size": 9, "color": "#5a7186"}},
               "bar": {"color": color, "thickness": .74},
               "bgcolor": "#eef4f8", "borderwidth": 0,
               "steps": [{"range": [0, 38], "color": "#e9f7ee"},
                         {"range": [38, 62], "color": "#fff6e5"},
                         {"range": [62, 100], "color": "#fdecea"}]}))
    fig.update_layout(height=height, margin=dict(l=14, r=14, t=36, b=6),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def donut(names, probs):
    if not PLOTLY:
        st.bar_chart(pd.DataFrame({"%": [p * 100 for p in probs]},
                                  index=[n.upper() for n in names])); return
    fig = px.pie(values=[p * 100 for p in probs],
                 names=[n.upper() for n in names], hole=.58,
                 color_discrete_sequence=px.colors.sequential.Teal_r)
    fig.update_traces(textinfo="percent", textfont_size=10)
    fig.update_layout(height=250, margin=dict(l=6, r=6, t=6, b=6),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(font=dict(size=10, color="#0f2438")))
    st.plotly_chart(fig, use_container_width=True)


COLS = ["Record ID", "Date", "Patient Name", "Age", "Sex", "Contact",
        "Lesion Site", "Symptoms", "Prediction", "Confidence (%)", "Risk Level",
        "Triage", "Risk Score", "Uncertainty", "Image Quality (%)",
        "Red Flags", "Image File", "Clinician"]


def load_records():
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            for c in COLS:
                if c not in df.columns: df[c] = ""
            return df[COLS]
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def save_records(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)


def load_clinician():
    if os.path.exists(DOC_PATH):
        try:
            with open(DOC_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh).get("name", "")
        except Exception:
            return ""
    return ""


def persist_clinician():
    """Sidebar callback — remembers the doctor's name permanently."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(DOC_PATH, "w", encoding="utf-8") as fh:
            json.dump({"name": st.session_state.get("clinician", "")}, fh)
        st.session_state["doc_msg"] = st.session_state.get("clinician", "")
    except Exception as ex:
        st.session_state["doc_err"] = str(ex)


def do_save(rec, clear_form=True):
    """Button callback — runs BEFORE the rerun, so the record is never lost."""
    st.session_state.records = pd.concat(
        [st.session_state.records, pd.DataFrame([rec])], ignore_index=True)
    save_records(st.session_state.records)
    st.session_state["saved_msg"] = rec["Record ID"]
    st.session_state["saved_name"] = rec["Patient Name"]
    if clear_form:
        st.session_state["clear_form"] = True


def reset_form():
    """Callback for the manual 'New Patient' button."""
    st.session_state["clear_form"] = True


def apply_clear():
    """Runs at the top of tab 1, before widgets are created."""
    if st.session_state.pop("clear_form", False):
        st.session_state["p_name"] = ""
        st.session_state["p_age"] = 35
        st.session_state["p_sex"] = "Male"
        st.session_state["p_contact"] = ""
        st.session_state["p_site"] = "Face"
        st.session_state["p_sym"] = ""
        st.session_state["upl_key"] = st.session_state.get("upl_key", 0) + 1
        st.session_state["cam_key"] = st.session_state.get("cam_key", 0) + 1


def build_report(rec, info, top5, q, sg):
    L = "=" * 64
    t5 = "\n".join(f"   {i+1}. {n.upper():<32} {p*100:6.2f} %"
                   for i, (n, p) in enumerate(top5))
    rc = "\n".join(f"   {i+1}. {s}" for i, s in enumerate(sg["steps"]))
    qn = "\n".join(f"   - {n}" for n in q["notes"])
    fl = ", ".join(sg["flags"]) if sg["flags"] else "None identified"
    return f"""{L}
        AI DIAGNOSTIC REPORT - DERMOSCOPIC SCREENING
{L}
Report ID      : {rec['Record ID']}
Date & Time    : {rec['Date']}
System         : AI-Based Skin Cancer Detection Using Deep
                 Learning on Dermoscopic Images (YOLOv11)
Reviewing Staff: {rec['Clinician'] or 'Not specified'}
Developed By   : Golam Muktadir Al Sabir (0242220005101761)
                 Jahidul Islam Jisan     (0242220005101743)

{L}
1. PATIENT DEMOGRAPHICS
{L}
Name             : {rec['Patient Name']}
Age / Sex        : {rec['Age']} / {rec['Sex']}
Contact          : {rec['Contact'] or '-'}
Lesion Site      : {rec['Lesion Site']}
Reported Symptoms: {rec['Symptoms'] or 'None reported'}
Red Flags        : {fl}

{L}
2. ANALYSIS SUMMARY
{L}
Analyzed File    : {rec['Image File']}
Detected Class   : {rec['Prediction'].upper()}  ({info['full']})
Confidence       : {rec['Confidence (%)']} %
Class Risk       : {rec['Risk Level']}
Composite Risk   : {rec['Risk Score']} / 100
Triage Category  : {sg['band']}  -  {sg['window']}
Model Uncertainty: {rec['Uncertainty']} (normalised entropy)

Differential probability ranking:
{t5}

{L}
3. IMAGE ACQUISITION QUALITY
{L}
Resolution    : {q['width']} x {q['height']} px
Quality Score : {q['score']:.1f} / 100
Brightness    : {q['brightness']:.1f}    Contrast: {q['contrast']:.1f}
Sharpness     : {q['sharpness']:.0f}
Observations:
{qn}

{L}
4. CLINICAL OBSERVATIONS
{L}
Morphological features, colour distribution and textural patterns
extracted by the convolutional network are most consistent with
{info['full'].upper()}.

Description: {info['desc']}
Suggested urgency: {info['urgency']}

{L}
5. AUTO-GENERATED CLINICAL SUGGESTIONS
{L}
{rc}

{L}
DISCLAIMER
{L}
AI-generated PRELIMINARY SCREENING output - not a medical
diagnosis. Not a validated medical device. Must not be the sole
basis of any clinical decision. Histopathology remains the
diagnostic gold standard. Always consult a dermatologist.
{L}
"""


# ------------------------------------------------------------
# SESSION DEFAULTS
# ------------------------------------------------------------
if "records" not in st.session_state:
    st.session_state.records = load_records()
if "clinician" not in st.session_state:
    st.session_state["clinician"] = load_clinician()
for k, v in [("p_name", ""), ("p_age", 35), ("p_sex", "Male"),
             ("p_contact", ""), ("p_site", "Face"), ("p_sym", ""),
             ("upl_key", 0), ("cam_key", 0)]:
    st.session_state.setdefault(k, v)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"<div style='text-align:center;padding:6px 0 2px'>{MED_LOGO}"
        "<h4 style='margin:8px 0 0;font-size:1.05rem'>DermAI Screening</h4>"
        "<p style='font-size:.72rem;opacity:.75;margin:2px 0 0'>"
        "Clinical Decision Support v5.1</p></div>", unsafe_allow_html=True)
    st.divider()

    st.text_input("🩺 Attending Clinician", key="clinician", placeholder="Dr. Name")
    st.button("💾 Remember this clinician", use_container_width=True,
              on_click=persist_clinician)
    if st.session_state.pop("doc_msg", None) is not None:
        st.success("Clinician name saved — it will load automatically next time.")
    clinician = st.session_state.get("clinician", "")

    conf_gate = st.slider("Confidence alert threshold (%)", 40, 95, 60, 5)
    show_adv = st.toggle("🧪 Advanced diagnostics", True)
    auto_on = st.toggle("🧠 Auto-suggestion engine", True)
    autosave = st.toggle("⚡ Auto-save every analysis", False,
                         help="Saves the record automatically without pressing Save.")
    st.divider()

    _d = st.session_state.records
    st.markdown("<div class='sb'><b>📊 Session Summary</b></div>", unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    s1.metric("Records", len(_d))
    s2.metric("High risk", int((_d["Risk Level"] == "HIGH").sum()) if len(_d) else 0)
    st.divider()
    st.markdown("""<div class='sb'><b>👨‍💻 Developed By</b><br><br>
    <b>Golam Muktadir Al Sabir</b><br><span style='opacity:.75'>ID: 0242220005101761</span>
    <br><br><b>Jahidul Islam Jisan</b><br><span style='opacity:.75'>ID: 0242220005101743</span>
    </div>""", unsafe_allow_html=True)
    st.warning("⚠️ **Medical Disclaimer** — educational / preliminary screening only. "
               "Not a medical device. Always consult a qualified dermatologist.")

# ------------------------------------------------------------
if THEME_CREATED:
    st.info("🎨 A light-theme configuration file was created at `.streamlit/config.toml`. "
            "Please stop the app (Ctrl+C) and run it again so the day-mode theme applies.")

# mobile-only hint (hidden automatically on desktop by the CSS above)
st.markdown("<div class='mob-hint'>☰ Tap the <b>»</b> icon at the top-left to open "
            "the sidebar — clinician name, confidence threshold and settings.</div>",
            unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🩺 AI-Based Skin Cancer Detection Using Deep Learning on Dermoscopic Images</h1>
  <p>Differential probability analysis, composite risk triage, automated clinical
     suggestions and structured patient record management.</p>
  <span class="tag">⚕️ YOLOv11 Engine</span>
  <span class="tag">🧠 Auto-Suggestion Triage</span>
  <span class="tag">📸 Live Camera Capture</span>
  <span class="tag">🗂️ Patient Records</span>
  <span class="tag">🔬 Dermoscopic Analysis</span>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔬 New Analysis", "🗂️ Patient Records", "📖 About & Method"])

# ============================================================
# TAB 1 — ANALYSIS
# ============================================================
with tab1:
    apply_clear()

    if st.session_state.get("saved_msg"):
        rid_done = st.session_state.pop("saved_msg")
        nm_done = st.session_state.pop("saved_name", "")
        st.success(f"✅ Record `{rid_done}` for **{nm_done}** saved to patient records. "
                   "The form is now cleared and ready for the next patient.")

    top_l, top_r = st.columns([4, 1])
    top_l.markdown("<div class='sec'>👤 Patient Registration</div>", unsafe_allow_html=True)
    top_r.button("🧹 New Patient (clear form)", use_container_width=True,
                 on_click=reset_form)

    with st.container(border=True):
        a, b, c, d = st.columns([2.1, .8, 1, 1.5])
        p_name = a.text_input("🧑 Patient Name *", key="p_name", placeholder="Full name")
        p_age = b.number_input("🎂 Age *", 0, 120, key="p_age")
        p_sex = c.selectbox("⚧ Sex *", ["Male", "Female", "Other"], key="p_sex")
        p_contact = d.text_input("📞 Contact / Hospital ID", key="p_contact",
                                 placeholder="Phone or ID")
        e, f = st.columns([1, 2.6])
        p_site = e.selectbox("📍 Lesion Site *", SITES, key="p_site")
        p_sym = f.text_input(
            "📝 Symptoms / History", key="p_sym",
            placeholder="e.g. bleeding, itching, enlarging over 3 months, family history")

    st.markdown("<div class='sec'>📷 Image Acquisition</div>", unsafe_allow_html=True)
    src = st.radio("Acquisition method",
                   ["📁  Upload dermoscopic image", "📸  Capture with camera"],
                   horizontal=True)
    if src.startswith("📁"):
        img_file = st.file_uploader("Supported formats — JPG · JPEG · PNG · BMP · WEBP",
                                    type=["jpg", "jpeg", "png", "bmp", "webp"],
                                    key=f"upl_{st.session_state['upl_key']}")
        fname = img_file.name if img_file else ""
    else:
        img_file = st.camera_input("📸 Centre the lesion in the frame and capture",
                                   key=f"cam_{st.session_state['cam_key']}")
        fname = f"camera_{datetime.datetime.now():%H%M%S}.jpg" if img_file else ""

    run = st.button("🚀 Run AI Analysis", use_container_width=True, type="primary")

    if run and img_file is None:
        st.error("Please upload or capture an image first.")
    elif run and not p_name.strip():
        st.error("Patient name is required before a report can be generated.")
    elif run:
        image = Image.open(img_file).convert("RGB")
        with st.spinner("Running deep-learning inference..."):
            res = load_model()(image, verbose=False)[0]

        if getattr(res, "probs", None) is None:
            st.error("`best.pt` is not a YOLO **classification** checkpoint.")
            st.stop()

        probs = res.probs.data.tolist()
        names = [res.names[i] for i in range(len(probs))]
        order = np.argsort(probs)[::-1]
        top5 = [(names[i], probs[i]) for i in order[:5]]
        cls_name, conf = top5[0][0], top5[0][1] * 100
        info = lesion_info(cls_name)
        unc = entropy_of(probs)
        q = image_quality(image)
        sg = auto_suggest(cls_name, conf, unc, info, p_age, p_sex, p_site,
                          p_sym, q, conf_gate)

        st.divider()
        L, R = st.columns([1, 1.5])
        with L:
            st.markdown("<div class='sec'>📸 Submitted Image</div>", unsafe_allow_html=True)
            st.image(image, use_container_width=True,
                     caption=f"{fname} • {q['width']}×{q['height']} px")
            gauge(sg["score"], "Composite Risk Score")
        with R:
            st.markdown("<div class='sec'>🔍 AI Analysis Result</div>", unsafe_allow_html=True)
            m = st.columns(4)
            for col, lbl, val in zip(
                    m, ["Prediction", "Confidence", "Uncertainty", "Image Quality"],
                    [cls_name.upper(), f"{conf:.1f}%", f"{unc:.3f}",
                     f"{q['score']:.0f}/100"]):
                col.markdown(f"<div class='mbox'><div class='lbl'>{lbl}</div>"
                             f"<div class='val'>{val}</div></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='rk {sg['css']}'>{sg['icon']} {sg['band']} · {sg['window']}"
                f"<small>{info['full']} — {info['desc']}</small></div>",
                unsafe_allow_html=True)
            if sg["flags"]:
                st.markdown("".join(f"<span class='flag'>🚩 {x}</span>" for x in sg["flags"]),
                            unsafe_allow_html=True)
            if conf < conf_gate:
                st.warning(f"Confidence {conf:.1f}% is below the {conf_gate}% threshold — "
                           "treat this result as inconclusive.")
            c1, c2 = st.columns([1.25, 1])
            with c1:
                st.caption("**Differential probability**")
                st.bar_chart(pd.DataFrame(
                    {"%": [p * 100 for p in probs]},
                    index=[n.upper() for n in names]).sort_values("%", ascending=False),
                    height=222)
            with c2:
                st.caption("**Class distribution**")
                donut(names, probs)

        if auto_on:
            st.markdown("<div class='sec'>🧠 Auto-Generated Clinical Suggestions</div>",
                        unsafe_allow_html=True)
            body = "".join(f"<div class='step'><div class='num'>{i}</div>"
                           f"<div>{s}</div></div>" for i, s in enumerate(sg["steps"], 1))
            chips = (f"<span class='chip'>🏥 Triage: {sg['band']}</span>"
                     f"<span class='chip'>📊 Risk {sg['score']}/100</span>"
                     f"<span class='chip'>⏱ {sg['window']}</span>"
                     f"<span class='chip'>🎂 Age {p_age} · {p_sex}</span>"
                     f"<span class='chip'>📍 {p_site}</span>")
            st.markdown(f"<div class='sugg'><h4>⚕️ Recommended clinical pathway</h4>"
                        f"{chips}<div style='height:8px'></div>{body}</div>",
                        unsafe_allow_html=True)
            with st.expander("🔤 ABCDE self-assessment reference"):
                st.markdown(
                    "**A — Asymmetry:** one half unlike the other.  \n"
                    "**B — Border:** irregular, scalloped or poorly defined edge.  \n"
                    "**C — Colour:** more than one shade, or uneven pigment.  \n"
                    "**D — Diameter:** larger than 6 mm, though melanoma can be smaller.  \n"
                    "**E — Evolving:** any change in size, shape, colour or symptoms.  \n\n"
                    "Any single positive criterion warrants a clinician's review.")

        if show_adv:
            st.markdown("<div class='sec'>🧪 Advanced Diagnostics</div>", unsafe_allow_html=True)
            g1, g2, g3 = st.columns([1.4, 1, 1])
            g1.caption("**Top-5 differential ranking**")
            g1.dataframe(pd.DataFrame({
                "Rank": range(1, len(top5) + 1),
                "Class": [n.upper() for n, _ in top5],
                "Prob (%)": [round(p * 100, 2) for _, p in top5],
                "Risk": [lesion_info(n)["risk"] for n, _ in top5]}),
                hide_index=True, use_container_width=True, height=210)
            g2.caption("**Acquisition metrics**")
            g2.dataframe(pd.DataFrame({
                "Metric": ["Brightness", "Contrast", "Sharpness", "Quality"],
                "Value": [f"{q['brightness']:.1f}", f"{q['contrast']:.1f}",
                          f"{q['sharpness']:.0f}", f"{q['score']:.1f}/100"]}),
                hide_index=True, use_container_width=True, height=210)
            g3.caption("**Quality observations**")
            for n in q["notes"]:
                g3.info(n, icon="ℹ️")

        st.markdown("<div class='sec'>📋 AI Diagnostic Report (for medical review)</div>",
                    unsafe_allow_html=True)
        rid = f"SCD-{datetime.datetime.now():%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"
        now = datetime.datetime.now().strftime("%d %B, %Y — %I:%M %p")
        saved = f"{rid}_{fname or 'image.jpg'}"
        try:
            image.save(os.path.join(IMG_DIR, saved))
        except Exception:
            saved = fname

        rec = {"Record ID": rid, "Date": now, "Patient Name": p_name, "Age": p_age,
               "Sex": p_sex, "Contact": p_contact, "Lesion Site": p_site,
               "Symptoms": p_sym, "Prediction": cls_name,
               "Confidence (%)": round(conf, 2), "Risk Level": info["risk"],
               "Triage": sg["band"], "Risk Score": sg["score"],
               "Uncertainty": round(unc, 3), "Image Quality (%)": round(q["score"], 1),
               "Red Flags": "; ".join(sg["flags"]), "Image File": saved,
               "Clinician": clinician}
        st.session_state["last_rec"] = rec

        with st.container(border=True):
            h = st.columns(4)
            h[0].markdown(f"**🆔 Report ID**  \n`{rid}`")
            h[1].markdown(f"**📅 Date**  \n{now}")
            h[2].markdown(f"**🩺 Clinician**  \n{clinician or '—'}")
            h[3].markdown(f"**🏥 Triage**  \n{sg['band']}")
            st.divider()
            st.markdown(f"**Patient:** {p_name} • **Age/Sex:** {p_age}/{p_sex} • "
                        f"**Site:** {p_site} • **Contact:** {p_contact or '—'}  \n"
                        f"**Symptoms:** {p_sym or 'None reported'}")
            st.markdown(f"**Detected:** {cls_name.upper()} ({info['full']}) • "
                        f"**Confidence:** {conf:.2f}% • **Class risk:** {info['risk']} • "
                        f"**Composite risk:** {sg['score']}/100 • **Entropy:** {unc:.3f}")
            st.info(f"Morphological, chromatic and textural features extracted by the "
                    f"network are most consistent with **{info['full'].upper()}**. "
                    f"{info['desc']} Suggested urgency: {info['urgency']}.")
            st.markdown("**⚕️ Suggested pathway**")
            for i, s in enumerate(sg["steps"], 1):
                st.markdown(f"{i}. {s}")
            st.caption("Auto-generated by the AI system developed by Golam Muktadir Al Sabir "
                       "& Jahidul Islam Jisan. Requires human medical verification.")

        txt = build_report(rec, info, top5, q, sg)
        d1, d2, d3 = st.columns(3)
        d1.download_button("📥 Report (.txt)", txt, f"AI_Skin_Report_{rid}.txt",
                           "text/plain", use_container_width=True)
        d2.download_button("📊 Record (.csv)", pd.DataFrame([rec]).to_csv(index=False),
                           f"Record_{rid}.csv", "text/csv", use_container_width=True)

        if autosave:
            do_save(rec, clear_form=False)
            st.success(f"⚡ Auto-saved as `{rid}` — see the Patient Records tab.")
            d3.button("🧹 Next Patient (clear form)", use_container_width=True,
                      on_click=reset_form)
        else:
            d3.button("💾 Save & Next Patient", use_container_width=True,
                      key="save_btn", on_click=do_save, args=(rec, True))
            st.caption("Press **Save & Next Patient** — the record is written to "
                       "`records/patient_records.csv` and the form clears automatically. "
                       "Download the report first if you need the .txt file.")

# ============================================================
# TAB 2 — RECORDS
# ============================================================
with tab2:
    st.markdown("<div class='sec'>🗂️ Patient Record Management</div>", unsafe_allow_html=True)
    df = st.session_state.records
    if df.empty:
        st.info("No records yet. Run an analysis, then press **Save & Next Patient** "
                "(or switch on Auto-save in the sidebar).")
    else:
        k = st.columns(5)
        k[0].metric("Total", len(df))
        k[1].metric("High risk", int((df["Risk Level"] == "HIGH").sum()))
        k[2].metric("Moderate", int((df["Risk Level"] == "MODERATE").sum()))
        k[3].metric("Mean conf.",
                    f"{pd.to_numeric(df['Confidence (%)'], errors='coerce').mean():.1f}%")
        k[4].metric("Mean risk",
                    f"{pd.to_numeric(df['Risk Score'], errors='coerce').mean():.1f}")

        s1, s2, s3, s4 = st.columns([2, 1.2, 1.1, 1.1])
        qtxt = s1.text_input("🔎 Search name / ID / prediction", "")
        frisk = s2.multiselect("🩸 Risk", ["HIGH", "MODERATE", "LOW", "UNDETERMINED"])
        fsex = s3.multiselect("⚧ Sex", ["Male", "Female", "Other"])
        ftri = s4.multiselect("🏥 Triage", ["URGENT REVIEW", "PRIORITY REVIEW",
                                            "ROUTINE MONITORING", "INCONCLUSIVE"])
        view = df.copy()
        if qtxt:
            view = view[view.apply(
                lambda r: qtxt.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
        if frisk: view = view[view["Risk Level"].isin(frisk)]
        if fsex: view = view[view["Sex"].isin(fsex)]
        if ftri: view = view[view["Triage"].isin(ftri)]

        st.caption(f"Showing {len(view)} of {len(df)} records — "
                   "tick 🗑️ then press Delete Selected.")
        view = view.copy(); view.insert(0, "🗑️", False)
        edited = st.data_editor(
            view, hide_index=True, use_container_width=True, height=340,
            column_config={"🗑️": st.column_config.CheckboxColumn("🗑️", width="small"),
                           "Record ID": st.column_config.TextColumn(disabled=True),
                           "Risk Score": st.column_config.ProgressColumn(
                               "Risk Score", min_value=0, max_value=100, format="%.0f")},
            key="editor")

        b1, b2, b3 = st.columns(3)
        if b1.button("🗑️ Delete Selected", use_container_width=True):
            drop = edited.loc[edited["🗑️"] == True, "Record ID"].tolist()
            if drop:
                st.session_state.records = df[~df["Record ID"].isin(drop)].reset_index(drop=True)
                save_records(st.session_state.records)
                st.success(f"Deleted {len(drop)} record(s)."); st.rerun()
            else:
                st.warning("No row selected.")
        b2.download_button("📊 Export All (.csv)", df.to_csv(index=False),
                           f"patient_records_{datetime.date.today()}.csv",
                           "text/csv", use_container_width=True)
        with b3.popover("⚠️ Clear All Records", use_container_width=True):
            st.write("This permanently deletes every stored record.")
            if st.button("Yes, delete everything", type="primary"):
                st.session_state.records = load_records().iloc[0:0]
                save_records(st.session_state.records); st.rerun()

        st.divider()
        st.markdown("<div class='sec'>🧾 Individual Patient History</div>",
                    unsafe_allow_html=True)
        who = st.selectbox("Select a patient to view their full visit history",
                           sorted(df["Patient Name"].astype(str).unique()))
        hist = df[df["Patient Name"].astype(str) == who]
        hc = st.columns(4)
        hc[0].metric("Visits", len(hist))
        hc[1].metric("Highest risk",
                     f"{pd.to_numeric(hist['Risk Score'], errors='coerce').max():.0f}")
        hc[2].metric("Latest triage", str(hist.iloc[-1]["Triage"]))
        hc[3].metric("Latest class", str(hist.iloc[-1]["Prediction"]).upper())
        st.dataframe(hist[["Record ID", "Date", "Age", "Sex", "Lesion Site",
                           "Prediction", "Confidence (%)", "Risk Level", "Triage",
                           "Risk Score", "Red Flags", "Clinician"]],
                     hide_index=True, use_container_width=True, height=200)
        if len(hist) > 1:
            st.caption("**Risk score trend across visits**")
            st.line_chart(pd.to_numeric(hist["Risk Score"], errors="coerce")
                          .reset_index(drop=True), height=190)

        st.divider()
        st.markdown("<div class='sec'>📈 Cohort Overview</div>", unsafe_allow_html=True)
        cA, cB, cC = st.columns(3)
        cA.caption("**By predicted class**")
        cA.bar_chart(df["Prediction"].str.upper().value_counts(), height=210)
        cB.caption("**Risk stratification**")
        cB.bar_chart(df["Risk Level"].value_counts(), height=210)
        cC.caption("**Triage distribution**")
        cC.bar_chart(df["Triage"].value_counts(), height=210)

# ============================================================
# TAB 3 — ABOUT
# ============================================================
with tab3:
    st.markdown("<div class='sec'>📖 Project Overview</div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""
**AI-Based Skin Cancer Detection Using Deep Learning on Dermoscopic Images**

This clinical decision-support prototype applies a YOLOv11 classification network to
dermoscopic images and produces a preliminary lesion category, a complete differential
probability distribution, a normalised uncertainty estimate and an objective image-quality
assessment. A rule-based suggestion engine then combines the model output with patient
context — age, sex, anatomical site and reported symptoms — to derive a composite risk
score and a triage category, and to generate a structured referral pathway for clinician
review.

**Pipeline:** acquisition (upload or live camera) → quality scoring → convolutional feature
extraction → soft-max classification → knowledge-base risk mapping → context-weighted
triage → report generation → record archiving and cohort analytics.

**Composite risk score:** weighted class malignancy potential scaled by model confidence,
plus contributions from detected red-flag symptoms, patient age band, UV-exposed anatomical
site, model entropy and image quality. Scores of 62 and above trigger urgent review, 38 to
61 priority review, and below 38 routine monitoring.

**Interpreting entropy:** values near zero indicate a confident peaked prediction; values
approaching one mean probability is spread across classes and the case is inconclusive.

**Responsive interface:** the layout adapts automatically to desktop, tablet and mobile
screens, so the same consultation workflow can be completed at the bedside on a phone.

**Data persistence:** every saved consultation is appended to `records/patient_records.csv`,
the submitted image is archived under `records/images/`, and the attending clinician's name
is remembered in `records/clinician.json` so it is restored on the next launch.
""")
        t1, t2 = st.columns(2)
        t1.info("**Golam Muktadir Al Sabir**\n\nID: 0242220005101761")
        t2.info("**Jahidul Islam Jisan**\n\nID: 0242220005101743")

    st.error("""**Important Limitations**

This is a student research prototype. It is not a certified medical device, has not undergone
clinical validation or regulatory approval, and its accuracy depends entirely on the training
data distribution. It can produce false negatives on malignant lesions and false positives on
benign ones, and the suggestion engine is rule-based rather than clinically validated. It must
never be the sole basis of any diagnostic or treatment decision, and every patient must be
evaluated by a qualified dermatologist.""")
