"""ui_parts.py - helper render kecil untuk aplikasi Streamlit."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

CSS = """
<style>
.big-letter {font-size: 88px; font-weight: 800; line-height: 1.05; margin: 0;}
.big-word {font-size: 72px; font-weight: 800; letter-spacing: 6px; min-height: 90px;}
.letter-ok {color: #1b9e4b;} .letter-unsure {color: #d98a00;} .letter-none {color: #c0392b;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def draw_box(img: Image.Image, box, color=(0, 200, 0), width: int = 5) -> Image.Image:
    if not box:
        return img
    out = img.copy()
    ImageDraw.Draw(out).rectangle(box, outline=color, width=width)
    return out


def border(img: Image.Image, ok: bool, width: int = 10) -> Image.Image:
    return ImageOps.expand(img, border=width, fill=(27, 158, 75) if ok else (192, 57, 43))


def topk_chart(topk):
    if not topk:
        return
    df = pd.DataFrame(topk, columns=["Huruf", "Probabilitas"])
    chart = (alt.Chart(df).mark_bar()
             .encode(y=alt.Y("Huruf:N", sort=None, title=None),
                     x=alt.X("Probabilitas:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                     tooltip=["Huruf", alt.Tooltip("Probabilitas:Q", format=".3f")])
             .properties(height=max(60, 34 * len(df))))
    st.altair_chart(chart, width="stretch")


def prediction_card(res: dict, threshold: float):
    """Huruf besar + keyakinan + bar top-k + latensi."""
    if res.get("hand_found") is False:
        st.markdown('<p class="big-letter letter-none">—</p>', unsafe_allow_html=True)
        st.warning("Tangan tidak terdeteksi, model tidak menebak.")
        return
    letter = res["letter"]
    cls = "letter-unsure" if letter == "?" else "letter-ok"
    st.markdown(f'<p class="big-letter {cls}">{letter}</p>', unsafe_allow_html=True)
    st.markdown(f"Keyakinan: **{res['conf']:.1%}**")
    if letter == "?":
        st.caption("Kurang yakin — coba foto lebih dekat / latar polos")
    topk_chart(res["topk"])
    st.caption(f"Latensi: {res['latency_ms']:.1f} ms")
