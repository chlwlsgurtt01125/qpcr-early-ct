import streamlit as st

# Data Catalog UI 렌더링 함수(네가 이미 만든 파일) 불러오기
from core.data_catalog_integration import render_data_catalog_section

st.set_page_config(page_title="Data Catalog", layout="wide")

st.title("📊 Data Quality Control & Catalog")
st.caption("QC 상태(PASS/FAIL/FLAG), Ct bin, excluded 사유를 한 번에 정리/다운로드하는 페이지")

try:
    render_data_catalog_section()
except Exception as e:
    st.error(f"Data Catalog 로드 실패: {e}")
    st.exception(e)
