import streamlit as st
import pandas as pd
import altair as alt

from core.registry import get_active_model_id, list_report_model_ids, resolve_report_paths
from core.storage import upsert_annotation, fetch_annotations, init_db

st.set_page_config(page_title="오차 큰 샘플 검토", layout="wide")
st.title("🧷 오차 큰 샘플(High-error) 검토")
st.caption("오차가 큰 후보군을 확인하고, 진짜 hard인지 검증/태그/코멘트를 저장합니다.")

init_db()

# --- Model selector ---
all_models = list_report_model_ids()
default_model = get_active_model_id()

model_id = st.selectbox(
    "모델 선택",
    options=all_models if all_models else ([default_model] if default_model else []),
    index=(all_models.index(default_model) if (all_models and default_model in all_models) else 0) if (all_models or default_model) else 0,
    placeholder="리포트 모델이 없습니다",
)

if not model_id:
    st.warning("리포트 모델이 없습니다. 먼저 data/reports/<model_id>/를 만들어주세요.")
    st.stop()

paths = resolve_report_paths(model_id)

if not paths["predictions_long"].exists():
    st.error(f"필수 파일이 없습니다: {paths['predictions_long']}")
    st.markdown(
        """
Hard Review는 샘플별 예측이 cutoff별로 쌓인 long-format이 필요합니다.

**필요 파일**
- `data/reports/<model_id>/predictions_long.parquet`

**권장 컬럼**
- `run_id` (str), `well_id` (str)
- `cutoff` (int)
- `true_ct` (float), `pred_ct` (float)
"""
    )
    st.stop()

pred_long = pd.read_parquet(paths["predictions_long"]).copy()
for c in ["true_ct", "pred_ct"]:
    if c not in pred_long.columns:
        st.error(f"predictions_long에 `{c}` 컬럼이 없습니다.")
        st.stop()

pred_long["abs_error"] = (pred_long["pred_ct"] - pred_long["true_ct"]).abs()

# --- Filters ---
col1, col2, col3, col4 = st.columns([1.3, 1.3, 1.2, 1.2])
with col1:
    cutoff_sel = st.selectbox("검토 cutoff", sorted(pred_long["cutoff"].unique()))
with col2:
    thr = st.number_input("high-error 기준(>=)", value=3.0, step=0.5)
with col3:
    status_filter = st.selectbox("상태", ["(전체)", "Unreviewed", "Confirmed", "SuspectLabel", "Resolved"])
with col4:
    reviewer = st.text_input("리뷰어(선택)", value="")

dfc = pred_long[pred_long["cutoff"] == cutoff_sel].copy()
dfc = dfc.sort_values("abs_error", ascending=False)

# join existing annotations for status/tags/comment display
ann = fetch_annotations(model_id=model_id, limit=5000)
ann_df = pd.DataFrame(ann) if ann else pd.DataFrame(columns=["run_id","well_id","cutoff","status","tags","comment"])
if not ann_df.empty:
    ann_df = ann_df[["run_id","well_id","cutoff","status","tags","comment"]]
    dfc = dfc.merge(ann_df, on=["run_id","well_id","cutoff"], how="left", suffixes=("","_ann"))

dfc["status"] = dfc["status"].fillna("Unreviewed")
dfc["tags"] = dfc["tags"].fillna("")
dfc["comment"] = dfc["comment"].fillna("")

# filter by threshold/status
dfc = dfc[dfc["abs_error"] >= float(thr)]
if status_filter != "(전체)":
    dfc = dfc[dfc["status"] == status_filter]

st.divider()

# --- Layout: left list / right detail ---
left, right = st.columns([2.2, 2.8])

with left:
    st.subheader("1) 후보 리스트")
    show_cols = [c for c in ["run_id","well_id","true_ct","pred_ct","abs_error","status","tags"] if c in dfc.columns]
    st.dataframe(dfc[show_cols].head(200), use_container_width=True, hide_index=True)

    if dfc.empty:
        st.info("조건에 해당하는 후보가 없습니다. threshold/status 필터를 조정해보세요.")
        st.stop()

    # 선택 UI: key 생성
    dfc["sample_key"] = dfc["run_id"].astype(str) + " | " + dfc["well_id"].astype(str)
    sample_key = st.selectbox("상세 보기 샘플 선택", dfc["sample_key"].head(200).tolist())

    sel = dfc[dfc["sample_key"] == sample_key].iloc[0]
    run_id = str(sel["run_id"])
    well_id = str(sel["well_id"])

with right:
    st.subheader("2) 상세 분석 + 리뷰 기록")

    # sample 전체 cutoff 트래젝토리
    s_all = pred_long[(pred_long["run_id"] == run_id) & (pred_long["well_id"] == well_id)].copy()
    s_all = s_all.sort_values("cutoff")
    s_all["abs_error"] = (s_all["pred_ct"] - s_all["true_ct"]).abs()

    k1, k2, k3 = st.columns(3)
    k1.metric("True Ct", float(sel["true_ct"]))
    k2.metric("Pred Ct", float(sel["pred_ct"]))
    k3.metric("Abs Error", float(sel["abs_error"]))

    st.markdown("**(핵심) cutoff 변화에 따른 예측 수렴(trajectory)**")
    traj = (
        alt.Chart(s_all)
        .mark_line(point=True)
        .encode(
            x=alt.X("cutoff:Q", title="Cutoff (cycle)"),
            y=alt.Y("pred_ct:Q", title="Pred Ct"),
            tooltip=["cutoff", "pred_ct", "true_ct", "abs_error"]
        )
    )
    # true Ct horizontal line
    true_ct = float(sel["true_ct"])
    hline = alt.Chart(pd.DataFrame({"y":[true_ct]})).mark_rule().encode(y="y:Q")
    st.altair_chart(traj + hline, use_container_width=True)

    st.markdown("**오차(Abs Error) 변화**")
    errc = (
        alt.Chart(s_all)
        .mark_line(point=True)
        .encode(
            x=alt.X("cutoff:Q", title="Cutoff (cycle)"),
            y=alt.Y("abs_error:Q", title="Abs Error"),
            tooltip=["cutoff", "abs_error"]
        )
    )
    st.altair_chart(errc, use_container_width=True)

    st.divider()
    st.markdown("### 3) 원인 태그/상태 저장")

    tag_options = [
        "no_or_weak_amplification",
        "baseline_drift",
        "noisy_early_phase",
        "late_ct_information_limit",
        "early_saturation",
        "possible_label_or_mapping_issue",
        "other",
    ]

    # 기존 저장된 값 로드(현재 cutoff 기준)
    existing_tags = []
    if isinstance(sel.get("tags", ""), str) and sel["tags"]:
        existing_tags = [t for t in sel["tags"].split(";") if t.strip()]

    status = st.selectbox(
        "상태(status)",
        options=["Unreviewed", "Confirmed", "SuspectLabel", "Resolved"],
        index=["Unreviewed", "Confirmed", "SuspectLabel", "Resolved"].index(sel["status"]) if sel["status"] in ["Unreviewed","Confirmed","SuspectLabel","Resolved"] else 0
    )
    tags = st.multiselect("태그(tags)", options=tag_options, default=existing_tags)
    comment = st.text_area("코멘트(comment)", value=str(sel.get("comment","")), height=120)

    if st.button("💾 저장", type="primary"):
        upsert_annotation(
            model_id=model_id,
            run_id=run_id,
            well_id=well_id,
            cutoff=int(cutoff_sel),
            true_ct=float(sel["true_ct"]),
            pred_ct=float(sel["pred_ct"]),
            abs_error=float(sel["abs_error"]),
            status=status,
            tags=tags,
            comment=comment,
            reviewer=reviewer,
        )
        st.success("저장 완료! (reviews/reviews.db)")
        st.rerun()

    with st.expander("최근 리뷰 기록 보기"):
        ann_latest = fetch_annotations(model_id=model_id, limit=50)
        if ann_latest:
            st.dataframe(pd.DataFrame(ann_latest), use_container_width=True, hide_index=True)
        else:
            st.info("아직 저장된 리뷰가 없습니다.")
