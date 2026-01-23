import streamlit as st
import pandas as pd
import altair as alt
import os

from core.registry import get_active_model_id, list_report_model_ids, resolve_report_paths

st.set_page_config(page_title="모델 성능", layout="wide")
st.title("📈 모델 성능 리포트")
st.caption("cutoff 변화에 따른 성능(MAE/RMSE)과 오류 패턴을 그래프로 확인합니다.")

# --- Model selector ---
all_models = list_report_model_ids()
default_model = get_active_model_id()

colA, colB, colC = st.columns([2, 1, 2])
with colA:
    model_id = st.selectbox(
        "모델 선택",
        options=all_models if all_models else ([default_model] if default_model else []),
        index=(all_models.index(default_model) if (all_models and default_model in all_models) else 0) if (all_models or default_model) else 0,
        placeholder="리포트 모델이 없습니다",
    )
with colB:
    step = st.selectbox("cutoff step", options=[1, 3, 5], index=1)
with colC:
    st.write("")

if not model_id:
    st.warning("data/reports/<model_id>/ 아래에 리포트 파일을 만들어주세요.")
    st.stop()

paths = resolve_report_paths(model_id)

# --- Load metrics_by_cutoff ---
if not paths["metrics_by_cutoff"].exists():
    st.error(f"리포트 파일이 없습니다: {paths['metrics_by_cutoff']}")
    st.markdown(
        """
**필요한 파일(최소 1개):**
- `data/reports/<model_id>/metrics_by_cutoff.parquet`

**권장 컬럼 예시:**
- `cutoff` (int)
- `mae_test` (float)
- `rmse_test` (float)
- `n_curves` (int)
- `n_runs` (int)
"""
    )
    st.stop()

m = pd.read_parquet(paths["metrics_by_cutoff"])
# 최소 컬럼 보호
needed = {"cutoff", "mae_test", "rmse_test"}
if not needed.issubset(set(m.columns)):
    st.error(f"metrics_by_cutoff에 필요한 컬럼이 부족합니다. 필요: {needed}, 현재: {set(m.columns)}")
    st.stop()

m = m.sort_values("cutoff").reset_index(drop=True)
m_plot = m[m["cutoff"] % step == 0] if step != 1 else m

# --- KPI cards ---
k1, k2, k3, k4 = st.columns(4)
best_row = m.loc[m["mae_test"].idxmin()]
k1.metric("추천 cutoff (MAE 최소)", int(best_row["cutoff"]))
k2.metric("최소 MAE", float(best_row["mae_test"]))
k3.metric("해당 cutoff RMSE", float(best_row["rmse_test"]))
if "n_curves" in m.columns:
    k4.metric("n_curves", int(best_row["n_curves"]))
else:
    k4.metric("cutoff 개수", int(m["cutoff"].nunique()))

st.divider()

# --- Plot: MAE/RMSE vs cutoff ---
left, right = st.columns([3, 2])

with left:
    st.subheader("1) cutoff에 따른 성능 변화")
    base = alt.Chart(m_plot).encode(x=alt.X("cutoff:Q", title="Cutoff (cycle)"))
    mae_line = base.mark_line().encode(y=alt.Y("mae_test:Q", title="MAE"))
    rmse_line = base.mark_line(strokeDash=[6, 4]).encode(y=alt.Y("rmse_test:Q", title="RMSE"))
    chart = alt.layer(mae_line, rmse_line).resolve_scale(y="independent")
    st.altair_chart(chart, use_container_width=True)

with right:
    st.subheader("2) cutoff별 테이블")
    show_cols = [c for c in ["cutoff", "mae_test", "rmse_test", "n_curves", "n_runs"] if c in m.columns]
    st.dataframe(m[show_cols], use_container_width=True, hide_index=True)

st.divider()

# --- Error pattern plots (optional if predictions_long exists) ---
if paths["predictions_long"].exists():
    st.subheader("3) 오류 패턴 분석(샘플 단위)")
    pred_long = pd.read_parquet(paths["predictions_long"])

    required = {"cutoff", "true_ct", "pred_ct"}
    if not required.issubset(set(pred_long.columns)):
        st.warning(f"predictions_long에 필요한 컬럼이 부족합니다. 필요: {required}")
    else:
        pred_long = pred_long.copy()
        pred_long["abs_error"] = (pred_long["pred_ct"] - pred_long["true_ct"]).abs()

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            cutoff_sel = st.selectbox("분석 cutoff 선택", sorted(pred_long["cutoff"].unique()))
        with c2:
            thr = st.number_input("high-error 기준(>=)", value=3.0, step=0.5)
        with c3:
            st.write("")

        dfc = pred_long[pred_long["cutoff"] == cutoff_sel].dropna(subset=["true_ct", "pred_ct"])
        if dfc.empty:
            st.info("선택한 cutoff에 해당하는 데이터가 없습니다.")
        else:
            # Scatter: Pred vs True
            colx, coly = st.columns(2)
            with colx:
                st.markdown("**Pred vs True**")
                sc = (
                    alt.Chart(dfc)
                    .mark_circle(size=60, opacity=0.6)
                    .encode(
                        x=alt.X("true_ct:Q", title="True Ct"),
                        y=alt.Y("pred_ct:Q", title="Pred Ct"),
                        tooltip=["true_ct", "pred_ct", "abs_error"]
                    )
                )
                # y=x line (domain based)
                minv = float(min(dfc["true_ct"].min(), dfc["pred_ct"].min()))
                maxv = float(max(dfc["true_ct"].max(), dfc["pred_ct"].max()))
                line = alt.Chart(pd.DataFrame({"x":[minv, maxv], "y":[minv, maxv]})).mark_line().encode(x="x:Q", y="y:Q")
                st.altair_chart(sc + line, use_container_width=True)

            with coly:
                st.markdown("**Abs Error vs True Ct**")
                ec = (
                    alt.Chart(dfc)
                    .mark_circle(size=60, opacity=0.6)
                    .encode(
                        x=alt.X("true_ct:Q", title="True Ct"),
                        y=alt.Y("abs_error:Q", title="Abs Error"),
                        tooltip=["true_ct", "pred_ct", "abs_error"]
                    )
                )
                st.altair_chart(ec, use_container_width=True)

            # Top high-error table
            st.markdown("**Top high-error candidates**")
            key_cols = [c for c in ["run_id", "well_id"] if c in dfc.columns]
            show = dfc.sort_values("abs_error", ascending=False).head(30)
            show_cols = key_cols + ["true_ct", "pred_ct", "abs_error"]
            st.dataframe(show[show_cols], use_container_width=True, hide_index=True)

else:
    st.info("샘플 단위 오류 분석을 보려면 `predictions_long.parquet`를 생성해 주세요.")
    st.caption(f"경로: {paths['predictions_long']}")
