# 客户端公共路径。CLIENT_ID 只允许 yb、23、7002、11；新增路径时补全对应分支。
case "${CLIENT_ID:?请先设置 CLIENT_ID=yb|23|7002|11}" in
  yb)
    CLIENT_ROOT=/root/data/yyf

    OGBENCH_ROOT=$CLIENT_ROOT/ogbench-new
    PYTHON_BIN=$OGBENCH_ROOT/.venv/bin/python
    DASHBOARD_ROOT=$CLIENT_ROOT/experiment-dashboard

    OGBENCH_DATA_DIR=$CLIENT_ROOT/ogbench-cache/data
    LEWM_DATA_ROOT=$CLIENT_ROOT/stable-worldmodel/datasets
    EGL_LIB_DIR=$CLIENT_ROOT/egl-runtime/root/usr/lib/x86_64-linux-gnu

    LEWM_RUNS_ROOT=$CLIENT_ROOT/lewm-runs
    OGBENCH_NATIVE_RUNS_ROOT=$CLIENT_ROOT/ogbench-native-runs
    HIQL_OFFICIAL_RUNS_ROOT=$CLIENT_ROOT/ogbench-hiql-official-runs
    ;;
  23|7002|11) echo "CLIENT_ID=$CLIENT_ID 的路径尚未登记" >&2; return 1 2>/dev/null || exit 1 ;;
  *) echo "CLIENT_ID 只允许 yb、23、7002、11" >&2; return 1 2>/dev/null || exit 1 ;;
esac
