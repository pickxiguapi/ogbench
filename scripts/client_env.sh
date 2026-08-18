# 客户端公共路径。实验脚本先 source 本文件；新增服务器时追加一个 case 分支。
case "${CLIENT_ID:-$(hostname)}" in
  yb|cs-3ab64-052a9-server)
    CLIENT_ID=yb
    CLIENT_ROOT=/root/data/yyf

    OGBENCH_ROOT=$CLIENT_ROOT/ogbench-new
    PYTHON_BIN=$OGBENCH_ROOT/.venv/bin/python
    DASHBOARD_ROOT=$CLIENT_ROOT/experiment-dashboard
    RECORDED_RUN=$DASHBOARD_ROOT/scripts/recorded_run.sh

    OGBENCH_DATA_DIR=$CLIENT_ROOT/ogbench-cache/data
    LEWM_DATA_ROOT=$CLIENT_ROOT/stable-worldmodel/datasets
    EGL_LIB_DIR=$CLIENT_ROOT/egl-runtime/root/usr/lib/x86_64-linux-gnu

    LEWM_RUNS_ROOT=$CLIENT_ROOT/lewm-runs
    OGBENCH_NATIVE_RUNS_ROOT=$CLIENT_ROOT/ogbench-native-runs
    HIQL_CHUNK_TWO_V_RUNS_ROOT=$CLIENT_ROOT/ogbench-hiql-chunk-two-v-runs
    HIQL_CHUNK_TWO_V_EVALS_ROOT=$CLIENT_ROOT/ogbench-hiql-chunk-two-v-evals
    HIQL_OFFICIAL_RUNS_ROOT=$CLIENT_ROOT/ogbench-hiql-official-runs
    ;;
  *) echo "未知服务器，请在 scripts/client_env.sh 中登记" >&2; return 1 2>/dev/null || exit 1 ;;
esac
