# 客户端公共路径。CLIENT_ID 只允许 yb、23、7002、11、node1、node2、node3、node4；新增路径时补全对应分支。
case "$CLIENT_ID" in
  yb)
    CLIENT_ROOT=/root/data/yyf

    OGBENCH_ROOT=${OGBENCH_ROOT:-$CLIENT_ROOT/ogbench-new}
    PYTHON_BIN=$OGBENCH_ROOT/.venv/bin/python

    OGBENCH_DATA_DIR=$CLIENT_ROOT/ogbench-cache/data
    LEWM_DATA_ROOT=$CLIENT_ROOT/stable-worldmodel/datasets
    GCIQL_DATA_ROOT=${GCIQL_DATA_ROOT:-$CLIENT_ROOT/stable-worldmodel/datasets}
    GCIQL_RUNS_ROOT=${GCIQL_RUNS_ROOT:-$CLIENT_ROOT/lewm-final}
    EGL_LIB_DIR=$CLIENT_ROOT/egl-runtime/root/usr/lib/x86_64-linux-gnu
    ;;
  23)
    CLIENT_ROOT=/home/dzb

    OGBENCH_ROOT=${OGBENCH_ROOT:-$CLIENT_ROOT/ogbench}
    PYTHON_BIN=$OGBENCH_ROOT/.venv/bin/python

    LEWM_DATA_ROOT=/data/dzb/stablewm-data/datasets
    RUN_DIR=/data/dzb/stablewm-data/
    ;;
  7002)
    CLIENT_ROOT=/home/yyf/yyf

    OGBENCH_ROOT=${OGBENCH_ROOT:-$CLIENT_ROOT/ogbench}
    PYTHON_BIN=$OGBENCH_ROOT/.venv-s23/bin/python
    ;;
  11)
    CLIENT_ROOT=/home/yyf

    OGBENCH_ROOT=${OGBENCH_ROOT:-$CLIENT_ROOT/ogbench}
    PYTHON_BIN=/data/yyf/H-LeWM/envs/ogbench/bin/python

    LEWM_DATA_ROOT=/data/yyf/H-LeWM/datasets
    ;;
  node1)
    CLIENT_ROOT=/data-training/yyf

    OGBENCH_ROOT=${OGBENCH_ROOT:-/home/yyf/ogbench-main}
    PYTHON_BIN=$CLIENT_ROOT/envs/ogbench/bin/python

    OGBENCH_DATA_DIR=$CLIENT_ROOT/ogbench-cache/data
    VISUAL_EVAL_ASSET_ROOT=$CLIENT_ROOT/lewm-gciql-visual-eval-assets
    VISUAL_EVAL_ROOT=$CLIENT_ROOT/lewm-gciql-visual-evals
    EGL_LIB_DIR=/usr/lib/x86_64-linux-gnu
    ;;
  node2)
    CLIENT_ROOT=/data-training/yyf

    OGBENCH_ROOT=${OGBENCH_ROOT:-/home/yyf/ogbench-main}
    PYTHON_BIN=$CLIENT_ROOT/envs/ogbench/bin/python

    OGBENCH_DATA_DIR=$CLIENT_ROOT/ogbench-cache/data
    LEWM_JAX_RUNS_ROOT=$CLIENT_ROOT/ogbench/lewm-jax-visual-runs
    EGL_LIB_DIR=/usr/lib/x86_64-linux-gnu
    ;;
  node3)
    CLIENT_ROOT=/data-training/yyf

    OGBENCH_ROOT=${OGBENCH_ROOT:-$CLIENT_ROOT/ogbench}
    PYTHON_BIN=$CLIENT_ROOT/envs/ogbench/bin/python

    OGBENCH_DATA_DIR=$CLIENT_ROOT/ogbench-cache/data
    EGL_LIB_DIR=/usr/lib/x86_64-linux-gnu
    ;;
  node4)
    CLIENT_ROOT=/data-training/yyf

    OGBENCH_ROOT=${OGBENCH_ROOT:-$CLIENT_ROOT/ogbench/clean-main}
    PYTHON_BIN=$CLIENT_ROOT/ogbench/.venv/bin/python

    OGBENCH_DATA_DIR=$CLIENT_ROOT/ogbench-cache/data
    LEWM_DATA_ROOT=$CLIENT_ROOT/datasets/latent-geometry
    LEWM_JAX_RUNS_ROOT=$CLIENT_ROOT/ogbench/lewm-jax-visual-runs
    EGL_LIB_DIR=/usr/lib/x86_64-linux-gnu
    ;;
esac
