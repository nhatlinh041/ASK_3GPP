#!/usr/bin/env bash
# Rebuild Knowledge Graph trong demo — dọn sạch zombie schema trước khi build.
#
# Usage (chạy từ demo/):
#   bash scripts/rebuild-kg.sh                 # full: clean + KG + embeddings
#   bash scripts/rebuild-kg.sh kg-only         # clean + KG (no embeddings)
#   bash scripts/rebuild-kg.sh embed-only      # chỉ tạo embeddings (KG đã có)
#   SKIP_RESTART=1 bash scripts/rebuild-kg.sh  # bỏ qua docker restart (giữ zombie tokens)
#
# Cleanup pipeline (full/kg-only):
#   1. Drop ALL constraints + indexes (kể cả vector + legacy)
#   2. Batched DETACH DELETE all nodes (10k/batch)
#   3. Restart Docker container neo4j-server (xoá zombie label/type tokens)
#   4. Setup schema mới (4 nodes + 6 edges đúng schema)
#   5. Load JSON → build graph
#   6. (full) Embedder + vector index
#
# Env vars (hoặc khai báo trong demo/.env):
#   JSON_DIR        — thư mục chứa processed JSON (default: ../3GPP_JSON_DOC/processed_json_v3)
#   NEO4J_URI       — (default: neo4j://localhost:7687)
#   NEO4J_USER      — (default: neo4j)
#   NEO4J_PASSWORD  — (default: password)
#   SKIP_RESTART    — set =1 để bỏ qua docker restart (giữ zombie metadata)
#   NEO4J_CONTAINER — container name (default: neo4j-server)

set -e
DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

ok()   { echo -e "\033[32m✓\033[0m $*"; }
info() { echo -e "\033[36mℹ\033[0m $*"; }
warn() { echo -e "\033[33m⚠\033[0m $*"; }
err()  { echo -e "\033[31m✗\033[0m $*"; exit 1; }

# Load .env nếu có
[ -f "$DEMO_DIR/.env" ] && { set -a; . "$DEMO_DIR/.env"; set +a; }

# Default JSON dir: dùng demo/3GPP_JSON_DOC nếu có (portable), fallback parent dir
if [ -d "$DEMO_DIR/3GPP_JSON_DOC/processed_json_v3" ]; then
  JSON_DIR_DEFAULT="$DEMO_DIR/3GPP_JSON_DOC/processed_json_v3"
else
  JSON_DIR_DEFAULT="$DEMO_DIR/../3GPP_JSON_DOC/processed_json_v3"
fi
JSON_DIR="${JSON_DIR:-$JSON_DIR_DEFAULT}"
MODE="${1:-full}"
NEO4J_CONTAINER="${NEO4J_CONTAINER:-neo4j-server}"
SKIP_RESTART="${SKIP_RESTART:-0}"

info "Demo dir:    $DEMO_DIR"
info "JSON dir:    $JSON_DIR"
info "Mode:        $MODE"
info "Container:   $NEO4J_CONTAINER"
info "Skip restart: $SKIP_RESTART"

# Activate venv nếu chưa active
if [ -z "$VIRTUAL_ENV" ]; then
  VENV="$DEMO_DIR/../.venv"
  [ -d "$VENV" ] || err "venv not found at $VENV. Run: python -m venv $VENV && pip install -r requirements.txt"
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
fi

# ── Helper: restart Neo4j Docker container để dọn zombie label/type tokens ──
restart_neo4j() {
  if [ "$SKIP_RESTART" = "1" ]; then
    warn "Skipping Docker restart (SKIP_RESTART=1) — zombie label/type tokens sẽ persist."
    return
  fi

  if ! command -v docker &>/dev/null; then
    warn "Docker không có sẵn — bỏ qua restart, zombie tokens sẽ persist."
    return
  fi

  if ! docker ps --filter "name=^${NEO4J_CONTAINER}$" --format "{{.Names}}" 2>/dev/null | grep -q "${NEO4J_CONTAINER}"; then
    warn "Container ${NEO4J_CONTAINER} không chạy — bỏ qua restart."
    return
  fi

  info "Restarting Docker container '${NEO4J_CONTAINER}' để dọn zombie tokens..."
  docker restart "$NEO4J_CONTAINER" >/dev/null

  # Đợi Neo4j HTTP ready (max 30s)
  info "Đợi Neo4j HTTP ready..."
  for i in $(seq 1 30); do
    if curl -sf http://localhost:7474 > /dev/null 2>&1; then
      ok "Neo4j ready sau ${i}s"
      # Đợi thêm 2s cho Bolt connection ổn định
      sleep 2
      return
    fi
    sleep 1
  done
  err "Neo4j không ready sau 30s. Check: docker logs ${NEO4J_CONTAINER}"
}

# ── Phase 1: Full clean (drop schema + delete data) ─────────────────────────
run_clean() {
  python - <<'PYEOF'
import sys, os
from pathlib import Path
demo_dir = Path(os.environ["DEMO_DIR"])
sys.path.insert(0, str(demo_dir))

from kg_builder import KGBuilder

builder = KGBuilder()
if not builder.verify_connection():
    sys.exit(1)
print("[kg] === Phase 1: Full clean ===")
builder.clean_all()
builder.close()
PYEOF
}

# ── Phase 2: Build KG (after restart, fresh schema) ─────────────────────────
run_build_kg() {
  python - <<'PYEOF'
import sys, os
from pathlib import Path
demo_dir = Path(os.environ["DEMO_DIR"])
sys.path.insert(0, str(demo_dir))

from kg_builder import KGBuilder

json_dir = Path(os.environ.get("JSON_DIR", ""))
builder = KGBuilder()
if not builder.verify_connection():
    sys.exit(1)
print("[kg] === Phase 2: Build KG ===")
builder.setup_schema()
builder.load_json_dir(json_dir)
builder.print_stats()
builder.validate()
builder.close()
PYEOF
}

# ── Phase 3: Embeddings + vector index ──────────────────────────────────────
run_embed() {
  python - <<'PYEOF'
import sys, os
from pathlib import Path
demo_dir = Path(os.environ["DEMO_DIR"])
sys.path.insert(0, str(demo_dir))

from kg_builder import Embedder

print("[kg] === Phase 3: Embeddings ===")
embedder = Embedder()
embedder.create_vector_index()
embedder.embed_all_chunks()
embedder.close()
PYEOF
}

export DEMO_DIR="$DEMO_DIR"
export JSON_DIR="$JSON_DIR"

# ── Main pipeline ───────────────────────────────────────────────────────────
case "$MODE" in
  full)
    run_clean
    restart_neo4j
    run_build_kg
    run_embed
    ;;
  kg-only)
    run_clean
    restart_neo4j
    run_build_kg
    ;;
  embed-only)
    run_embed
    ;;
  clean-only)
    run_clean
    restart_neo4j
    ;;
  *)
    err "Unknown mode: $MODE. Use: full | kg-only | embed-only | clean-only"
    ;;
esac

ok "Rebuild complete."
