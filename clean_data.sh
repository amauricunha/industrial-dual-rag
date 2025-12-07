#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$ROOT_DIR/data"

detect_compose() {
  if docker compose version >/dev/null 2>&1; then
    echo "docker compose"
    return
  fi
  if docker-compose --version >/dev/null 2>&1; then
    echo "docker-compose"
    return
  fi
  echo "Nenhum comando docker compose encontrado." >&2
  exit 1
}

COMPOSE_CMD="$(detect_compose)"

if [[ "${1:-}" != "--yes" ]]; then
  echo "Este procedimento vai parar os contêineres e apagar todo o conteúdo da pasta ./data."
  read -r -p "Deseja continuar? [y/N] " ANSWER
  case "$ANSWER" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Operação cancelada."; exit 0 ;;
  esac
fi

cd "$ROOT_DIR"

if [[ -d "$DATA_DIR" ]]; then
  echo "Parando serviços Docker..."
  $COMPOSE_CMD down

  echo "Removendo conteúdos de $DATA_DIR ..."
  rm -rf "$DATA_DIR"
else
  echo "Pasta $DATA_DIR não existe; nada para remover."
fi

echo "Recriando estrutura mínima..."
mkdir -p \
  "$DATA_DIR/api/chromadb" \
  "$DATA_DIR/api/faiss_index" \
  "$DATA_DIR/api/summaries" \
  "$DATA_DIR/api/uploads" \
  "$DATA_DIR/weaviate"

cat <<"MSG"
Limpeza concluída.
Execute "$COMPOSE_CMD up --build -d" para subir novamente e reprocessar os PDFs antes de novos testes.
MSG
