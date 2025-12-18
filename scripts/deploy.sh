#!/usr/bin/env bash

# Guard: this script uses bash features (arrays, [[ ]], pipefail).
# If someone runs it via `sh`/`zsh` or sources it, we fail fast with a clear message.
if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "ERROR: This script must be run with bash (do not 'source' it)." >&2
  echo "Run: bash \"$0\"" >&2
  # If sourced, return; otherwise exec bash.
  if (return 0 2>/dev/null); then
    return 1
  fi
  exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

#
# Build & push Lambda container image to ECR, then deploy via Terraform (interactive apply).
#
# Requirements:
# - aws CLI v2 configured (credentials + permissions)
# - docker
# - terraform
#
# Usage:
#   ./scripts/deploy.sh
#

say() { printf "%s\n" "$*"; }
die() { printf "ERROR: %s\n" "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

prompt_default() {
  # prompt_default "Question" "default"
  local q="$1"
  local d="$2"
  local v=""
  read -r -p "${q} [${d}]: " v
  if [[ -z "${v}" ]]; then
    printf "%s" "${d}"
  else
    printf "%s" "${v}"
  fi
}

prompt_yes_no_default_yes() {
  # returns 0 for yes, 1 for no
  local q="$1"
  local v=""
  read -r -p "${q} [Y/n]: " v
  case "${v}" in
    ""|Y|y|YES|yes) return 0 ;;
    N|n|NO|no) return 1 ;;
    *) die "Please answer Y or n." ;;
  esac
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
tf_dir="${repo_root}/infra/terraform"

need_cmd aws
need_cmd docker
need_cmd terraform

export AWS_PAGER=""

say "Repo: ${repo_root}"
say "Terraform dir: ${tf_dir}"
[[ -d "${tf_dir}" ]] || die "Terraform directory not found: ${tf_dir}"

say ""
say "Checking Docker daemon..."
if ! docker info >/dev/null 2>&1; then
  die "Docker daemon is not running. Please start Docker Desktop (or your Docker engine) and re-run."
fi

default_region="ap-southeast-2"
default_name="text-insight-clustering-service"

say ""
say "### AWS settings"
aws_profile="$(prompt_default "AWS profile (leave empty to use default credentials chain)" "")"
aws_region="$(prompt_default "AWS region" "${default_region}")"

aws_cli=(aws)
if [[ -n "${aws_profile}" ]]; then
  aws_cli+=(--profile "${aws_profile}")
fi
aws_cli+=(--region "${aws_region}")
aws_cli+=(--no-cli-pager)

say ""
say "Checking AWS identity..."
account_id="$("${aws_cli[@]}" sts get-caller-identity --query Account --output text 2>/dev/null || true)"
[[ -n "${account_id}" && "${account_id}" != "None" ]] || die "Unable to determine AWS account id (check credentials/profile/region)."
say "Using AWS account: ${account_id}"

say ""
say "### Image / ECR settings"
tf_name="$(prompt_default "Terraform var.name (resource base name)" "${default_name}")"
ecr_repo_name="$(prompt_default "ECR repository name" "${tf_name}")"

default_tag="$(date -u +"%Y%m%d-%H%M%S")"
image_tag="$(prompt_default "Image tag" "${default_tag}")"

registry="${account_id}.dkr.ecr.${aws_region}.amazonaws.com"
image_uri="${registry}/${ecr_repo_name}:${image_tag}"

say ""
say "Planned image URI:"
say "  ${image_uri}"

say ""
say "Ensuring ECR repo exists: ${ecr_repo_name}"
if ! "${aws_cli[@]}" ecr describe-repositories --repository-names "${ecr_repo_name}" >/dev/null 2>&1; then
  say "ECR repo not found; creating..."
  "${aws_cli[@]}" ecr create-repository --repository-name "${ecr_repo_name}" >/dev/null
else
  say "ECR repo exists."
fi

say ""
say "Logging Docker into ECR registry: ${registry}"

docker_config_dir=""
cleanup() {
  if [[ -n "${docker_config_dir}" ]]; then
    rm -rf "${docker_config_dir}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if command -v docker-credential-ecr-login >/dev/null 2>&1; then
  say "Detected Amazon ECR credential helper; using it (no docker login needed)."
  docker_config_dir="$(mktemp -d)"
  cat > "${docker_config_dir}/config.json" <<EOF
{
  "credHelpers": {
    "${registry}": "ecr-login"
  }
}
EOF
else
  say "ECR credential helper not found; performing docker login via AWS token..."
  docker_config_dir="$(mktemp -d)"
  say "Fetching ECR auth token (aws ecr get-login-password)..."
  ecr_password="$("${aws_cli[@]}" ecr get-login-password 2>/dev/null || true)"
  if [[ -z "${ecr_password}" ]]; then
    die "Failed to fetch ECR auth token. Try: aws sts get-caller-identity && aws ecr get-login-password --region ${aws_region}"
  fi

  printf "%s" "${ecr_password}" \
    | DOCKER_CONFIG="${docker_config_dir}" docker login --username AWS --password-stdin "${registry}" >/dev/null
  unset ecr_password
fi

say ""
say "### Docker build & push"
say "Building image from ${repo_root}/Dockerfile ..."
#
# IMPORTANT (Lambda container image compatibility):
# Lambda is picky about manifest/layer media types. In practice, the most reliable path is:
# - use buildx to PUSH directly with Docker media types + gzip compression
#   (avoids OCI index/zstd layers that Lambda may reject)
#
#
# Some environments have a `docker buildx` shim that exists but can't actually parse buildx flags.
# We consider buildx usable only if it can successfully parse `--platform` on the build subcommand.
# You can force-disable buildx by setting FORCE_NO_BUILDX=1.
#
buildx_ok=false
if [[ "${FORCE_NO_BUILDX:-}" == "1" ]]; then
  buildx_ok=false
else
  if docker buildx >/dev/null 2>&1; then
    if docker buildx build --platform linux/amd64 --help >/dev/null 2>&1; then
      buildx_ok=true
    fi
  fi
fi

if [[ "${buildx_ok}" == "true" ]]; then
  say "Building & pushing with docker buildx (linux/amd64)..."

  # Prefer forcing docker mediatypes + gzip if supported by this buildx version.
  # Not all versions expose these knobs, so we fall back gracefully.
  if docker buildx build --help 2>/dev/null | grep -q "oci-mediatypes"; then
    buildx_cmd=(
      docker buildx build
      --platform linux/amd64
      --provenance=false
      --sbom=false
      --output "type=image,name=${image_uri},push=true,oci-mediatypes=false,compression=gzip"
      "${repo_root}"
    )
  else
    buildx_cmd=(
      docker buildx build
      --platform linux/amd64
      --provenance=false
      --sbom=false
      --push
      -t "${image_uri}"
      "${repo_root}"
    )
  fi

  if ! DOCKER_CONFIG="${docker_config_dir}" "${buildx_cmd[@]}"; then
    say "WARNING: docker buildx failed (exit $?) — falling back to classic docker build." >&2
    buildx_ok=false
  fi
fi

if [[ "${buildx_ok}" != "true" ]]; then
  say "WARNING: docker buildx is unavailable or doesn't support --platform; falling back to classic docker build."
  say "Tip: this fallback sets DOCKER_BUILDKIT=0 to avoid OCI/zstd media types that Lambda may reject."
  DOCKER_BUILDKIT=0 docker build -t "${ecr_repo_name}:${image_tag}" "${repo_root}"
  DOCKER_CONFIG="${docker_config_dir}" docker tag "${ecr_repo_name}:${image_tag}" "${image_uri}"
  say "Pushing image: ${image_uri}"
  DOCKER_CONFIG="${docker_config_dir}" docker push "${image_uri}"
fi

say ""
say "### Terraform deploy"
say "Terraform will be applied INTERACTIVELY; you'll need to type 'yes' to confirm."

#
# tfvars handling:
# - Preferred: `infra/terraform/lambda_env.auto.tfvars` (auto-loaded by Terraform)
# - Override: set TFVARS_FILE=/path/to/file.tfvars when running this script
#
tf_var_file="${TFVARS_FILE:-}"
default_tfvars="${tf_dir}/lambda_env.auto.tfvars"
if [[ -z "${tf_var_file}" && -f "${default_tfvars}" ]]; then
  tf_var_file="${default_tfvars}"
fi

pushd "${tf_dir}" >/dev/null
terraform init

apply_args=(
  -var "aws_region=${aws_region}"
  -var "name=${tf_name}"
  -var "lambda_image_uri=${image_uri}"
)
if [[ -n "${tf_var_file}" ]]; then
  # Allow relative or absolute paths
  if [[ ! -f "${tf_var_file}" ]]; then
    # try resolving relative to repo root
    if [[ -f "${repo_root}/${tf_var_file}" ]]; then
      tf_var_file="${repo_root}/${tf_var_file}"
    else
      die "tfvars file not found: ${tf_var_file}"
    fi
  fi
  apply_args+=(-var-file "${tf_var_file}")
fi

set +e
terraform apply "${apply_args[@]}"
tf_exit=$?
set -e

say ""
if [[ ${tf_exit} -ne 0 ]]; then
  die "terraform apply failed with exit code ${tf_exit}"
fi

say "Deployment complete."
say ""
say "Useful outputs:"
terraform output
popd >/dev/null

