set -euo pipefail

YELLOW='\033[0;33m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "🔍 Validating required environment variables…"

# Check for missing env vars
missing=()
[[ -z "${APP_CONFIG_ENDPOINT:-}" ]] && missing+=("APP_CONFIG_ENDPOINT")

if [[ ${#missing[@]} -gt 0 ]]; then
  echo -e "${YELLOW}⚠️  Missing required environment variables:${NC}"
  for var in "${missing[@]}"; do
    echo "    • $var"
  done
  echo
  echo "Please set them before running this script, e.g.:"
  echo "  export APP_CONFIG_ENDPOINT=<your-value>"
  echo "Or use: azd env set APP_CONFIG_ENDPOINT <your-value>"
  exit 1
fi

echo -e "${GREEN}✅ All required azd env values are set.${NC}"
echo

echo -e "${BLUE}📦 Creating temporary virtual environment…${NC}"
python -m venv evaluation/.venv_temp
chmod a+r evaluation/.venv_temp/bin/activate
source evaluation/.venv_temp/bin/activate
echo -e "${BLUE}⬇️  Installing requirements…${NC}"
pip install --upgrade pip
pip install -r evaluation/requirements.txt


echo -e "${BLUE}🚀 Running evaluate.py…${NC}"
python -m evaluation.evaluate
echo -e "${GREEN}✅ Finished evaluation.${NC}"

# clean up venv only if we created it
if [[ -n "${AZURE_APP_CONFIG_ENDPOINT:-}" ]]; then
  echo
  echo -e "${BLUE}🧹 Cleaning up…${NC}"
  deactivate
  rm -rf evaluation/.venv_temp
fi