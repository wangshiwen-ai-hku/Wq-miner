TOPIC="$1"
if [ -z "$TOPIC" ]; then
    TOPIC="WorldQuant Brain News 情感数据与短期价格反转的 Alpha"
fi
python run_research.py --topic "$TOPIC" -v