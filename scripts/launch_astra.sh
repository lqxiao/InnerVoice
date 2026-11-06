#!/bin/bash
cd /local/home/liqiangx/STP/think
pkill -f "gpt-6-astra" ; sleep 2
rm -f data_math/self/gen_gpt6astra.jsonl data_math/self/gen_gpt6astra.jsonl.rejected.jsonl
EXTRA='IMPORTANT: the <reasoning> section is mandatory and must be substantial: at least 200 words of genuine step-by-step working (set up, compute intermediate quantities with their values, check a special case or alternative route, then conclude). The <answer> section follows it. In the final boxed expression put ONLY the final value asked for (a number, Yes/No, a closed form), never a sentence or an equation like X=Y unless the question asks for an equation.'
nohup python3 latent/bedrock_gen.py --model us.openai.gpt-6-astra --effort "" --temperature -1 --sys_extra "$EXTRA" --out data_math/self/gen_gpt6astra.jsonl --workers 12 > /tmp/bedrock_astra.log 2>&1 &
echo launched
