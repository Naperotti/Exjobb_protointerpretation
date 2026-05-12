# External GPU Workflow (Tailscale + SSH)

Quick reference for running code on the remote GPU machine.

---

## 1) One-time setup

### 1.1 Connect to remote machine
```bash
ssh tony@100.121.67.110
```

### 1.2 Clone project on remote
```bash
git clone https://github.com/Naperotti/Exjobb_protointerpretation.git
cd ~/Exjobb_protointerpretation
```

### 1.3 Create Python environment
```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.4 Verify GPU is available
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Should print `True`.

### 1.5 Add Hugging Face token (optional)
Create token at https://huggingface.co/settings/tokens, then:
```bash
source ~/venv/bin/activate
huggingface-cli login
```

---

## 2) Quick reference: local vs remote

- **Edit code**: Local machine (VS Code)
- **Git operations** (`git add`, `git commit`, `git push`): Local machine
- **Run Python scripts**: Remote machine (via SSH)
- **Commands with `C:\...` paths**: Local machine only

---

## 3) Update dependencies

If `requirements.txt` changed locally, update remote:

```bash
ssh tony@100.121.67.110 "source ~/venv/bin/activate && cd ~/Exjobb_protointerpretation && pip install -r requirements.txt"
```

---

## 4) Run generation + embeddings (OOM-safe chunked pipeline)

**This is the recommended way to run large jobs.** It breaks the work into smaller chunks to avoid GPU out-of-memory.

### 4.1 Configure locally

Edit `run_chunked_pipeline.py`:

```python
TOTAL_RETURNS = 3000        # Total sequences per prompt
CHUNK_RETURNS = 500         # Per-chunk size (decrease if OOM)
BASE_SETTINGS_NAME = "Bank_prompts_40tokens_3000returns"
```

### 4.2 Push to remote

```bash
git add run_chunked_pipeline.py settings.py
git commit -m "update chunk config"
git push
```

### 4.3 Run on remote (background)

### Pull first
ssh tony@100.121.67.110 "cd ~/Exjobb_protointerpretation && git fetch origin && git switch iterative-embeddings && git reset --hard origin/iterative-embeddings"
```bash
ssh tony@100.121.67.110 "cd ~/Exjobb_protointerpretation && source ~/venv/bin/activate && nohup python -u run_chunked_pipeline.py > chunked_run.log 2>&1 &"
```

### 4.4 Monitor progress

```bash
ssh tony@100.121.67.110 "cd ~/Exjobb_protointerpretation && tail -f chunked_run.log"
```

Exit with `Ctrl+C` (log continues remotely).

### 4.5 Outputs

When complete, final merged files appear in:
- `data/BASE_SETTINGS_NAME.npz` — merged sequences + entropies
- `embeddings/aligned_va_embeddings_BASE_SETTINGS_NAME.npy` — merged embeddings
- `embeddings/aligned_va_metadata_BASE_SETTINGS_NAME.json` — merged metadata

Chunk files (`part1`, `part2`, etc.) are kept for debugging.

### 4.6 Run precompute_umap.py (on remote, after embeddings are done)

Once embeddings merge completes, precompute UMAP + OPTICS on the remote GPU:

```bash
ssh tony@100.121.67.110 "cd ~/Exjobb_protointerpretation && source ~/venv/bin/activate && nohup python -u precompute_umap.py > precompute_run.log 2>&1 &"
```

Watch progress:

```bash
ssh tony@100.121.67.110 "cd ~/Exjobb_protointerpretation && tail -f precompute_run.log"
```

This generates 5 output files in `embeddings/` directory:
- `umap_projections.npy` `[num_tokens, n_prompts, num_return, 2]`
- `optics_labels.npy`
- `optics_reachability.npy`
- `optics_orderings.npy`
- `optics_metrics.npz`

These are loaded by `visualize_umap_optics_local.py` for instant slider rendering.

---

## 5) Check job status

### Check if jobs are running

```bash
ssh tony@100.121.67.110 "ps aux | grep python"
```

Look for `generate.py`, `aligned_VA_embeddings.py`, or `run_chunked_pipeline.py`.

### View GPU usage

```bash
ssh tony@100.121.67.110 "nvidia-smi"
```

Or memory summary only:

```bash
ssh tony@100.121.67.110 "nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv"
```

---

## 6) Stop a running job

### By process name

Stop chunked pipeline:
```bash
ssh tony@100.121.67.110 "pkill -f 'run_chunked_pipeline.py'"
```

Stop generate:
```bash
ssh tony@100.121.67.110 "pkill -f 'python.*generate.py'"
```

Stop embeddings:
```bash
ssh tony@100.121.67.110 "pkill -f 'python.*aligned_VA_embeddings.py'"
```

Kill all Python:
```bash
ssh tony@100.121.67.110 "pkill python"
```

### By process ID

If PID is `12345`:

```bash
ssh tony@100.121.67.110 "kill 12345"
```

Force kill:
```bash
ssh tony@100.121.67.110 "kill -9 12345"
```

---

## 7) View files on remote

### List generated data

```bash
ssh tony@100.121.67.110 "ls -lh ~/Exjobb_protointerpretation/data/"
```

### List embeddings

```bash
ssh tony@100.121.67.110 "ls -lh ~/Exjobb_protointerpretation/embeddings/"
```

### Find recently modified files (last 60 min)

```bash
ssh tony@100.121.67.110 "cd ~/Exjobb_protointerpretation && find . -type f -mmin -60 -ls"
```

### Check disk usage

```bash
ssh tony@100.121.67.110 "df -h ~"
```

---

## 8) Download results to local machine (from git bash or terminal)

From your **local machine** (git bash or terminal), copy merged outputs:

### Download merged sequences (git bash)

```bash
scp tony@100.121.67.110:~/Exjobb_protointerpretation/data/Bank_prompts_40tokens_3000returns.npz /c/Users/naper/OneDrive/Dokument/GitHub/Exjobb_protointerpretation/data/
```

(Replace filename with your `BASE_SETTINGS_NAME`)

### Download merged embeddings (git bash)

```bash
scp tony@100.121.67.110:~/Exjobb_protointerpretation/embeddings/aligned_va_embeddings_Bank_prompts_40tokens_3000returns.npy /c/Users/naper/OneDrive/Dokument/GitHub/Exjobb_protointerpretation/embeddings/
scp tony@100.121.67.110:~/Exjobb_protointerpretation/embeddings/aligned_va_metadata_Bank_prompts_40tokens_3000returns.json /c/Users/naper/OneDrive/Dokument/GitHub/Exjobb_protointerpretation/embeddings/
```

### Visualize locally (after files are downloaded)

```bash
python visualize_umap_optics_local.py
python visualize_entropy.py
```

---

## 9) Sync code reliably (avoid merge conflicts)

When pulling on remote (especially if tracked artifacts exist):

**From local:**
```bash
git add -A
git commit -m "your message"
git push
```

**From local (pull remote to match):**
```bash
ssh tony@100.121.67.110 "cd ~/Exjobb_protointerpretation && git fetch origin && git switch iterative-embeddings && git reset --hard origin/iterative-embeddings"
```

This ensures remote always matches remote branch exactly, avoiding conflicts.

---

## Quick command cheatsheet

| Task | Command |
|------|---------|
| Start chunked run | `ssh tony@100.121.67.110 "cd ~/Exjobb_protointerpretation && source ~/venv/bin/activate && nohup python -u run_chunked_pipeline.py > chunked_run.log 2>&1 &"` |
| Watch chunked log | `ssh tony@100.121.67.110 "tail -f ~/Exjobb_protointerpretation/chunked_run.log"` |
| Run precompute | `ssh tony@100.121.67.110 "cd ~/Exjobb_protointerpretation && source ~/venv/bin/activate && nohup python -u precompute_umap.py > precompute_run.log 2>&1 &"` |
| Watch precompute log | `ssh tony@100.121.67.110 "tail -f ~/Exjobb_protointerpretation/precompute_run.log"` |
| Check GPU | `ssh tony@100.121.67.110 "nvidia-smi"` |
| List running jobs | `ssh tony@100.121.67.110 "ps aux \| grep python"` |
| Stop job | `ssh tony@100.121.67.110 "pkill -f run_chunked_pipeline.py"` |
| List data | `ssh tony@100.121.67.110 "ls -lh ~/Exjobb_protointerpretation/data/"` |
| Sync remote | `ssh tony@100.121.67.110 "cd ~/Exjobb_protointerpretation && git fetch origin && git switch iterative-embeddings && git reset --hard origin/iterative-embeddings"` |




### download

RUN=Bank_prompts_40tokens_3000returns
LOCAL=/c/Users/naper/OneDrive/Dokument/GitHub/Exjobb_protointerpretation


scp tony@100.121.67.110:~/Exjobb_protointerpretation/embeddings/aligned_va_metadata_${RUN}.json ${LOCAL}/embeddings/
scp tony@100.121.67.110:~/Exjobb_protointerpretation/embeddings/umap_projections_${RUN}*.npy ${LOCAL}/embeddings/
scp tony@100.121.67.110:~/Exjobb_protointerpretation/embeddings/optics_labels_${RUN}*.npy ${LOCAL}/embeddings/
scp tony@100.121.67.110:~/Exjobb_protointerpretation/embeddings/optics_reachability_${RUN}*.npy ${LOCAL}/embeddings/
scp tony@100.121.67.110:~/Exjobb_protointerpretation/embeddings/optics_orderings_${RUN}*.npy ${LOCAL}/embeddings/
scp tony@100.121.67.110:~/Exjobb_protointerpretation/embeddings/optics_metrics_${RUN}*.npz ${LOCAL}/embeddings/

scp tony@100.121.67.110:~/Exjobb_protointerpretation/data/${RUN}.npz ${LOCAL}/data/
scp tony@100.121.67.110:~/Exjobb_protointerpretation/embeddings/aligned_va_embeddings_${RUN}.npy ${LOCAL}/embeddings/