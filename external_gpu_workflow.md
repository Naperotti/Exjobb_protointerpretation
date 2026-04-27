# External GPU Workflow (Tailscale + SSH)

Use this guide to run your project on the remote GPU machine.

## 1) One-time setup

### 1.1 Connect to remote machine
Run from your local computer:

```bash
ssh tony@100.121.67.110
```

### 1.2 Clone project on remote
Run on the remote machine:

```bash
git clone https://github.com/Naperotti/Exjobb_protointerpretation.git
cd ~/Exjobb_protointerpretation
```

### 1.3 Create Python environment on remote
Run on the remote machine:

```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.4 Verify GPU is visible to PyTorch
Run on the remote machine:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `True`, GPU is available.

## 2) Daily routine while developing

### 2.1 Edit code locally
Use VS Code on your local machine.

### 2.2 Push local changes
Run from local project folder:

```bash
git add -A
git commit -m "update"
git push
```

### 2.3 Pull and run on remote GPU
Run from local machine (single command):

```bash
ssh tony@100.121.67.110 "cd ~/Exjobb_protointerpretation && source ~/venv/bin/activate && git pull && python generate.py"
```

For embeddings script instead:

```bash
ssh tony@100.121.67.110 "cd ~/Exjobb_protointerpretation && source ~/venv/bin/activate && git pull && python aligned_VA_embeddings.py"
```

## 3) When to run commands locally vs remotely

- Local machine: `git add`, `git commit`, `git push`.
- Remote machine (SSH): `python`, `pip install`, `git pull`, and running scripts.

Quick rule:
- If command contains local Windows paths like `C:\...`, run locally.
- If command should use remote GPU, run in SSH.

## 4) Long job mode (keep running after disconnect)

Start background run from local machine:

```bash
ssh tony@100.121.67.110 "cd ~/Exjobb_protointerpretation && source ~/venv/bin/activate && nohup python aligned_VA_embeddings.py > run.log 2>&1 &"
```

Check logs:

```bash
ssh tony@100.121.67.110 "cd ~/Exjobb_protointerpretation && tail -f run.log"
```

## 5) If dependencies changed

After changing `requirements.txt`, run once on remote:

```bash
ssh tony@100.121.67.110 "source ~/venv/bin/activate && cd ~/Exjobb_protointerpretation && pip install -r requirements.txt"
```

## 6) Download generated data from remote to local

If you are currently inside SSH (`tony@spark-228b...`), that shell is running on remote Linux.

- You only need to exit SSH if you want to run a command that writes to a local Windows path like `C:\Users\...`.
- You can also stay connected and open a second local terminal, then run the copy command there.

Why:

- `C:\Users\...` exists on your local machine, not on the remote machine.
- So a copy command targeting your local Windows folder must be started from a local terminal.

On remote, leave SSH:

```bash
exit
```

On your local machine, run one of these:

PowerShell:

```powershell
scp tony@100.121.67.110:~/Exjobb_protointerpretation/data/all_prompts_test.json "C:\Users\naper\OneDrive\Dokument\GitHub\Exjobb_protointerpretation\data\"
```

Git Bash:

```bash
scp tony@100.121.67.110:~/Exjobb_protointerpretation/data/all_prompts_test.json /c/Users/naper/OneDrive/Dokument/GitHub/Exjobb_protointerpretation/data/
```

That will copy the remote file down to your local `data` folder and overwrite the old one.
