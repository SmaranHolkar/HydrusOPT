import subprocess, sys, os

prompts = [
    "If all cats are dogs and all dogs are fish, what is a cat?",
    "What is the color of silence?",
    "Can God create a rock so heavy that even God cannot lift it?",
    "What happens after you die before you are born?",
    "Is this statement false: this statement is false?",
]

for p in prompts:
    print(f"\n{'='*60}")
    print(f"PROMPT: {p}")
    print('='*60)
    try:
        result = subprocess.run(
            [sys.executable, "Hydrusopt_test.py",
             "--model", "Qwen/Qwen2.5-3B-Instruct",
             "--profile", "fast",
             "--skip-linearise",
             "--no-bench",
             "--no-retrieval",
             "--check-consistency", p,
             "--guard-mode", "post"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        out = result.stdout + result.stderr
        printed = False
        for line in out.splitlines():
            l = line.strip()
            if any(k in l for k in ["Clean  :", "Match  :", "Run 1  :", "Run 2  :", "[POST]", "[SAFE]", "CONSISTENT", "INCONSISTENT"]):
                print(line)
                printed = True
        if not printed:
            print(f"[no matching output — exit code {result.returncode}]")
            # print last 20 lines for debugging
            for line in out.splitlines()[-20:]:
                print("  >> " + line)
    except Exception as e:
        print(f"[EXCEPTION] {e}")
