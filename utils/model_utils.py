import multiprocessing as mp
import time
import random
import traceback
from typing import Any, Dict, List, Optional

# Global Cohere client (initialized in main)
import cohere
co = cohere.ClientV2()  # initialize once, shared with workers


def _init_worker(global_co):
    """
    Called once per worker process. 
    Sets the global 'co' inside that process.
    """
    global co
    co = global_co


def _chat_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    One task = one co.chat call with retry logic.
    """
    i = task["index"]
    model = task["model"]
    messages = task["messages"]
    kwargs = task["kwargs"]
    max_retries = task["max_retries"]
    backoff_base = task["backoff_base"]
    jitter = task["jitter"]

    if co is None:
        return {
            "index": i,
            "ok": False,
            "response": None,
            "error": "Cohere client not initialized in worker",
            "attempts": 0,
            "elapsed_s": 0.0,
        }

    start = time.time()
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            resp = co.chat(model=model, messages=messages, **kwargs)
            return {
                "index": i,
                "ok": True,
                "response": resp,  # safe serialization
                "error": None,
                "attempts": attempt,
                "elapsed_s": time.time() - start,
            }
        except Exception:
            last_err = traceback.format_exc()
            if attempt >= max_retries:
                return {
                    "index": i,
                    "ok": False,
                    "response": None,
                    "error": last_err,
                    "attempts": attempt,
                    "elapsed_s": time.time() - start,
                }
            sleep_s = (backoff_base * (2 ** (attempt - 1))) + random.uniform(0, jitter)
            time.sleep(sleep_s)

    return {
        "index": i,
        "ok": False,
        "response": None,
        "error": "Unexpected retry loop exit.",
        "attempts": attempt,
        "elapsed_s": time.time() - start,
    }


def chat_n_times(
    n: int,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_workers: Optional[int] = None,
    max_retries: int = 3,
    backoff_base: float = 0.5,
    jitter: float = 0.25,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Run `co.chat` n times in parallel using multiprocessing.
    Uses a global Cohere client shared to workers via initializer.
    """
    if n <= 0:
        return []

    tasks = [
        {
            "index": i,
            "model": model,
            "messages": messages,
            "kwargs": kwargs,
            "max_retries": max_retries,
            "backoff_base": backoff_base,
            "jitter": jitter,
        }
        for i in range(n)
    ]

    if max_workers is None or max_workers < 1:
        max_workers = mp.cpu_count()

    with mp.Pool(
        processes=max_workers, 
        initializer=_init_worker, 
        initargs=(co,), 
        maxtasksperchild=100
    ) as pool:
        results = pool.map(_chat_worker, tasks)

    results.sort(key=lambda r: r["index"])
    return results


# ---------- Example ----------
if __name__ == "__main__":

    # Parallel calls
    outs = chat_n_times(
        n=5,
        model="command-a-03-2025",
        messages=[{"role": "user", "content": "what is 987987 * 123123. just give the answer and nothing else"}],
        temperature=1.0,
    )
    for o in outs:
        if o["ok"]:
            print(f"[{o['index']}] OK: {o['response'].message.content[-1].text}")
        else:
            print(f"[{o['index']}] FAIL: {o['error']}")
