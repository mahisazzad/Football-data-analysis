import json, time, urllib.request, urllib.error, os, sys

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

def fetch(url, path, max_retries=8):
    if os.path.exists(path) and os.path.getsize(path) > 100:
        return True
    delay = 3
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-script/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(path, "wb") as f:
                f.write(data)
            json.loads(data)
            return True
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"  retry {attempt+1} for {path}: {e}")
            time.sleep(delay); delay = min(delay*1.7, 30)
    return False

def download_batch(matches_file, events_dir, threesixty_dir, start, limit):
    matches = json.load(open(matches_file))
    match_ids = [m["match_id"] for m in matches]
    batch = match_ids[start:start+limit]
    print(f"{len(batch)} matches to fetch this run")
    for i, mid in enumerate(batch):
        fetch(f"{BASE}/events/{mid}.json", f"{events_dir}/{mid}.json"); time.sleep(0.35)
        fetch(f"{BASE}/three-sixty/{mid}.json", f"{threesixty_dir}/{mid}.json"); time.sleep(0.35)
        print(f"[{i+1}/{len(batch)}] match {mid} done", flush=True)

if __name__ == "__main__":
    matches_file, events_dir, threesixty_dir, start, limit = sys.argv[1:6]
    download_batch(matches_file, events_dir, threesixty_dir, int(start), int(limit))
