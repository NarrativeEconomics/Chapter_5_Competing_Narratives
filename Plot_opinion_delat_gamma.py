import csv
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ---- 1) Parse your "log-style CSV" into tidy rows: time | agent | opinion | gamma | input | attention ----
path = "bse_H2_Case1_delta_gamma_hours005_i05_0001_opinion.csv"

records_by_time = defaultdict(list)

def to_float(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")

with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        tok = [x.strip() for x in row if x is not None and x.strip() != ""]

        # row starts: t=,0,...
        if len(tok) < 2 or tok[0] != "t=":
            continue

        try:
            t = int(float(tok[1]))
        except ValueError:
            continue

        i = 2
        # each agent chunk is 10 tokens:
        # id=, AGENT, opinion=, OP, gamma=, GAM, input=, IN, attention=, ATT
        while i + 9 < len(tok):
            if tok[i] != "id=":
                i += 1
                continue

            # strict label check for your exact format
            if (tok[i+2] != "opinion=" or
                tok[i+4] != "gamma=" or
                tok[i+6] != "delta=" or
                tok[i+8] != "attention="):
                i += 1
                continue

            agent = tok[i+1]
            opinion = to_float(tok[i+3])
            gamma = to_float(tok[i+5])
            delta = to_float(tok[i+7])
            attention = to_float(tok[i+9])

            records_by_time[t].append({
                "agent": agent,
                "opinion": opinion,
                "gamma": round(gamma, 2),
                "delta": round(delta, 2),
                "attention": attention
            })

            i += 10  # move to next agent block

# flatten
records = []
for t, agents in records_by_time.items():
    for agent_record in agents:
        records.append({"time": t, **agent_record})

# guarantee columns exist even if empty
df = pd.DataFrame(records, columns=["time","agent","opinion","gamma","delta","attention"])

# convert seconds->minutes (your original)
df["time"] = (df["time"] / 60).astype(int)

print(df.head())
print("rows:", len(df), "unique agents:", df["agent"].nunique(), "unique times:", df["time"].nunique())

# ---------------- plots ----------------

# Plot opinion over time
plt.figure()
for agent, group in df.groupby("agent"):
    plt.plot(group["time"], group["opinion"])
plt.ylim(-3, 3)
plt.xlabel("Time (minutes)")
plt.ylabel("Opinion")
plt.title("Agent Opinions Over Time")
plt.grid(True)
plt.show()

# Plot gamma and input over time (two lines per agent will overwrite color if you force it)
plt.figure()
for agent, group in df.groupby("agent"):
    plt.plot(group["time"], group["gamma"], color="orange")
for agent, group in df.groupby("agent"):
    plt.plot(group["time"], group["delta"], color="green")
plt.xlabel("Time (minutes)")
plt.ylabel("gamma and delta")
plt.title("Agent gamma and delta Over Time")
plt.grid(True)
plt.show()

# Plot attention over time
plt.figure()
for agent, group in df.groupby("agent"):
    plt.plot(group["time"], group["attention"])
plt.xlabel("Time (minutes)")
plt.ylabel("Attention")
plt.title("Agent Attention Over Time")
plt.grid(True)
plt.show()
