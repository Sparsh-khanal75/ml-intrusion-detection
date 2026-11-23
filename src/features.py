"""NSL-KDD schema, attack taxonomy and preprocessing."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
DROP = ["num_outbound_cmds"]  # constant zero in NSL-KDD
NUMERIC = [c for c in COLUMNS[:-2] if c not in CATEGORICAL + DROP]

# Standard 4-category taxonomy (Tavallaee et al., 2009). Includes attacks that
# appear ONLY in the test set, which is what makes KDDTest+ a generalisation test.
ATTACK_CATEGORY = {
    # Denial of Service
    **dict.fromkeys(["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
                     "mailbomb", "processtable", "udpstorm"], "DoS"),
    # Probing / reconnaissance
    **dict.fromkeys(["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"], "Probe"),
    # Remote-to-Local
    **dict.fromkeys(["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
                     "warezclient", "warezmaster", "named", "sendmail", "snmpgetattack",
                     "snmpguess", "worm", "xlock", "xsnoop", "httptunnel"], "R2L"),
    # User-to-Root (privilege escalation)
    **dict.fromkeys(["buffer_overflow", "loadmodule", "perl", "rootkit", "ps",
                     "sqlattack", "xterm"], "U2R"),
    "normal": "Normal",
}
CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# MITRE ATT&CK context for each category, used in reports
MITRE = {
    "DoS": "T1498/T1499 Network & Endpoint Denial of Service",
    "Probe": "T1046 Network Service Discovery",
    "R2L": "T1110 Brute Force / T1190 Exploit Public-Facing Application",
    "U2R": "T1068 Exploitation for Privilege Escalation",
}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, names=COLUMNS)
    df["category"] = df["label"].map(ATTACK_CATEGORY)
    unknown = df["category"].isna()
    if unknown.any():
        raise ValueError(f"Unmapped labels: {sorted(df.loc[unknown, 'label'].unique())}")
    df["is_attack"] = (df["category"] != "Normal").astype(int)
    return df


def make_preprocessor() -> ColumnTransformer:
    # handle_unknown='ignore' so services never seen in training don't crash inference
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])


def features(df: pd.DataFrame) -> pd.DataFrame:
    return df[CATEGORICAL + NUMERIC]
