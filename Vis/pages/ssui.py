import streamlit as st
import os
import subprocess

st.set_page_config(layout="wide")

"# Available resources"

"## CPU"
"### `os.cpu_count()`"
f"This machine has {os.cpu_count()} CPU available"

"### `lscpu`"
lscpu = subprocess.run(args=["lscpu"], capture_output=True).stdout.decode("utf-8")
st.code(f"{lscpu}")

"## Disk"
"### `df -H`"
dfH = subprocess.run(args=["df", "-H"], capture_output=True).stdout.decode("utf-8")
st.code(dfH)

"## Memory"
"### `free -h`"
freeg = subprocess.run(args=["free", "-h"], capture_output=True).stdout.decode("utf-8")
st.code(f"{freeg}")

"## Processes"
"### `top -b -n 1 | head -n 30`"
sortBy = st.radio("Sort by", ["%MEM", "%CPU"], horizontal=True)
topn = (
    subprocess.run(
        args=f"top -bc -w 300 -n 1 -o +{sortBy}".split(), capture_output=True
    )
    .stdout.decode("utf-8")
    .splitlines()[4:50]
)
st.code("\n".join(topn))

"## IP addresses"

"From [this discussion](https://discuss.streamlit.io/t/ip-addresses-for-streamlit-community-cloud/75304/)"

st.code("""
    35.230.127.150
    35.203.151.101
    34.19.100.134
    34.83.176.217
    35.230.58.211
    35.203.187.165
    35.185.209.55
    34.127.88.74
    34.127.0.121
    35.230.78.192
    35.247.110.67
    35.197.92.111
    34.168.247.159
    35.230.56.30
    34.127.33.101
    35.227.190.87
    35.199.156.97
    34.82.135.155
    """,
    language=None,
    line_numbers=True,
)
# "### `ifconfig`"
# "*requires `net-tools`*"

# ipa = subprocess.run(args=["ifconfig"], capture_output=True).stdout.decode('utf-8').splitlines()
# st.code("\n".join(ipa))

"### Python `socket` (?)"
with st.echo():
    import socket

    s = socket.socket(
        family=socket.AF_INET,  # (host, port)
        type=socket.SOCK_DGRAM,
    )

    s.connect(("8.8.8.8", 1))
    ip, port = s.getsockname()

    st.metric("**IP**", ip)
    st.metric("**Port**", port)
    s.close()