import streamlit as st
import os
import subprocess
import socket
import psutil  # Import psutil for memory info

st.set_page_config(layout="wide")

"# Available resources"

## CPU Information
"## CPU"
"### `os.cpu_count()`"
st.write(f"This machine has {os.cpu_count()} CPU cores available.")

"### `lscpu`"
lscpu = subprocess.run(args=["lscpu"], capture_output=True, text=True)
st.code(lscpu.stdout)

## Disk Information
"## Disk"
"### `df -H`"
dfH = subprocess.run(args=["df", "-H"], capture_output=True, text=True)
st.code(dfH.stdout)

## Memory Information
"## Memory"
"### Memory Usage via psutil"

# Get memory details using psutil
memory = psutil.virtual_memory()

# Display memory stats
st.write(f"Total Memory: {memory.total / (1024 ** 3):.2f} GB")
st.write(f"Available Memory: {memory.available / (1024 ** 3):.2f} GB")
st.write(f"Used Memory: {memory.used / (1024 ** 3):.2f} GB")
st.write(f"Memory Percent: {memory.percent}%")

## Processes
"## Processes"
"### `top -b -n 1 | head -n 30`"

sortBy = st.radio("Sort by", ["%MEM", "%CPU"], horizontal=True)

topn = subprocess.run(
    args=f"top -b -n 1 -o +{sortBy}".split(),
    capture_output=True,
    text=True
).stdout.splitlines()[4:50]

st.code("\n".join(topn))

## IP Addresses
"## IP addresses"

"From [this discussion](https://discuss.streamlit.io/t/ip-addresses-for-streamlit-community-cloud/75304/)"

st.code("""35.230.127.150
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
34.82.135.155""", language=None, line_numbers=True)

## Python `socket` method to get IP
"### Python `socket` method"

with st.echo():
    s = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 1))  # Connect to a remote server
    ip, port = s.getsockname()  # Get local IP address and port
    s.close()

    st.metric("**IP Address**", ip)
    st.metric("**Port**", port)
