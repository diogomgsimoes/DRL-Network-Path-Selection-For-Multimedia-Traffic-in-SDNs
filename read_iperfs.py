import os

log_files = [f for f in os.listdir('.') if f.endswith('.log')]

for file in log_files:
    iperf_successful = False
    if "client" in file:
        with open(file, "r") as iperf_report:
            for line in iperf_report.readlines():
                if "iperf Done." in line:
                    iperf_successful = True
            if iperf_successful:
                iperf_report.seek(0)
                for line in iperf_report.readlines():
                    if "receiver" in line:
                        data = line.split()
                        print(file, data[6])
