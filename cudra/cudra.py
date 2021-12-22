import os
import subprocess
import time

import datetime
import pynvml


class cudra(object):
    def __init__(self, gpu_ids, sec):

        self.gpu_ids = gpu_ids
        self.sec = sec

        pynvml.nvmlInit()
        handler = list()
        for i in gpu_ids:
            handler.append((i, pynvml.nvmlDeviceGetHandleByIndex(i)))
        self.handler = handler

        self.lt_process = list()

    def run(self, script: str, arguments: list(), p=False, log_prefix="./"):
        self.script = script
        root = os.path.join(log_prefix, ".cudra/log")
        assert os.access(root, os.W_OK), f"CUDRA cannot write {root}."

        os.makedirs(root, exist_ok=True)
        log_file = f'{datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")}_exec.txt'
        logPath = os.path.join(root, log_file)
        pipe = open(logPath, "a")
        self.logPath = logPath
        self.arguments = arguments

        cnt = 0
        while True:
            empty_gpus = list()
            for idx, handle in self.handler:
                process = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                if len(process) == 0:
                    empty_gpus.append(idx)
            if len(empty_gpus) > 0:
                for gpu in empty_gpus:
                    argument = arguments[cnt]
                    cmd = (
                        f"CUDA_VISIBLE_DEVICES={gpu}",
                        f"exec python",
                        script,
                    )
                    for key, value in argument.items():
                        cmd += (f"--{key}", str(value))

                    if p:
                        cmd = ("echo",) + cmd
                    else:
                        print(" ".join(cmd))

                    process = subprocess.Popen(
                        " ".join(cmd),
                        shell=True,
                        text=True,
                        start_new_session=True,
                        stdout=open(os.devnull, "w"),
                        stderr=pipe,
                        stdin=open(os.devnull, "w"),
                    )
                    self.lt_process.append(process)

                    cnt += 1

                    if cnt >= len(arguments):
                        break

            else:
                pass

            if cnt >= len(arguments):
                break

            time.sleep(self.sec)

        for p in self.lt_process:
            p.communicate()

    def __del__(self):
        pynvml.nvmlShutdown()
        for p in self.lt_process:
            p.kill()

        error_execute = 0
        f = open(self.logPath, mode="r")
        for line in f.readlines():
            if "Error" in line:
                error_execute += 1

        msg = (
            f"[CUDRA]: ERROR {error_execute}/{len(self.arguments)}",
            f"Log File Path: {self.logPath}",
        )
        print("\n".join(msg))
