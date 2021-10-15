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

    def run(self, script: str, arguments: list(), p=False):
        self.script = script

        cnt = 0
        while cnt < len(arguments):
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

                    process = subprocess.Popen(
                        " ".join(cmd), shell=True, text=True, start_new_session=True
                    )
                    self.lt_process.append(process)

                    cnt += 1

                    if cnt >= len(arguments):
                        break

            else:
                pass

            # result = process.communicate()
            time.sleep(self.sec)


    def __del__(self):
        pynvml.nvmlShutdown()
        for p in self.lt_process:
            p.kill()
            