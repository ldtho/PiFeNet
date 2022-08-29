import sys
import trace
from second.pytorch.traintest import train


def main():
    train(config_path='/home/starlet/kaggle/code/secondpytorch/second/configs/transpillars/car/xyres_16.config',
          model_dir='/home/starlet/kaggle/code/secondpytorch/results')
# create a Trace object, telling it what to ignore, and whether to
# do tracing or line-counting or both.
print(sys.prefix, sys.exec_prefix)
tracer = trace.Trace(
    ignoredirs=[sys.prefix, sys.exec_prefix],
    trace=1,
)
tracer.runfunc(main)
r = tracer.results()
print(tracer)
r.write_results(summary=True,coverdir='coverdir2')

