# import pyplot as plt
import matplotlib.pyplot as plt
import statistics
import numpy
import itertools
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

d = '{}/results'.format(BASE_DIR)
prefixes = ['LSTM', 'peepLSTM']
names = ['loss', 'accuracy']
for prefix in prefixes:
    for name in names: 
        xs, ys = [], []
        
        for num in (5, 10, 20):
            filename = '{}/{}_{}_{}.txt'.format(d, prefix, name, num)
            ts = []
            steps = []
            values = []
            ## you

            with open(filename, 'r') as file:
                ts = file.read().split('\n')
                print(ts[0].split(','))
                for t in ts:
                    ts = [list(map(str, t.split(','))) for t in ts]
            # my
            ts = ['0,0.69921875', '60,0.671875', '120,0.67578125']
            ts = [list(map(float, t.split(','))) for t in ts]
            
            for i, t in enumerate(ts):
                x, y = t
                if num == 5:
                    xs.append(x)
                    ys.append(y)
                else:
                    ys[i] += y
                        
        ys = [round(y/3, 2) for y in ys]
            
        fig, axes = plt.subplots()
        axes.set_xlabel('step')  # 横轴名称
        axes.set_ylabel('accuracy')
        
        std = round(statistics.stdev(ys), 2)
        axes.set_title('{}_{}, std:{}'.format(prefix, name, ))  # 图形名称

        axes.plot(xs, ys)
        axes.legend(name, loc=0)  # 图例
        
## you
plt.show()