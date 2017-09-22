from jellyfish_remove_x import run
import sys

k_vals = [x for x in range(13, 22)]
lower_limits = [1, 5, 10, 15, 20, 25]


f1 = open('run_times.txt', 'a')
f2 = open('percentages.txt', 'a')

for k in k_vals:
    for l in lower_limits:
        f1.write("\nk:\t%d\nl:\t%d\n" % (k, l))
        f2.write("\nk:\t%d\nl:\t%d\n" % (k, l))
        run_times = []
        percentages = []
        for x in range(5):
            percent, time = run(k, l)

            print "K:\t%d\nL:\t%d\nTime:\t%d\n%%:\t%d" % (k, l, time, percent)

            run_times.append(time)
            percentages.append(percent)

        f1.write("%s\n" % ' '.join([str(x) for x in run_times]))
        f2.write("%s\n" % ' '.join([str(x) for x in percentages]))

f1.close()
f2.close()
