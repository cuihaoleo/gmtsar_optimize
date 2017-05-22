#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor
import tempfile
import logging
import math
import itertools
import re
import os
import shutil
import argparse
import subprocess

shutil._use_fd_functions = False
logging.basicConfig(level=logging.DEBUG)

IMG_PREFIX = "IMG-HH-"
LED_PREFIX = "LED-"
DATA_SUFFIX = "-H1.0__A"


class PathHelper:
    def __init__(self, *compo):
        self.name = compo[-1]
        if len(compo) > 1:
            self.directory = os.path.join(*compo[:-1])
        else:
            self.directory = ""

    @property
    def IMG(self):
        return os.path.join(self.directory, IMG_PREFIX + self.name + DATA_SUFFIX)

    @property
    def PRM(self):
        return self.IMG + ".PRM"

    @property
    def RAW(self):
        return self.IMG + ".raw"

    @property
    def LED(self):
        return os.path.join(self.directory, LED_PREFIX + self.name + DATA_SUFFIX)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='data directory')
    parser.add_argument('-f','--frame', nargs=3, help='start/end/step of frame number', required=True, type=int)
    parser.add_argument('-o','--orbit', nargs='+', help='orbit numbers of candidate data', required=True, type=int)
    args = parser.parse_args()

    if len(args.orbit) < 2:
        raise Exception("Must specify more than two orbit numbers")

    return args


def symlink_plus(src, dst):
    logging.info("Symlink '%s' to '%s'", src, dst)

    if os.path.exists(dst) and os.path.islink(dst):
        os.remove(dst)

    if os.path.exists(src):
        os.symlink(src, dst)
    else:
        raise Exception("'%s' not found!" % src)



def threadMerging(workdir, name1, name2, outname):
    logging.info("Merging '%s' and '%s' to '%s'", name1, name2, outname)
    p = subprocess.Popen(
            ["ALOS_merge",
            PathHelper(name1).PRM,
            PathHelper(name2).PRM,
            "-output", PathHelper(outname).IMG, "-V"],
            #stdout=subprocess.PIPE,
            #stderr=subprocess.PIPE,
            cwd=workdir)

    if p.wait() != 0:
        raise Exception("ALOS_merge failed!")


def threadMeasurePerpBaseline(dataroot, master_name, slave_name):
    tmpdir = os.path.join(dataroot, "tmp/")

    with tempfile.TemporaryDirectory(dir=tmpdir) as workdir:
        for name in (master_name, slave_name):
            subdir = name + "-L1.0"
            infile = PathHelper(dataroot, subdir, name)
            outfile = PathHelper(workdir, name)

            symlink_plus(infile.IMG, outfile.IMG)
            symlink_plus(infile.LED, outfile.LED)

        try:
            threadPreprocess(dataroot, workdir, master_name, slave_name)
        except Exception:
            return float("inf")

        master = PathHelper(master_name)
        slave = PathHelper(slave_name)
        p = subprocess.Popen(
                ["ALOS_baseline", master.PRM, slave.PRM],
                stdout=subprocess.PIPE,
                #stderr=subprocess.PIPE
                cwd=workdir)

        for line in p.stdout:
            line = line.decode()
            if line.startswith("B_perpendicular"):
                _, latter = line.strip().split("=")
                result = abs(float(latter))
                break
 
        if p.wait() != 0:
            raise Exception("ALOS_baseline failed!")

    logging.info("Perp-baseline between %s and %s is %f",
                   master_name, slave_name, result)
    
    return result


def threadPreprocess(dataroot, workdir, master_name, slave_name):
    logging.info("Pre-processing '%s' and '%s'", master_name, slave_name)
    tmpdir = os.path.join(dataroot, "tmp/")

    for name in (master_name, slave_name):
        subdir = name + "-L1.0"
        infile = PathHelper(dataroot, subdir, name)
        outfile = PathHelper(workdir, name)

        symlink_plus(infile.IMG, outfile.IMG)
        symlink_plus(infile.LED, outfile.LED)

    master = PathHelper(master_name)
    slave = PathHelper(slave_name)
    cmd = ["pre_proc.csh", "ALOS", master.IMG, slave.IMG]
    p = subprocess.Popen(
            cmd,
            #stdout=subprocess.PIPE,
            #stderr=subprocess.PIPE
            cwd=workdir)

    if p.wait() != 0:
        raise Exception(" ".join(cmd) + " failed!")


def main():
    args = parse_arguments()
    dataroot = os.path.abspath(args.directory)
    tmpdir = os.path.join(dataroot, "tmp/")
    rawdir = os.path.join(dataroot, "raw/")

    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)

    shutil.rmtree(rawdir, ignore_errors=True)
    os.mkdir(rawdir)

    frame_list = list(range(args.frame[0], args.frame[1]+1, args.frame[2]))

    with ThreadPoolExecutor(24) as executor:
        frame = frame_list[len(frame_list) // 2]
        tasks = []

        for (slave, master) in itertools.combinations(args.orbit, 2):
            master_name = "ALPSRP%05d%04d" % (master, frame)
            slave_name = "ALPSRP%05d%04d" % (slave, frame)
            future = executor.submit(threadMeasurePerpBaseline,
                            dataroot, master_name, slave_name)
            tasks.append((master, slave, future))

        min_perp_baseline = float("+inf")
        pair = None

        for master, slave, future in tasks:
            result = future.result()
            if result > 10 and result < min_perp_baseline:
                min_perp_baseline = result
                pair = (master, slave)

    logging.info("Orbit number of master and slave: (%d, %d)", *pair)

    with ThreadPoolExecutor(24) as executor, \
         tempfile.TemporaryDirectory(dir=tmpdir) as workdir:
        tasks = []
        data_pairs = []

        for frame in range(args.frame[0], args.frame[1]+1, args.frame[2]):
            master_name = "ALPSRP%05d%04d" % (pair[0], frame)
            slave_name = "ALPSRP%05d%04d" % (pair[1], frame)
            future = executor.submit(threadPreprocess,
                            dataroot, workdir, master_name, slave_name)
            tasks.append(future)
            data_pairs.append((master_name, slave_name))

        for future in tasks:
            future.result()

        offset = 1
        while offset < len(data_pairs):
            mask = (offset << 1) - 1
            tasks = []

            for local_id in range(len(data_pairs)):
                if (local_id & mask) == 0:
                    tmp_pair = ("merge-%d-%d-master" % (local_id, local_id + offset),
                                "merge-%d-%d-slave" % (local_id, local_id + offset))
                    try:
                        other = data_pairs[local_id + offset]
                        this = data_pairs[local_id]
                    except IndexError:
                        continue
                    
                    future = executor.submit(threadMerging,
                                workdir, this[0], other[0], tmp_pair[0])
                    tasks.append(future)

                    future = executor.submit(threadMerging,
                                workdir, this[1], other[1], tmp_pair[1])
                    tasks.append(future)

                    data_pairs[local_id] = tmp_pair

            offset <<= 1

            for future in tasks:
                future.result()

        for src_name, dst_name in zip(tmp_pair, ["master", "slave"]):
            source = PathHelper(workdir, src_name)
            target = PathHelper(rawdir, dst_name)
            os.link(source.RAW, target.RAW)
     
            with open(source.PRM, "r") as fin, \
                 open(target.PRM, "w") as fout:

                for raw_line in fin:
                    line = raw_line.strip()

                    if line.startswith("led_file"):
                        p1, p2 = line.split("=", 1)
                        led_filename = p2.strip()
                        print("led_file = ", os.path.basename(target.LED), file=fout)
                    elif line.startswith("input_file"):
                        print("input_file = ", os.path.basename(target.RAW), file=fout)
                    else:
                        print(line, file=fout)

                os.link(os.path.join(workdir, led_filename),
                        target.LED)
 

if __name__ == "__main__":
    main()
