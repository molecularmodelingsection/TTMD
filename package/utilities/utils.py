import os
import sys
import MDAnalysis as mda
import multiprocessing
import tqdm




class resume:
    def __init__(self, device):
        self.device = device
        self.resume = self.gpu_info()



    def gpu_check(self):
        try:
            os.system(f'nvidia-smi -i {self.device} --query-gpu=name --format=csv,noheader -f gpu_check.info')
        except Exception:
            print('Check your device or your nvidia drivers!')
            sys.exit(0)

        with open('gpu_check.info', 'r') as f:
            lines = f.readlines()
            try:
                current_gpu = lines[0].rstrip('\n')
                print('\ncurrent gpu =', current_gpu)

            except Exception:
                current_gpu = ''
                print('\ncurrent gpu =', current_gpu)

        os.system('rm gpu_check.info')

        return current_gpu



    def gpu_info(self):
        if os.path.exists('gpu.info'):

            with open(f'gpu.info', 'r') as i:
                line = i.readlines()
            
            try:
                gpu = line[0]

            except Exception:
                gpu = ''

            current_gpu = self.gpu_check()

            if current_gpu == gpu:
                resume = True
                print(f'\n——————————\nResume ON\n——————————\n')

            else:
                resume = False
                with open(f'gpu.info', 'w') as i:
                    i.write(current_gpu)
                print(f'\n——————————\nResume OFF\n——————————\n')

        else:
            resume = False
            with open(f'gpu.info', 'w') as i:
                current_gpu = self.gpu_check()
                i.write(current_gpu)
                
        if resume == True:
            r = 'on'
        elif resume == False:
            r = 'off'

        return r



class pid:
    def __init__(self):
        self.pid()
        pass

    def pid(self):
        main_pid = os.getpid()
        with open('main_pid.log', 'w') as p:
            p.write(str(main_pid))

        print(f'parent_pid = {main_pid}\n')

        with open('kill.py', 'w') as kill:
            kill.write(f'''import os

with open('main_pid.log', 'r') as main:
    main_pid = int(main.readlines()[0])

os.system(f'pgrep -g {{main_pid}}> pid.log')

with open('pid.log', 'r') as log:
    lines = log.readlines()
    for line in lines:
        pid = int(line.rstrip('\\n'))
        os.system(f'kill {{pid}}')
''')
        return main_pid



class check_trj_len:
    def __init__(self, vars):
        self.__dict__ = vars
        self.cfactor = self.timestep * self.dcdfreq / 1000000



    def frame_to_ns(self, frame_or_ns):
        frame = frame_or_ns * self.cfactor
        return frame



    def ns_to_frame(self, frame_or_ns):
        ns = frame_or_ns / self.cfactor
        return ns



    def check(self, top, trj, len_trj):
        print(f'\n    Checking {trj} lenght')

        check_u = mda.Universe(top, trj)
        check_frames = int(len(check_u.trajectory))

        ref_frames = int(self.ns_to_frame(len_trj))

        print(f'      Trajectory has {check_frames} frames (expected = {ref_frames})')

        if check_frames == ref_frames:
            print('      Trajectory integrity checked\n')
            checked = True

        elif check_frames != ref_frames:
            print('      Error in trajectory lenght')
            checked = False

        return checked
