"""
Evaluates a driving model on a set of CARLA routes wherein each route is evaluated on a separate machine in parallel.
This script generates the necessary shell files to run this on a SLURM cluster.
It also monitors the evaluation and resubmits crashed routes.
At the end all results files are aggregated and parsed.
Best run inside a tmux terminal.
"""

import subprocess
import time
from pathlib import Path
import os
import fnmatch
import ujson
import sys

# Our centOS is missing some c libraries.
# Usually miniconda has them, so we tell the linker to look there as well.
# newlib = '/path/to/miniconda3/lib/'
# if not newlib in os.environ['LD_LIBRARY_PATH']:
#   os.environ['LD_LIBRARY_PATH'] += ':' + newlib


def create_run_eval_bash(bash_save_dir, results_save_dir, route_path, route, checkpoint, logs_save_dir,
                         carla_tm_port_start, benchmark, carla_root):
  Path(f'{results_save_dir}').mkdir(parents=True, exist_ok=True)
  with open(f'{bash_save_dir}/eval_{route}.sh', 'w', encoding='utf-8') as rsh:
    rsh.write(f'''\
export CARLA_ROOT={carla_root}
export CARLA_SERVER=${{CARLA_ROOT}}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=/cluster/work/andrebw/carla_garage/scenario_runner
export LEADERBOARD_ROOT=/cluster/work/andrebw/carla_garage/leaderboard
export PYTHONPATH="${{CARLA_ROOT}}/PythonAPI/carla/":"${{SCENARIO_RUNNER_ROOT}}":"${{LEADERBOARD_ROOT}}":${{PYTHONPATH}}
''')
    rsh.write(f"""
export PORT=$1
echo 'World Port:' $PORT
export TM_PORT=`comm -23 <(seq {carla_tm_port_start} {carla_tm_port_start+400} | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
echo 'TM Port:' $TM_PORT
export ROUTES={route_path}
# export TEAM_AGENT=team_code/sensor_agent.py
export TEAM_AGENT=/cluster/work/andrebw/carla_garage/team_code/data_agent.py
export TEAM_CONFIG=logs/{checkpoint}/
export CHALLENGE_TRACK_CODENAME=MAP
export REPETITIONS=1
export RESUME=1
export CHECKPOINT_ENDPOINT={results_save_dir}/{route}.json
export DEBUG_CHALLENGE=1
export DATAGEN=0
export SAVE_PATH={logs_save_dir}
export DIRECT=0
export UNCERTAINTY_WEIGHT=1
export UNCERTAINTY_THRESHOLD=0.5
export HISTOGRAM=0
export BLOCKED_THRESHOLD=180
export TMP_VISU=0
export VISU_PLANT=0
export SLOWER=1
export STOP_CONTROL=1
export TP_STATS=0
export BENCHMARK={benchmark}

echo "Loading modules..."
module load Anaconda3/2024.02-1
module load libjpeg-turbo/2.1.5.1-GCCcore-12.3.0
              
echo "Activating conda environment..."
conda activate garage
conda env list
""")
    rsh.write('''
python3 -u ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=1 \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--timeout=120 \
--traffic-manager-port=${TM_PORT}
''')


def make_jobsub_file(commands, job_number, exp_name, exp_root_name, partition):
  os.makedirs(f'evaluation/{exp_root_name}/{exp_name}/run_files/logs', exist_ok=True)
  os.makedirs(f'evaluation/{exp_root_name}/{exp_name}/run_files/job_files', exist_ok=True)
  job_file = f'evaluation/{exp_root_name}/{exp_name}/run_files/job_files/{job_number}.sh'
  qsub_template = f"""#!/bin/bash
#SBATCH --job-name={job_number}-{exp_name}
#SBATCH --partition={partition}
#SBATCH --account=share-ie-idi
#SBATCH -o evaluation/{exp_root_name}/{exp_name}/run_files/logs/qsub_out{job_number}.log
#SBATCH -e evaluation/{exp_root_name}/{exp_name}/run_files/logs/qsub_err{job_number}.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=20gb
#SBATCH --time=01-12:00:00
#SBATCH --gres=gpu:1
# #SBATCH --constraint=(a100)
"""
  for cmd in commands:
    qsub_template = qsub_template + f"""
{cmd}

"""

  with open(job_file, 'w', encoding='utf-8') as f:
    f.write(qsub_template)
  return job_file


def get_num_jobs(job_name, username):
  len_usrn = len(username)
  num_running_jobs = int(
      subprocess.check_output(
          f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
          shell=True,
      ).decode('utf-8').replace('\n', ''))
  with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
    max_num_parallel_jobs = int(f.read())

  return num_running_jobs, max_num_parallel_jobs


def main():
  import glob
  num_repetitions = 1
  benchmark = 'full'
  experiment = 'pdm_lite'
  model_dir = '/cluster/work/andrebw/carla_garage_main/logs'
  code_root = '/cluster/work/andrebw/carla_garage'
  carla_root = '/cluster/work/andrebw/carla_garage/carla/CARLA_Leaderboard_20'
  partition = 'GPUQ'
  username = 'andrebw'
  experiment_name_stem = f'{experiment}_{benchmark}'
  exp_names_tmp = []
  for i in range(num_repetitions):
    exp_names_tmp.append(experiment_name_stem + f'_e{i}')
  # route_path = f'leaderboard/data/{benchmark}_split/'
  # route_path = '/cluster/work/andrebw/carla_garage/data/50x36_town_13'
  # route_path = '/cluster/work/andrebw/carla_garage/data/town13_selection'
  route_path = '/cluster/work/andrebw/carla_garage/leaderboard/data_validation'
  route_pattern = '*.xml'

  carla_world_port_start = 10000
  carla_streaming_port_start = 20000
  carla_tm_port_start = 30000

  epochs = ['model_0030']
  job_nr = 0
  for epoch in epochs:
    # Root folder in which each of the evaluation seeds will be stored
    experiment_name_root = experiment_name_stem + '_' + epoch
    exp_names = []
    for name in exp_names_tmp:
      exp_names.append(name + '_' + epoch)

    checkpoint = experiment
    checkpoint_new_name = checkpoint + '_' + epoch

    # Links the model file into team_code
    copy_model = False

    if copy_model:
      # copy checkpoint to my folder
      cmd = f'mkdir -p {code_root}/team_code/checkpoints/{checkpoint_new_name}'
      print(cmd)
      os.system(cmd)
      cmd = f'cp {model_dir}/{checkpoint}/config.pickle team_code/checkpoints/{checkpoint_new_name}/'
      print(cmd)
      os.system(cmd)
      cmd = f'ln -sf {model_dir}/{checkpoint}/{epoch}.pth team_code/checkpoints/{checkpoint_new_name}/model.pth'
      print(cmd)
      os.system(cmd)

    route_files = glob.glob(f"{route_path}/**/*.xml", recursive=True) 
    print("Num route files:", len(route_files))
    
    for exp_name in exp_names:
      bash_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/run_bashs')
      results_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/results')
      logs_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/logs')
      bash_save_dir.mkdir(parents=True, exist_ok=True)
      results_save_dir.mkdir(parents=True, exist_ok=True)
      logs_save_dir.mkdir(parents=True, exist_ok=True)

    meta_jobs = {}

    for exp_name in exp_names:
      for route_file in route_files:
        route = Path(route_file).stem

        bash_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/run_bashs')
        results_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/results')
        logs_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/logs')

        commands = []

        # Finds a free port
        commands.append(
            f'FREE_WORLD_PORT=`comm -23 <(seq {carla_world_port_start} {carla_world_port_start + 400} | sort) '
            f'<(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`')
        commands.append("echo 'World Port:' $FREE_WORLD_PORT")
        commands.append(
            f'FREE_STREAMING_PORT=`comm -23 <(seq {carla_streaming_port_start} {carla_streaming_port_start + 400} '
            f'| sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`')
        commands.append('echo "Launching CARLA with world-port=$FREE_WORLD_PORT and streaming-port=$FREE_STREAMING_PORT"')
        commands.append(f'sh {carla_root}/CarlaUE4.sh --world-port=$FREE_WORLD_PORT -RenderOffScreen -nosound -graphicsadapter=0 -carla-streaming-port=$FREE_STREAMING_PORT &')
        commands.append('sleep 20')  # Waits for CARLA to finish starting
        create_run_eval_bash(bash_save_dir,
                             results_save_dir,
                             route_file,
                             route,
                             checkpoint,
                             logs_save_dir,
                             carla_tm_port_start,
                             benchmark=benchmark,
                             carla_root=carla_root)
        commands.append(f'chmod u+x {bash_save_dir}/eval_{route}.sh')
        commands.append(f'./{bash_save_dir}/eval_{route}.sh $FREE_WORLD_PORT')
        commands.append('sleep 2')

        # carla_world_port_start += 50
        # carla_streaming_port_start += 50
        # carla_tm_port_start += 50

        job_file = make_jobsub_file(commands=commands,
                                    job_number=job_nr,
                                    exp_name=experiment_name_stem,
                                    exp_root_name=experiment_name_root,
                                    partition=partition)
        result_file = f'{results_save_dir}/{route}.json'

        # Wait until submitting new jobs that the #jobs are at below max
        num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=experiment_name_stem, username=username)
        print(f'{num_running_jobs}/{max_num_parallel_jobs} jobs are running...')
        while num_running_jobs >= max_num_parallel_jobs:
          num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=experiment_name_stem, username=username)
          time.sleep(1)
        time.sleep(1)
        print(f'Submitting job {job_nr}/{len(route_files) * num_repetitions}: {job_file}')
        jobid = subprocess.check_output(f'sbatch {job_file}', shell=True).decode('utf-8').strip().rsplit(' ',
                                                                                                         maxsplit=1)[-1]
        meta_jobs[jobid] = (False, job_file, result_file, 0)

        job_nr += 1

  training_finished = False
  while not training_finished:
    num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=experiment_name_stem, username=username)
    print(f'{num_running_jobs} jobs are running...')
    time.sleep(10)

    # resubmit unfinished jobs
    for k in list(meta_jobs.keys()):
      job_finished, job_file, result_file, resubmitted = meta_jobs[k]
      need_to_resubmit = False
      if not job_finished and resubmitted < 5:
        # check whether job is running
        if int(subprocess.check_output(f'squeue | grep {k} | wc -l', shell=True).decode('utf-8').strip()) == 0:
          # check whether result file is finished?
          if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f_result:
              evaluation_data = ujson.load(f_result)
            progress = evaluation_data['_checkpoint']['progress']

            if len(progress) < 2 or progress[0] < progress[1]:
              need_to_resubmit = True
            else:
              for record in evaluation_data['_checkpoint']['records']:
                if record['status'] == 'Failed - Agent couldn\'t be set up':
                  need_to_resubmit = True
                  print('Resubmit - Agent not setup')
                elif record['status'] == 'Failed':
                  need_to_resubmit = True
                elif record['status'] == 'Failed - Simulation crashed':
                  need_to_resubmit = True
                elif record['status'] == 'Failed - Agent crashed':
                  need_to_resubmit = True

            if not need_to_resubmit:
              # delete old job
              print(f'Finished job {job_file}')
              meta_jobs[k] = (True, None, None, 0)
          else:
            need_to_resubmit = True

      if need_to_resubmit:
        # Remove crashed results file
        if os.path.exists(result_file):
          print('Remove file: ', result_file)
          Path(result_file).unlink()
        num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=experiment_name_stem, username=username)
        print(f'{num_running_jobs}/{max_num_parallel_jobs} jobs are running...')
        while num_running_jobs >= max_num_parallel_jobs:
          print("Waiting for other jobs...")
          num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=experiment_name_stem, username=username)
          time.sleep(5)
        time.sleep(1)
        print(f'resubmit sbatch {job_file}')
        jobid = subprocess.check_output(f'sbatch {job_file}', shell=True).decode('utf-8').strip().rsplit(' ',
                                                                                                         maxsplit=1)[-1]
        meta_jobs[jobid] = (False, job_file, result_file, resubmitted + 1)
        meta_jobs[k] = (True, None, None, 0)
      time.sleep(2)
    time.sleep(8)

    if num_running_jobs == 0:
      training_finished = True

  print('Evaluation finished.')
  # print('Evaluation finished. Start parsing results.')
  # eval_root = f'{code_root}/evaluation/{experiment_name_root}'
  # subprocess.check_call(
  #     f'python {code_root}/tools/result_parser.py --xml {code_root}/leaderboard/data/{benchmark}.xml '
  #     f'--results {eval_root} --log_dir {eval_root} --town_maps {code_root}/leaderboard/data/town_maps_xodr '
  #     f'--map_dir {code_root}/leaderboard/data/town_maps_tga --device cpu '
  #     f'--map_data_folder {code_root}/tools/proxy_simulator/map_data --subsample 1 --strict --visualize_infractions',
  #     stdout=sys.stdout,
  #     stderr=sys.stderr,
  #     shell=True)


if __name__ == '__main__':
  main()
